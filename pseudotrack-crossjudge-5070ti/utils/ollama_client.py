"""Ollama client for the McMiner pseudocode track.

This replaces the four-provider `llm_clients.py` this bundle inherited. Nothing
here talks to a hosted API: the only endpoint is a local Ollama server, reached
through its NATIVE /api/chat, with `requests` as the sole dependency. There is
no OpenAI / Anthropic / Gemini / vLLM SDK involved, and no OpenAI-compatible
/v1 shim in between.

Why the native API rather than Ollama's /v1 endpoint
----------------------------------------------------
The /v1 shim has no first-class way to turn reasoning off, so the previous
version of this bundle smuggled per-family switches through the OpenAI request
body: a top-level `reasoning_effort` for gpt-oss, and a nested
`extra_body.chat_template_kwargs.enable_thinking` for qwen3.6. Both worked, but
neither is checkable -- if a switch failed to reach the model you found out from
a 10-minute call or an empty answer, not from an error.

/api/chat takes a `think` field directly, and the reply separates `thinking`
from `content`, so this module can *see* whether the model reasoned and say so.

The `think` value is per-model, and getting it wrong is expensive
----------------------------------------------------------------
Measured against Ollama 0.32.14 on this dataset:

    gpt-oss, think=false   -> content='' , thinking='The user says...',
                              done_reason='length'. The model reasoned until the
                              budget ran out and returned NOTHING. The miner
                              records that as "no misconception predicted" and
                              the judge parser turns it into a score. A silent
                              false negative, not an error.
    gpt-oss, think='low'   -> content='OK' in 17 tokens, 5.5s.

So `think: false` does not mean "do not think" for a reasoning-only model like
gpt-oss; it means "no explicit level", and the model falls back to its default.
Reasoning models need a LEVEL. Non-reasoning-by-default models like qwen3.6 need
the boolean. THINK_BY_MODEL below encodes that, and `think_for()` is the single
place it is decided.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests

DEFAULT_HOST = "http://localhost:11434"

# Per-model reasoning control. Matched on a prefix of the model name so that
# 'gpt-oss-judge:latest', 'gpt-oss:20b' and 'gpt-oss-mcminer' all resolve --
# Ollama reports models with a :tag suffix, and an exact-match lookup silently
# returns nothing, leaving the model at its default. That failure is invisible.
THINK_BY_MODEL = {
    "gpt-oss": "low",   # reasoning model: needs a LEVEL, not False (see module docstring)
    "qwen": False,      # thinking off; measured >10 min/call with it on
}


def think_for(model: str) -> Any:
    """Reasoning setting for a model name, or None to leave the default alone.

    OLLAMA_THINK overrides everything, for measuring the cost of thinking:
        OLLAMA_THINK=low    OLLAMA_THINK=false    OLLAMA_THINK=default
    """
    override = os.getenv("OLLAMA_THINK")
    if override is not None:
        low = override.strip().lower()
        if low in ("default", ""):
            return None
        if low in ("false", "0", "no", "off"):
            return False
        if low in ("true", "1", "yes", "on"):
            return True
        return low  # a level: low / medium / high

    name = (model or "").lower()
    for prefix, value in THINK_BY_MODEL.items():
        if name.startswith(prefix):
            return value
    return None


class OllamaError(RuntimeError):
    """An Ollama request failed, or returned something unusable."""


class OllamaClient:
    """One local Ollama model, over /api/chat.

    Deliberately small: this pipeline only ever sends a single user message and
    reads back text. Everything else the old multi-provider client carried
    (tools, batch APIs, guided decoding, the Responses API) had no call site
    here.
    """

    def __init__(self, model: str, host: Optional[str] = None,
                 timeout: Optional[int] = None, retries: int = 2):
        if not model:
            raise ValueError("OllamaClient needs a model name")
        self.model_name = model
        # Accept an OpenAI-style .../v1 base url and strip it, so a stale
        # OLLAMA_BASE_URL from an older config does not produce a confusing 404.
        raw = (host or os.getenv("OLLAMA_HOST_URL") or DEFAULT_HOST).rstrip("/")
        self.host = raw[:-3].rstrip("/") if raw.endswith("/v1") else raw
        self.timeout = timeout or int(os.getenv("OLLAMA_TIMEOUT", "900"))
        self.retries = retries
        self._warned_empty = False

    # ------------------------------------------------------------------ core
    def create_message(self, messages: List[Dict[str, str]],
                       kwargs: Optional[Dict[str, Any]] = None,
                       **_ignored) -> str:
        """Send one chat request, return the assistant's text.

        `**_ignored` swallows the reasoning/budget_tokens/reasoning_effort
        arguments the inherited call sites still pass for other providers.
        Reasoning is controlled by `think` here, not by those.
        """
        kwargs = kwargs or {}
        model = kwargs.get("model") or self.model_name

        options: Dict[str, Any] = {}
        if kwargs.get("temperature") is not None:
            options["temperature"] = float(kwargs["temperature"])
        # Ollama calls the response cap num_predict, not max_tokens.
        if kwargs.get("max_tokens") is not None:
            options["num_predict"] = int(kwargs["max_tokens"])

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if options:
            payload["options"] = options
        think = think_for(model)
        if think is not None:
            payload["think"] = think

        data = self._post("/api/chat", payload)
        message = data.get("message") or {}
        content = (message.get("content") or "").strip()

        if not content:
            # Distinguish the two ways this happens, because the fix differs and
            # the caller cannot tell them apart from an empty string. Neither is
            # allowed to pass silently: the miner would record "no misconception
            # found" and the judge parser would turn it into a score.
            thinking = (message.get("thinking") or "").strip()
            reason = data.get("done_reason")
            if thinking and reason == "length":
                raise OllamaError(
                    "%s spent its whole %s-token budget reasoning and returned no answer "
                    "(done_reason=length, think=%r). Reasoning models need a LEVEL "
                    "('low'), not False. Set OLLAMA_THINK=low or fix THINK_BY_MODEL."
                    % (model, options.get("num_predict", "?"), think))
            raise OllamaError(
                "%s returned empty content (done_reason=%r, think=%r). Raise the token "
                "budget or check the model." % (model, reason, think))
        return content

    def create_batch_messages(self, batch: List[List[Dict[str, str]]],
                              **kwargs) -> List[str]:
        """Sequential fallback for the --use-batch flag.

        Ollama has no batch endpoint and serves one request at a time anyway, so
        'batch' here means 'a loop'. Kept so --use-batch does not crash.
        """
        inner = kwargs.pop("kwargs", None) or kwargs
        return [self.create_message(m, kwargs=inner) for m in batch]

    # ------------------------------------------------------------------ http
    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.host + path
        last = None
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(url, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                last = OllamaError("cannot reach Ollama at %s: %s" % (url, e))
            else:
                if r.status_code == 200:
                    try:
                        data = r.json()
                    except ValueError as e:
                        raise OllamaError("Ollama returned non-JSON: %s" % e)
                    if isinstance(data, dict) and data.get("error"):
                        raise OllamaError("Ollama error: %s" % data["error"])
                    return data
                # A model that cannot be allocated comes back as a 500 with the
                # allocator's message. Surface it verbatim -- it names the exact
                # buffer size that failed, which is what tells you whether to
                # free RAM or lower num_ctx.
                body = (r.text or "")[:500]
                last = OllamaError("Ollama HTTP %d: %s" % (r.status_code, body))
                if r.status_code < 500:
                    raise last
            if attempt < self.retries:
                time.sleep(2 * (attempt + 1))
        raise last


def create_client(model: str, host: Optional[str] = None) -> OllamaClient:
    return OllamaClient(model=model, host=host)


def probe(model: str, host: Optional[str] = None) -> Dict[str, Any]:
    """One tiny real call. Returns timing and whether the model reasoned.

    Used by preflight: it is the only way to know a `think` setting took effect
    before committing to hours of mining.
    """
    client = OllamaClient(model=model, host=host)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "stream": False,
        "options": {"num_predict": 300, "temperature": 0},
    }
    think = think_for(model)
    if think is not None:
        payload["think"] = think
    t0 = time.time()
    data = client._post("/api/chat", payload)
    message = data.get("message") or {}
    return {
        "model": model,
        "think": think,
        "seconds": time.time() - t0,
        "content": (message.get("content") or "").strip(),
        "thinking": (message.get("thinking") or "").strip(),
        "done_reason": data.get("done_reason"),
        "eval_count": data.get("eval_count"),
    }


def list_models(host: Optional[str] = None) -> List[str]:
    raw = (host or os.getenv("OLLAMA_HOST_URL") or DEFAULT_HOST).rstrip("/")
    if raw.endswith("/v1"):
        raw = raw[:-3].rstrip("/")
    r = requests.get(raw + "/api/tags", timeout=15)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


def show(model: str, host: Optional[str] = None) -> Dict[str, Any]:
    raw = (host or os.getenv("OLLAMA_HOST_URL") or DEFAULT_HOST).rstrip("/")
    if raw.endswith("/v1"):
        raw = raw[:-3].rstrip("/")
    r = requests.post(raw + "/api/show", json={"model": model}, timeout=30)
    r.raise_for_status()
    return r.json()


def num_ctx_of(model: str, host: Optional[str] = None) -> Optional[int]:
    """What context window was this model actually built with?

    A prompt longer than num_ctx is truncated by the server with no error, and
    the model then answers confidently about a program it only half saw.
    """
    try:
        info = show(model, host)
    except Exception:  # noqa: BLE001
        return None
    for line in (info.get("parameters") or "").splitlines():
        bits = line.split()
        if len(bits) == 2 and bits[0] == "num_ctx":
            return int(bits[1])
    for key, value in (info.get("model_info") or {}).items():
        if key.endswith(".context_length"):
            return int(value)
    return None


def unload(model: str, host: Optional[str] = None) -> None:
    """Drop a model from memory now, instead of waiting out Ollama's 5-minute
    keep-alive. The miner and judge cannot co-reside on a small card."""
    raw = (host or os.getenv("OLLAMA_HOST_URL") or DEFAULT_HOST).rstrip("/")
    if raw.endswith("/v1"):
        raw = raw[:-3].rstrip("/")
    try:
        requests.post(raw + "/api/chat",
                      json={"model": model, "messages": [], "keep_alive": 0},
                      timeout=60)
    except requests.RequestException:
        pass
