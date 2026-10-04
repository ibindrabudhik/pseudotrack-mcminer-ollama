#!/usr/bin/env python3
"""McMiner-S: mine one misconception per code, with a local Ollama model.

Usage:
    python run_infer_misc.py
        --ollama-model qwen3.6-mcminer:latest
        --template zeroshot
        --template-dir prompt_templates/mining-pseudocode
        --problems-file dataset/pseudocode_track/problems_pseudocode.json
        --input-dir dataset/pseudocode_track/pseudocode_codes
        --output-dir results/single

    Correct (NONE) codes, appended to an existing predictions.json:
        ... --input-dir dataset/pseudocode_track/pseudocode_codes_none
            --append-results

Local Ollama only: no provider switch, no API key. Reasoning is controlled per
model by THINK_BY_MODEL in utils/ollama_client.py, not by a flag here.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import xml.etree.ElementTree as ET
from tqdm import tqdm
import re
from datetime import datetime
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ollama_client
from utils.ollama_client import OllamaClient, OllamaError
import rag_retrieval  # RAG retrieval loader / context formatter (src/rag_retrieval.py)
import ref_retrieval  # APR reference-code loader / context formatter (src/ref_retrieval.py)


def load_prompt_template(template_name: str = "zeroshot", template_dir: str = "prompt_templates/mining") -> str:
    """Load mining prompt template from external markdown file."""
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Map template names to files
    template_files = {
        "zeroshot": "zeroshot.md",
        "zeroshot-rag": "zeroshot-rag.md",  # RAG-augmented (retrieved-context) mining
        "zeroshot-ref": "zeroshot-ref.md",  # APR-reference-augmented ({reference_code}) mining
        "zeroshot-rag-ref": "zeroshot-rag-ref.md",  # combined: RAG top-k + APR reference
        "zeroshot-no-reasoning": "zeroshot-no-reasoning.md",
        "fewshot": "fewshot.md",  # backward compatibility
        "fewshot-no-reasoning": "fewshot-no-reasoning.md"  # backward compatibility
    }
    
    if template_name not in template_files:
        raise ValueError(f"Unknown template name: {template_name}. Available: {list(template_files.keys())}")
    
    template_file = template_files[template_name]
    template_path = os.path.join(script_dir, template_dir, template_file)
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        # Remove markdown header if present (first line starting with #)
        lines = template_content.split('\n')
        if lines and lines[0].startswith('#'):
            lines = lines[1:]  # Remove header line
            if lines and lines[0].strip() == '':  # Remove empty line after header
                lines = lines[1:]
        
        return '\n'.join(lines)
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Template file not found: {template_path}")


def load_json_data(file_path: str) -> Dict[str, Any]:
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_corrupted_codes(input_dir: str) -> List[Dict[str, Any]]:
    """Load all corrupted code files from the input directory."""
    corrupted_codes = []
    
    # Look for JSON files in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.json') and filename not in ['summary.json', 'filtering_report.json']:
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                # Extract problem_id from filename if not in data
                if 'problem_id' not in data:
                    match = re.search(r'problem_(\d+)', filename)
                    if match:
                        data['problem_id'] = int(match.group(1))
                
                # Add filename for tracking
                data['source_file'] = filename
                corrupted_codes.append(data)
    
    return corrupted_codes


def create_mining_prompt(template: str, problem: Dict[str, Any], student_code: str,
                         rag_ctx: str = "", ref_ctx: str = "") -> str:
    """Create a prompt for inferring misconceptions from code."""
    prompt = template

    # Replace placeholders
    prompt = prompt.replace("{problem_description}", problem.get("description", ""))
    prompt = prompt.replace("{problem_title}", f"Problem {problem.get('id', 'Unknown')}")
    prompt = prompt.replace("{student_code}", student_code)
    # RAG: the {retrieved_context} slot exists only in the -rag template; harmless no-op otherwise.
    prompt = prompt.replace("{retrieved_context}", rag_ctx)
    # REF: the {reference_code} slot exists only in the -ref template; harmless no-op otherwise.
    prompt = prompt.replace("{reference_code}", ref_ctx)

    return prompt


def parse_mining_response(response: str) -> Dict[str, Any]:
    """Parse LLM response for inferred misconceptions."""
    # Some providers (e.g. qwen3/OpenRouter reasoning runs) can return a None/empty body when the
    # token budget is spent on thinking. Coerce to a string so the regex parsing below never raises
    # "expected string or bytes-like object" — an empty body simply parses as no misconception.
    if not isinstance(response, str):
        response = "" if response is None else str(response)

    result = {
        "reasoning": "",
        "misconceptions": [],
        "metadata": {},
        "raw_response": response,
        "parse_success": False,
        "analysis": ""
    }

    try:
        # Extract reasoning (if present)
        reasoning_match = re.search(r'<reasoning>\s*(.*?)\s*</reasoning>', response, re.DOTALL)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
        
        # Extract misconception directly (no wrapper tag)
        misconception_match = re.search(r'<misconception>\s*(.*?)\s*</misconception>', response, re.DOTALL)
        if misconception_match:
            misconception_text = misconception_match.group(1).strip()
            
            if misconception_text.upper() == "NONE":
                result["misconceptions"] = []
                result["no_predicted_misconceptions"] = True
            else:
                misconception = {}
                
                # Extract fields (description and explanation plus metadata fields)
                desc_match = re.search(r'<description>\s*(.*?)\s*</description>', misconception_text, re.DOTALL)
                if desc_match:
                    misconception['description'] = desc_match.group(1).strip()
                
                explanation_match = re.search(r'<explanation>\s*(.*?)\s*</explanation>', misconception_text, re.DOTALL)
                if explanation_match:
                    misconception['explanation'] = explanation_match.group(1).strip()
                
                # Extract metadata fields (for consistency with multi-code version)
                # First check if they're in the misconception block
                type_match = re.search(r'<type>\s*(.*?)\s*</type>', misconception_text, re.DOTALL)
                if type_match:
                    misconception['misconception_type'] = type_match.group(1).strip()
                
                error_match = re.search(r'<error_type>\s*(.*?)\s*</error_type>', misconception_text, re.DOTALL)
                if error_match:
                    misconception['error_type'] = error_match.group(1).strip()
                    
                confidence_match = re.search(r'<confidence>\s*(.*?)\s*</confidence>', misconception_text, re.DOTALL)
                if confidence_match:
                    misconception['confidence'] = confidence_match.group(1).strip()
                
                if 'description' in misconception:  # Only add if we found at least a description
                    result["misconceptions"].append(misconception)
                
                result["no_predicted_misconceptions"] = len(result["misconceptions"]) == 0
        
        # Extract metadata
        metadata_match = re.search(r'<metadata>\s*(.*?)\s*</metadata>', response, re.DOTALL)
        if metadata_match:
            metadata_text = metadata_match.group(1).strip()
            result["metadata"]["raw"] = metadata_text
            
            # Parse simple key: value pairs
            for line in metadata_text.split('\n'):
                line = line.strip()
                if ':' in line and not line.startswith('#'):
                    key, value = line.split(':', 1)
                    result["metadata"][key.strip()] = value.strip()
            
            # If misconceptions exist but are missing metadata fields, copy from global metadata
            if result["misconceptions"]:
                for misconception in result["misconceptions"]:
                    if "misconception_type" not in misconception and "misconception_type" in result["metadata"]:
                        misconception["misconception_type"] = result["metadata"]["misconception_type"]
                    if "error_type" not in misconception and "error_type" in result["metadata"]:
                        misconception["error_type"] = result["metadata"]["error_type"]
                    if "confidence" not in misconception and "confidence_level" in result["metadata"]:
                        misconception["confidence"] = result["metadata"]["confidence_level"]
        
        # Check if parsing was successful
        if misconception_match:
            result["parse_success"] = True
            
        # Ensure no_predicted_misconceptions field is always present
        if "no_predicted_misconceptions" not in result:
            result["no_predicted_misconceptions"] = len(result["misconceptions"]) == 0
            
    except Exception as e:
        result["metadata"]["parse_error"] = str(e)
        print(f"Warning: Failed to parse LLM response: {e}")
    
    return result


def get_model_name(args, llm_client=None) -> str:
    """The Ollama model tag this run used."""
    return args.ollama_model


def get_llm_kwargs(args) -> Dict[str, Any]:
    """Generation settings. Reasoning is NOT set here.

    It is decided per model by think_for() in utils/ollama_client.py, because
    the right value differs by model family and getting it wrong fails silently
    rather than loudly. See that module's docstring.
    """
    return {
        "model": args.ollama_model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }


def log_model_access(args) -> None:
    think = ollama_client.think_for(args.ollama_model)
    print(f"🔌 OLLAMA -> model={args.ollama_model}  host={args.ollama_host}  "
          f"think={think!r}")


def create_llm_client(args) -> Any:
    """The only client this bundle has. Local Ollama, native API, no vendor SDK."""
    return ollama_client.OllamaClient(model=args.ollama_model, host=args.ollama_host)


def generate_mining_batches(corrupted_codes: List[Dict[str, Any]], problems: Dict[str, Any],
                              template: str, none_only: bool = False,
                              rag_index=None, rag_top_k: int = 3,
                              ref_index=None) -> List[Tuple[Dict[str, Any], List[Dict[str, str]]]]:
    """Generate batches of prompts for inferring misconceptions."""
    batches = []
    
    mode_str = "NONE-only" if none_only else "all"
    print(f"Preparing mining prompts ({mode_str} mode)...")
    
    for code_data in tqdm(corrupted_codes, desc="Processing corrupted codes"):
        problem_id = code_data.get('problem_id')
        
        # Skip if no problem_id
        if problem_id is None:
            continue
        
        # Get problem context
        problem = None
        
        # Handle dict format with string or int keys
        if isinstance(problems, dict):
            # Try string key first
            problem = problems.get(str(problem_id))
            # Try int key if string didn't work
            if not problem:
                problem = problems.get(problem_id)
        elif isinstance(problems, list):
            # Handle list format
            for p in problems:
                if p.get('id') == problem_id:
                    problem = p
                    break
        
        # If no problem found, create minimal context
        if not problem:
            problem = {'id': problem_id, 'description': f'Problem {problem_id}'}
        
        # Process each solution in the corrupted code file
        solutions = code_data.get('solutions', [])
        for sol_idx, solution in enumerate(solutions):
            generated_code = solution.get('generated_code', '')
            original_code = generated_code  # Store original value for metadata
            
            # Check if this is a NONE sample
            is_none_sample = not original_code or original_code == 'NONE'
            
            # Apply filtering based on mode
            if none_only and not is_none_sample:
                # Skip non-NONE samples when in NONE-only mode
                continue
            elif not none_only and is_none_sample:
                # Handle NONE cases by substituting with correct solution (existing behavior)
                if problem and 'solutions' in problem and problem['solutions']:
                    # Use the first correct solution as substitute
                    generated_code = problem['solutions'][0]
                    print(f"Substituting NONE with correct solution for problem {problem_id}, solution {sol_idx}")
                else:
                    # If no correct solution available, skip
                    print(f"Warning: No correct solution available for problem {problem_id}, skipping NONE sample")
                    continue
            elif none_only and is_none_sample:
                # Process NONE samples in NONE-only mode - substitute with correct solution
                if problem and 'solutions' in problem and problem['solutions']:
                    generated_code = problem['solutions'][0]
                    print(f"Processing NONE sample with correct solution for problem {problem_id}, solution {sol_idx}")
                else:
                    print(f"Warning: No correct solution available for problem {problem_id}, skipping NONE sample")
                    continue
            
            # RAG: fetch retrieved-context block for this code (corrupted -> submission CSV,
            # NONE-substituted correct code -> correct CSV). Empty string when RAG is off.
            rag_ctx = ""
            if rag_index is not None:
                rag_ctx = rag_index.format_context(
                    problem_id,
                    misconception_id=code_data.get('misconception_id'),
                    is_correct=is_none_sample,
                    top_k=rag_top_k,
                )

            # REF: fetch the APR reference-code block for this code. Empty string when REF is off,
            # NO_REFERENCE line for correct/NONE-substituted codes (no misconception_id to join on).
            ref_ctx = ""
            if ref_index is not None:
                ref_ctx = (ref_retrieval.NO_REFERENCE if is_none_sample
                           else ref_index.format_context(
                               problem_id, misconception_id=code_data.get('misconception_id')))

            # Create mining prompt
            prompt = create_mining_prompt(template, problem, generated_code, rag_ctx, ref_ctx)

            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Store metadata
            # For NONE-substituted samples, the ground truth becomes "no misconception" since we're analyzing correct code
            if is_none_sample:
                # Original labels (what failed to generate)
                original_misc_id = code_data.get('misconception_id')
                original_misc_desc = code_data.get('misconception_description', '')
                # Ground truth for analysis (correct code = no misconception)
                gt_misconception = "NONE"
                gt_misconception_desc = "No misconception - correct code"
            else:
                # Regular corrupted code - ground truth matches original
                original_misc_id = code_data.get('misconception_id')
                original_misc_desc = code_data.get('misconception_description', '')
                gt_misconception = original_misc_id
                gt_misconception_desc = original_misc_desc
            
            metadata = {
                "source_file": code_data.get('source_file', ''),
                "problem_id": problem_id,
                "solution_index": sol_idx,
                "original_misconception_id": original_misc_id,
                "original_misconception_desc": original_misc_desc,
                "gt_misconception": gt_misconception,  # Ground truth for evaluation
                "gt_misconception_desc": gt_misconception_desc,
                "was_none_substituted": is_none_sample,
                "code_type": "correct" if is_none_sample else "corrupted",
            }
            
            batches.append((metadata, messages))
    
    print(f"Prepared {len(batches)} mining prompts ({mode_str} mode)")
    return batches


def save_mining_results(results: List[Dict[str, Any]], output_dir: str, 
                          llm_info: Dict[str, str], run_info: Dict[str, Any], 
                          append_mode: bool = False):
    """Save mining results to JSON files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual results
    new_predictions = []
    
    for i, result in enumerate(results):
        metadata = result["metadata"]
        parsed = result["parsed_response"]
        
        # Copy metadata fields into each misconception for compatibility
        enhanced_misconceptions = []
        for misc in parsed["misconceptions"]:
            enhanced_misc = misc.copy()
            # Add metadata fields that are now outside misconception tag
            if "misconception_type" in parsed["metadata"]:
                enhanced_misc["misconception_type"] = parsed["metadata"]["misconception_type"]
            if "error_type" in parsed["metadata"]:
                enhanced_misc["error_type"] = parsed["metadata"]["error_type"]
            if "confidence_level" in parsed["metadata"]:
                enhanced_misc["confidence"] = parsed["metadata"]["confidence_level"]  # Map to 'confidence' for compatibility
            enhanced_misconceptions.append(enhanced_misc)
        
        prediction = {
            "prediction_id": f"{metadata['source_file']}_{metadata['solution_index']}",
            "source_file": metadata["source_file"],
            "problem_id": metadata["problem_id"],
            "solution_index": metadata["solution_index"],
            "original_misconception": {
                "id": metadata.get("original_misconception_id"),
                "description": metadata.get("original_misconception_desc")
            },
            "ground_truth_misconception": {
                "id": metadata.get("gt_misconception"),
                "description": metadata.get("gt_misconception_desc")
            },
            "predicted_misconceptions": enhanced_misconceptions,
            "no_predicted_misconceptions": parsed.get("no_predicted_misconceptions", False),
            "reasoning": parsed["reasoning"],
            "analysis": parsed["analysis"],
            "parse_success": parsed["parse_success"],
            "was_none_substituted": metadata.get("was_none_substituted", False),
            "code_type": metadata.get("code_type", "corrupted"),
            "metadata": parsed["metadata"]
        }
        
        new_predictions.append(prediction)
    
    # Handle appending to existing predictions
    output_file = os.path.join(output_dir, "predictions.json")
    all_predictions = []
    
    if append_mode and os.path.exists(output_file):
        # Load existing predictions
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_predictions = json.load(f)
            
            # Create set of existing prediction IDs to avoid duplicates
            existing_ids = {pred["prediction_id"] for pred in existing_predictions}
            
            # Add existing predictions
            all_predictions.extend(existing_predictions)
            
            # Add new predictions, skipping duplicates
            duplicates_skipped = 0
            for pred in new_predictions:
                if pred["prediction_id"] not in existing_ids:
                    all_predictions.append(pred)
                else:
                    duplicates_skipped += 1
            
            print(f"Appended {len(new_predictions) - duplicates_skipped} new predictions, skipped {duplicates_skipped} duplicates")
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load existing predictions ({e}), saving new predictions only")
            all_predictions = new_predictions
    else:
        all_predictions = new_predictions
    
    # Save all predictions to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(all_predictions)} total predictions to {output_file}")
    
    # Calculate statistics - use new predictions for run-specific stats
    new_predictions_count = len(new_predictions)
    total_predictions = len(all_predictions)
    successful_parses = sum(1 for p in new_predictions if p["parse_success"])
    total_misconceptions_found = sum(len(p["predicted_misconceptions"]) for p in new_predictions)
    avg_misconceptions = total_misconceptions_found / new_predictions_count if new_predictions_count > 0 else 0
    
    # Count NONE substitutions in new predictions
    none_substitutions = sum(1 for result in results if result["metadata"].get("was_none_substituted", False))
    
    # Count code types
    correct_codes = sum(1 for p in new_predictions if p.get("code_type") == "correct")
    corrupted_codes = sum(1 for p in new_predictions if p.get("code_type") == "corrupted")
    
    # Count codes with no predicted misconceptions by type
    correct_codes_no_misc = sum(1 for p in new_predictions if p.get("code_type") == "correct" and len(p["predicted_misconceptions"]) == 0)
    corrupted_codes_no_misc = sum(1 for p in new_predictions if p.get("code_type") == "corrupted" and len(p["predicted_misconceptions"]) == 0)
    
    # Save summary
    summary = {
        "run_timestamp": run_info["timestamp"],
        "input_directory": run_info["input_dir"],
        "output_directory": output_dir,
        "llm_provider": llm_info["provider"],
        "llm_model": llm_info["model"],
        "processing_mode": llm_info["processing_mode"],
        "reasoning_enabled": llm_info.get("reasoning_enabled", False),
        "template_type": run_info["template_type"],
        "statistics": {
            "new_codes_analyzed": new_predictions_count,
            "total_codes_in_file": total_predictions,
            "correct_codes_analyzed": correct_codes,
            "corrupted_codes_analyzed": corrupted_codes,
            "successful_parses": successful_parses,
            "parse_success_rate": successful_parses / new_predictions_count if new_predictions_count > 0 else 0,
            "total_misconceptions_found": total_misconceptions_found,
            "average_misconceptions_per_code": avg_misconceptions,
            "codes_with_no_misconceptions": sum(1 for p in new_predictions if len(p["predicted_misconceptions"]) == 0),
            "correct_codes_no_misconceptions": correct_codes_no_misc,
            "corrupted_codes_no_misconceptions": corrupted_codes_no_misc,
            "none_substitutions": none_substitutions,
            "none_substitution_rate": none_substitutions / new_predictions_count if new_predictions_count > 0 else 0,
            "append_mode": append_mode
        }
    }
    
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary Statistics:")
    print(f"  - New codes analyzed: {summary['statistics']['new_codes_analyzed']}")
    print(f"  - Total codes in file: {summary['statistics']['total_codes_in_file']}")
    print(f"  - Code types: {summary['statistics']['correct_codes_analyzed']} correct, {summary['statistics']['corrupted_codes_analyzed']} corrupted")
    print(f"  - Parse success rate: {summary['statistics']['parse_success_rate']:.2%}")
    print(f"  - Total misconceptions found: {total_misconceptions_found}")
    print(f"  - Average misconceptions per code: {avg_misconceptions:.2f}")
    print(f"  - Codes with no misconceptions: {summary['statistics']['codes_with_no_misconceptions']}")
    if correct_codes > 0:
        print(f"    - Correct codes: {summary['statistics']['correct_codes_no_misconceptions']}/{correct_codes} ({summary['statistics']['correct_codes_no_misconceptions']/correct_codes*100:.1f}%)")
    if corrupted_codes > 0:
        print(f"    - Corrupted codes: {summary['statistics']['corrupted_codes_no_misconceptions']}/{corrupted_codes} ({summary['statistics']['corrupted_codes_no_misconceptions']/corrupted_codes*100:.1f}%)")
    print(f"  - NONE substitutions: {summary['statistics']['none_substitutions']} ({summary['statistics']['none_substitution_rate']:.1%})")
    if append_mode:
        print(f"  - Mode: Appended to existing results")


def main():
    parser = argparse.ArgumentParser(description="Infer misconceptions from corrupted code using LLMs")
    
    # Model selection -- local Ollama only. There is no provider switch: this
    # bundle has exactly one backend, and the flag that used to choose between
    # five of them was the main source of "which endpoint am I actually hitting"
    # confusion.
    parser.add_argument("--ollama-model", required=True,
                       help="Ollama model tag, e.g. qwen3.6-mcminer:latest")
    parser.add_argument("--ollama-host", default="http://localhost:11434",
                       help="Ollama server. A trailing /v1 is stripped: this uses the "
                            "native /api/chat, not the OpenAI-compatible shim.")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=4000,
                       help="Response cap (sent to Ollama as options.num_predict)")

    # Reasoning settings
    parser.add_argument("--reasoning", action="store_true",
                       help="Enable reasoning mode (only for compatible models)")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="medium",
                       help="Reasoning effort level for OpenAI models")
    parser.add_argument("--no-reasoning-template", action="store_true",
                       help="Use the no-reasoning variant of the mining template")
    
    # Template settings
    parser.add_argument("--template",
                       choices=["zeroshot", "zeroshot-rag", "zeroshot-ref", "zeroshot-rag-ref", "zeroshot-no-reasoning", "fewshot", "fewshot-no-reasoning"],
                       help="Specify template name explicitly (overrides --no-reasoning-template)")
    parser.add_argument("--template-dir", default="prompt_templates/mining",
                       help="Directory containing template files")
    
    # Data paths
    parser.add_argument("--input-dir", 
                       default="mining_misconceptions/data/corrupted_codes/corrupted_codes_anthropic",
                       help="Directory containing corrupted code files")
    parser.add_argument("--problems-file", 
                       default="mining_misconceptions/data/problems_processed.json",
                       help="Path to problems JSON file")
    
    # Output
    parser.add_argument("--output-dir", 
                       default="mining_misconceptions/test_results/mined_misconceptions",
                       help="Output directory for results")
    
    # Processing options
    parser.add_argument("--use-batch", action="store_true",
                       help="Enable batch processing")
    parser.add_argument("--max-files", type=int, default=None,
                       help="Limit number of files to process (for testing)")
    parser.add_argument("--debug-prompt", action="store_true",
                       help="Save first generated prompt to debug_mining.txt")
    parser.add_argument("--none-only", action="store_true",
                       help="Process only NONE samples (codes that were previously skipped)")
    parser.add_argument("--append-results", action="store_true",
                       help="Append results to existing predictions file instead of overwriting")

    # RAG: retrieval-augmented mining. Omit --rag-csv for the exact non-RAG baseline.
    parser.add_argument("--rag-csv", default=None,
                       help="Submission retrieval CSV; enables RAG context. Use with a *-rag template.")
    parser.add_argument("--rag-correct-csv", default=None,
                       help="Correct-code retrieval CSV (for NONE-substituted correct codes)")
    parser.add_argument("--rag-top-k", type=int, default=3,
                       help="Number of retrieved misconceptions to inject (default 3)")

    # REF: APR reference-code injection. Omit --ref-csv for the exact non-reference baseline.
    parser.add_argument("--ref-csv", default=None,
                       help="APR reference CSV; enables reference-code context. Use with a *-ref template.")
    parser.add_argument("--ref-column", default="Reference_Code",
                       help="Which reference column to inject (Reference_Code | Best_Reference | Repaired_Code)")

    args = parser.parse_args()
    
    # Validate argument combinations
    if args.none_only and not args.append_results:
        print("⚠️  Warning: --none-only is typically used with --append-results to add to existing predictions")
    
    if args.none_only:
        print("🔍 Running in NONE-only mode: processing only samples that were previously 'NONE'")
    
    # Load data
    print("Loading data...")
    try:
        problems = load_json_data(args.problems_file)
        corrupted_codes = load_corrupted_codes(args.input_dir)
        
        if args.max_files:
            corrupted_codes = corrupted_codes[:args.max_files]
            print(f"Limited to {len(corrupted_codes)} files for testing")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    print(f"Loaded {len(corrupted_codes)} corrupted code files")
    print(f"Loaded {len(problems)} problems for context")
    
    # Load appropriate template
    if args.template:
        template_type = args.template
    else:
        template_type = "zeroshot-no-reasoning" if args.no_reasoning_template else "zeroshot"
    
    try:
        template = load_prompt_template(template_type, args.template_dir)
    except FileNotFoundError as e:
        print(f"❌ Template loading error: {e}")
        return 1

    # Build RAG index if requested (omit --rag-csv => exact non-RAG baseline)
    rag_index = None
    if args.rag_csv:
        rag_index = rag_retrieval.load_index(args.rag_csv, args.rag_correct_csv, args.rag_top_k)
        if "{retrieved_context}" not in template:
            print("⚠️  --rag-csv set but template has no {retrieved_context} slot; "
                  "use a *-rag template (e.g. --template zeroshot-rag) for context to be injected.")

    # Build REF index if requested (omit --ref-csv => exact non-reference baseline)
    ref_index = None
    if args.ref_csv:
        ref_index = ref_retrieval.load_reference_index(args.ref_csv, args.ref_column)
        if "{reference_code}" not in template:
            print("⚠️  --ref-csv set but template has no {reference_code} slot; "
                  "use a *-ref template (e.g. --template zeroshot-ref) for the reference to be injected.")

    # Create LLM client
    log_model_access(args)
    try:
        llm_client = create_llm_client(args)
    except Exception as e:
        print(f"Error creating LLM client: {e}")
        return 1
    
    # Generate mining prompts
    batches = generate_mining_batches(corrupted_codes, problems, template, args.none_only,
                                      rag_index=rag_index, rag_top_k=args.rag_top_k,
                                      ref_index=ref_index)
    
    if not batches:
        print("No valid codes to analyze.")
        return 1
    
    # Debug first prompt if requested
    if args.debug_prompt and batches:
        debug_prompt = batches[0][1][0]["content"]
        with open("debug_mining.txt", 'w', encoding='utf-8') as f:
            f.write(debug_prompt)
        print(f"🔍 Debug: First prompt saved to debug_mining.txt ({len(debug_prompt)} chars)")
    
    # Process mining requests
    all_results = []

    print(f"🔄 Processing {len(batches)} requests")
    kwargs = get_llm_kwargs(args)
    failures = 0
    for metadata, messages in tqdm(batches, desc="Inferring misconceptions"):
        try:
            response = llm_client.create_message(messages, kwargs=kwargs)
        except OllamaError as e:
            # An empty or truncated reply is an ERROR here, not a "no
            # misconception found". The old client returned "" for both, and the
            # parser recorded that as a confident negative -- a silent false
            # negative that looked like a real result.
            print(f"\n  ! mining error: {e}")
            response = ""
            failures += 1
        parsed_response = parse_mining_response(response)
        all_results.append({
            "metadata": metadata,
            "parsed_response": parsed_response
        })
    if failures:
        print(f"\n⚠️  {failures}/{len(batches)} requests failed and were recorded as "
              f"empty predictions. Do not read those as negatives.")

    # Save results
    print("\nSaving results...")
    
    llm_info = {
        "provider": "ollama",
        "model": get_model_name(args, llm_client),
        "processing_mode": "individual",
        "think": repr(ollama_client.think_for(args.ollama_model))
    }
    
    run_info = {
        "timestamp": datetime.now().isoformat(),
        "input_dir": args.input_dir,
        "template_type": template_type
    }
    
    save_mining_results(all_results, args.output_dir, llm_info, run_info, args.append_results)
    
    print("✅ Misconception Mining completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())