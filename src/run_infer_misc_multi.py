#!/usr/bin/env python3
"""
Modified version of run_infer_misc.py that can handle multiple codes per misconception.

This script groups corrupted codes by misconception and processes them together,
allowing the LLM to identify patterns across multiple examples of the same misconception.
Codes with "NONE" as the generated code are replaced with correct solutions, ensuring bags contain only
actual code implementations.

Usage:
    # Group by misconception (default behavior)
    python run_infer_misc_multi.py --llm anthropic --template zeroshot-no-reasoning-multi
    
    # Control grouping parameters
    python run_infer_misc_multi.py --max-codes-per-group 5 --max-problems-per-group 3
    
    # Include correct codes (no misconceptions)
    python run_infer_misc_multi.py --include-correct-codes --correct-code-ratio 0.2
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
from collections import defaultdict
import random

from dotenv import load_dotenv
load_dotenv(override=True)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_clients import OpenAIClient, VLLMClient, AnthropicClient, GeminiClient
import rag_retrieval  # RAG retrieval loader / context formatter (src/rag_retrieval.py)
import ref_retrieval  # APR reference-code loader / context formatter (src/ref_retrieval.py)


def load_prompt_template(template_name: str = "zeroshot-no-reasoning-multi", template_dir: str = "prompt_templates/mining") -> str:
    """Load mining prompt template from external markdown file."""
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Map template names to files
    template_files = {
        "zeroshot-no-reasoning-multi": "zeroshot-no-reasoning-multi.md",
        "zeroshot-no-reasoning-multi-rag": "zeroshot-no-reasoning-multi-rag.md",  # RAG-augmented
        "zeroshot-no-reasoning-multi-ref": "zeroshot-no-reasoning-multi-ref.md",  # APR-reference-augmented
        "zeroshot-no-reasoning-multi-rag-ref": "zeroshot-no-reasoning-multi-rag-ref.md",  # combined RAG + reference
        "zeroshot-no-reasoning-general": "zeroshot-no-reasoning-general.md",
        "zeroshot-multi": "zeroshot-multi.md",
        "fewshot-no-reasoning-multi": "fewshot-no-reasoning-multi.md",
        "zeroshot": "zeroshot.md",
        "zeroshot-no-reasoning": "zeroshot-no-reasoning.md",
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


def get_correct_solutions(problems: Dict[str, Any]) -> Dict[int, List[str]]:
    """Extract correct solutions from problems data."""
    correct_solutions = {}
    
    # Handle both list and dict formats
    if isinstance(problems, list):
        # Handle list format
        for problem_data in problems:
            if isinstance(problem_data, dict) and 'id' in problem_data:
                problem_id = problem_data['id']
                solutions = []
                
                # Check various possible fields for solutions
                if 'solutions' in problem_data:
                    sol_data = problem_data['solutions']
                    if isinstance(sol_data, list):
                        solutions.extend(sol_data)
                    elif isinstance(sol_data, str):
                        solutions.append(sol_data)
                elif 'solution' in problem_data:
                    solutions.append(problem_data['solution'])
                elif 'code' in problem_data:
                    solutions.append(problem_data['code'])
                
                if solutions:
                    correct_solutions[problem_id] = solutions
                    
    elif isinstance(problems, dict):
        # Handle dict format (original logic)
        for problem_id, problem_data in problems.items():
            # Handle both string and int problem IDs
            pid = int(problem_id) if isinstance(problem_id, str) and problem_id.isdigit() else problem_id
            
            solutions = []
            # Look for correct solutions in the problem data
            if isinstance(problem_data, dict):
                # Check various possible fields for solutions
                if 'solutions' in problem_data:
                    sol_data = problem_data['solutions']
                    if isinstance(sol_data, list):
                        solutions.extend(sol_data)
                    elif isinstance(sol_data, str):
                        solutions.append(sol_data)
                elif 'solution' in problem_data:
                    solutions.append(problem_data['solution'])
                elif 'code' in problem_data:
                    solutions.append(problem_data['code'])
            
            if solutions:
                correct_solutions[pid] = solutions
    
    return correct_solutions


def create_correct_only_bags(problems: Dict[str, Any], num_bags: int,
                           bag_size_mode: str, bag_size_min: int = None,
                           bag_size_max: int = None, bag_size_fixed: int = None,
                           cover_all: bool = False, passes: int = 1) -> List[List[Dict[str, Any]]]:
    """Create bags containing only correct code samples.

    cover_all: if True, ignore num_bags and instead PARTITION every available correct code
      (one per problem) into bags of bag_size, so every correct code is evaluated at least once.
      `passes` re-partitions the pool that many times (different shufflings) for more bags /
      statistical weight. Default (cover_all=False) keeps the original ratio-based random sampling.
    """
    correct_solutions = get_correct_solutions(problems)
    bags = []

    # Get all available problem IDs with correct solutions
    available_problems = list(correct_solutions.keys())

    def _make_code(problem_id):
        solution = random.choice(correct_solutions[problem_id])
        return {
            'problem_id': problem_id,
            'misconception_id': None,  # No misconception
            'misconception_description': 'No misconception - correct code',
            'solutions': [{'generated_code': solution, 'is_correct': True}],
            'source_file': f'correct_problem_{problem_id}',
        }

    if cover_all:
        if not available_problems:
            return bags
        bag_size = bag_size_max if bag_size_mode == "range" else bag_size_fixed
        bag_size = bag_size or 5
        for _ in range(max(1, passes)):
            pool = available_problems[:]
            random.shuffle(pool)
            for i in range(0, len(pool), bag_size):
                chunk = pool[i:i + bag_size]
                if chunk:
                    bags.append([_make_code(pid) for pid in chunk])
        return bags

    for _ in range(num_bags):
        if not available_problems:
            break
            
        # Determine bag size
        if bag_size_mode == "range":
            bag_size = random.randint(bag_size_min, bag_size_max)
        else:
            bag_size = bag_size_fixed
        
        # Sample problems for this bag
        num_problems = min(bag_size, len(available_problems))
        selected_problems = random.sample(available_problems, num_problems)
        
        bag = []
        for problem_id in selected_problems:
            solutions = correct_solutions[problem_id]
            # Pick one solution randomly
            solution = random.choice(solutions)
            
            # Create a code data structure similar to corrupted codes
            code_data = {
                'problem_id': problem_id,
                'misconception_id': None,  # No misconception
                'misconception_description': 'No misconception - correct code',
                'solutions': [{
                    'generated_code': solution,
                    'is_correct': True
                }],
                'source_file': f'correct_problem_{problem_id}'
            }
            bag.append(code_data)
        
        if bag:
            bags.append(bag)
    
    return bags


def replace_none_with_correct(code_group: List[Dict[str, Any]], 
                            correct_solutions: Dict[int, List[str]]) -> Tuple[List[Dict[str, Any]], bool]:
    """Replace NONE codes with correct solutions and track if all were NONE."""
    updated_group = []
    all_were_none = True
    
    for code_data in code_group:
        problem_id = code_data.get('problem_id')
        solutions = code_data.get('solutions', [])
        
        updated_solutions = []
        for solution in solutions:
            if solution.get('generated_code') == 'NONE' and problem_id in correct_solutions:
                # Replace NONE with a correct solution
                correct_code = random.choice(correct_solutions[problem_id])
                updated_solutions.append({
                    'generated_code': correct_code,
                    'is_correct': True,
                    'original_was_none': True
                })
            else:
                # This solution was not NONE
                if solution.get('generated_code') != 'NONE':
                    all_were_none = False
                updated_solutions.append(solution)
        
        # Update the code data
        updated_code_data = code_data.copy()
        updated_code_data['solutions'] = updated_solutions
        updated_group.append(updated_code_data)
    
    return updated_group, all_were_none


def group_codes_by_misconception(corrupted_codes: List[Dict[str, Any]], 
                                bag_size_mode: str,
                                bag_size_min: int = None,
                                bag_size_max: int = None,
                                bag_size_fixed: int = None,
                                max_problems_per_group: int = None) -> Dict[int, List[Dict[str, Any]]]:
    """Group corrupted codes by misconception ID. Bag size limiting happens later during batch generation."""
    
    misconception_groups = defaultdict(list)
    
    for code_data in corrupted_codes:
        misc_id = code_data.get('misconception_id')
        if misc_id is not None:
            misconception_groups[misc_id].append(code_data)
    
    # Apply max_problems_per_group constraint if specified (but not bag size limits)
    if max_problems_per_group is not None:
        filtered_groups = {}
        for misc_id, codes in misconception_groups.items():
            # Shuffle to get diverse problem coverage
            random.shuffle(codes)
            
            # Group by problem to ensure diversity
            problem_groups = defaultdict(list)
            for code in codes:
                problem_groups[code.get('problem_id')].append(code)
            
            # Select codes ensuring problem diversity (but keeping ALL codes within the problem limit)
            selected_codes = []
            problems_used = 0
            
            for problem_id, problem_codes in problem_groups.items():
                if problems_used >= max_problems_per_group:
                    break
                
                # Take first code from this problem
                if problem_codes:
                    selected_codes.append(problem_codes[0])
                    problems_used += 1
            
            filtered_groups[misc_id] = selected_codes
        
        return filtered_groups
    else:
        # No problem limit - return all codes grouped by misconception
        return dict(misconception_groups)


def create_multi_mining_prompt(template: str, problems: Dict[str, Any],
                              code_group: List[Dict[str, Any]],
                              rag_index=None, rag_top_k: int = 3,
                              ref_index=None) -> str:
    """Create a prompt for inferring misconceptions from multiple code examples."""
    
    # Prepare problem context
    problem_contexts = []
    for code_data in code_group:
        problem_id = code_data.get('problem_id')
        problem = get_problem_by_id(problems, problem_id)
        if problem:
            problem_contexts.append(f"**Problem {problem_id}:** {problem.get('description', f'Problem {problem_id}')}")
    
    # Remove duplicates while preserving order
    seen_problems = set()
    unique_contexts = []
    for context in problem_contexts:
        if context not in seen_problems:
            unique_contexts.append(context)
            seen_problems.add(context)
    
    problem_context = "\n\n".join(unique_contexts)
    
    # Prepare code examples
    code_examples = []
    for i, code_data in enumerate(code_group):
        problem_id = code_data.get('problem_id')
        solutions = code_data.get('solutions', [])
        
        for sol_idx, solution in enumerate(solutions):
            generated_code = solution.get('generated_code', '')
            if generated_code and generated_code != 'NONE':
                block = (
                    f"**Student Code {len(code_examples) + 1} for Problem {problem_id}:**\n"
                    f"```python\n{generated_code}\n```"
                )
                # RAG: append this code's retrieved-context shortlist (corrupted -> submission CSV,
                # correct code -> correct CSV). Only when RAG is enabled.
                if rag_index is not None:
                    is_correct = (str(code_data.get('source_file', '')).startswith('correct_')
                                  or code_data.get('misconception_id') in (None, "", "None"))
                    ctx = rag_index.format_context(
                        problem_id,
                        misconception_id=code_data.get('misconception_id'),
                        is_correct=is_correct,
                        top_k=rag_top_k,
                    )
                    block += f"\n\n*Retrieved Context for Student Code {len(code_examples) + 1}:*\n{ctx}"
                # REF: append this code's APR reference solution. Correct-only bag codes have no
                # misconception_id, so they never join and render the NO_REFERENCE line.
                if ref_index is not None:
                    is_correct = (str(code_data.get('source_file', '')).startswith('correct_')
                                  or code_data.get('misconception_id') in (None, "", "None"))
                    ref_ctx = (ref_retrieval.NO_REFERENCE if is_correct
                               else ref_index.format_context(
                                   problem_id, misconception_id=code_data.get('misconception_id')))
                    block += f"\n\n*Reference Solution for Student Code {len(code_examples) + 1}:*\n{ref_ctx}"
                code_examples.append(block)
    
    corrupted_codes = "\n\n".join(code_examples)
    num_codes = len(code_examples)
    
    # Determine misconception block format for single misconception
    misconception_block = """<description>[Clear description of the ONE shared misconception, starting with "The student believes"]</description>
<explanation>[Explain how the given code exhibits the misconception]</explanation>"""
    
    # Replace placeholders
    prompt = template.replace("{problem_context}", problem_context)
    prompt = prompt.replace("{num_codes}", str(num_codes))
    prompt = prompt.replace("{corrupted_codes}", corrupted_codes)
    prompt = prompt.replace("{misconception_block}", misconception_block)

    return prompt


def load_frozen_bags(reuse_path: str, corrupted_codes: List[Dict[str, Any]],
                     problems: Dict[str, Any]) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Reconstruct the EXACT bags used in a prior multi_predictions.json run, keyed off each
    record's group_info.source_files, instead of re-sampling bag composition. Use this to
    re-mine with a different --rag-csv/--ref-csv (or a new template) without paying for a fresh
    random bag draw and without invalidating the pairing with the prior run's results.
    """
    prior = json.load(open(reuse_path, 'r', encoding='utf-8'))
    by_source = {c.get('source_file'): c for c in corrupted_codes}
    correct_solutions = get_correct_solutions(problems)

    frozen = []
    missing = 0
    for rec in prior:
        group_info = rec.get('group_info', {}) or {}
        source_files = group_info.get('source_files', [])
        code_group = []
        for sf in source_files:
            if sf.startswith('correct_problem_'):
                pid = int(sf.replace('correct_problem_', ''))
                sols = correct_solutions.get(pid)
                if not sols:
                    missing += 1
                    continue
                code_group.append({
                    'problem_id': pid, 'misconception_id': None,
                    'misconception_description': 'No misconception - correct code',
                    'solutions': [{'generated_code': sols[0], 'is_correct': True}],
                    'source_file': sf,
                })
            else:
                cd = by_source.get(sf)
                if cd is None:
                    missing += 1
                    continue
                code_group.append(cd)
        frozen.append((rec, code_group))

    if missing:
        print(f"⚠️  reuse-bags-from: {missing} source file(s) referenced in {reuse_path} were not "
              f"found in --input-dir / correct pool and were skipped from their bag.")
    print(f"🔒 reuse-bags-from: loaded {len(frozen)} frozen bags from {reuse_path} "
          f"(bag composition NOT re-sampled)")
    return frozen


def generate_frozen_multi_mining_batches(frozen_bags: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]],
                                        problems: Dict[str, Any], template: str,
                                        rag_index=None, rag_top_k: int = 3,
                                        ref_index=None) -> List[Tuple[Dict[str, Any], List[Dict[str, str]]]]:
    """Build (metadata, messages) batches for a frozen bag set. Preserves each bag's original
    group_type / misconception_id / gt_misconception so alignment + eval scripts downstream see
    the same shape as a normal (freshly-sampled) run.
    """
    batches = []
    for rec, code_group in tqdm(frozen_bags, desc="Re-mining frozen bags"):
        if not code_group:
            continue
        prompt = create_multi_mining_prompt(template, problems, code_group,
                                           rag_index=rag_index, rag_top_k=rag_top_k, ref_index=ref_index)
        messages = [{"role": "user", "content": prompt}]
        gi = rec.get('group_info', {}) or {}
        metadata = {
            "group_type": rec.get('group_type'),
            "misconception_id": rec.get('misconception_id'),
            "original_misconception_desc": rec.get('original_misconception', ''),
            "num_codes": gi.get('num_codes', sum(len(c.get('solutions', [])) for c in code_group)),
            "num_problems": gi.get('num_problems', len(set(c.get('problem_id') for c in code_group))),
            "source_files": gi.get('source_files', [c.get('source_file', '') for c in code_group]),
            "problem_ids": gi.get('problem_ids', [c.get('problem_id') for c in code_group]),
            "gt_misconception": gi.get('gt_misconception', rec.get('misconception_id', 'NONE')),
        }
        batches.append((metadata, messages))
    return batches


def get_problem_by_id(problems: Dict[str, Any], problem_id: int) -> Optional[Dict[str, Any]]:
    """Get problem by ID from problems data."""
    if isinstance(problems, dict):
        # Try string key first
        problem = problems.get(str(problem_id))
        # Try int key if string didn't work
        if not problem:
            problem = problems.get(problem_id)
        return problem
    elif isinstance(problems, list):
        # Handle list format
        for p in problems:
            if p.get('id') == problem_id:
                return p
    return None


def parse_multi_mining_response(response: str) -> Dict[str, Any]:
    """Parse LLM response for inferred misconceptions (multi-code version)."""
    # Some providers (e.g. qwen3/OpenRouter reasoning runs) can return a None/empty body when the
    # token budget is spent on thinking. Coerce to a string so the regex parsing below never raises
    # "expected string or bytes-like object" — an empty body simply parses as no misconception.
    if not isinstance(response, str):
        response = "" if response is None else str(response)

    result = {
        "reasoning": "",
        "analysis": "",
        "misconceptions": [],
        "metadata": {},
        "raw_response": response,
        "parse_success": False
    }

    try:
        # Extract reasoning (if present)
        reasoning_match = re.search(r'<reasoning>\s*(.*?)\s*</reasoning>', response, re.DOTALL)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
            
        # Extract analysis (if present)
        analysis_match = re.search(r'<analysis>\s*(.*?)\s*</analysis>', response, re.DOTALL)
        if analysis_match:
            result["analysis"] = analysis_match.group(1).strip()
        
        # Extract misconception directly (no wrapper tag)
        misconception_match = re.search(r'<misconception>\s*(.*?)\s*</misconception>', response, re.DOTALL)
        if misconception_match:
            misconception_text = misconception_match.group(1).strip()
            
            if misconception_text.upper() == "NONE":
                result["misconceptions"] = []
                result["no_predicted_misconceptions"] = True
            else:
                misconception = {}
                
                # Extract fields
                desc_match = re.search(r'<description>\s*(.*?)\s*</description>', misconception_text, re.DOTALL)
                if desc_match:
                    misconception['description'] = desc_match.group(1).strip()
                
                explanation_match = re.search(r'<explanation>\s*(.*?)\s*</explanation>', misconception_text, re.DOTALL)
                if explanation_match:
                    misconception['explanation'] = explanation_match.group(1).strip()
                
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
        
        # Check if parsing was successful
        if misconception_match:
            result["parse_success"] = True
            
    except Exception as e:
        result["metadata"]["parse_error"] = str(e)
        print(f"Warning: Failed to parse LLM response: {e}")
    
    return result


def generate_multi_mining_batches(misconception_groups: Dict[int, List[Dict[str, Any]]], 
                                 problems: Dict[str, Any], template: str,
                                 correct_bags_ratio: float = 0.15,
                                 bag_size_mode: str = "fixed",
                                 bag_size_min: int = None,
                                 bag_size_max: int = None,
                                 bag_size_fixed: int = None,
                                 bags_per_misconception: int = None,
                                 correct_bags_only: bool = False,
                                 num_correct_bags_override: int = None,
                                 correct_bags_cover_all: bool = False,
                                 correct_bags_passes: int = 1,
                                 rag_index=None,
                                 rag_top_k: int = 3,
                                 ref_index=None) -> List[Tuple[Dict[str, Any], List[Dict[str, str]]]]:
    """Generate batches of prompts for multi-code misconception inference.

    correct_bags_only: if True, emit ONLY correct-only bags (no misconception bags) — used to
      add correct-bag evaluation incrementally on top of an existing misconception run.
    num_correct_bags_override: if set, use this exact number of correct-only bags instead of
      deriving it from correct_bags_ratio * misconception_bags.
    """
    batches = []
    
    print("Preparing multi-code mining prompts...")
    
    # Get correct solutions for NONE replacement
    correct_solutions = get_correct_solutions(problems)
    
    # Process misconception groups first to count actual bags
    actual_misconception_bags = 0
    
    # First pass: count how many bags we'll actually create
    for misc_id, code_group in misconception_groups.items():
        if not code_group:
            continue
            
        if bags_per_misconception is None:
            # Calculate maximum possible bags
            min_bag_size = bag_size_min if bag_size_mode == "range" else bag_size_fixed
            if min_bag_size and len(code_group) >= min_bag_size:
                # For maximum bags, we need unique problems
                unique_problems = list(set(code.get('problem_id') for code in code_group))
                max_possible_bags = len(unique_problems) // min_bag_size
                actual_misconception_bags += max_possible_bags
        else:
            actual_misconception_bags += bags_per_misconception
    
    # Calculate number of correct-only bags based on actual misconception bags
    num_correct_bags = int(actual_misconception_bags * correct_bags_ratio)
    if num_correct_bags_override is not None:
        num_correct_bags = num_correct_bags_override

    # Create correct-only bags
    if correct_bags_cover_all:
        print(f"Creating correct-only bags in COVER-ALL mode ({correct_bags_passes} pass(es) over all "
              f"correct codes) — every correct code is bagged at least once...")
    else:
        print(f"Attempting to create {num_correct_bags} correct-only bags ({correct_bags_ratio:.0%} of {actual_misconception_bags} estimated misconception bags)...")
    correct_only_bags = create_correct_only_bags(
        problems, num_correct_bags,
        bag_size_mode, bag_size_min, bag_size_max, bag_size_fixed,
        cover_all=correct_bags_cover_all, passes=correct_bags_passes
    )
    
    # Process correct-only bags
    actual_correct_bags = 0
    for i, bag in enumerate(correct_only_bags):
        prompt = create_multi_mining_prompt(template, problems, bag, rag_index=rag_index, rag_top_k=rag_top_k, ref_index=ref_index)
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        metadata = {
            "group_type": "correct_only",
            "misconception_id": None,
            "original_misconception_desc": "No misconception - all correct codes",
            "num_codes": sum(len(code.get('solutions', [])) for code in bag),
            "num_problems": len(set(code.get('problem_id') for code in bag)),
            "source_files": [code.get('source_file', '') for code in bag],
            "problem_ids": [code.get('problem_id') for code in bag],
            "gt_misconception": "NONE"  # Ground truth: no misconception
        }
        
        batches.append((metadata, messages))
        actual_correct_bags += 1

    if correct_bags_only:
        print(f"correct-bags-only mode: prepared {actual_correct_bags} correct-only bag(s); skipping misconception bags")
        return batches

    # Process misconception groups with multiple bags per misconception
    for misc_id, code_group in tqdm(misconception_groups.items(), desc="Processing misconception groups"):
        if not code_group:
            continue
        
        # Determine number of bags for this misconception
        if bags_per_misconception is None:
            # Create as many bags as possible with unique problems
            # Group codes by problem ID
            codes_by_problem = defaultdict(list)
            for code in code_group:
                codes_by_problem[code.get('problem_id')].append(code)
            
            # Get list of unique problems with their codes
            unique_problem_codes = []
            for problem_id, problem_codes in codes_by_problem.items():
                # Take the first code for each problem
                unique_problem_codes.append(problem_codes[0])
            
            # Shuffle for variety
            random.shuffle(unique_problem_codes)
            
            # Create bags until we run out of codes
            bag_idx = 0
            remaining_codes = unique_problem_codes.copy()
            
            while remaining_codes:
                # Determine bag size
                if bag_size_mode == "range":
                    current_bag_size = random.randint(bag_size_min, bag_size_max)
                    min_size = bag_size_min
                else:
                    current_bag_size = bag_size_fixed
                    min_size = bag_size_fixed
                
                # Check if we have enough codes left
                if len(remaining_codes) < min_size:
                    break
                
                # Take codes for this bag
                bag_codes = remaining_codes[:current_bag_size]
                remaining_codes = remaining_codes[current_bag_size:]
                
                # Replace NONE codes with correct solutions
                processed_codes, all_were_none = replace_none_with_correct(bag_codes, correct_solutions)
                
                # Create mining prompt for this group
                prompt = create_multi_mining_prompt(template, problems, processed_codes, rag_index=rag_index, rag_top_k=rag_top_k, ref_index=ref_index)

                messages = [
                    {"role": "user", "content": prompt}
                ]

                # Store metadata
                metadata = {
                    "group_type": "misconception",
                    "misconception_id": misc_id,
                    "bag_index": bag_idx,
                    "original_misconception_desc": code_group[0].get('misconception_description', ''),
                    "num_codes": sum(len(code.get('solutions', [])) for code in processed_codes),
                    "num_problems": len(set(code.get('problem_id') for code in processed_codes)),
                    "source_files": [code.get('source_file', '') for code in processed_codes],
                    "problem_ids": [code.get('problem_id') for code in processed_codes],
                    "gt_misconception": "NONE" if all_were_none else misc_id  # Ground truth
                }

                batches.append((metadata, messages))
                bag_idx += 1

        else:
            # Fixed number of bags per misconception
            for bag_idx in range(bags_per_misconception):
                # For multiple bags, resample the codes
                if bags_per_misconception > 1 and len(code_group) > 1:
                    # Determine bag size
                    if bag_size_mode == "range":
                        current_bag_size = random.randint(bag_size_min, bag_size_max)
                    else:
                        current_bag_size = bag_size_fixed
                    
                    # Sample codes for this bag (with replacement if necessary)
                    if len(code_group) >= current_bag_size:
                        sampled_codes = random.sample(code_group, current_bag_size)
                    else:
                        # If we don't have enough codes, sample with replacement
                        sampled_codes = random.choices(code_group, k=current_bag_size)
                else:
                    sampled_codes = code_group
                
                # Replace NONE codes with correct solutions
                processed_codes, all_were_none = replace_none_with_correct(sampled_codes, correct_solutions)
                
                # Create mining prompt for this group
                prompt = create_multi_mining_prompt(template, problems, processed_codes, rag_index=rag_index, rag_top_k=rag_top_k, ref_index=ref_index)

                messages = [
                    {"role": "user", "content": prompt}
                ]

                # Store metadata
                metadata = {
                    "group_type": "misconception",
                    "misconception_id": misc_id,
                    "bag_index": bag_idx,
                    "original_misconception_desc": code_group[0].get('misconception_description', ''),
                    "num_codes": sum(len(code.get('solutions', [])) for code in processed_codes),
                    "num_problems": len(set(code.get('problem_id') for code in processed_codes)),
                    "source_files": [code.get('source_file', '') for code in processed_codes],
                    "problem_ids": [code.get('problem_id') for code in processed_codes],
                    "gt_misconception": "NONE" if all_were_none else misc_id  # Ground truth
                }

                batches.append((metadata, messages))

    # Calculate actual counts
    actual_misconception_bags = len(batches) - actual_correct_bags
    
    print(f"Prepared {len(batches)} multi-code mining prompts ({actual_correct_bags} correct-only, {actual_misconception_bags} misconception)")
    if actual_correct_bags != num_correct_bags:
        print(f"  Note: Intended {num_correct_bags} correct-only bags, but created {actual_correct_bags} (limited by available problems)")
    
    return batches


def log_model_access(args) -> None:
    """Print exactly which provider/model/endpoint this run is about to hit."""
    if args.llm == "anthropic":
        model, endpoint = args.anthropic_model, "api.anthropic.com (native)"
    elif args.llm == "openai":
        model, endpoint = args.openai_model, "api.openai.com (native)"
    elif args.llm == "gemini":
        model, endpoint = args.gemini_model, "generativelanguage.googleapis.com (native)"
    elif args.llm == "openrouter":
        model, endpoint = args.openrouter_model, args.openrouter_base_url
    else:  # vllm
        model, endpoint = (args.vllm_model or "server-detected"), args.vllm_base_url
    print(f"🔌 LLM ACCESS -> provider={args.llm}  model={model}  endpoint={endpoint}  "
          f"reasoning={getattr(args, 'reasoning', False)}")


def create_llm_client(args) -> Any:
    """Create the appropriate LLM client based on arguments."""
    if args.llm == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return AnthropicClient(api_key=api_key)
    
    elif args.llm == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return OpenAIClient(api_key=api_key)
    
    elif args.llm == "vllm":
        if args.vllm_model:
            # Offline mode with direct model loading
            print(f"Initializing vLLM in offline mode with model: {args.vllm_model}")
            return VLLMClient(
                model=args.vllm_model,
                offline_mode=True,
                gpu_memory_utilization=0.9,
                max_model_len=32768 if '8b' in args.vllm_model.lower() else None
            )
        else:
            # Server mode (existing behavior)
            return VLLMClient(
                api_key=args.vllm_api_key or "NONE",
                base_url=args.vllm_base_url
            )
    
    elif args.llm == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        return GeminiClient(api_key=api_key, model=args.gemini_model)

    elif args.llm == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        # OpenRouter is OpenAI-compatible; force Chat Completions (it has no Responses API).
        return OpenAIClient(api_key=api_key, model=args.openrouter_model,
                            base_url=args.openrouter_base_url, use_responses_api=False)

    else:
        raise ValueError(f"Unsupported LLM: {args.llm}")


def get_llm_kwargs(args) -> Dict[str, Any]:
    """Get LLM-specific kwargs based on the selected provider."""
    if args.llm == "anthropic":
        # Check if model supports reasoning and if reasoning is enabled
        reasoning_supported = args.anthropic_model in ["claude-3-7-sonnet-latest", "claude-sonnet-4-0", "claude-opus-4-0", "claude-sonnet-4-5"]
        use_reasoning = args.reasoning and reasoning_supported
        return {
            "model": args.anthropic_model,
            "temperature": 0.1 ,
            "max_tokens": 4000,
            "reasoning": use_reasoning,
            "budget_tokens": args.thinking_budget if use_reasoning else 1000
        }
    elif args.llm == "openai":
        # Check if model supports reasoning and if reasoning is enabled
        reasoning_supported = args.openai_model in ["o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini"]
        use_reasoning = args.reasoning and reasoning_supported
        return {
            "model": args.openai_model,
            "temperature": 0.1,
            "max_tokens": 4000,
            "reasoning": use_reasoning,
            "reasoning_effort": args.reasoning_effort
        }
    elif args.llm == "gemini":
        # Check if model supports thinking/reasoning and if reasoning is enabled
        gemini_thinking_models = ["gemini-2.5-flash-preview", "gemini-2.5-flash-exp", "gemini-2.5-flash",
                                 "gemini-2.5-pro", "gemini-2.5-pro-exp", "gemini-2.5-pro-exp-03-25"]
        reasoning_supported = any(thinking_model in args.gemini_model for thinking_model in gemini_thinking_models)
        use_reasoning = args.reasoning and reasoning_supported
        return {
            "model": args.gemini_model,
            "temperature": 0.1,
            "max_tokens": 4000,
            "reasoning": use_reasoning,
            "thinking_budget": args.thinking_budget if use_reasoning else 1000
        }
    elif args.llm == "openrouter":
        return {
            "model": args.openrouter_model,
            "temperature": 0.1,
            "max_tokens": 4000,
            "reasoning": args.reasoning,
            "reasoning_effort": args.reasoning_effort
        }
    else:  # vllm
        # Use Qwen3-recommended parameters if thinking mode is enabled
        if hasattr(args, 'vllm_enable_thinking') and args.vllm_enable_thinking:
            # Qwen3 thinking mode parameters
            # https://qwen.readthedocs.io/en/latest/deployment/vllm.html#thinking-non-thinking-modes
            return {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "max_tokens": 4000
            }
        else:
            # Qwen3 non-thinking or standard vLLM parameters
            return {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "max_tokens": 4000
        }


def get_model_name(args, llm_client=None) -> str:
    """Get the model name being used."""
    if args.llm == "anthropic":
        return args.anthropic_model
    elif args.llm == "openai":
        return args.openai_model
    elif args.llm == "gemini":
        return args.gemini_model
    elif args.llm == "openrouter":
        return args.openrouter_model
    else:  # vllm
        if llm_client and hasattr(llm_client, 'model_name') and llm_client.model_name:
            return llm_client.model_name
        else:
            return "vllm-served-model"


def save_multi_mining_results(results: List[Dict[str, Any]], output_dir: str,
                             llm_info: Dict[str, str], run_info: Dict[str, Any],
                             append: bool = False):
    """Save multi-code mining results to JSON files.

    append: if True, merge into an existing multi_predictions.json. Existing correct_only groups
      are dropped (replaced by the new ones); existing misconception groups are preserved.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Save individual results
    all_predictions = []
    
    for i, result in enumerate(results):
        metadata = result["metadata"]
        parsed = result["parsed_response"]
        
        # Copy metadata fields into each misconception for compatibility
        enhanced_misconceptions = []
        for misc in parsed["misconceptions"]:
            enhanced_misc = misc.copy()
            # Add metadata fields if not already present
            if "misconception_type" in parsed["metadata"] and "misconception_type" not in enhanced_misc:
                enhanced_misc["misconception_type"] = parsed["metadata"]["misconception_type"]
            if "error_type" in parsed["metadata"] and "error_type" not in enhanced_misc:
                enhanced_misc["error_type"] = parsed["metadata"]["error_type"]
            if "confidence_level" in parsed["metadata"] and "confidence" not in enhanced_misc:
                enhanced_misc["confidence"] = parsed["metadata"]["confidence_level"]  # Map to 'confidence' for compatibility
            enhanced_misconceptions.append(enhanced_misc)
        
        prediction = {
            "prediction_id": f"group_{metadata.get('group_type', 'unknown')}_{metadata.get('misconception_id', i)}_{metadata.get('bag_index', 0)}",
            "group_type": metadata.get("group_type", "unknown"),
            "misconception_id": metadata["misconception_id"],
            "problem_id": metadata["problem_ids"][0] if metadata["problem_ids"] else None,  # First problem for compatibility
            "original_misconception": {
                "id": metadata["misconception_id"],
                "description": metadata.get("original_misconception_desc")
            },
            "group_info": {
                "num_codes": metadata["num_codes"],
                "num_problems": metadata["num_problems"],
                "source_files": metadata["source_files"],
                "problem_ids": metadata["problem_ids"],
                "gt_misconception": metadata.get("gt_misconception"),  # Ground truth
                "bag_index": metadata.get("bag_index", 0)
            },
            "predicted_misconceptions": enhanced_misconceptions,
            "no_predicted_misconceptions": parsed.get("no_predicted_misconceptions", False),
            "reasoning": parsed.get("reasoning", ""),
            "analysis": parsed["analysis"],
            "parse_success": parsed["parse_success"],
            "metadata": parsed["metadata"]
        }
        
        all_predictions.append(prediction)

    # Save all predictions to a single file
    output_file = os.path.join(output_dir, "multi_predictions.json")

    if append and os.path.exists(output_file):
        try:
            existing = json.load(open(output_file, encoding="utf-8"))
        except Exception:
            existing = []
        new_types = {p.get("group_type") for p in all_predictions}
        # Keep existing groups whose type is NOT being regenerated in this run
        # (e.g. appending correct_only replaces old correct_only, preserves misconception).
        kept = [p for p in existing if p.get("group_type") not in new_types]
        print(f"Append mode: kept {len(kept)} existing group(s) of other type(s), "
              f"replacing {len(existing) - len(kept)} group(s) of type {new_types}.")
        all_predictions = kept + all_predictions

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_predictions)} group predictions to {output_file}")
    
    # Calculate statistics (use .get() defaults so a merged/appended predictions list with
    # slightly older-schema entries never crashes the summary).
    total_groups = len(all_predictions)
    successful_parses = sum(1 for p in all_predictions if p.get("parse_success"))
    total_misconceptions_found = sum(len(p.get("predicted_misconceptions", [])) for p in all_predictions)
    avg_misconceptions = total_misconceptions_found / total_groups if total_groups > 0 else 0
    total_codes_analyzed = sum(p.get("group_info", {}).get("num_codes", 0) for p in all_predictions)

    # Separate statistics for different group types
    correct_only_groups = [p for p in all_predictions if p.get("group_type") == "correct_only"]
    misconception_groups = [p for p in all_predictions if p.get("group_type") == "misconception"]

    # Count groups with no predicted misconceptions
    groups_with_no_predicted_misconceptions = sum(1 for p in all_predictions if p.get("no_predicted_misconceptions", False) or len(p.get("predicted_misconceptions", [])) == 0)
    correct_groups_no_misc = sum(1 for p in correct_only_groups if p.get("no_predicted_misconceptions", False) or len(p.get("predicted_misconceptions", [])) == 0)
    misconception_groups_no_misc = sum(1 for p in misconception_groups if p.get("no_predicted_misconceptions", False) or len(p.get("predicted_misconceptions", [])) == 0)
    
    # Save summary
    summary = {
        "run_timestamp": run_info["timestamp"],
        "input_directory": run_info["input_dir"],
        "output_directory": output_dir,
        "llm_provider": llm_info["provider"],
        "llm_model": llm_info["model"],
        "processing_mode": "multi-code-grouping",
        "template_type": run_info["template_type"],
        "grouping_params": {
            "bag_size_mode": run_info.get("bag_size_mode"),
            "bag_size_config": run_info.get("bag_size_config"),
            "max_problems_per_group": run_info.get("max_problems_per_group"),
            "correct_bags_ratio": run_info.get("correct_bags_ratio"),
            "bags_per_misconception": run_info.get("bags_per_misconception")
        },
        "statistics": {
            "total_groups": total_groups,
            "correct_only_groups": len(correct_only_groups),
            "misconception_groups": len(misconception_groups),
            "total_codes_analyzed": total_codes_analyzed,
            "successful_parses": successful_parses,
            "parse_success_rate": successful_parses / total_groups if total_groups > 0 else 0,
            "total_misconceptions_found": total_misconceptions_found,
            "average_misconceptions_per_group": avg_misconceptions,
            "groups_with_no_predicted_misconceptions": groups_with_no_predicted_misconceptions,
            "correct_groups_no_predicted_misconceptions": correct_groups_no_misc,
            "misconception_groups_no_predicted_misconceptions": misconception_groups_no_misc
        }
    }
    
    with open(os.path.join(output_dir, "multi_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary Statistics:")
    print(f"  - Total groups: {total_groups} ({len(correct_only_groups)} correct-only, {len(misconception_groups)} misconception)")
    print(f"  - Parse success rate: {summary['statistics']['parse_success_rate']:.2%}")  
    print(f"  - Total codes analyzed: {total_codes_analyzed}")
    print(f"  - Total misconceptions found: {total_misconceptions_found}")
    print(f"  - Average misconceptions per group: {avg_misconceptions:.2f}")
    print(f"  - Groups with no predicted misconceptions: {groups_with_no_predicted_misconceptions} ({groups_with_no_predicted_misconceptions/total_groups*100:.1f}%)")
    print(f"    - Correct-only groups: {correct_groups_no_misc}/{len(correct_only_groups)} ({correct_groups_no_misc/len(correct_only_groups)*100:.1f}%)" if correct_only_groups else "")
    print(f"    - Misconception groups: {misconception_groups_no_misc}/{len(misconception_groups)} ({misconception_groups_no_misc/len(misconception_groups)*100:.1f}%)" if misconception_groups else "")


def main():
    parser = argparse.ArgumentParser(description="Infer misconceptions from multiple corrupted codes using LLMs")
    
    # LLM selection
    parser.add_argument("--llm", choices=["anthropic", "openai", "vllm", "gemini", "openrouter"],
                       default="anthropic", help="LLM provider to use")
    parser.add_argument("--openrouter-model", default="anthropic/claude-sonnet-4.5",
                       help="OpenRouter model id, e.g. anthropic/claude-sonnet-4.5, openai/gpt-4o, google/gemini-2.5-flash")
    parser.add_argument("--openrouter-base-url", default="https://openrouter.ai/api/v1",
                       help="OpenRouter (OpenAI-compatible) base URL")

    # LLM-specific arguments
    parser.add_argument("--vllm-base-url", default="http://localhost:8000/v1",
                       help="Base URL for vLLM server (server mode)")
    parser.add_argument("--vllm-api-key", default="NONE",
                       help="API key for vLLM server (if needed)")
    parser.add_argument("--vllm-model", default=None,
                       help="vLLM model for offline mode (e.g., Qwen/Qwen3-8B). If specified, uses offline mode instead of server.")
    parser.add_argument("--vllm-enable-thinking", action="store_true",
                       help="Enable thinking mode for Qwen3 models in vLLM offline mode")
    parser.add_argument("--openai-model", default="gpt-4o",
                       help="OpenAI model name")
    parser.add_argument("--anthropic-model", default="claude-sonnet-4-5",
                       help="Anthropic model name")
    parser.add_argument("--gemini-model", 
                       choices=["gemini-2.5-pro-preview-06-05", "gemini-2.5-flash-preview-05-20", "gemini-2.0-flash",
                               "gemini-2.5-flash-preview", "gemini-2.5-flash-exp", "gemini-2.5-flash",
                               "gemini-2.5-pro", "gemini-2.5-pro-exp", "gemini-2.5-pro-exp-03-25"],
                       default="gemini-2.5-flash-preview",
                       help="Gemini model name")
    
    # Template settings
    parser.add_argument("--template",
                       choices=["zeroshot-no-reasoning-multi", "zeroshot-no-reasoning-multi-rag", "zeroshot-no-reasoning-multi-ref", "zeroshot-no-reasoning-multi-rag-ref", "zeroshot-no-reasoning-general", "zeroshot-multi", "fewshot-no-reasoning-multi"],
                       default="zeroshot-no-reasoning-multi",
                       help="Template for multi-code analysis")
    parser.add_argument("--template-dir", default="prompt_templates/mining",
                       help="Directory containing template files")
    
    # Data paths
    parser.add_argument("--input-dir", 
                       default="mining_misconceptions/data/corrupted_codes/corrupted_codes_sonnet-4-sample",
                       help="Directory containing corrupted code files")
    parser.add_argument("--problems-file", 
                       default="mining_misconceptions/data/problems_processed.json",
                       help="Path to problems JSON file")
    
    # Output
    parser.add_argument("--output-dir", 
                       default="mining_misconceptions/test_results/multi_mined_misconceptions",
                       help="Output directory for results")
    
    # Grouping parameters
    parser.add_argument("--codes-per-group", type=int, default=None,
                       help="Fixed number of codes per group (overridden if min/max specified)")
    parser.add_argument("--min-codes-per-group", type=int, default=None,
                       help="Minimum number of codes per group (for random sampling)")
    parser.add_argument("--max-codes-per-group", type=int, default=None,
                       help="Maximum number of codes per group (for random sampling)")
    parser.add_argument("--max-problems-per-group", type=int, default=None,
                       help="Maximum number of different problems per group")
    parser.add_argument("--min-codes-for-processing", type=int, default=2,
                       help="Minimum number of codes in a group to process it")
    parser.add_argument("--correct-bags-ratio", type=float, default=0.15,
                       help="Ratio of bags containing only correct codes (default: 0.15)")
    parser.add_argument("--bags-per-misconception", type=int, default=None,
                       help="Number of bags to create per misconception (default: None - create as many as possible)")
    parser.add_argument("--correct-bags-only", action="store_true",
                       help="Emit ONLY correct-only bags (skip misconception bags). Use with "
                            "--append-results to add correct-bag evaluation on top of an existing run.")
    parser.add_argument("--num-correct-bags", type=int, default=None,
                       help="Exact number of correct-only bags (overrides ratio-derived count). "
                            "Required/most useful with --correct-bags-only.")
    parser.add_argument("--correct-bags-cover-all", action="store_true",
                       help="Partition EVERY correct code (one per problem) into correct-only bags so "
                            "all are evaluated, instead of random sampling. Ignores --num-correct-bags.")
    parser.add_argument("--correct-bags-passes", type=int, default=1,
                       help="With --correct-bags-cover-all, how many shuffled full passes over the "
                            "correct pool to bag (default 1). Use >1 for more correct bags / statistical weight.")
    parser.add_argument("--append-results", action="store_true",
                       help="Merge new predictions into an existing multi_predictions.json instead of "
                            "overwriting. Existing correct_only groups are replaced; misconception groups kept.")
    
    # Reasoning settings
    parser.add_argument("--reasoning", action="store_true",
                       help="Enable reasoning mode (only for compatible models)")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="medium",
                       help="Reasoning effort level for OpenAI models")
    parser.add_argument("--thinking-budget", type=int, default=2000,
                       help="Token budget for thinking/reasoning (Anthropic and Gemini)")
    
    # Processing options
    parser.add_argument("--use-batch", action="store_true",
                       help="Enable batch processing")
    parser.add_argument("--debug-prompt", action="store_true",
                       help="Save first generated prompt to debug_multi_mining.txt")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for reproducible grouping")
    parser.add_argument("--max-requests", type=int, default=None,
                       help="Maximum number of requests to process (for debugging)")

    # RAG: retrieval-augmented mining. Omit --rag-csv for the exact non-RAG baseline.
    parser.add_argument("--rag-csv", default=None,
                       help="Submission retrieval CSV; enables RAG context. Use with a *-rag template.")
    parser.add_argument("--rag-correct-csv", default=None,
                       help="Correct-code retrieval CSV (for correct-only bags)")
    parser.add_argument("--rag-top-k", type=int, default=3,
                       help="Number of retrieved misconceptions to inject per code (default 3)")

    # Reuse an existing run's EXACT bag composition instead of re-sampling bags. Use this when
    # re-mining with a different --rag-csv/--ref-csv/template so bag composition (and its cost/
    # randomness) doesn't change between runs -- only the injected context does.
    parser.add_argument("--reuse-bags-from", default=None,
                       help="Path to a prior multi_predictions.json; freezes bag composition from "
                            "its group_info.source_files instead of re-sampling bags.")

    # REF: APR reference-code injection. Omit --ref-csv for the exact non-reference baseline.
    parser.add_argument("--ref-csv", default=None,
                       help="APR reference CSV; enables reference-code context. Use with a *-ref template.")
    parser.add_argument("--ref-column", default="Reference_Code",
                       help="Which reference column to inject (Reference_Code | Best_Reference | Repaired_Code)")

    args = parser.parse_args()
    
    # Validate reasoning argument compatibility
    if args.reasoning:
        # Define compatible models for reasoning
        openai_reasoning_models = ["o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini"]
        anthropic_reasoning_models = ["claude-3-7-sonnet-latest", "claude-sonnet-4-5", "claude-opus-4-0"]
        gemini_thinking_models = ["gemini-2.5-flash-preview", "gemini-2.5-flash-exp", "gemini-2.5-flash",
                                 "gemini-2.5-pro", "gemini-2.5-pro-exp", "gemini-2.5-pro-exp-03-25"]
        
        # Validate reasoning is only used with no-reasoning templates
        if "no-reasoning" not in args.template:
            print(f"❌ Error: --reasoning should be used with 'no-reasoning' templates for best results.")
            print(f"   Current template: {args.template}")
            print(f"   Recommended: Use a template with 'no-reasoning' in the name when using --reasoning mode.")
        
        if args.llm == "openai":
            if args.openai_model not in openai_reasoning_models:
                print(f"❌ Error: --reasoning is only compatible with OpenAI reasoning models: {', '.join(openai_reasoning_models)}")
                print(f"   Current model: {args.openai_model}")
                return 1
        elif args.llm == "anthropic":
            if args.anthropic_model not in anthropic_reasoning_models:
                print(f"❌ Error: --reasoning is only compatible with Anthropic reasoning models: {', '.join(anthropic_reasoning_models)}")
                print(f"   Current model: {args.anthropic_model}")
                return 1
        elif args.llm == "gemini":
            if not any(thinking_model in args.gemini_model for thinking_model in gemini_thinking_models):
                print(f"❌ Error: --reasoning is only compatible with Gemini thinking models: {', '.join(gemini_thinking_models)}")
                print(f"   Current model: {args.gemini_model}")
                return 1
        elif args.llm == "openrouter":
            # OpenRouter's unified `reasoning` param fans out to whatever the
            # underlying model supports — no per-model whitelist to check here.
            pass
        else:
            print(f"❌ Error: --reasoning is only compatible with OpenAI, Anthropic, Gemini, and OpenRouter LLMs.")
            print(f"   Current LLM: {args.llm}")
            print(f"   Compatible OpenAI models: {', '.join(openai_reasoning_models)}")
            print(f"   Compatible Anthropic models: {', '.join(anthropic_reasoning_models)}")
            print(f"   Compatible Gemini models: {', '.join(gemini_thinking_models)}")
            return 1
    
    # Set random seed
    random.seed(args.random_seed)
    
    # Validate and determine bag size configuration
    bag_size_min = None
    bag_size_max = None  
    bag_size_fixed = None
    
    if args.min_codes_per_group is not None and args.max_codes_per_group is not None:
        if args.min_codes_per_group > args.max_codes_per_group:
            print(f"Error: min-codes-per-group ({args.min_codes_per_group}) cannot be greater than max-codes-per-group ({args.max_codes_per_group})")
            return 1
        bag_size_mode = "range"
        bag_size_min = args.min_codes_per_group
        bag_size_max = args.max_codes_per_group
        print(f"Using variable bag sizes: [{bag_size_min}, {bag_size_max}]")
    elif args.codes_per_group is not None:
        bag_size_mode = "fixed"
        bag_size_fixed = args.codes_per_group
        print(f"Using fixed bag size: {bag_size_fixed}")
    else:
        # Default to fixed size of 5 if nothing specified
        bag_size_mode = "fixed"
        bag_size_fixed = 5
        print(f"Using default fixed bag size: {bag_size_fixed}")
    
    # Load data
    print("Loading data...")
    try:
        problems = load_json_data(args.problems_file)
        corrupted_codes = load_corrupted_codes(args.input_dir)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    print(f"Loaded {len(corrupted_codes)} corrupted code files")
    print(f"Loaded {len(problems)} problems for context")
    
    filtered_groups = {}
    if not args.reuse_bags_from:
        # Group codes by misconception (skipped in --reuse-bags-from mode: bag composition comes
        # from the frozen prior run instead of a fresh random sample).
        print("Grouping codes by misconception...")
        misconception_groups = group_codes_by_misconception(
            corrupted_codes,
            bag_size_mode,
            bag_size_min,
            bag_size_max,
            bag_size_fixed,
            args.max_problems_per_group
        )

        # Filter groups by minimum size
        filtered_groups = {k: v for k, v in misconception_groups.items() if len(v) >= args.min_codes_for_processing}

        print(f"Created {len(filtered_groups)} misconception groups (min {args.min_codes_for_processing} codes each)")

        # Show bag size configuration
        if bag_size_mode == "range":
            print(f"Bag size configuration: {bag_size_min}-{bag_size_max} codes per bag (variable)")
        else:
            print(f"Bag size configuration: {bag_size_fixed} codes per bag (fixed)")

        # Show total codes available per misconception
        group_sizes = [len(codes) for codes in filtered_groups.values()]
        if group_sizes:
            print(f"Codes per misconception: min={min(group_sizes)}, max={max(group_sizes)}, avg={sum(group_sizes)/len(group_sizes):.1f}")
    else:
        print(f"--reuse-bags-from set: skipping bag sampling; bags will be frozen from {args.reuse_bags_from}")
    
    # Load template
    try:
        template = load_prompt_template(args.template, args.template_dir)
    except FileNotFoundError as e:
        print(f"❌ Template loading error: {e}")
        return 1

    # Build RAG index if requested (omit --rag-csv => exact non-RAG baseline)
    rag_index = None
    if args.rag_csv:
        rag_index = rag_retrieval.load_index(args.rag_csv, args.rag_correct_csv, args.rag_top_k)
        if "-rag" not in args.template:
            print("⚠️  --rag-csv set but template name lacks '-rag'; use zeroshot-no-reasoning-multi-rag "
                  "so the retrieved-context framing is present.")

    # Build REF index if requested (omit --ref-csv => exact non-reference baseline)
    ref_index = None
    if args.ref_csv:
        ref_index = ref_retrieval.load_reference_index(args.ref_csv, args.ref_column)
        if "-ref" not in args.template:
            print("⚠️  --ref-csv set but template name lacks '-ref'; use zeroshot-no-reasoning-multi-ref "
                  "so the reference-solution framing is present.")

    # Create LLM client
    print(f"Initializing {args.llm} LLM client...")
    log_model_access(args)
    if args.reasoning:
        if args.llm == "openai":
            print(f"🧠 Reasoning mode enabled with {args.openai_model} (effort: {args.reasoning_effort})")
        elif args.llm == "anthropic":
            print(f"🧠 Reasoning mode enabled with {args.anthropic_model} (budget: {args.thinking_budget} tokens)")
        elif args.llm == "gemini":
            print(f"🧠 Reasoning mode enabled with {args.gemini_model} (budget: {args.thinking_budget} tokens)")
    try:
        llm_client = create_llm_client(args)
    except Exception as e:
        print(f"Error creating LLM client: {e}")
        return 1
    
    # Generate mining prompts
    if args.reuse_bags_from:
        frozen_bags = load_frozen_bags(args.reuse_bags_from, corrupted_codes, problems)
        batches = generate_frozen_multi_mining_batches(frozen_bags, problems, template,
                                                        rag_index=rag_index, rag_top_k=args.rag_top_k,
                                                        ref_index=ref_index)
    else:
        batches = generate_multi_mining_batches(filtered_groups, problems, template, args.correct_bags_ratio, bag_size_mode, bag_size_min, bag_size_max, bag_size_fixed, args.bags_per_misconception, correct_bags_only=args.correct_bags_only, num_correct_bags_override=args.num_correct_bags, correct_bags_cover_all=args.correct_bags_cover_all, correct_bags_passes=args.correct_bags_passes, rag_index=rag_index, rag_top_k=args.rag_top_k, ref_index=ref_index)
    
    if not batches:
        print("No valid groups to analyze.")
        return 1
    
    # Limit requests for debugging if specified
    if args.max_requests:
        original_count = len(batches)
        batches = batches[:args.max_requests]
        print(f"🔧 Debug mode: Limited to {len(batches)} requests (from {original_count} total)")
    
    # Debug first prompt if requested
    if args.debug_prompt and batches:
        debug_prompt = batches[0][1][0]["content"]
        with open("debug_multi_mining.txt", 'w', encoding='utf-8') as f:
            f.write(debug_prompt)
        print(f"🔍 Debug: First prompt saved to debug_multi_mining.txt ({len(debug_prompt)} chars)")
    
    # Process mining requests
    all_results = []
    
    if args.use_batch and args.llm != "gemini":
        print(f"🔄 Processing {len(batches)} requests in batch mode")
        
        all_metadata = [item[0] for item in batches]
        all_messages = [item[1] for item in batches]
        
        try:
            kwargs = get_llm_kwargs(args)
            print(f"Submitting batch to {args.llm}...")
            
            if args.llm == "anthropic":
                reasoning = kwargs.pop("reasoning", False)
                budget_tokens = kwargs.pop("budget_tokens", 1000)
                responses = llm_client.create_batch_messages(all_messages, reasoning=reasoning, budget_tokens=budget_tokens, **kwargs)
            elif args.llm == "openai":
                reasoning = kwargs.pop("reasoning", False)
                reasoning_effort = kwargs.pop("reasoning_effort", "medium")
                responses = llm_client.create_batch_messages(all_messages, reasoning=reasoning, reasoning_effort=reasoning_effort, **kwargs)
            elif args.llm == "gemini":
                reasoning = kwargs.pop("reasoning", False)
                thinking_budget = kwargs.pop("thinking_budget", 1000)
                responses = llm_client.create_batch_messages(all_messages, reasoning=reasoning, thinking_budget=thinking_budget, **kwargs)
            elif args.llm == "openrouter":
                reasoning = kwargs.pop("reasoning", False)
                reasoning_effort = kwargs.pop("reasoning_effort", "medium")
                responses = llm_client.create_batch_messages(all_messages, reasoning=reasoning, reasoning_effort=reasoning_effort, **kwargs)
            elif args.llm == "vllm" and hasattr(args, 'vllm_model') and args.vllm_model:
                # Use vLLM offline mode with thinking control for Qwen3
                enable_thinking = getattr(args, 'vllm_enable_thinking', False)
                responses = llm_client.create_batch_messages_with_thinking(all_messages, enable_thinking=enable_thinking, **kwargs)
            else:
                responses = llm_client.create_batch_messages(all_messages, **kwargs)
                
            print(f"✅ Batch processing completed")
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
            return 1
        
        # Parse responses
        for metadata, response in zip(all_metadata, responses):
            parsed_response = parse_multi_mining_response(response)
            all_results.append({
                "metadata": metadata,
                "parsed_response": parsed_response
            })
    
    else:
        # Individual processing
        print(f"🔄 Processing {len(batches)} requests individually")
        
        for metadata, messages in tqdm(batches, desc="Inferring misconceptions"):
            try:
                kwargs = get_llm_kwargs(args)
                
                if hasattr(llm_client, 'create_message'):
                    if args.llm == "anthropic":
                        reasoning = kwargs.pop("reasoning", False)
                        budget_tokens = kwargs.pop("budget_tokens", 1000)
                        response = llm_client.create_message(messages, kwargs=kwargs, reasoning=reasoning, budget_tokens=budget_tokens)
                    elif args.llm == "openai":
                        reasoning = kwargs.pop("reasoning", False)
                        reasoning_effort = kwargs.pop("reasoning_effort", "medium")
                        response = llm_client.create_message(messages, kwargs=kwargs, reasoning=reasoning, reasoning_effort=reasoning_effort)
                    elif args.llm == "gemini":
                        reasoning = kwargs.pop("reasoning", False)
                        thinking_budget = kwargs.pop("thinking_budget", 1000)
                        response = llm_client.create_message(messages, kwargs=kwargs, reasoning=reasoning, thinking_budget=thinking_budget)
                    elif args.llm == "openrouter":
                        reasoning = kwargs.pop("reasoning", False)
                        reasoning_effort = kwargs.pop("reasoning_effort", "medium")
                        response = llm_client.create_message(messages, kwargs=kwargs, reasoning=reasoning, reasoning_effort=reasoning_effort)
                    else:
                        response = llm_client.create_message(messages, kwargs=kwargs)
                else:
                    # Fallback to batch with single item
                    if args.llm == "anthropic":
                        reasoning = kwargs.pop("reasoning", False)
                        budget_tokens = kwargs.pop("budget_tokens", 1000)
                        responses = llm_client.create_batch_messages([messages], reasoning=reasoning, budget_tokens=budget_tokens, **kwargs)
                    elif args.llm == "openai":
                        reasoning = kwargs.pop("reasoning", False)
                        reasoning_effort = kwargs.pop("reasoning_effort", "medium")
                        responses = llm_client.create_batch_messages([messages], reasoning=reasoning, reasoning_effort=reasoning_effort, **kwargs)
                    elif args.llm == "gemini":
                        reasoning = kwargs.pop("reasoning", False)
                        thinking_budget = kwargs.pop("thinking_budget", 1000)
                        responses = llm_client.create_batch_messages([messages], reasoning=reasoning, thinking_budget=thinking_budget, **kwargs)
                    elif args.llm == "openrouter":
                        reasoning = kwargs.pop("reasoning", False)
                        reasoning_effort = kwargs.pop("reasoning_effort", "medium")
                        responses = llm_client.create_batch_messages([messages], reasoning=reasoning, reasoning_effort=reasoning_effort, **kwargs)
                    elif args.llm == "vllm" and hasattr(args, 'vllm_model') and args.vllm_model:
                        # Use vLLM offline mode with thinking control for Qwen3
                        enable_thinking = getattr(args, 'vllm_enable_thinking', False)
                        responses = llm_client.create_batch_messages_with_thinking([messages], enable_thinking=enable_thinking, **kwargs)
                    else:
                        responses = llm_client.create_batch_messages([messages], **kwargs)
                    response = responses[0]
                    
            except Exception as e:
                print(f"Error processing request: {e}")
                response = ""
            
            parsed_response = parse_multi_mining_response(response)
            all_results.append({
                "metadata": metadata,
                "parsed_response": parsed_response
            })
    
    # Save results
    print("\nSaving results...")
    
    llm_info = {
        "provider": args.llm,
        "model": get_model_name(args, llm_client),
        "processing_mode": "batch" if args.use_batch else "individual",
        "reasoning_enabled": args.reasoning,
        "reasoning_effort": args.reasoning_effort if args.reasoning and args.llm in ["openai", "openrouter"] else None,
        "thinking_budget": args.thinking_budget if args.reasoning and args.llm in ["anthropic", "gemini"] else None
    }
    
    run_info = {
        "timestamp": datetime.now().isoformat(),
        "input_dir": args.input_dir,
        "template_type": args.template,
        "bag_size_mode": bag_size_mode,
        "bag_size_config": {
            "min_codes_per_group": bag_size_min,
            "max_codes_per_group": bag_size_max,
            "fixed_codes_per_group": bag_size_fixed
        },
        "max_problems_per_group": args.max_problems_per_group,
        "correct_bags_ratio": args.correct_bags_ratio,
        "bags_per_misconception": args.bags_per_misconception
    }
    
    save_multi_mining_results(all_results, args.output_dir, llm_info, run_info, append=args.append_results)
    
    print("✅ Multi-code Misconception Mining completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 