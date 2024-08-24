"""agentless/run_instance.py

Example command:
```
python -m agentless.run_instance --instance_id=django__django-11039 --output_folder=single_instance_test
```
"""

import argparse
import json
import os
from typing import Dict, Any

from datasets import load_dataset

from agentless.fl.localize import localize_instance
from agentless.repair.repair import repair, post_process_repair
from agentless.repair.rerank import normalize_patches, _load_results, majority_voting
from agentless.util.utils import setup_logger

def process_single_instance(
    instance_id: str,
    swe_bench_data: Dict[str, Any],
    args: argparse.Namespace,
    logger: Any
) -> Dict[str, Any]:
    """
    Process a single instance through localization, repair, and rerank steps.
    
    Args:
    instance_id (str): The ID of the instance to process.
    swe_bench_data (Dict[str, Any]): The SWE-bench dataset.
    args (argparse.Namespace): Command line arguments.
    logger (Any): Logger object for logging messages.
    
    Returns:
    Dict[str, Any]: Results of the processing steps.
    """
    logger.info(f"Processing instance {instance_id}")

    bug = next(bug for bug in swe_bench_data if bug['instance_id'] == instance_id)
    
    # Step 1: Localization
    args.output_file = os.path.join(args.output_folder, "location", f"{instance_id}_loc_outputs.jsonl")
    # Create the dir if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    if not args.skip_loc:
        localization_result = localize_instance(  # Result is already written to file, we don't need to use it
            bug, args, swe_bench_data, start_file_locs=None, existing_instance_ids=set()
        )
    # Prep outputs as inputs for the repair step.
    args.loc_file = os.path.join(args.output_folder, "location", f"{instance_id}_loc_outputs.jsonl")
    
    # Step 2: Repair
    args.output_file = os.path.join(args.output_folder, "repair", f"{instance_id}_repair.jsonl")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    if not args.skip_repair:
        repair(args)

        args.raw_output_file = args.output_file
        for i in range(args.max_samples):
            args.output_file = os.path.join(os.path.dirname(args.output_file), f"output_{i}_processed.jsonl")
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            args.select_id = i
            post_process_repair(args)
    
    # Step 3: Rerank
    args.patch_folder = os.path.join(args.output_folder, "repair")
    args.num_samples = 10
    # first normalize
    normalize_patches(args)
    # then load results
    _load_results(args)
    # then rerank
    args.output_file = os.path.join(args.output_folder, f"all_preds.jsonl")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    majority_voting(args)
    
    return {
        "instance_id": instance_id,
    }

def main():
    parser = argparse.ArgumentParser(description="Process a single instance through localization, repair, and rerank steps.")

    # Args for just this file
    parser.add_argument("--skip_loc", action="store_true", help="Skip localization and use existing localization file")
    parser.add_argument("--skip_repair", action="store_true", help="Skip repair and use existing repair file")
    parser.add_argument("--skip_rerank", action="store_true", help="Skip rerank and use existing rerank file")

    # Common args
    parser.add_argument("--instance_id", type=str, required=True, help="ID of the instance to process")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save output files")
    parser.add_argument("--output_file", type=str, default="loc_outputs.jsonl")
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18", choices=["gpt-4o-2024-05-13", "deepseek-coder", "gpt-4o-mini-2024-07-18"], help="Model to use for processing")
    parser.add_argument("--backend", type=str, default="openai", choices=["openai", "deepseek"], help="Backend to use for API calls")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode without making actual API calls")
    
    # Add other common arguments from localize.py, repair.py, and rerank.py
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--context_window", type=int, default=10)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--target_id", type=str)

    # Args for localize.py
    parser.add_argument("--file_level", action="store_true", default=True)
    parser.add_argument("--related_level", action="store_true", default=True)
    parser.add_argument("--fine_grain_line_level", action="store_true", default=True)
    parser.add_argument("--compress", action="store_true", default=True)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--add_space", action="store_true")
    parser.add_argument("--no_line_number", action="store_true")
    parser.add_argument("--sticky_scroll", action="store_true")
    parser.add_argument(
        "--match_partial_paths",
        action="store_true",
        help="Whether to match model generated files based on subdirectories of original repository if no full matches can be found",
    )

    # Args for repair.py
    parser.add_argument(
        "--stop_at_n_unique_valid_samples",
        type=int,
        default=-1,
        help="Early stop when we get N unique valid samples, set to -1 if don't want to do early stopping.",
    )
    parser.add_argument("--gen_and_process", action="store_true", default=True)
    parser.add_argument("--loc_interval", action="store_true", default=True)
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument(
        "--select_id",
        type=int,
        default=-1,
        help="Index the selected samples during post-processing.",
    )
    parser.add_argument(
        "--only_correct", action="store_true"
    )  # only work on correct loc files (saves time)
    parser.add_argument("--post_process", action="store_true")
    parser.add_argument("--cot", action="store_true", default=True)
    parser.add_argument("--fine_grain_loc_only", action="store_true")
    parser.add_argument("--diff_format", action="store_true", default=True)
    parser.add_argument("--skip_greedy", action="store_true")
    
    # Args for rerank.py
    parser.add_argument("--patch_folder", type=str)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--deduplicate", action="store_true", default=True)
    parser.add_argument("--plausible", action="store_true", default=True)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "localization_logs"), exist_ok=True)
    logger = setup_logger(os.path.join(args.output_folder, f"{args.instance_id}.log"))
    
    swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    
    result = process_single_instance(args.instance_id, swe_bench_data, args, logger)
    
    output_file = os.path.join(args.output_folder, f"{args.instance_id}_full_process.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Full process result for instance {args.instance_id} saved to {output_file}")

if __name__ == "__main__":
    main()