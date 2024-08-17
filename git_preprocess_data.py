# multiprocessing app to process all the files in the repo and save the data
# for faster agent processing.

import argparse
import json
from multiprocessing import Pool
import os
import sys
from datasets import load_dataset
from get_repo_structure.get_repo_structure import (
    get_project_structure_from_scratch,
)

def save_project_structure(args):
    bug, output_folder = args
    d = get_project_structure_from_scratch(
        bug["repo"], bug["base_commit"], bug["instance_id"], "playground"
    )
    # save the project structure as json dict to the output folder
    with open(os.path.join(output_folder, f"{bug['instance_id']}.json"), "w") as f:
        json.dump(d, f)
    print(f"Saved project structure for {bug['instance_id']}")


if __name__ == "__main__":
    # use parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--output_folder", type=str, default="playground")
    parser.add_argument("--run_top_n", type=int, default=1)
    parser.add_argument("--repo_filter", type=str, default="")
    args = parser.parse_args()

    swe_bench_data = load_dataset(args.dataset_id, split=args.split_name)
    if args.repo_filter != "":
        swe_bench_data = swe_bench_data.filter(lambda x : x["repo"] == args.repo_filter)

    if args.run_top_n > 0:
        swe_bench_data = swe_bench_data.select(range(args.run_top_n))
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    # do multiprocessing here to save the project structure for each bug
    with Pool(10) as p:
        p.map(save_project_structure, [(bug, args.output_folder) for bug in swe_bench_data])