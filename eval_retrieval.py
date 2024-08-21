"""
Runs retrieval evaluation on the given dataset.
"""


import pandas as pd
import re
import random
import argparse
import json
from multiprocessing import Pool
import os
import sys
from collections import defaultdict
from datasets import load_dataset
from typing import List


def load_jsonl(filepath):
    """
    Load a JSONL file from the given filepath.

    Arguments:
    filepath -- the path to the JSONL file to load

    Returns:
    A list of dictionaries representing the data in each line of the JSONL file.
    """
    with open(filepath, "r") as file:
        return [json.loads(line) for line in file]


def get_affected_files(patch_string):
    pattern = r'diff --git a/(.*?) b/(.*?)$'
    matches = re.findall(pattern, patch_string, re.MULTILINE)
    affected_files = set()
    for match in matches:
        affected_files.add(match[0])  # 'a' path
        affected_files.add(match[1])  # 'b' path
    
    return list(affected_files)


def get_retrieval_eval_results(swe_bench_data, pred_jsonl_path, top_k_list: List[int] = [1,3,5,7,10]):
    """
    Runs retrieval evaluation on the given dataset.
    """
    
    # read the jsonl file and for each instance_id, search the dataset for the instance_id and get the context
    # then run the retrieval evaluation on the context and the query
    # return the average precision, recall, and f1 score
    pred_data = load_jsonl(pred_jsonl_path)
    avg_recall = 0
    count = 0
    avg_recalls = {k : 0 for k in top_k_list}
    for pred in pred_data:
        instance_id = pred["instance_id"]
        lite_dataset = swe_bench_data.filter(lambda x: x["instance_id"] in [instance_id])
        if len(lite_dataset) == 0:
            continue
        gt_patch = lite_dataset[0]["patch"]
        pred_files = pred["found_files"]
        gt_files = get_affected_files(gt_patch)
        for _k in top_k_list:
            pred_files_k = pred_files[:_k]
            
            # get the recall.
            intersection = set(pred_files_k) & set(gt_files)
            recall_at_k = len(intersection) / len(set(gt_files))
            avg_recalls[_k] += recall_at_k
        count += 1
    if count == 0:
        print("No instances found in the dataset.")
        return {k: 0 for k in top_k_list}, 0
    # Calculate average recall for each k
    avg_recalls = {k: recall / count for k, recall in avg_recalls.items()}
    
    return avg_recalls, count


if __name__ == "__main__":
    # use parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, default="exploiter345/SWE-bench_Verified_50")
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--preds_path", type=str, default="")
    
    args = parser.parse_args()

    assert args.preds_path != ""

    # load the dataset 
    swe_bench_data = load_dataset(args.dataset_id, split=args.split_name)

    avg_recalls, count = get_retrieval_eval_results(swe_bench_data, args.preds_path)
    for k, recall in avg_recalls.items():
        print(f"Average recall@{k}: {recall:.4f}")
    print(f"Count: {count}")