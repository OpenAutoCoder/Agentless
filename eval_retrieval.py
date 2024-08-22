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

from agentless.retreival.VectorDB import QdrantDB
from agentless.retreival.rerank_files import rerank_files
from agentless.retreival.upload_swe_bench_to_qdrant import init_repo


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


def find_relevant_files_using_vector_search(name:str, code_dict:dict, problem_statement: str, top_n: int = 15):
    # filter out all non python files
    code_dict = {k: v for k, v in code_dict.items() if k.endswith('.py')}

    # filter out test files
    code_dict = {k: v for k, v in code_dict.items() if not k.startswith('test')}

    print(f"Repo name: {name}")
    
    # Initialize vectordb
    vectordb = QdrantDB(
        repo_name=name,
        codebase_dict=code_dict,
        include_chunk_content=True,
        regenerate_embeddings=False,
    )
    
    # Search for relevant files
    vector_search_results = vectordb.search_code(problem_statement, top_n=top_n)
    print(f"Vector search results: {vector_search_results}")

    vector_search_results_files = [result['file_path'] for result in vector_search_results]

    code_dict_for_vector_search = {}
    for file in vector_search_results_files:
        code_dict_for_vector_search[file] = code_dict[file]
        
    reranked_files = rerank_files(problem_statement, code_dict_for_vector_search)
    
    return reranked_files

def create_vector_search_predictions(swe_bench_data,pred_jsonl_path, top_k_list: List[int] = [1,3,5,7,10], instance_ids: List[str] = None):
    """
    Creates the predictions for the given dataset and prediction jsonl file.
    """
    predictions = []
    
    if instance_ids is not None:
        filtered_swe_bench_data = swe_bench_data.filter(lambda x: x["instance_id"] in instance_ids)
    else:
        filtered_swe_bench_data = swe_bench_data

    for instance in filtered_swe_bench_data:
        instance_id = instance["instance_id"]
        problem_statement = instance["problem_statement"]
        owner, repo_name = instance["repo"].split('/')
        # print(f"Repo name: {repo_name}")

        # Use custom_init_repo to initialize the repository
        fs = init_repo(owner, repo_name, instance['base_commit'])

        code_dict = fs.get_code_dict()
        
        # Find relevant files using vector search
        relevant_files = find_relevant_files_using_vector_search(
            name=repo_name,
            code_dict=code_dict,
            problem_statement=problem_statement,
            top_n=max(top_k_list)
        )

        relevant_files_paths = [file for file in relevant_files]
        
        # Create prediction entry for each k in top_k_list
        for k in top_k_list:
            prediction = {
                "instance_id": instance_id,
                "found_files": relevant_files_paths[:k]
            }
            predictions.append(prediction)
    
    # Save predictions to jsonl file
    with open(pred_jsonl_path, "w") as f:
        for pred in predictions:
            json.dump(pred, f)
            f.write("\n")
    
    print(f"Predictions saved to {pred_jsonl_path}")


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

    create_vector_search_predictions(swe_bench_data, args.preds_path, instance_ids=["pylint-dev__pylint-4604"])

