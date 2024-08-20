# Does stratified sampling from the swe_bench verified dataset
# based on the difficulty level distribution.

import pandas as pd
import random
import argparse
import json
from multiprocessing import Pool
import os
import sys
from collections import defaultdict
from datasets import load_dataset


def get_difficult_distribution(swe_bench_data):
    # get the distribution of the difficulty level
    difficult_level_map = defaultdict(list)
    # output the name of instance ids for each difficult level
    for bug in swe_bench_data:
        difficult_level_map[bug["difficulty"]].append(bug["instance_id"])
    return difficult_level_map


def create_lite_dataset(swe_bench_data, distr, lite_dataset_size):
    # based on the distribution, sample the number of instances that are needed to reach the lite dataset size
    # first figure out the number of instances to sample for each difficult level
    difficult_level_distribution = {}
    for difficult_level in distr:
        difficult_level_distribution[difficult_level] = len(distr[difficult_level]) / sum(len(distr[difficult_level]) for difficult_level in distr)
    sampled_instances_by_difficult_level = {}
    all_sampled_instance_ids = []
    total_sampled_instances = 0
    for difficult_level in distr:
        sample_size = int(difficult_level_distribution[difficult_level] * lite_dataset_size)
        sampled_instances_by_difficult_level[difficult_level] = random.sample(distr[difficult_level], sample_size)
        all_sampled_instance_ids.extend(sampled_instances_by_difficult_level[difficult_level])
        total_sampled_instances += sample_size
    
    while total_sampled_instances < lite_dataset_size:
        for difficult_level in distr:
            if total_sampled_instances >= lite_dataset_size:
                break
            if len(sampled_instances_by_difficult_level[difficult_level]) < len(distr[difficult_level]):
                remaining = set(distr[difficult_level]) - set(sampled_instances_by_difficult_level[difficult_level])
                sample = random.choice(list(remaining))
                sampled_instances_by_difficult_level[difficult_level].append(sample)
                all_sampled_instance_ids.append(sample)
                total_sampled_instances += 1

    # create a lite dataset of the same type as the swe_bench_data
    lite_dataset = swe_bench_data.filter(lambda x: x["instance_id"] in all_sampled_instance_ids)
    return lite_dataset

    

if __name__ == "__main__":
    # use parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--annotations_path", type=str, default="")
    parser.add_argument("--lite_dataset_size", type=int, default=50)
    
    args = parser.parse_args()

    assert args.annotations_path != ""

    # load the annotations csv
    annotations = pd.read_csv(args.annotations_path)
    
    # load the dataset 
    swe_bench_data = load_dataset(args.dataset_id, split=args.split_name)

    # get difficult level from the annoations and add that as a column to the dataset
    def get_difficulty(instance_id, annotations_df):
        matching_annotation = annotations_df[annotations_df["instance_id"] == instance_id]
        if not matching_annotation.empty:
            return matching_annotation["difficulty"].iloc[0]
        else:
            return "unknown"  # or any default value you prefer

    swe_bench_data = swe_bench_data.map(
        lambda x: {"difficulty": get_difficulty(x["instance_id"], annotations)}
    )    
    distr = get_difficult_distribution(swe_bench_data)

    # create a lite dataset of a given size using stratified sampling
    lite_dataset = create_lite_dataset(swe_bench_data, distr, args.lite_dataset_size)

    # upload to huggingface
    lite_dataset.push_to_hub(f"exploiter345/SWE-bench_Verified_{args.lite_dataset_size}")
