import argparse
import json
import os
from collections import Counter, OrderedDict
from pathlib import Path

from tqdm import tqdm

from agentless.util.postprocess_data import extract_python_blocks, normalize_patch
from agentless.util.utils import load_json, load_jsonl

execution_results = dict()


def _load_results(args):
    global execution_results

    roots = [Path(folder) for folder in args.patch_folder.split(",")]

    # assumes interval
    intervals = [(0, int(args.num_samples / len(roots)) - 1) for _ in range(len(roots))]

    for index, root in enumerate(roots):
        interval = intervals[index]
        for i in range(interval[0], interval[1] + 1):
            patches = load_jsonl(root / f"output_{i}_normalized.jsonl")
            print(
                f"Loaded {len(patches)} patches from {root / f'output_{i}_normalized.jsonl'}"
            )
            for patch in patches[:300]:
                try:
                    execution_results.setdefault(patch["instance_id"], []).append(
                        {
                            "normalized_patch": patch["normalized_patch"].strip(),
                            "patch": patch["model_patch"],
                            "plausible": True,  # default to TRUE for now, TODO: add plausible execution.
                        }
                    )
                except:
                    print(i)
                    print(patch)
                    exit(-1)


def get_sample(instance_id, sample_id) -> tuple[str, bool]:
    """Returns the diff and pass status."""
    return execution_results[instance_id][sample_id]


def get_all_patches(instance_id, num_samples, deduplicate) -> list[str]:
    """Returns all unique patches."""
    patches = [execution_results[instance_id][i]["patch"] for i in range(num_samples)]
    if deduplicate:
        patch_keys = [
            execution_results[instance_id][i]["normalized_patch"]
            for i in range(num_samples)
        ]
    else:
        patch_keys = [
            execution_results[instance_id][i]["patch"] for i in range(num_samples)
        ]
    unique_patches = set()
    patch_ids = []
    for i in range(num_samples):
        patch_key = patch_keys[i].strip()
        if patch_key and patch_key not in unique_patches:
            unique_patches.add(patch_key)
            patch_ids.append(i)
    return [(id, patches[id]) for id in patch_ids]


def get_all_patches_num(instance_id, num_samples, deduplicate) -> list[str]:
    """Returns all unique patches with number."""
    # print(f"{len(execution_results)}")
    patches = [execution_results[instance_id][i]["patch"] for i in range(num_samples)]
    if deduplicate:
        patch_keys = [
            execution_results[instance_id][i]["normalized_patch"]
            for i in range(num_samples)
        ]
    else:
        patch_keys = [
            execution_results[instance_id][i]["patch"] for i in range(num_samples)
        ]
    unique_patches = {}
    total_patch_num = {}
    patch_ids = []
    for i in range(num_samples):
        if patch_keys[i] and patch_keys[i] not in unique_patches:
            unique_patches[patch_keys[i]] = i
            patch_ids.append(i)
            total_patch_num[i] = 0
        if patch_keys[i]:
            total_patch_num[unique_patches[patch_keys[i]]] += 1

    return [(id, patches[id], total_patch_num[id]) for id in patch_ids]


######

import json


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def majority_voting(args):
    with open(args.output_file, "w") as f:
        for instance_id in execution_results:
            if len(execution_results[instance_id]) < args.num_samples:
                print(
                    f"There were only {len(execution_results[instance_id])} patches for {instance_id} instead of the full {args.num_samples}"
                )

            patch_keys = [
                execution_results[instance_id][i]["normalized_patch"]
                for i in range(len(execution_results[instance_id]))
            ]
            plausible = [
                execution_results[instance_id][i]["plausible"]
                for i in range(len(execution_results[instance_id]))
            ]
            raw_patches = [
                execution_results[instance_id][i]["patch"]
                for i in range(len(execution_results[instance_id]))
            ]

            if args.plausible:
                patch_ids = [
                    i
                    for i in range(len(execution_results[instance_id]))
                    if patch_keys[i].strip() and plausible[i]
                ]
            else:
                patch_ids = [
                    i
                    for i in range(len(execution_results[instance_id]))
                    if patch_keys[i].strip()
                ]

            if not patch_ids:
                # just vote on all patches
                if not all([x.strip() == "" for x in raw_patches]):
                    vote = Counter()
                    first_appear_idx = dict()
                    valid_indices = []
                    for i in range(len(execution_results[instance_id])):
                        sample = get_sample(instance_id, i)
                        patch_key = sample["normalized_patch"]
                        if patch_key != "":
                            valid_indices.append(i)
                            vote[patch_key] += 1
                            if patch_key not in first_appear_idx:
                                first_appear_idx[patch_key] = i
                    maj_selected_id = max(
                        valid_indices,
                        key=lambda i: (
                            vote[patch_keys[i]],
                            -first_appear_idx[patch_keys[i]],
                        ),
                    )
                    patch = get_sample(instance_id, maj_selected_id)["patch"]
                    result = {
                        "model_name_or_path": "agentless",
                        "instance_id": instance_id,
                        "model_patch": patch,
                    }
                else:
                    print(f"No raw patches valid for {instance_id}")
                    result = {
                        "model_name_or_path": "agentless",
                        "instance_id": instance_id,
                        "model_patch": "",
                    }
                f.write(json.dumps(result) + "\n")
                continue

            vote = Counter()
            first_appear_idx = dict()
            for i in patch_ids:
                sample = get_sample(instance_id, i)
                patch_key, patch = sample["normalized_patch"], sample["patch"]
                vote[patch_key] += 1
                if patch_key not in first_appear_idx:
                    first_appear_idx[patch_key] = i

            maj_selected_id = max(
                patch_ids,
                key=lambda i: (vote[patch_keys[i]], -first_appear_idx[patch_keys[i]]),
            )

            if args.target is not None and instance_id == args.target:
                for patch in vote:
                    print(
                        "=" * 20,
                        vote[patch],
                        "=" * 20,
                    )
                    print(patch)
                    print("=" * 50)

            sample = get_sample(instance_id, maj_selected_id)
            result = {
                "model_name_or_path": "agentless",
                "instance_id": instance_id,
                "model_patch": sample["patch"],
            }
            f.write(json.dumps(result) + "\n")


def normalize_patches(args):
    # separate the patch folders
    output_folders = [Path(folder) for folder in args.patch_folder.split(",")]
    num_folders = len(output_folders)
    # output_folder = Path(args.patch_folder)
    selected_ids = list(range(int(args.num_samples / num_folders)))

    # print(num_folders, output_folders)

    for output_folder in output_folders:
        for i in selected_ids:
            if os.path.exists(output_folder / f"output_{i}_normalized.jsonl"):
                # skip
                continue
            patches = load_jsonl(output_folder / f"output_{i}_processed.jsonl")
            for d in patches:
                instance_id = d["instance_id"]
                patch = d["model_patch"]
                original_file_content = d["original_file_content"]
                normalized_patch = normalize_patch(
                    instance_id, patch, original_file_content
                )
                d["normalized_patch"] = normalized_patch
            with open(output_folder / f"output_{i}_normalized.jsonl", "w") as f:
                for d in patches:
                    f.write(json.dumps(d) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_folder", type=str)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=11)
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--plausible", action="store_true")
    parser.add_argument("--output_file", type=str, default="all_preds.jsonl")
    args = parser.parse_args()

    # first normalize
    normalize_patches(args)
    # then load results
    _load_results(args)
    # then rerank
    majority_voting(args)


if __name__ == "__main__":
    main()
#
