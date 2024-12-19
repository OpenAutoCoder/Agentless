import argparse
import json
import os
from collections import Counter
from pathlib import Path

from agentless.util.postprocess_data import normalize_patch
from agentless.util.utils import load_json, load_jsonl

execution_results = dict()


def _load_results(args):
    global execution_results

    roots = [Path(folder) for folder in args.patch_folder.split(",")]

    intervals = [(0, int(args.num_samples / len(roots)) - 1) for _ in range(len(roots))]

    interval = intervals[0]
    for i in range(interval[0], interval[1] + 1):
        for _, root in enumerate(roots):
            patches = load_jsonl(root / f"output_{i}_normalized.jsonl")
            print(
                f"Loaded {len(patches)} patches from {root / f'output_{i}_normalized.jsonl'}"
            )
            if args.regression:
                regression_test_results = load_jsonl(
                    root / f"output_{i}_regression_test_results.jsonl"
                )
            if args.reproduction:
                reproduction_test_results = load_jsonl(
                    root / f"output_{i}_reproduction_test_results.jsonl"
                )

            for patch in patches[:]:
                if args.regression:
                    regression_test_result = [
                        x
                        for x in regression_test_results
                        if x["instance_id"] == patch["instance_id"]
                    ][0].get("regression", [0] * 10000)
                    regression_test_result = len(regression_test_result)
                else:
                    regression_test_result = 0

                if args.reproduction:
                    if (
                        len(
                            [
                                x
                                for x in reproduction_test_results
                                if x["instance_id"] == patch["instance_id"]
                            ]
                        )
                        == 1
                    ):
                        reproduction_test_result = [
                            x
                            for x in reproduction_test_results
                            if x["instance_id"] == patch["instance_id"]
                        ][0].get("reproduction", False)
                    else:
                        reproduction_test_result = False
                else:
                    reproduction_test_result = True

                execution_results.setdefault(patch["instance_id"], []).append(
                    {
                        "normalized_patch": patch["normalized_patch"].strip(),
                        "patch": patch["model_patch"],
                        "regression_test_result": regression_test_result,
                        "reproduction_test_result": reproduction_test_result,
                    }
                )


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


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def modified_length(normalized_patch):
    changed_length = 0

    for line in normalized_patch.splitlines():
        if len(line) > 3 and (line.startswith("---") or line.startswith("+++")):
            continue

        if line.startswith("-"):
            changed_length += 1
        if line.startswith("+"):
            changed_length += 1

    assert changed_length != 0

    return changed_length


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
            regression_tests = [
                execution_results[instance_id][i]["regression_test_result"]
                for i in range(len(execution_results[instance_id]))
            ]

            min_tests = min(regression_tests)
            regression_tests = [
                True if x == min_tests else False for x in regression_tests
            ]

            reproduction_tests = [
                execution_results[instance_id][i]["reproduction_test_result"]
                for i in range(len(execution_results[instance_id]))
            ]
            raw_patches = [
                execution_results[instance_id][i]["patch"]
                for i in range(len(execution_results[instance_id]))
            ]

            if args.regression and not args.reproduction:
                patch_ids = [
                    i
                    for i in range(len(execution_results[instance_id]))
                    if patch_keys[i].strip() and regression_tests[i]
                ]
            elif args.reproduction:
                patch_ids = [
                    i
                    for i in range(len(execution_results[instance_id]))
                    if patch_keys[i].strip()
                    and regression_tests[i]
                    and reproduction_tests[i]
                ]
                if len(patch_ids) == 0:
                    # reset to just using the all regression passing patches
                    patch_ids = [
                        i
                        for i in range(len(execution_results[instance_id]))
                        if patch_keys[i].strip() and regression_tests[i]
                    ]
            else:
                patch_ids = [
                    i
                    for i in range(len(execution_results[instance_id]))
                    if patch_keys[i].strip()
                ]

            if not patch_ids:
                # just vote on all patches
                if not all([x.strip() == "" for x in raw_patches]) and not all(
                    [x.strip() == "" for x in patch_keys]
                ):
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
            changed_length_idx = dict()
            for i in patch_ids:
                sample = get_sample(instance_id, i)
                patch_key, patch = sample["normalized_patch"], sample["patch"]
                vote[patch_key] += 1
                if patch_key not in first_appear_idx:
                    first_appear_idx[patch_key] = i
                    changed_length_idx[patch_key] = modified_length(patch_key)

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
    selected_ids = list(range(int(args.num_samples / num_folders)))

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
                new_file_content = d["new_file_content"]
                edited_files = d["edited_files"]
                normalized_patch = normalize_patch(
                    instance_id,
                    patch,
                    original_file_content,
                    new_file_content,
                    edited_files,
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
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--reproduction", action="store_true")
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
