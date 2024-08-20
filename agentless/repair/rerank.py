import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from Agentless.agentless.test.run_tests import run_tests
from Agentless.agentless.util.postprocess_data import normalize_patch
from Agentless.agentless.util.regression_tests import (
    get_execution_result_files,
    load_existing_results,
    remove_duplicates,
)
from Agentless.agentless.util.utils import load_jsonl

execution_results = dict()


def normalize_patches(args):
    # separate the patch folders
    output_folders = [Path(folder) for folder in args.patch_folder.split(",")]
    num_folders = len(output_folders)
    selected_ids = list(range(int(args.num_samples / num_folders)))

    for output_folder in output_folders:
        for i in selected_ids:
            if os.path.exists(output_folder / f"output_{i}_normalized.jsonl"):
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


def order_normalized_patches(data, folder_path, args):
    result = {}
    os.makedirs(folder_path, exist_ok=True)
    jsonl_file_path = os.path.join(folder_path, "ordered_normalized_patches.jsonl")

    grouped_by_instance = defaultdict(list)
    for entry in data:
        instance_id = entry["instance_id"]
        grouped_by_instance[instance_id].append(entry)

    with open(jsonl_file_path, "w") as jsonl_file:
        for instance_id, entries in grouped_by_instance.items():
            patch_counter = Counter()
            patch_example = {}

            for entry in entries:
                if args.deduplicate:
                    normalized_patch = entry["normalized_patch"].strip()
                else:
                    normalized_patch = entry["model_patch"].strip()

                model_patch = entry["model_patch"]
                patch_counter[normalized_patch] += 1
                if normalized_patch not in patch_example:
                    patch_example[normalized_patch] = model_patch

            sorted_patches = [
                {
                    "normalized_patch": patch,
                    "frequency": count,
                    "patch": patch_example[patch],
                    "instance_id": instance_id,
                }
                for patch, count in patch_counter.most_common()
                if patch != ""
            ]

            if sorted_patches == []:
                sorted_patches = [
                    {
                        "normalized_patch": patch,
                        "frequency": count,
                        "patch": patch_example[patch],
                        "instance_id": instance_id,
                    }
                    for patch, count in patch_counter.most_common()
                ]

            result[instance_id] = sorted_patches
            jsonl_file.write(json.dumps({instance_id: sorted_patches}) + "\n")

    return result


def _select_patches(args):
    global execution_results

    execution_results_files = get_execution_result_files(
        os.path.dirname(args.output_file)
    )
    print(f"getting previously evaluated patches from {execution_results_files}")
    total_number_of_results = 0
    for execution_file in execution_results_files:
        additional_results = load_existing_results(execution_file)
        for key in additional_results.keys():
            total_number_of_results += len(additional_results[key])
        execution_results.update(additional_results)
    print(f"{total_number_of_results} execution results already found")

    # Get all the patches
    max_patches = 0
    roots = [Path(folder) for folder in args.patch_folder.split(",")]
    intervals = [(0, int(args.num_samples / len(roots)) - 1) for _ in range(len(roots))]
    initial_patches = []
    for index, root in enumerate(roots):
        interval = intervals[index]
        for i in range(interval[0], interval[1] + 1):
            max_patches += 1
            initial_patches.extend(
                remove_duplicates(load_jsonl(root / f"output_{i}_normalized.jsonl"))
            )

    # Sort the patches for each instance id
    ordered_normalized_patches = order_normalized_patches(
        initial_patches, os.path.dirname(args.output_file), args
    )
    patches_with_only_ones = 0
    for key in ordered_normalized_patches.keys():
        if ordered_normalized_patches[key][0]["frequency"] == 1:
            patches_with_only_ones += 1
    print("Instances with no duplicate patches: ", patches_with_only_ones)

    for i in range(max_patches):
        # Get the patches for the instance_id
        patches = []
        for key in ordered_normalized_patches.keys():
            if len(ordered_normalized_patches[key]) > i:
                patches.append(ordered_normalized_patches[key][i])

        # Filter the normalized patches based on if they were already evaluated
        # For plausible add them if the normalized_patch isn't in the past execution results
        # for testing patches add them
        previously_evaluated_normalized_patches = {}
        patches_to_evaluate = []
        for patch in patches:
            previously_evaluated_normalized_dicts = execution_results.get(
                patch.get("instance_id"), []
            )
            found_previous_passing = False
            for entry in previously_evaluated_normalized_dicts:
                if entry["plausible"]:
                    previously_evaluated_normalized_patches[
                        patch["instance_id"]
                    ] = entry
                    found_previous_passing = True
                    print(
                        f"a more frequent plausible patch has already been found for iteration {i} and instance {patch['instance_id']}"
                    )
            if not found_previous_passing:
                patches_to_evaluate.append(patch)

        run_id = f"regression_tests_{args.output_file.replace('.jsonl','').split('/')[-1]}_{i}"

        instance_ids = [patch["instance_id"] for patch in patches_to_evaluate]
        patches = [patch["patch"] for patch in patches_to_evaluate]

        if args.plausible:
            print(args.instance_ids)
            instance_to_resolved = run_tests(
                instance_ids,
                patches,
                args.num_workers,
                run_id,
                args.regression_tests,
                args.instance_ids,
                args.timeout,
                not args.testing,
                args.apply_test_patch,
                args.run_all_tests,
            )
        else:
            instance_to_resolved = {}
            for instance in instance_ids:
                instance_to_resolved[instance] = True

        for patch in patches_to_evaluate:
            if patch["instance_id"] in instance_to_resolved:
                is_plausible = instance_to_resolved[patch["instance_id"]]
                new_patch = patch
                if not args.deduplicate:
                    new_patch["model_patch"] = new_patch["normalized_patch"]
                new_patch["plausible"] = is_plausible
                execution_results.setdefault(patch["instance_id"], []).append(patch)
            else:
                is_plausible = False
                new_patch = patch
                if not args.deduplicate:
                    new_patch["model_patch"] = new_patch["normalized_patch"]
                new_patch["plausible"] = is_plausible
                execution_results.setdefault(patch["instance_id"], []).append(patch)

        jsonl_filename = args.output_file.replace(
            ".jsonl", f"_{i}_ordered_regression_results.jsonl"
        )
        with open(jsonl_filename, "w") as jsonl_file:
            for instance_id, results in execution_results.items():
                for result in results:
                    if result["plausible"]:
                        jsonl_file.write(
                            json.dumps({"instance_id": instance_id, **result}) + "\n"
                        )
                        break

    with open(args.output_file, "w") as jsonl_file:
        for instance_id, results in execution_results.items():
            found_plausible = False
            for result in results:
                if result["plausible"] and (not found_plausible):
                    jsonl_file.write(
                        json.dumps({"instance_id": instance_id, **result}) + "\n"
                    )
                    found_plausible = True

            if not found_plausible:
                print("couldn't find a plausible case")
                for result in results:
                    jsonl_file.write(
                        json.dumps({"instance_id": instance_id, **result}) + "\n"
                    )
                    break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_folder", type=str)
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=11)
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--plausible", action="store_true")
    parser.add_argument(
        "--testing", action="store_true", help="If true don't apply the model patch"
    )
    parser.add_argument(
        "--run_all_tests",
        action="store_true",
        help="Instead of just running the passing tests, runs all the PASS_TO_PASS tests",
    )
    parser.add_argument(
        "--apply_test_patch",
        action="store_true",
        help="If true we apply the patch to include the tests added with the new patch",
    )
    parser.add_argument("--output_file", type=str, default="results/all_preds.jsonl")
    parser.add_argument(
        "--regression_tests", type=str, default="successful_tests.jsonl"
    )
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument(
        "--timeout", type=int, default=1200, help="Timeout for running tests in seconds"
    )
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to run (space separated)",
    )
    args = parser.parse_args()

    # first normalize
    normalize_patches(args)

    # then load and run regression tests on the results
    _select_patches(args)


if __name__ == "__main__":
    main()
