import argparse
import json
import os

from datasets import load_dataset

from agentless.test.run_tests import run_reproduction_tests, txt_file_contains_string
from agentless.util.utils import load_jsonl

execution_results = dict()


def run_reproduction_for_each_instance(args, lines, run_id, test_jsonl):

    instance_ids = [line["instance_id"] for line in lines]
    patches = [line["model_patch"] for line in lines]

    results = run_reproduction_tests(
        instance_ids,
        patches,
        args.num_workers,
        run_id,
        args.instance_ids,
        args.timeout,
        testing_patches=False,
        apply_model_patch=True,
        test_jsonl=test_jsonl,
        dataset_name=args.dataset,
    )
    return results


def _run_reproduction_tests(args):
    if args.testing:
        # for reproduction test selection
        # run on original repo to select tests which can reproduce the issue
        ds = load_dataset(args.dataset)
        instance_ids = ds["test"]["instance_id"]
        patches = [
            {"instance_id": instance_id, "patch": "", "normalized_patch": ""}
            for instance_id in instance_ids
        ]

        evaluation_tests = load_jsonl(args.test_jsonl)

        results = run_reproduction_tests(
            instance_ids,
            patches,
            args.num_workers,
            args.run_id,
            args.instance_ids,
            args.timeout,
            testing_patches=True,
            apply_model_patch=False,
            test_jsonl=args.test_jsonl,
            dataset_name=args.dataset,
        )

        with open(args.test_jsonl.replace(".jsonl", "_verified.jsonl"), "w") as file:
            for evaluation_test in evaluation_tests:
                instance_id = evaluation_test["instance_id"]
                if instance_id in results and results[instance_id]:
                    evaluation_test["verified"] = True
                    file.write(json.dumps(evaluation_test) + "\n")

    elif args.predictions_path == "gold":
        # check on groundtruth patches
        # for evaluation purposes
        ds = load_dataset(args.dataset)
        instance_ids = ds["test"]["instance_id"]
        patches = ds["test"]["patch"]

        results = run_reproduction_tests(
            instance_ids,
            patches,
            args.num_workers,
            args.run_id,
            args.instance_ids,
            args.timeout,
            testing_patches=False,
            apply_model_patch=True,
            test_jsonl=args.test_jsonl,
            dataset_name=args.dataset,
        )

        with open(
            "gold_production_test_results.json",
            "w",
        ) as file:
            file.write(json.dumps(results))

    else:
        # run on the agentless generated patches
        assert args.predictions_path.endswith("_processed.jsonl")
        with open(args.predictions_path, "r") as file:
            data_lines = [json.loads(line) for line in file]

        if args.load:
            reproduction_dict = {}
            for data in data_lines:
                instance_id = data["instance_id"]
                expected_output = "Issue resolved"
                path_to_log = f"logs/run_evaluation/{args.run_id}/test/{instance_id}/test_output.txt"
                if os.path.isfile(path_to_log):
                    passes_tests = txt_file_contains_string(
                        path_to_log, expected_output
                    )
                    reproduction_dict[instance_id] = passes_tests
                else:
                    reproduction_dict[instance_id] = False
        else:
            reproduction_dict = run_reproduction_for_each_instance(
                args, data_lines, args.run_id, args.test_jsonl
            )

        updated_data_lines = []
        for data in data_lines:
            instance_id = data["instance_id"]
            if instance_id in reproduction_dict:
                data["reproduction"] = reproduction_dict[instance_id]
            updated_data_lines.append(data)

        with open(
            args.predictions_path.replace(
                "processed.jsonl", "reproduction_test_results.jsonl"
            ),
            "w",
        ) as file:
            for data in updated_data_lines:
                file.write(json.dumps(data) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument(
        "--testing", action="store_true", help="If true don't apply the model patch"
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        help="Patch file",
    )
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument(
        "--timeout", type=int, default=600, help="Timeout for running tests in seconds"
    )
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to run (space separated)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        choices=["princeton-nlp/SWE-bench_Lite", "princeton-nlp/SWE-bench_Verified"],
    )
    parser.add_argument("--test_jsonl", type=str)
    parser.add_argument("--load", action="store_true")

    args = parser.parse_args()

    # then load and run production tests on the results
    _run_reproduction_tests(args)


if __name__ == "__main__":
    main()
