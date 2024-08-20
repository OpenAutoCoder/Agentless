import argparse
import json

import jsonlines
from datasets import load_dataset


def main(output_jsonl_path, input_folder_path):
    ds = load_dataset("princeton-nlp/SWE-bench_Lite")

    with jsonlines.open(output_jsonl_path, mode="w") as writer:
        for entry in ds["test"]:
            instance_id = entry["instance_id"]
            file_path = f"{input_folder_path}/test/{instance_id}/report.json"
            tests = eval(entry["PASS_TO_PASS"])

            try:
                with open(file_path, "r") as file:
                    data = json.load(file)
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                continue

            tests_which_pass = (
                data.get(instance_id, {})
                .get("tests_status", {})
                .get("PASS_TO_PASS", {})
                .get("success", [])
            )

            if tests_which_pass == []:
                print(
                    f"{instance_id} didn't get any passing tests for its original {len(tests)} tests"
                )

            # Create the dictionary for each entry
            result_entry = {
                "instance_id": instance_id,
                "tests_passing_in_original_repo": tests_which_pass,
                "original_pass_to_pass": tests,
            }

            # Write the entry to the JSONL file
            writer.write(result_entry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some paths.")
    parser.add_argument(
        "--output_jsonl_path",
        type=str,
        required=True,
        help="Path to the output JSONL file",
    )
    parser.add_argument(
        "--input_folder_path",
        type=str,
        required=True,
        help="Path to the input folder containing the logs",
    )

    args = parser.parse_args()
    main(args.output_jsonl_path, args.input_folder_path)
