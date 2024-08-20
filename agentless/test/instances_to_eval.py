import json

import jsonlines
from datasets import load_dataset

ds = load_dataset("princeton-nlp/SWE-bench_Lite")

with jsonlines.open("successful_tests.jsonl", mode="w") as writer:
    for entry in ds["test"]:
        instance_id = entry["instance_id"]
        file_path = f"logs/run_evaluation/regression_tests_get_passing_tests_0/test/{instance_id}/report.json"
        tests = eval(entry["PASS_TO_PASS"])

        with open(file_path, "r") as file:
            data = json.load(file)
        tests_which_pass = (
            data.get(instance_id, {})
            .get("tests_status", {})
            .get("PASS_TO_PASS", {})
            .get("success", [])
        )

        if tests_which_pass == []:
            print(
                f" {instance_id} didn't get any passing tests for its original {len(tests)}"
            )

        # Create the dictionary for each entry
        result_entry = {
            "instance_id": instance_id,
            "tests_passing_in_original_repo": tests_which_pass,
            "original_pass_to_pass": tests,
        }

        # Write the entry to the JSONL file
        writer.write(result_entry)
