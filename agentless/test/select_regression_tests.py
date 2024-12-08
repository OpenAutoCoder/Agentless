import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm

from agentless.util.api_requests import num_tokens_from_messages
from agentless.util.model import make_model
from agentless.util.utils import load_jsonl, setup_logger

MAX_CONTEXT_LENGTH = 128000


select_tests_prompt_template = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Below are a list of existing tests in the repository.
```
{passing_tests}
```

Please identify the tests that should not be run after applying the patch to fix the issue.
These tests should be excluded as the original functionality may change due to the patch.

### Example
```
test1
test2
test5
```
Return only the selected tests.
"""


def select_test(instance_id, args, swe_bench_data, prev_o, passing_tests):
    def _parse_model_return_lines(content: str) -> list[str]:
        if content:
            return content.strip().split("\n")

    if args.target_id is not None:
        if args.target_id != instance_id:
            return

    log_file = os.path.join(
        args.output_folder, "select_test_logs", f"{instance_id}.log"
    )
    logger = setup_logger(log_file)
    found = False
    for o in prev_o:
        if o["instance_id"] == instance_id:
            found = True
            break

    if found:
        logger.info(f"skipping {instance_id}")
        return None

    if len(passing_tests) == 0:
        with open(args.output_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "raw_output": "",
                        "tests_passing_in_original_repo": passing_tests,
                        "original_regressions": passing_tests,
                        "traj": {},
                    }
                )
                + "\n"
            )
        return

    logger.info(f"================ selecting test for {instance_id} ================")

    bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
    problem_statement = bench_data["problem_statement"]

    raw_output = ""

    prompt_template = select_tests_prompt_template
    message = prompt_template.format(
        problem_statement=problem_statement, passing_tests="\n".join(passing_tests)
    ).strip()

    logger.info(f"prompting with message:\n{message}")

    # get greedy sample
    model = make_model(
        model=args.model,
        logger=logger,
        backend=args.backend,
        max_tokens=1024,
        temperature=0,
        batch_size=1,
    )

    def message_too_long(message):
        return num_tokens_from_messages(message, args.model) >= MAX_CONTEXT_LENGTH

    while message_too_long(message):
        # half it
        # TODO: we can prompt the model multiple times to select from different subset of tests
        passing_tests = passing_tests[: len(passing_tests) // 2]
        message = prompt_template.format(
            problem_statement=problem_statement, passing_tests="\n".join(passing_tests)
        ).strip()

    greedy_traj = model.codegen(message, num_samples=1)[0]
    greedy_traj["prompt"] = message
    raw_output = greedy_traj["response"]

    logger.info(raw_output)

    model_identified_tests = _parse_model_return_lines(raw_output)
    subset_regression_tests = []

    for x in passing_tests:
        if x not in model_identified_tests:
            subset_regression_tests.append(x)

    with open(args.output_file, "a") as f:
        f.write(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "raw_output": raw_output,
                    "tests_passing_in_original_repo": subset_regression_tests,
                    "original_regressions": passing_tests,
                    "traj": greedy_traj,
                }
            )
            + "\n"
        )


def select_tests(args):
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    instance_test_dict = {}
    with open(args.passing_tests, "r") as file:
        for line in file:
            json_obj = json.loads(line.strip())
            instance_id = json_obj["instance_id"]
            test = json_obj["tests_passing_in_original_repo"]
            instance_test_dict[instance_id] = test

    swe_bench_data = load_dataset(args.dataset, split="test")
    instance_ids = (
        swe_bench_data["instance_id"]
        if args.instance_ids is None
        else args.instance_ids
    )
    prev_o = load_jsonl(args.output_file) if os.path.exists(args.output_file) else []

    results = []

    for instance_id in tqdm(instance_ids, total=len(instance_ids), colour="MAGENTA"):
        result = select_test(
            instance_id, args, swe_bench_data, prev_o, instance_test_dict[instance_id]
        )
        if result is not None:
            results.append(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=[
            "gpt-4o-2024-05-13",
            "deepseek-coder",
            "gpt-4o-mini-2024-07-18",
            "claude-3-5-sonnet-20241022",
        ],
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=["openai", "deepseek", "anthropic"],
    )
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--target_id", type=str)
    parser.add_argument(
        "--mock", action="store_true", help="Mock run to compute prompt tokens."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        choices=["princeton-nlp/SWE-bench_Lite", "princeton-nlp/SWE-bench_Verified"],
    )
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to run (space separated), if not provided, all instances will be run",
    )
    parser.add_argument("--passing_tests", type=str, required=True)

    args = parser.parse_args()

    assert (not "deepseek" in args.model) or (
        args.backend == "deepseek"
    ), "Must specify `--backend deepseek` if using a DeepSeek model"

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    if not os.path.exists(os.path.join(args.output_folder, "select_test_logs")):
        os.makedirs(os.path.join(args.output_folder, "select_test_logs"))

    args.output_file = os.path.join(args.output_folder, "output.jsonl")
    select_tests(args)


if __name__ == "__main__":
    main()
