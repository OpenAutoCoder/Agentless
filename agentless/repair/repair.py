import argparse
import concurrent.futures
import json
import os
from difflib import unified_diff
from threading import Lock

from datasets import load_dataset
from tqdm import tqdm

from agentless.util.api_requests import num_tokens_from_messages
from agentless.util.model import make_model
from agentless.util.postprocess_data import (
    check_code_differ_by_just_empty_lines,
    check_syntax,
    extract_python_blocks,
    fake_git_repo,
    lint_code,
    parse_diff_edit_commands,
    parse_edit_commands,
    parse_str_replace_edit_commands,
    split_edit_multifile_commands,
)
from agentless.util.preprocess_data import (
    get_full_file_paths_and_classes_and_functions,
    get_repo_structure,
    line_wrap_content,
    transfer_arb_locs_to_locs,
)
from agentless.util.utils import cleanup_logger, load_jsonl, setup_logger

repair_relevant_file_instruction = """
Below are some code segments, each from a relevant file. One or more of these files may contain bugs.
"""
repair_prompt_combine_topn = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

{repair_relevant_file_instruction}
--- BEGIN FILE ---
```
{content}
```
--- END FILE ---

Please generate `edit_file` commands to fix the issue.

The `edit_file` command takes four arguments:

edit_file(filename: str, start: int, end: int, content: str) -> None:
    Edit a file. It replaces lines `start` through `end` (inclusive) with the given text `content` in the open file.
    Args:
    filename: str: The full file name to edit.
    start: int: The start line number. Must satisfy start >= 1.
    end: int: The end line number. Must satisfy start <= end <= number of lines in the file.
    content: str: The content to replace the lines with.

Please note that THE `edit_file` FUNCTION REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the `edit_file` command in blocks ```python...```.
"""


repair_prompt_combine_topn_cot = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

{repair_relevant_file_instruction}
--- BEGIN FILE ---
```
{content}
```
--- END FILE ---

Please first localize the bug based on the issue statement, and then generate `edit_file` commands to fix the issue.

The `edit_file` command takes four arguments:

edit_file(filename: str, start: int, end: int, content: str) -> None:
    Edit a file. It replaces lines `start` through `end` (inclusive) with the given text `content` in the open file.
    Args:
    filename: str: The full file name to edit.
    start: int: The start line number. Must satisfy start >= 1.
    end: int: The end line number. Must satisfy start <= end <= number of lines in the file.
    content: str: The content to replace the lines with.

Please note that THE `edit_file` FUNCTION REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the `edit_file` command in blocks ```python...```.
"""


repair_prompt_combine_topn_cot_diff = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

{repair_relevant_file_instruction}
--- BEGIN FILE ---
```
{content}
```
--- END FILE ---

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
"""

repair_prompt_combine_topn_cot_str_replace = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

{repair_relevant_file_instruction}
--- BEGIN FILE ---
```
{content}
```
--- END FILE ---

Please first localize the bug based on the issue statement, and then generate editing commands to fix the issue.
"""


def _post_process_multifile_repair(
    raw_output: str,
    file_contents: dict[str, str],
    logger,
    file_loc_intervals: dict[str, list],
    diff_format=False,
    str_replace_format=False,
) -> tuple[list[str], list[str]]:
    if not str_replace_format:
        edit_multifile_commands = extract_python_blocks(raw_output)
    else:
        edit_multifile_commands = raw_output
    edited_files = []
    new_contents = []
    try:
        file_to_commands = split_edit_multifile_commands(
            edit_multifile_commands,
            diff_format=diff_format,
            str_replace_format=str_replace_format,
        )
    except Exception as e:
        logger.error(e)
        return edited_files, new_contents

    logger.info("=== file_to_commands: ===")
    logger.info(json.dumps(file_to_commands, indent=2))

    for edited_file_key in file_to_commands:
        edited_file = ""
        new_content = ""
        try:
            logger.info(f"=== edited_file: {edited_file_key} ===")
            edit_commands = file_to_commands[edited_file_key]
            logger.info("=== edit_commands: ===")
            for c in edit_commands:
                logger.info(c)
                logger.info("\n" + "-" * 40)
            edited_file = eval(edited_file_key)  # convert '"file.py"' to 'file.py'
            content = file_contents[edited_file]
            if diff_format:
                new_content = parse_diff_edit_commands(
                    edit_commands, content, file_loc_intervals[edited_file]
                )
            elif str_replace_format:
                new_content = parse_str_replace_edit_commands(
                    edit_commands, content, file_loc_intervals[edited_file]
                )
            else:
                new_content = parse_edit_commands(edit_commands, content)
        except Exception as e:
            logger.error(e)
            edited_file = ""
            new_content = ""

        if edited_file == "" or new_content == "":
            continue
        edited_files.append(edited_file)
        new_contents.append(new_content)
        diff = list(
            unified_diff(
                content.split("\n"),
                new_content.split("\n"),
                fromfile=edited_file,
                tofile=edited_file,
                lineterm="",
            )
        )

        logger.info(f"extracted patch:")
        logger.info("\n".join(diff))
        print("\n".join(diff))

    return edited_files, new_contents


def construct_topn_file_context(
    file_to_locs,
    pred_files,
    file_contents,
    structure,
    context_window: int,
    loc_interval: bool = True,
    fine_grain_loc_only: bool = False,
    add_space: bool = False,
    sticky_scroll: bool = False,
    no_line_number: bool = True,
):
    """Concatenate provided locations to form a context.

    loc: {"file_name_1": ["loc_str_1"], ...}
    """
    file_loc_intervals = dict()
    topn_content = ""

    for pred_file, locs in file_to_locs.items():
        content = file_contents[pred_file]
        line_locs, context_intervals = transfer_arb_locs_to_locs(
            locs,
            structure,
            pred_file,
            context_window,
            loc_interval,
            fine_grain_loc_only,
            file_content=file_contents[pred_file] if pred_file in file_contents else "",
        )

        if len(line_locs) > 0:
            # Note that if no location is predicted, we exclude this file.
            file_loc_content = line_wrap_content(
                content,
                context_intervals,
                add_space=add_space,
                no_line_number=no_line_number,
                sticky_scroll=sticky_scroll,
            )
            topn_content += f"### {pred_file}\n{file_loc_content}\n\n\n"
            file_loc_intervals[pred_file] = context_intervals

    return topn_content, file_loc_intervals


def process_loc(loc, args, swe_bench_data, prev_o, write_lock=None):
    instance_id = loc["instance_id"]

    if args.target_id is not None:
        if args.target_id != instance_id:
            return

    log_file = os.path.join(args.output_folder, "repair_logs", f"{instance_id}.log")
    logger = setup_logger(log_file)
    found = False
    for o in prev_o:
        if o["instance_id"] == instance_id:
            found = True
            break

    if found:
        logger.info(f"skipping {instance_id} since patch already generated")
        return None

    logger.info(f"================ repairing {instance_id} ================")
    if len(loc["found_files"]) == 0:
        if write_lock is not None:
            write_lock.acquire()
        with open(args.output_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "raw_output": [""],
                        "try_count": [0],
                        "all_generations": [[]],
                        "traj": [],
                        "prev_content": [[]],
                        "file_names": [[]],
                    }
                )
                + "\n"
            )
        if write_lock is not None:
            write_lock.release()
        return

    pred_files = loc["found_files"][: args.top_n]
    bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
    problem_statement = bench_data["problem_statement"]
    structure = get_repo_structure(
        instance_id, bench_data["repo"], bench_data["base_commit"], "playground"
    )
    files, _, _ = get_full_file_paths_and_classes_and_functions(structure)
    raw_outputs, counts, all_generations, traj, prev_contents, file_names = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    raw_output = ""
    topn_content = ""
    # Construct file contents
    file_contents = dict()
    for i, pred_file in enumerate(pred_files):
        content = None
        for file_content in files:
            if file_content[0] == pred_file:
                content = "\n".join(file_content[1])
                file_contents[pred_file] = content
                break

        assert content is not None, f"{pred_file} file not found"
    # Construct top-n file context
    file_to_edit_locs = dict()

    if "found_edit_locs" in loc:
        file_to_edit_locs = loc["found_edit_locs"]

    topn_content, file_loc_intervals = construct_topn_file_context(
        file_to_edit_locs,
        pred_files,
        file_contents,
        structure,
        context_window=args.context_window,
        loc_interval=args.loc_interval,
        fine_grain_loc_only=args.fine_grain_loc_only,
        add_space=args.add_space,
        no_line_number=args.diff_format or args.str_replace_format,
        sticky_scroll=args.sticky_scroll,
    )

    if topn_content.strip() == "":
        if write_lock is not None:
            write_lock.acquire()
        with open(args.output_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "raw_output": [""],
                        "try_count": [0],
                        "all_generations": [[]],
                        "traj": [],
                        "prev_content": [[]],
                        "file_names": [[]],
                    }
                )
                + "\n"
            )
        if write_lock is not None:
            write_lock.release()
        return

    prompt_template = (
        repair_prompt_combine_topn_cot_str_replace
        if args.cot and args.str_replace_format
        else repair_prompt_combine_topn_cot_diff
        if args.cot and args.diff_format
        else repair_prompt_combine_topn_cot
        if args.cot
        else repair_prompt_combine_topn
    )
    file_instruction = repair_relevant_file_instruction
    message = prompt_template.format(
        repair_relevant_file_instruction=file_instruction,
        problem_statement=problem_statement,
        content=topn_content.rstrip(),
    ).strip()
    logger.info(f"prompting with message:\n{message}")

    all_generations, counts, traj, prev_contents, file_names = [], [], [], [], []
    sample_responses = []
    # get greedy sample
    model = make_model(
        model=args.model,
        logger=logger,
        backend=args.backend,
        max_tokens=1024,
        temperature=0,
        batch_size=1,
    )
    if args.skip_greedy:
        greedy_traj = {
            "response": "",
            "usage": {
                "completion_tokens": 0,
                "prompt_tokens": 0,
            },
        }
    else:
        if args.mock:
            greedy_traj = {
                "response": "",
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, args.model),
                },
            }
        else:
            if args.str_replace_format:
                greedy_traj = model.codegen_w_tool(
                    message, num_samples=1, prompt_cache=args.max_samples > 1
                )[0]
            else:
                greedy_traj = model.codegen(
                    message, num_samples=1, prompt_cache=args.max_samples > 1
                )[0]

    sample_responses.append(greedy_traj)
    # get temperature samples
    model = make_model(
        model=args.model,
        logger=logger,
        backend=args.backend,
        max_tokens=1024,
        temperature=0.8,
        batch_size=args.max_samples - 1,  # minus the 1 greedy sample
    )

    if args.mock:
        first_traj = {
            "response": "",
            "usage": {
                "prompt_tokens": num_tokens_from_messages(message, args.model),
            },
        }
        later_traj = {
            "response": "",
            "usage": {"prompt_tokens": 0},
        }
        if args.max_samples - 1:
            sample_trajs = [first_traj] + [later_traj] * (args.max_samples - 2)
        else:
            sample_trajs = []
    else:
        if args.max_samples - 1:
            # always use cached prompt if possible for later samples
            if args.str_replace_format:
                sample_trajs = model.codegen_w_tool(
                    message, num_samples=args.max_samples - 1, prompt_cache=True
                )
            else:
                sample_trajs = model.codegen(
                    message, num_samples=args.max_samples - 1, prompt_cache=True
                )
        else:
            sample_trajs = []

    sample_responses.extend(sample_trajs)

    count = 0
    while count < args.max_samples:
        print(f"trying the {count + 1}-th sample ...")
        ret = sample_responses[count]
        count += 1
        traj.append({**ret, "prompt": message})

        if args.mock:
            continue

        raw_output = ret["response"]
        logger.info(f"raw output:\n{raw_output}")
        all_generations.append(raw_output)
        edited_files, new_contents = _post_process_multifile_repair(
            raw_output,
            file_contents,
            logger,
            file_loc_intervals,
            diff_format=args.diff_format,
            str_replace_format=args.str_replace_format,
        )

        if len(new_contents) == 0:
            prev_contents.append("")
            file_names.append("")
        else:
            prev_content = [file_contents[edited_file] for edited_file in edited_files]
            prev_contents.append(prev_content)
            file_names.append(edited_files)

        counts.append(count)
        raw_outputs.append(raw_output)

    if write_lock is not None:
        write_lock.acquire()
    with open(args.output_file, "a") as f:
        f.write(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "raw_output": raw_outputs,
                    "all_generations": [all_generations],
                    "try_count": counts,
                    "traj": traj,
                    "prev_content": [prev_contents],
                    "file_names": [file_names],
                }
            )
            + "\n"
        )
    if write_lock is not None:
        write_lock.release()


def repair(args):
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    swe_bench_data = load_dataset(args.dataset, split="test")
    locs = load_jsonl(args.loc_file)
    prev_o = load_jsonl(args.output_file) if os.path.exists(args.output_file) else []

    with open(f"{args.output_folder}/used_locs.jsonl", "w") as f:
        for loc in locs:
            f.write(json.dumps(loc) + "\n")

    if args.num_threads == 1:
        for loc in tqdm(locs, total=len(locs), colour="MAGENTA"):
            process_loc(loc, args, swe_bench_data, prev_o)
    else:
        write_lock = Lock()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_threads
        ) as executor:
            futures = {
                executor.submit(
                    process_loc, loc, args, swe_bench_data, prev_o, write_lock
                ): loc
                for loc in locs
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(locs),
                colour="MAGENTA",
            ):
                future.result()


def post_process_raw_output(
    raw_output_text, file_contents, logger, file_loc_intervals, args
):
    git_diffs = ""
    raw_git_diffs = ""
    edited_files, new_contents, contents = [], [], []
    try:
        edited_files, new_contents = _post_process_multifile_repair(
            raw_output_text,
            file_contents,
            logger,
            file_loc_intervals,
            diff_format=args.diff_format,
            str_replace_format=args.str_replace_format,
        )

        contents = [file_contents[edited_file] for edited_file in edited_files]

        git_diff = fake_git_repo("playground", edited_files, contents, new_contents)

        raw_git_diffs += "\n" + git_diff.replace("\ No newline at end of file\n", "")

        syntax_success = check_syntax(new_contents)

        differ_by_empty_lines = check_code_differ_by_just_empty_lines(
            new_contents, contents
        )

        logger.info(f"{differ_by_empty_lines = }")
        if syntax_success and not differ_by_empty_lines:
            git_diffs = raw_git_diffs
        else:
            git_diffs = ""  # no need to evaluate
    except Exception as e:
        print(raw_output_text)
        print(e)

    return git_diffs, raw_git_diffs, contents, edited_files, new_contents


def post_process_repair(args):
    """
    apply some diff formatting.
    """
    raw_outputs = load_jsonl(args.raw_output_file)
    locs = load_jsonl(args.loc_file)

    for raw_output in raw_outputs:
        instance_id = raw_output["instance_id"]
        log_file = os.path.join(args.output_folder, "repair_logs", f"{instance_id}.log")
        logger = setup_logger(log_file)

        if raw_output["raw_output"] == "":
            with open(args.output_file, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "model_name_or_path": "agentless",
                            "instance_id": instance_id,
                            "model_patch": "",
                        }
                    )
                    + "\n"
                )
            continue

        if args.select_id == -1:
            # Use the last generation
            assert False, "not implemented for now"
        else:
            # Use the indexed generation
            generation_idx = args.select_id
            try:
                raw_output_text = raw_output["all_generations"][0][generation_idx]
                original_file_content = raw_output["prev_content"][0][generation_idx]
                pred_file = raw_output["file_names"][0][generation_idx]

                pred_files = [loc for loc in locs if loc["instance_id"] == instance_id][
                    0
                ]["found_files"][: args.top_n]

                git_diffs = ""
                raw_git_diffs = ""
                if isinstance(raw_output["raw_output"], str):
                    # for backward compatibility
                    raw_output["raw_output"] = [raw_output["raw_output"]]

                if isinstance(original_file_content, str):
                    original_file_content = [original_file_content]
                    pred_file = [pred_file]

                file_contents = {
                    file_name: o_file_content
                    for file_name, o_file_content in zip(
                        pred_file, original_file_content
                    )
                }

                file_loc_intervals = dict()

                loc = [loc for loc in locs if loc["instance_id"] == instance_id][0]

                for i, tmp_pred_file in enumerate(pred_files):
                    if tmp_pred_file not in pred_file:
                        continue
                    if (
                        "found_edit_locs" in loc
                        and tmp_pred_file in loc["found_edit_locs"]
                    ):
                        line_locs, context_intervals = transfer_arb_locs_to_locs(
                            loc["found_edit_locs"][tmp_pred_file],
                            None,
                            loc["found_files"][i],
                            args.context_window,
                            args.loc_interval,
                            args.fine_grain_loc_only,
                            file_content=file_contents[tmp_pred_file]
                            if tmp_pred_file in file_contents
                            else "",
                        )
                    else:
                        line_locs, context_intervals = [], []  # default values.

                    file_loc_intervals[tmp_pred_file] = context_intervals
            except Exception as e:
                logger.info(e)
                print(e)
                raw_output_text = ""

        if raw_output_text:
            (
                git_diffs,
                raw_git_diffs,
                content,
                edited_files,
                new_contents,
            ) = post_process_raw_output(
                raw_output_text, file_contents, logger, file_loc_intervals, args
            )
        else:
            git_diffs = ""
            raw_git_diffs = ""
            content = []
            edited_files = []
            new_contents = []

        with open(args.output_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "model_name_or_path": "agentless",
                        "instance_id": instance_id,
                        "model_patch": git_diffs.lstrip(),
                        "raw_model_patch": raw_git_diffs.lstrip(),
                        "original_file_content": content,
                        "edited_files": edited_files,
                        "new_file_content": new_contents,
                    }
                )
                + "\n"
            )
        cleanup_logger(logger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc_file", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=1)
    parser.add_argument("--loc_interval", action="store_true")
    parser.add_argument("--context_window", type=int, default=10)
    parser.add_argument("--gen_and_process", action="store_true")
    parser.add_argument("--max_samples", type=int, default=20, help="Sampling budget.")
    parser.add_argument(
        "--select_id",
        type=int,
        default=-1,
        help="Index the selected samples during post-processing.",
    )
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
    parser.add_argument("--post_process", action="store_true")
    parser.add_argument("--add_space", action="store_true")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--fine_grain_loc_only", action="store_true")
    parser.add_argument("--diff_format", action="store_true")
    parser.add_argument("--str_replace_format", action="store_true")
    parser.add_argument("--skip_greedy", action="store_true")
    parser.add_argument("--sticky_scroll", action="store_true")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for creating API requests",
    )
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

    args = parser.parse_args()

    assert (not "deepseek" in args.model) or (
        args.backend == "deepseek"
    ), "Must specify `--backend deepseek` if using a DeepSeek model"

    # diff_format and str_replace_format cannot be both True
    assert not (
        args.diff_format and args.str_replace_format
    ), "Cannot use both diff_format and str_replace_format"

    # str_replace_format only supported with anthropic backend
    assert not (
        args.str_replace_format and args.backend != "anthropic"
    ), "str_replace_format only supported with anthropic backend"

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    if not os.path.exists(os.path.join(args.output_folder, "repair_logs")):
        os.makedirs(os.path.join(args.output_folder, "repair_logs"))

    args.output_file = os.path.join(args.output_folder, "output.jsonl")

    if args.post_process:
        args.raw_output_file = args.output_file
        if args.select_id == -1:
            args.output_file = args.raw_output_file.replace(
                ".jsonl", "_processed.jsonl"
            )
        else:
            args.output_file = args.raw_output_file.replace(
                ".jsonl", f"_{args.select_id}_processed.jsonl"
            )
        post_process_repair(args)
    elif args.gen_and_process:
        repair(args)
        args.raw_output_file = args.output_file
        for i in range(args.max_samples):
            args.output_file = args.raw_output_file.replace(
                ".jsonl", f"_{i}_processed.jsonl"
            )
            args.select_id = i
            post_process_repair(args)
    else:
        repair(args)


if __name__ == "__main__":
    main()
