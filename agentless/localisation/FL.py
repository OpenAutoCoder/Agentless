import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Tuple, List, Any, Dict

from Agentless.agentless.util.compress_file import get_skeleton
from Agentless.agentless.util.model import make_model
from Agentless.agentless.util.postprocess_data import extract_code_blocks, extract_locs_for_files
from Agentless.agentless.util.preprocess_data import transfer_arb_locs_to_locs, line_wrap_content, \
    show_project_structure, get_full_file_paths_and_classes_and_functions, get_repo_files
from apps.helper import read_file


class FL(ABC):
    def __init__(self, instance_id, structure, requirement, test_step, **kwargs):
        self.requirement = requirement
        self.structure = structure
        self.instance_id = instance_id
        self.test_step = test_step

    @abstractmethod
    def localize(self, top_n=1, mock=False) -> tuple[list, list, list, any]:
        pass


def construct_topn_file_context(
        file_to_locs,
        file_contents,
        structure,
        context_window: int,
        loc_interval: bool = True,
        fine_grain_loc_only: bool = False,
        sticky_scroll: bool = False,
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
                sticky_scroll=sticky_scroll,
            )
            topn_content += f"### {pred_file}\n{file_loc_content}\n\n\n"
            file_loc_intervals[pred_file] = context_intervals

    return topn_content, file_loc_intervals


def _parse_model_return_lines(content: str) -> list[str]:
    return content.strip().split("\n")


class LLMFL(FL):
    obtain_relevant_files_prompt = """
You are an expert in writing test code that covers specific criteria within the automotive zone controller domain using a private framework repository called TAF (Test Automotive Framework).

The repository TAF provides the necessary tools for developing test code related to the zone controller. Your task is to analyze the following requirement and identify the files needed to create test code that fulfills this specific requirement.

### Requirement ###
{requirement}

###

### Repository Structure ###
{structure}

###

Please provide the full paths of up to 5 files required to write the test code for the specified requirement. The files should be listed in order of importance, from most to least relevant, and each file path should be enclosed in triple backticks.

Here Some Examples:
{examples}

## Strict Guidelines:
- **Do not invent or fabricate any file names or paths**; only include those present in the provided repository structure.
- **Focus solely on the requirement provided** and identify files that directly contribute to fulfilling this requirement.
- Adherence to these guidelines is critical. Any deviation, such as creating non-existent file names, will lead to immediate disqualification from the task.
"""

    file_content_in_block_template = """
### File: {file_name} ###
```python
{file_content}
```
"""
    obtain_relevant_code_combine_top_n_prompt = """
You are an expert in test code implementation within the automotive zone controller domain, utilizing a private test framework repository known as TAF (Test Automotive Framework). Your task is to analyze the provided test step and file contents to determine the exact method names needed to implement the test step. The requirement is included solely to provide additional context.

### Requirement (for context) ###
{requirement}

### Test Step ###
{test_step}

### File Contents ###
{file_contents}

###

Your goal is to identify and list the method names that are relevant to the implementation of the test step. These method names should come directly from the provided file contents.

### Formatting Guidelines:
- Return the locations in the following format, including the full file path, class name, and method name.
- List all relevant method names under each file path, ordered by their occurrence in the file.

### Examples:
```
full_path1/file1.py
full_path1.file1.MyClass1: my_method

full_path2/file2.py
full_path2.file2.MyClass2: my_method2
full_path2.file2.MyClass2: my_method2_2
full_path2.file2.MyClass2: my_method2_3


full_path3/file3.py
full_path3.file3.MyClass3: my_method3
full_path2.file2.MyClass2: my_method3_2


full_path4/file4.py
full_path4.file4.MyClass4: my_method4
```

Return only the location(s) in the format shown above.

## Strict Rules:
- **Do not invent or fabricate any method names**; only use those that are found in the provided file contents.
- **Return the results in the exact format specified** above.
- **Focus solely on what supports the specific test step**, using the requirement as contextual information only.
- Adherence to these rules is mandatory. Any deviation, such as generating method names not present in the files, will result in immediate termination of the task.
"""
    obtain_relevant_functions_and_vars_from_compressed_files_prompt_more = """
You are an expert in test code implementation within the automotive zone controller domain, using a private test framework repository known as TAF (Test Automotive Framework). Your task is to analyze the provided test step and the skeleton of relevant files to identify all the necessary locations (method names) required to cover the test step. The requirement is included solely to provide additional context.

### Requirement (for context) ###
{requirement}

###

### Test Step ###
{test_step}

### Skeleton of Relevant Files ###
{file_contents}

###

Your objective is to identify and list all method names relevant to implementing the test step, including directly related methods and any potentially related methods. The method names should be derived only from the provided file skeleton.

### Formatting Guidelines:
- Return the locations in the following format, including the full file path, class name, and method name.
- List all relevant method names under each file path, ordered by their occurrence in the file.

### Examples:
```
full_path1/file1.py
full_path1.file1.MyClass1: my_method

full_path2/file2.py
full_path2.file2.MyClass2: my_method2
full_path2.file2.MyClass2: my_method2_2
full_path2.file2.MyClass2: my_method2_3


full_path3/file3.py
full_path3.file3.MyClass3: my_method3
full_path2.file2.MyClass2: my_method3_2


full_path4/file4.py
full_path4.file4.MyClass4: my_method4
```


Return only the locations.

## Strict Rules:
- **Do not invent or fabricate any method names**; only use those found in the provided file skeleton.
- **Return the results in the exact format specified** above.
- **Focus solely on what supports the specific test step**, using the requirement as contextual information only.
- Adherence to these rules is mandatory. Any deviation, such as generating method names not present in the files, will result in immediate termination of the task.

"""

    def __init__(
            self, instance_id, structure, requirement, test_step, model_name,
    ):
        super().__init__(instance_id, structure, requirement=requirement, test_step=test_step)
        self.max_tokens = 300
        self.model_name = model_name

    def extract_examples(self, current):
        final_examples = ""
        tools = os.listdir(
            "datasets/datasets/required_tools"
        )
        for i in range(len(tools)):
            tool = tools[i]
            if tool.replace(".json", "") == current:
                continue
            path_req = os.path.join(
                "datasets/datasets/requirements", tool.replace(".json", ".txt"))
            final_examples += f"\n- example {i + 1} \n\n**Requirement:**\n"
            requirement = read_file(path_req)
            final_examples += requirement + "\n\n**Result:**\n```"
            tools_val = json.loads(open(os.path.join("datasets/datasets/required_tools", tool)).read())
            for p in tools_val.keys():
                struct = str(p).split(".")

                if "/".join(struct).lower() == "taf/core/time":
                    file = "/".join(struct) + "/__init__.py"
                else:
                    final = struct[-1]
                    if final[0].isupper():
                        struct.pop()
                    else:
                        struct.append("__init__")
                    file = "/".join(struct) + ".py"

                final_examples += f"\n{file}"

            final_examples += "\n```"
        return final_examples

    def localize(self, current, top_n=1) -> tuple[list[Any], dict[str, Any], Any]:

        found_files = []
        examples = self.extract_examples(current)

        message = self.obtain_relevant_files_prompt.format(
            requirement=self.requirement,
            structure=show_project_structure(self.structure).strip(),
            examples=examples,
        ).strip()
        print(f"prompting with message:\n{message}")
        print("=" * 80)

        model = make_model(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]
        model_found_files = _parse_model_return_lines(raw_output)

        files, classes, functions = get_full_file_paths_and_classes_and_functions(
            self.structure
        )

        for file_content in files:
            file = file_content[0]
            if file in model_found_files:
                found_files.append(file)

        # sort based on order of appearance in model_found_files
        found_files = sorted(found_files, key=lambda x: model_found_files.index(x))

        print(raw_output)

        return (
            found_files,
            {"raw_output_files": raw_output},
            traj,
        )

    def localize_function_from_compressed_files(self, file_names):

        file_contents = get_repo_files(self.structure, file_names)
        compressed_file_contents = {
            fn: get_skeleton(code) for fn, code in file_contents.items()
        }
        contents = [
            self.file_content_in_block_template.format(file_name=fn, file_content=code)
            for fn, code in compressed_file_contents.items()
        ]
        file_contents = "".join(contents)
        template = (
            self.obtain_relevant_functions_and_vars_from_compressed_files_prompt_more
        )
        message = template.format(
            requirement=self.requirement, test_step=self.test_step, file_contents=file_contents
        )
        logging.info(f"prompting with message:\n{message}")
        logging.info("=" * 80)

        model = make_model(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]
        model_found_locs = extract_code_blocks(raw_output)
        model_found_locs_separated = extract_locs_for_files(
            model_found_locs, file_names
        )

        logging.info(f"==== raw output ====")
        logging.info(raw_output)
        logging.info("=" * 80)
        logging.info(f"==== extracted locs ====")
        for loc in model_found_locs_separated:
            logging.info(loc)
        logging.info("=" * 80)

        print(raw_output)

        return model_found_locs_separated, {"raw_output_loc": raw_output}, traj

    def localize_line_from_coarse_function_locs(
            self,
            file_names,
            coarse_locs,
            context_window: int,
            sticky_scroll: bool,
            temperature: float = 0.0,
            num_samples: int = 1,
    ):

        file_contents = get_repo_files(self.structure, file_names)
        topn_content, file_loc_intervals = construct_topn_file_context(
            coarse_locs,
            file_contents,
            self.structure,
            context_window=context_window,
            loc_interval=True,
            sticky_scroll=sticky_scroll,
        )
        template = self.obtain_relevant_code_combine_top_n_prompt
        message = template.format(
            requirement=self.requirement, test_step=self.test_step, file_contents=topn_content
        )
        logging.info(f"prompting with message:\n{message}")
        logging.info("=" * 80)

        model = make_model(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=temperature,
            batch_size=num_samples,
        )
        raw_trajs = model.codegen(message, num_samples=num_samples)

        # Merge trajectories
        raw_outputs = [raw_traj["response"] for raw_traj in raw_trajs]
        traj = {
            "prompt": message,
            "response": raw_outputs,
            "usage": {  # merge token usage
                "completion_tokens": sum(
                    raw_traj["usage"]["completion_tokens"] for raw_traj in raw_trajs
                ),
                "prompt_tokens": sum(
                    raw_traj["usage"]["prompt_tokens"] for raw_traj in raw_trajs
                ),
            },
        }
        model_found_locs_separated_in_samples = []
        for raw_output in raw_outputs:
            model_found_locs = extract_code_blocks(raw_output)
            model_found_locs_separated = extract_locs_for_files(
                model_found_locs, file_names
            )
            model_found_locs_separated_in_samples.append(model_found_locs_separated)

            logging.info(f"==== raw output ====")
            logging.info(raw_output)
            logging.info("=" * 80)
            print(raw_output)
            print("=" * 80)
            logging.info(f"==== extracted locs ====")
            for loc in model_found_locs_separated:
                logging.info(loc)
            logging.info("=" * 80)
        logging.info("==== Input coarse_locs")
        coarse_info = ""
        for fn, found_locs in coarse_locs.items():
            coarse_info += f"### {fn}\n"
            if isinstance(found_locs, str):
                coarse_info += found_locs + "\n"
            else:
                coarse_info += "\n".join(found_locs) + "\n"
        logging.info("\n" + coarse_info)
        if len(model_found_locs_separated_in_samples) == 1:
            model_found_locs_separated_in_samples = (
                model_found_locs_separated_in_samples[0]
            )

        return (
            model_found_locs_separated_in_samples,
            {"raw_output_loc": raw_outputs},
            traj,
        )
