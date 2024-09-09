import json
import logging
import os
from abc import ABC, abstractmethod
from json import JSONDecodeError
from typing import Any

from langsmith import traceable

from Agentless.agentless.util.compress_file import get_skeleton
from Agentless.agentless.util.model import make_model
from Agentless.agentless.util.postprocess_data import extract_code_blocks, extract_locs_for_files
from Agentless.agentless.util.preprocess_data import transfer_arb_locs_to_locs, line_wrap_content, \
    show_project_structure, get_full_file_paths_and_classes_and_functions, get_repo_files
from apps.helper import read_file
from apps.services.code_skeleton_extractor import filtered_nodes_by_label


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
You are an expert in test code implementation within the automotive zone controller domain, using a custom test framework repository called TAF (Test Automotive Framework). 
Your task is to analyze the provided test step and test framework file contents to accurately identify the method names needed to implement the test step. Please note that the requirements are provided for context only.
Keep in mind the differing perspectives: the requirements are written from the perspective of the Device Under Test (DUT), while the test framework methods are written from the perspective of the test system interacting with the DUT.

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

    create_skeleton_code = """
    You are an expert in test code implementation within the automotive zone controller domain, using a custom test framework repository called TAF (Test Automotive Framework). 
    Keep in mind the differing perspectives: the requirements are written from the perspective of the Device Under Test (DUT), while the test framework methods are written from the perspective of the test system interacting with the DUT.
    We have extracted unordered object of classes and methods from the TAF repository that can potentially be used to write the test code of the requirement provided.

    You are tasked to write a pseudocode of the test code that fulfills a specific test step of the requirement provided. The pseudocode should include the classes and methods that are relevant to the test step of the requirement and the process of testing.

    ### Requirement ###
    {requirement}
    
    ### test step ###
    {test_step}

    ### Reference Classes and Methods ###
    {classes}

    ### example output ###
    ```
    1 - {{ "step_explication": "stimulation: un event 1" , "methods_used":["full_path3.file3.MyClass3: my_method3(param1), "full_path2.file2.MyClass2: my_method2(param1, param2)"] }}
    2 - {{ "step_explication": "retrieval: value triggered by event 1" , "methods_used":["full_path3.file3.MyClass3: my_method3(param1), "full_path2.file2.MyClass2: my_method2(param1, param2)"] }}
    3 - {{ "step_explication": "report: value triggered by event 1" , "methods_used":["full_path3.file3.MyClass3: my_method3(param1), "full_path2.file2.MyClass2: my_method2(param1, param2)"] }}
    4 - {{ "step_explication": "stimulation: un event 2" , "methods_used":["full_path3.file3.MyClass3: my_method3(param1)] }}
    5 - {{ "step_explication": "retrieval: value triggered by event 2" , "methods_used":["full_path3.file3.MyClass3: my_method3(param1)] }}
    6 - {{ "step_explication": "report: value triggered by event 2" , "methods_used":["full_path3.file3.MyClass3: my_method3(param1)] }}
    ```

    ## Strict Rules:
    - **the event can be only 'stimulation', 'retrieval', 'report'
    - **Do not invent or fabricate any method names**; only use those found in the provided file.
    - **Return the results in the exact format specified** above.
    - **the list should be well ordered
    - **in the pseudocode after each result expected in the test there is reporting of the reseal
    - **do not include any extra information in the output
    - **in the output of details should be valid json objects with the keys "step_explication" and "methods_used" and the values should be strings with the correct format like specified in the example
    - Adherence to these rules is mandatory. Any deviation, such as generating method names not present in the files, events don't exist, will result in immediate termination of the task.
    """

    verify_tools = """
    You are an expert in test code implementation within the automotive zone controller domain, using a custom test framework repository called TAF (Test Automotive Framework). 
    Keep in mind the differing perspectives: the requirements are written from the perspective of the Device Under Test (DUT), while the test framework methods are written from the perspective of the test system interacting with the DUT.    
    
    You have been provided with a list of tools that are required to write the test code for the specified requirement. Your task is to verify the correctness of the tools provided.
    You will be provided with the full TAF repository structure, the requirement, and the list of tools required to write the test code. 
    Your goal is to confirm whether the tools provided are necessary for fulfilling the requirement and correct it if necessary.
    
    ### Requirement ###
    {requirement}
    
    ### TAF Repository Structure ###
    {structure}
    
    ### Tools Required ###
    {tools}
    
    ## Example Output:
    ```
    full_path1.file1.MyClass1: my_method
    full_path2.file2.MyClass2: my_method2
    full_path2.file2.MyClass2: my_method2_2
    full_path2.file2.MyClass2: my_method2_3
    full_path3.file3.MyClass3: my_method3
    full_path2.file2.MyClass2: my_method3_2
    full_path4.file4.MyClass4: my_method4
    ```
    
    ## Strict Guidelines:
    - **Do not invent or fabricate any tool names**; only include those present in the provided repository structure.
    - **Focus solely on the requirement provided** and identify tools that directly contribute to fulfilling this requirement.
    - **Return the results in the exact format specified** above.
    - **do not include any extra information in the output
    - **in the output the tools should not include the parameters of the methods only the name like specified in the example
    - **give the correct path of the file from the object given by the repository structure given
    - Adherence to these guidelines is critical. Any deviation, such as creating non-existent tool names, will lead to immediate disqualification from the task.

    """
    verification_of_use_right_tools = """
    You are a top-tier expert in writing test code for the automotive zone controller domain, specifically utilizing a private framework repository known as TAF (Test Automotive Framework). Your deep understanding of both TAF and Python allows you to assess code with exceptional precision.
    
    I have reviewed the TAF source code and identified a set of tools that may be relevant for implementing a specific test step for a requirement. A developer has already written a test code for this, and I need you to evaluate whether the developer has:
    1. Used the correct TAF tools if any TAF tools are used in the code.
    2. Correctly used native Python methods where TAF tools are not necessary.
    
    ### Important:
    - If the developer uses TAF tools, ensure they are from the provided list of relevant TAF tools. Using tools outside this list should return `false`.
    - If the developer uses native Python methods and does not require TAF tools for the task, the code should still meet the requirement.
    - Only return `false` if incorrect tools or incorrect logic are used.
    
    ### Requirement ###
    {requirement}
    
    ### Test Step ###
    {test_step}
    
    ### Suggested TAF Tools ###
    {tools}
    
    ### Developer's Code specific for the test step ###
    ```python
    {code}
    ```
    
    ### The full code ###
    {code_full}
    
    ### Examples Output ###
    ```
    {{"result": true,"explanation": "The developer correctly used native Python methods without needing TAF tools."}}
    ```
    
    
    ```
    {{"result": false,"explanation": "The developer used the wrong TAF tools, or incorrect logic was applied."}}
    ```
    
    ### Strict Guidelines:
    - **Your result must be a boolean answering the question: "Has the developer met the requirement correctly?".
    - **If TAF tools are used, verify that they come from the provided list. If a tool is used but is not on the list, return false.
    - **If no TAF tools are used and native Python logic is sufficient, return true.
    - **You must not provide any additional information or explanations beyond the boolean result and the 'explanation' string.
    - **The result should be returned in the exact format shown above (a JSON object with 'result' as a boolean and 'explanation' as a string).
    - **Strictly adhere to these guidelines to ensure the highest level of code quality and correctness. 
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

    @traceable(
        name="generate pseudo code "
    )
    def give_skeleton(self, files_struct):
        template = self.create_skeleton_code
        message = template.format(
            requirement=self.requirement, classes=files_struct, test_step=self.test_step
        )
        logging.info(f"prompting with message:\n{message}")
        logging.info("=" * 80)

        model = make_model(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        result = False
        output = None
        while not result:
            traj = model.codegen(message, num_samples=1)[0]
            row_output = traj["response"]
            output, result = self.extract_skleton(row_output)
        return output

    def verify_tools_by_line(self, test_step, tools, label, graph):
        taf = filtered_nodes_by_label(graph, label)
        message = self.verify_tools.format(
            requirement=test_step,
            structure=json.dumps(taf),
            tools=json.dumps(tools),
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
        raw_output = traj["response"].replace("`", "")
        list = raw_output.split("\n")
        result = []
        for el in list:
            if el == "":
                continue
            seq = el.split(":")
            if len(seq) < 2:
                continue
            if seq[0].strip() == "":
                continue
            if seq[1].strip() == "":
                continue
            path = seq[0].strip()
            path = path.replace('/', '.').replace('.py_', '.')
            seq_ver = path.split("_")
            if len(seq_ver) > 1:
                end = seq_ver[-1]
                if end[0].isupper():
                    interface = seq_ver.pop()
                    path = ".".join(seq_ver)
                    path = path + "." + interface
            if path.endswith(".py"):
                path = path.replace(".py", "")
            else:
                path = path.replace(".py", ".")
            result.append(
                f"{path}: {''.join(seq[1:]).strip()}"
            )
        return result

    def verify_tools_in_code(self,tools, code, full_code):
        prompt = self.verification_of_use_right_tools.format(
            requirement=self.requirement,
            tools=json.dumps(tools),
            code=code,
            test_step=self.test_step,
            code_full=full_code
        )
        model = make_model(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        traj = model.codegen(prompt, num_samples=1)[0]
        return traj["response"]


    def extract_skleton(self, raw_output):
        output_list = raw_output.strip().replace("json", "").replace("`", "").split("\n")
        result = {}
        for el in output_list:
            if el == "":
                continue
            seq = el.split("-")
            if len(seq) < 2:
                continue
            number = int(seq[0].strip())
            try:
                json_parsed = json.loads("".join(seq[1:]).strip())
            except JSONDecodeError as e:
                print("error in json")
                return None, False
            result[number] = json_parsed
        res = []
        for i in range(len(result) + 1):
            if i not in result:
                continue
            res.append(result[i])

        return res, True

    @traceable
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
