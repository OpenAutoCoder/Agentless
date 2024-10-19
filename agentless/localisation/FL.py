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
from apps.services.open_ia_llm import OpenIA_LLM


class FL(ABC):
    def __init__(self, instance_id, structure, requirement, test_step, **kwargs):
        self.requirement = requirement
        self.structure = structure
        self.instance_id = instance_id
        self.test_step = test_step

    @abstractmethod
    def localize_files(self, top_n=1, mock=False) -> tuple[list, list, list, any]:
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
    You are an expert in writing test code for the automotive zone controller domain using a private framework repository called TAF (Test Automotive Framework). Your role is to generate pseudocode based on the extracted classes, methods, and logic ('code_glue') that will fulfill specific test steps for a given requirement.

    ### Objective:
    Your task is to write pseudocode that outlines the test code necessary to implement a specific test step for the provided requirement. The pseudocode must include relevant classes, methods, and logic ('code_glue') necessary to execute the test process.

    ### Event Types:
    Each step of the pseudocode should belong to one of the following event types, each representing a different phase of the test process:

    1. **stimulation**: Actions or triggers that simulate inputs or events within the system. This typically starts the test by interacting with the system under test.

    2. **retrieval**: Gathering the results or system states after the stimulation event. This is the data or response collected as a result of the initial trigger.

    3. **report**: Logging or documenting the retrieved values, outputs, or system states to ensure traceability and validation of the test results.

    4. **code_glue**: Pure Python logic used to manipulate internal data, prepare inputs, or format outputs. These are non-API-related steps that handle test data, calculations, or other internal code logic. **No methods or classes are involved in this step**.

    ### Guidelines:
    - **Requirement**: The requirement describes what the test is verifying or validating within the system.
    - **Test Step**: The specific step or sequence you need to test from the requirement.
    - **Reference Classes and Methods**: The extracted classes and methods available for use in the pseudocode.

    ### Example Test Flow:
    You are expected to organize the pseudocode using the following format, ensuring each step is explained clearly with a description and any applicable methods. If it's a `code_glue` step, ensure it contains pure logic with no associated methods.

    ### Requirement ###
    {requirement}

    ### Test Step ###
    {test_step}

    ### Reference Classes, Methods, and Code Glue ###
    {classes}

    ### Pseudocode Example Output ###
    ```
    1 - {{ "step_explication": "stimulation: trigger event 1 to initiate test", "methods_used": ["full_path3.file3.MyClass3: my_method3(param1)", "full_path2.file2.MyClass2: my_method2(param1, param2)"] }} 
    2 - {{ "step_explication": "code_glue: prepare input data for retrieval", "methods_used": [] }} 
    3 - {{ "step_explication": "retrieval: gather results from event 1", "methods_used": ["full_path3.file3.MyClass3: my_method3(param1)", "full_path2.file2.MyClass2: my_method2(param1, param2)"] }} 
    4 - {{ "step_explication": "report: log gathered results", "methods_used": ["full_path3.file3.MyClass3: my_method3(param1)", "full_path2.file2.MyClass2: my_method2(param1, param2)"] }} 
    5 - {{ "step_explication": "stimulation: trigger event 2 to validate next condition", "methods_used": ["full_path3.file3.MyClass3: my_method3(param1)"] }} 
    6 - {{ "step_explication": "code_glue: format result for further processing", "methods_used": [] }} 
    7 - {{ "step_explication": "retrieval: obtain system response for event 2", "methods_used": ["full_path3.file3.MyClass3: my_method3(param1)"] }} 
    8 - {{ "step_explication": "report: log final output for event 2", "methods_used": ["full_path3.file3.MyClass3: my_method3(param1)"] }}
    ```


    ### Mandatory Rules:
    1. **Event Types**: The pseudocode must adhere to the event types defined above (`stimulation`, `retrieval`, `report`, and `code_glue`).
    2. **code_glue Rules**: No methods or classes should be associated with `code_glue` steps. These steps should only contain internal logic manipulations.
    3. **Strict Method Usage**: Only use methods and classes listed in the reference. Do not create or fabricate method names or class names not provided.
    4. **Structured Format**: Return results in the exact format specified. Each step must be a valid JSON object with the keys `"step_explication"` and `"methods_used"`, where `"methods_used"` is either a list of methods or an empty list for `code_glue` steps.
    5. **Order of Execution**: Ensure the pseudocode is well-ordered and sequential, as required by the test logic.
    6. **No Extra Information**: The output must contain only the requested pseudocode, no additional commentary or details.
    
    Adherence to these rules is mandatory. Any deviation, such as fabricating methods or classes, or not following the output format, will result in task rejection.
    """



    map_pseudo_code = """
You are an expert in writing test code for the automotive zone controller domain using a private framework repository called TAF (Test Automotive Framework). 
Your task is to verify if the pseudocode of a requirement can be mapped to one or multiple lines of test code, with each pseudocode segment mapped to one and only one TAF API call.
  
### What You Will Receive:
1. The **requirement**, which provides context for the functionality expected from the Device Under Test (DUT).
2. The **pseudocode**, which represents the steps from the DUT perspective.
3. The **test code**, which is written from the perspective of the test system interacting with the DUT.
4. A **list of nodes** already taken to prevent overlap in your mappings.

### Your Task:
1. **Map the pseudocode** to one or multiple lines of the test code, ensuring that each pseudocode action corresponds to a single API call from TAF.
2. **Ensure non-overlapping mappings**: The lines you map must not overlap with previously mapped or reserved line. 
The list of already taken lines will be provided.
3. If a pseudocode action cannot be mapped to any test code line, return an empty response for that action.
  
### Key Considerations:
- Each pseudocode segment can map to **one or multiple lines** in the test code.
- Only one **TAF API call** should be mapped per pseudocode action (that means in the output there is only one method or function from the TAF framework not including the reporting)
- **Do not include any setup or teardown actions** unless they directly contribute to the main pseudocode action.
- If no appropriate line exists in the test code for a particular pseudocode action, return an empty response for that action.

### Inputs Provided:
- **Requirement**: 
{requirement}


- **Pseudocode**: 
{pseudo_code}


- **Test Code**: 
{test_code}


- **lines Taken**: 
{nodes_taken}



### Example Output:
Example 1:
```
line 5: methode(param)
line 8: self.__tc_id = self.__class__.__name__
``` 

example 2:
```
```
example 3:
```
line 6: self.__power_path_ctrl.bypass_limphome_control(True)
line 7: self._reporting.add_report_message_info(f"Set resistive load")
line 8: voltage = self.__power_path_ctrl.measure_actual_voltage(POWER_PATH_CHANNEL)
```
    

### Strict Guidelines:
- **Return valid lines** from the test code only. Do not generate or hallucinate lines that do not exist.
- **Do not overlap** your result should not include any line already provided in the "lines Taken" list.
- **No extra information** should be included in the output.
- Focus only on the **primary action** of the pseudocode. Ignore setup or context lines that do not directly contribute to the primary action.
- Each pseudocode action should map to **one and only one TAF API call** in the test code.
- If no valid mapping exists, return an **empty response**.
- Adherence to these guidelines is critical. Any deviation, such as generating non-existent lines or choosing any line already exist in the **lines Taken**, will lead to immediate disqualification from the task.

"""


    verify_tools = """
    You are an expert in writing test code that covers specific criteria within the automotive zone controller domain using a private framework repository called TAF (Test Automotive Framework).
    You have been provided with a list of tools that are required to write the test code for the specified requirement. Your task is to verify the correctness of the tools provided.
    You will be provided with the full TAF repository structure, the requirement, and the list of tools required to write a line of code of a requirement from a pseudo code. 
    Your goal is to confirm whether the tools provided are necessary for fulfilling the requirement and correct it if necessary.
    
    ### Requirement ###
    {requirement}
    
    ### pseudo code ###
    {line_code}
    
    
    
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
    - ** at least extract one tool 
    - **Main action focus: Only map the lines that represent the primary action in the pseudo code. Ignore setup or context actions that occur before or after.
    - **the line format should be as the example path seperated by '.' and separation between the path and method should be by ": "
    - Adherence to these guidelines is critical. Any deviation, such as creating non-existent tool names or returning empty result or any line of the result don't match the format specified, will lead to immediate disqualification from the task.

    """
    verification_of_use_right_tools = """
    You are a top-tier expert in writing test code for the automotive zone controller domain, specifically utilizing a private framework repository known as TAF (Test Automotive Framework). Your deep understanding of both TAF and Python allows you to assess code with exceptional precision.

    I have reviewed the TAF source code and identified a set of tools that may be relevant for implementing a specific test step for a requirement. A developer has already written a test code for this, and I need you to evaluate whether the developer has:
    1. Used the correct TAF tools if any TAF tools are used in the code.
    2. Used native Python correctly if TAF tools are not needed for the specific test step.

    ### Important:
    - If the developer uses TAF tools, they must be from the provided list of relevant TAF tools. Using other TAF tools is incorrect.
    - If no TAF tools are used, verify that native Python methods are sufficient and correctly implemented for the task.

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
    - **If TAF tools are used, they must be from the provided list. If tools not on the list are used, return false.
    - **You must not provide any additional information or explanations beyond the boolean result and explanation.
    - **The result should be returned in the exact format shown above (a JSON object with 'result' as a boolean and 'explanation' as a string).
    - **Strictly adhere to these guidelines to ensure the highest level of code quality and correctness.  
    
    """
    skeleton_verification = """
You are an expert in writing test code for the automotive zone controller domain using a private framework repository called TAF (Test Automotive Framework).
I have created a pseudocode for implementing specific requirements, but it was written without full knowledge of how TAF works. Your task is to review and optimize this pseudocode for proper integration with TAF.

### What you will receive:
1. The **requirement**, which provides the context for the functionality expected from the Device Under Test (DUT).
2. The **existing pseudocode**, which outlines the steps and tools I initially planned for implementing the requirements.
3. The **full TAF repository structure** and API descriptions, including available classes, methods, and tools for implementing the tests.

### Your Tasks:
1. **Verify** the existing pseudocode against the TAF framework:
    - Identify **missing steps** that need to be added for proper functionality with TAF.
    - Identify **unnecessary steps** that TAF handles automatically and can be removed from the pseudocode.
2. **Optimize** the pseudocode:
    - Adjust any steps to better align with TAF's tools and methods.
    - Ensure the pseudocode is coherent, streamlined, and follows TAF conventions.
3. If any changes are needed, **update the pseudocode** and return the optimized version.
4. **If no changes are needed**, return the exact same pseudocode without any modifications.

### Key Considerations:
- The **requirement** is written from the perspective of the DUT.
- The **TAF methods** describe how the test system interacts with the DUT.
- Your final pseudocode should be efficient, removing any redundant steps and adding any necessary ones based on TAF’s capabilities.

### Inputs Provided:
- **Requirement:** 
{requirement}


- **Existing Pseudocode:** 
{pseudocode}


- **Reference Classes and Methods from TAF:** 
{classes}

### Example Output (if changes are needed):
    ```
    [ {{ "step_explication": "stimulation: un event 1" , "methods_used":["full_path3.file3.MyClass3: my_method3(param1), "full_path2.file2.MyClass2: my_method2(param1, param2)"] }},
    {{ "step_explication": "retrieval: value triggered by event 1" , "methods_used":["full_path3.file3.MyClass3: my_method3(param1), "full_path2.file2.MyClass2: my_method2(param1, param2)"] }},
    {{ "step_explication": "report: value triggered by event 1" , "methods_used":["full_path3.file3.MyClass3: my_method3(param1), "full_path2.file2.MyClass2: my_method2(param1, param2)"] }},
    {{ "step_explication": "stimulation: un event 2" , "methods_used":["full_path3.file3.MyClass3: my_method3(param1)] }},
    {{ "step_explication": "retrieval: value triggered by event 2" , "methods_used":["full_path3.file3.MyClass3: my_method3(param1)] }},
    {{ "step_explication": "report: value triggered by event 2" , "methods_used":["full_path3.file3.MyClass3: my_method3(param1)] }}]
    ```
    

### Strict Rules:
- **Events** must be 'stimulation', 'retrieval', or 'report'.
- Only use **existing methods** from the provided TAF files—do **not** create new ones.
- If changes are needed, update the pseudocode while keeping the overall structure logical and streamlined.
- If **no changes** are needed, return the exact same pseudocode without any modification.
- Ensure that the **output format** is valid and strictly adheres to the example provided.
- Return  the output in the exact format specified above** without any extra information
- The **order** of steps must be logical and complete, with a proper reporting step after each task.
- **Output** must be valid JSON objects with the exact keys "step_explication" and "methods_used."
"""




    def __init__(
            self, instance_id, structure, requirement, test_step,
    ):
        super().__init__(instance_id, structure, requirement=requirement, test_step=test_step)
        self.max_tokens = 16000

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

    @traceable(
        name="7.1.localize files names for test steps"
    )
    def localize_files(self, current, top_n=1) -> tuple[list[Any], dict[str, Any], Any]:

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
            model=OpenIA_LLM.get_version_model("localize_files"),
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
        name="7.3.2.verify the pseudo code using the TAF"
    )
    def verify_skeleton(self,skeleton, graph):
        full_taf = filtered_nodes_by_label(graph)
        template = self.skeleton_verification

        message = template.format(
            requirement=self.test_step,
            classes=json.dumps(full_taf, indent=4),
            pseudocode=json.dumps(skeleton,indent=4)
        )
        print(f"prompting with message:\n{message}")
        print("=" * 80)
        model = make_model(
            model=OpenIA_LLM.get_version_model("verify_skeleton"),
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )

        output = []
        for i in range(5):
            traj = model.codegen(message, num_samples=1)[0]
            row_output = traj["response"]
            print(row_output)
            if str(row_output).startswith("```"):
                row_output = row_output[3:]
            if row_output.startswith("json"):
                row_output = row_output[4:]
            if str(row_output).endswith("```"):
                row_output = row_output[:-3]
            try:
                output = json.loads(row_output)
                done = True
            except JSONDecodeError as e:
                print("error in json of skeleton verification")
                done = False
            if done:
                break
        return output

    @traceable(
        name="7.3.1.generate pseudo code "
    )
    def give_skeleton(self, files_struct):
        template = self.create_skeleton_code
        message = template.format(
            requirement=self.requirement, classes=files_struct, test_step=self.test_step
        )
        logging.info(f"prompting with message:\n{message}")
        logging.info("=" * 80)

        model = make_model(
            model=OpenIA_LLM.get_version_model("give_skeleton"),
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
    @traceable(
        name="7.3.2.verify tools by pseudo code line"
    )
    def verify_tools_by_line(self, test_step, line, label, graph):
        taf = filtered_nodes_by_label(graph, label)
        message = self.verify_tools.format(
            requirement=test_step,
            structure=json.dumps(taf),
            tools=json.dumps(line['methods_used']),
            line_code=line['step_explication']
        ).strip()
        print(f"prompting with message:\n{message}")
        print("=" * 80)

        model = make_model(
            model=OpenIA_LLM.get_version_model("verify_tools_by_line"),
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
            seq = el.split(":") if len(el.split(":"))>=2  else el.split(" ")
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
                if len(end) > 0 and end[0].isupper():
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
    @traceable(
        name="7.4.1.map pseudo code to code"
    )
    def map_pseudo_code_to_code(self, test_code, pseudo_code, nodes_taken):
        prompt = self.map_pseudo_code.format(
            requirement=self.test_step,
            test_code=test_code,
            pseudo_code=pseudo_code,
            nodes_taken=json.dumps([node.properties['reference'] for node in nodes_taken], indent=4)
        )
        model = make_model(
            model=OpenIA_LLM.get_version_model("map_pseudo_code_to_code"),
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        traj = model.codegen(prompt, num_samples=1)[0]["response"].replace("`", "")
        lines = []
        for res in traj.split("\n"):
            if not res or len(res) == 0:
                continue
            if str(res).strip().startswith("line "):
                seq = res.split(":")
                seq.pop(0)
                if len(seq) ==0:
                    continue
                lines.append(":".join(seq).strip() if len(seq)> 1 else seq[0].strip())
            else:
                lines.append(res.strip())
        return lines


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

    @traceable(
        name="7.2.localize functions names from files corresponding to test step"
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
            model=OpenIA_LLM.get_version_model("localize_function_from_compressed_files"),
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

