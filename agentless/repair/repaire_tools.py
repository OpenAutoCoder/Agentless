from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from apps.services.code_skeleton_extractor import filtered_methods_by_file_name_function
from apps.services.open_ia_llm import OpenIA_LLM


query_get_test_steps = """
Match (n:Test_step) where n.doc_ref = '{doc_ref}' return n
"""

query_get_pseudo_code_by_test_step = """
Match (n:Pseudo_Code)<-[:USE_PSEUDO_CODE]-(k:Test_step) where n.doc_ref = '{doc_ref}' and k.doc_ref = '{doc_ref}' and k.id = '{id_test_step}'
    return n
"""

query_get_tools_from_pseudo_code = """
Match (n:Pseudo_Code)-[:HAS_TOOL_SUGGESTION]->(k:Tool_Suggestion)
where n.doc_ref = '{doc_ref}' and k.doc_ref = '{doc_ref}' and n.id = '{id_node_code}'
return k
"""

query_get_related_code = """
Match (n:Test_step)-[:TEST_STEP_CODE_MAP]->(k)
 where n.doc_ref = '{doc_ref}' and k.doc_ref = '{doc_ref}' and n.id = '{id_node_code}'
    return k
"""


prompt_generate_pseudo_code_implementation = """
You are a highly skilled test engineer tasked with writing **automated test scripts** for individual **test steps** based on a set of predefined actions.

Each test step is part of a larger system requirement but should be handled as a standalone piece. Your job is to generate test code for the **specific test step only**, not the entire requirement.

### Input Format:
You will receive a JSON object structured as follows:

{{
  "test_step_explanation": "Explanation of the overall test step",
  "pseudocode_steps": [
    {{
      "description": "Explanation of the pseudocode line",
      "methods": [
        {{
          "path": "path_to_file",
          "method": "method_signature",
          "doc_ref": "doc_reference",
          "documentation": "method_documentation",
          "id": "unique_identifier",
          "label": "method_label",
          "class": "class_name"
        }},
        ...
      ]
    }},
    ...
  ]
}}

### Your Task:
1. Parse the JSON object.
2. Implement the test code by sequentially interpreting the pseudocode lines, mapping each line to the provided TAF methods.
3. Use **only** the provided methods for each pseudocode line, ensuring that you select the appropriate method based on the descriptions.
4. The output should be a **single, continuous block of code**, with no extraneous explanations, comments, or non-relevant actions.
5. Ensure code correctness and continuity while adhering strictly to the given methods.

### Example Input:
{{
  "test_step_explanation": "Check AUTOSAR system for errors",
  "pseudocode_steps": [
    {{
      "description": "Set the trackErrors variable to True",
      "methods": [
        {{
          "path": "taf_tools/utilities/system_debug/interface.py",
          "method": "def set_variable(self, var_name, value)",
          "doc_ref": "taf_docs",
          "documentation": "Set the value of a SW variable.",
          "id": "taf_tools/utilities/system_debug/interface.py_Interface_def set_variable(self, var_name, value)",
          "label": "stimulation",
          "class": "Interface"
        }}
      ]
    }},
    {{
      "description": "Sleep for 10 seconds",
      "methods": [
        {{
          "path": "taf/core/time/__init__.py",
          "method": "def sleep(seconds)",
          "doc_ref": "taf_docs",
          "documentation": "Sleep for a given time.",
          "id": "taf/core/time/__init__.py_sleep",
          "label": "stimulation",
          "class": "Interface"
        }}
      ]
    }}
  ]
}}

### Example Output:
self.__system_debug.set_variable(TRACK_ERRORS_VAR, True)
taf.core.time.sleep(10)

### Strict Guidelines:
- **Use only the provided methods**: No extra methods or functions should be used.
- **Ensure code continuity**: The output must be a continuous block of code, logically connected without gaps.
- **No additional output**: Do not include explanations, comments, or anything outside of the code itself.
- **Focus on the primary action of each step**: Ignore any setup or teardown unless explicitly mentioned in the pseudocode.
- **Return the full test code**: Ensure that the output includes all lines of code for the test step.
- **do not include any extra information or explication in the output
- **Precondition Validation: Each test case must validate the Device Under Test (DUT) state before beginning any action. Assume that the DUT may already be operational; therefore, do not assume it boots from scratch at the start of each test case. Ensure test steps include proper state checks in the precondition.
- **critically important to adhere to these guidelines**. Any deviation, such as adding extra information or using methods not provided, will result in disqualification from the task.

"""

default_prompt_generate_code = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                prompt_generate_pseudo_code_implementation,
            ),
            (
                "human",
                (
                    "Tip: Make sure to answer in the correct format and do "
                    "not include any extra information in the output."
                    "the input: \n{input}"
                ),
            ),
        ]
    )


prompt_to_compare_versions_implementation = """
You are an expert in test code implementation for automotive zone controller systems, working with a specialized Test Automotive Framework (TAF). I need your expertise to **compare two implementations** of a specific test step and modify only the test step from the old implementation where necessary.

### Context:
I have developed a **new implementation** of a test step using updated methods and tools, while the **old implementation** follows a more established logical structure. Your task is to modify only the **test step portion** of the old implementation based on insights from the new one, while preserving as much of the old code logic as possible.

The **test step** is part of a larger requirement, but you are required to focus solely on that specific step.

### Inputs:
1. **Requirement**: The larger system behavior being tested, written from the perspective of the Device Under Test (DUT). 
2. **Test Step**: The specific action or check that needs to be tested. 
3. **My New Implementation**: The test code I have recently written using updated methods. 
4. **Old Implementation**: The previous test code that you will modify. 
5. **Full Test Code**: The complete test code, provided for context. 
6. **TAF Description**: Details about the Test Automotive Framework methods and their usage. 

### Task:
- **Analyze both implementations** to understand their differences and the strengths of each.
- Modify **only the test step in the old implementation**, incorporating updated tools and methods from the new implementation, while keeping the old logic intact.
- Ensure that the modified test step adheres to the **exact action** described in the test step.

### Guidelines:
1. **Do not modify anything beyond the specific test step** in the old implementation.
2. **Merge tools and logic**: Preserve the reliable logic from the old implementation while integrating the advanced tools from the new implementation.
3. **Focus solely on the test step**: Do not make changes to setup, cleanup, or other unrelated portions of the code.
4. **Strict Use of Provided Methods**: Only use the methods provided in either the old or new implementation—do not introduce external methods or tools.
5. **Full Test Code Return**: Return the **entire test code**, but ensure that only the specific test step is modified. If no change is needed, return the full test code unchanged.
6. **Probabilistic Approach**: The logic from the old implementation carries more weight, but the tools from the new implementation should be integrated wherever they improve efficiency or correctness.

### Important Notes:
- **No new tools or methods**: Stick to the available methods and tools provided.
- **Main Action Focus**: Implement only the primary action required for the test step. Avoid modifying unrelated parts of the code.
- **Strict Adherence**: Ensure you don’t add unnecessary logic or methods beyond the scope of the task. Only the test step should be changed if needed.

### Example Workflow:
- **Old Logic**: Preserved unless tools from the new implementation provide significant improvement.
- **New Tools**: Integrated only where they improve the test step.
- **Outcome**: A modification of the old implementation, keeping its logic but using better tools from the new code.

The final output should be the **complete test code**, with changes made only to the test step as described.

"""

default_prompt_merge_code = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                prompt_to_compare_versions_implementation,
            ),
            (
                "human",
                (
                    "Tip: Make sure to answer in the correct format and do "
                    "not include any extra information in the output."
                    "requirement: \n{requirement}\n\n"
                    "test step: \n{test_step}\n\n"
                    "My New Implementation: \n{generated}\n\n"
                    "old implementation: \n{old}\n\n"
                    "Full Test Code: \n{test_code}\n\n"
                    "TAF Description: \n{taf}"
                ),
            ),
        ]
    )




def retrieve_code_related(doc_ref, id_test_step, graph):
    res = graph.query(query_get_related_code.format(doc_ref=doc_ref,id_node_code=id_test_step))
    return [a['k'] for a in res]

def retrieve_tools_related(doc_ref, line, graph):
    res = graph.query(query_get_tools_from_pseudo_code.format(doc_ref=doc_ref,id_node_code=line["id"]))
    return [a['k'] for a in res]

def retrieve_test_steps(doc_ref, graph):
    res = graph.query(query_get_test_steps.format(doc_ref=doc_ref))
    return [a['n'] for a in res]

def retrieve_pseudo_code_by_test_step(doc_ref, id_test_step, graph):
    res = graph.query(query_get_pseudo_code_by_test_step.format(doc_ref=doc_ref, id_test_step=id_test_step))
    return [a['n'] for a in res]


def treat_path(path):
    path = path.replace(".py","").replace(".","/")
    seq = path.split("/")
    if str(seq[-1][0]).isupper():
        seq = seq[:-1]
    return '/'.join(seq)

@traceable(
    name="generate a version of code implemented with the right tools found"
)
def generate_version_with_tool(input):
    model = OpenIA_LLM.get_model(OpenIA_LLM.get_version_model("generate_version_with_tool"))
    chain = default_prompt_generate_code | model
    res = chain.invoke({
        "input": input
    })
    return res.content

@traceable(
    name="merge the 2 versions of the code"
)
def merge_two_versions_of_code(test_step,taf,old_version,new_version,test_code_text, requirement_text):
    model = OpenIA_LLM.get_model(OpenIA_LLM.get_version_model("merge_two_versions_of_code"))
    chain = default_prompt_merge_code | model
    res = chain.invoke({
        "requirement":requirement_text,
        "test_step":test_step,
        "generated":new_version,
        "old":old_version,
        "test_code":test_code_text,
        "taf":taf
    })
    return res.content



@traceable(
    name = "repair taf implementation"
)
def repair_taf_implementation(doc_ref, test_code_text,requirement, graph):
    code_corrected = test_code_text
    test_steps = retrieve_test_steps(doc_ref, graph)
    for test_step in test_steps:
        pseudo_codes = retrieve_pseudo_code_by_test_step(doc_ref, test_step['id'], graph)
        pseudo_codes.sort(key=lambda x: int(x["number"] if "number" in x.keys() else 0))
        lines = []
        tools_for_test_step = []
        code_related = retrieve_code_related(doc_ref, test_step['id'], graph)
        old_code = ''
        number = 0
        if len(code_related) > 0:
            code_related.sort(key=lambda x: int(x["number"] if "number" in x.keys() else 0))
            number = 0
            if "number" in code_related[0].keys():
                number  = int(code_related[0]['number'])
        for code_line in code_related:
            if  ("number" in code_line.keys())and(int(code_line["number"]) - number > 1):
                old_code += "\n#other code...   \n\n"
            old_code += code_line['reference']
            old_code += '\n'

        for line in pseudo_codes:
         tools_related = retrieve_tools_related(doc_ref, line, graph)
         tools_name = [tool['function'].split("(")[0] for tool in tools_related]
         paths = [treat_path(tool["path"]) for tool in tools_related]
         description_tools = filtered_methods_by_file_name_function(graph, paths, tools_name)
         lines.append({
             "description":line["explanation"],
             "methods":description_tools
         })
         for tool in description_tools:
            tools_for_test_step.append(tool)

         generated_code = generate_version_with_tool({"test_step_explanation":test_step['explanation'],"pseudocode_steps": lines})
         code_corrected = merge_two_versions_of_code(test_step['explanation'], tools_for_test_step, old_code, generated_code, code_corrected, requirement)
    print("corrected code after taf tool repair:")
    print(code_corrected)
    return code_corrected


