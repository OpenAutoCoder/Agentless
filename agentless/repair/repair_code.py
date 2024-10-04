import argparse
import json
import os

from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from apps.helper import read_file
from apps.services.neo4jDB.graphDB_dataAccess import create_graph_database_connection
from apps.services.open_ia_llm import OpenIA_LLM

query_get_code = """
MATCH (ce:Code_error)
WHERE ce.doc_ref = '{}'
RETURN ce   
"""

default_prompt_to_fix_code = """
# Code Correction Instructions for GPT-4

## 1. Overview
You will be provided with the following:
- **Error Explanation**: A description of the problem or error in the code.
- **Potential Error Line**: The specific line or lines where the error might exist.
- **Full Code**: The complete Python code that contains the issue.

Your task is to correct the code for the specific problem identified and return the full corrected code. Do not provide any additional information beyond the corrected code.

## 2. Instructions
- Analyze the error explanation and the potential source of the error in the provided line.
- Apply the necessary correction to the code to fix the problem.
- Return the entire corrected code.

## 3. Output Requirements
- Return the full code with the applied correction.
- Provide no additional explanations, comments, or information beyond the corrected code itself.

## 4. Strict Compliance
- Do **not** include any explanations or comments in the output.
- Adhere strictly to the error explanation provided, correcting only the specific problem identified.
- Return the full corrected code in the exact format as provided.

"""

prompt_fix_code = ChatPromptTemplate.from_messages([
    (
        "system",
        default_prompt_to_fix_code,
    ),
    (
        "human",
        (
            """
            we have identified an issue in the code below:
              --- Error Explanation ---
              {error_explanation}
              
              --- Potential Error Line ---
              ```
              {potential_error_line}
              ```
              --- BEGIN FULL CODE FILE ---
              ```
              {full_code}
              ```
              --- END FILE ---
              
              ## Strict Guidelines:
              - **Do not invent or fabricate any tool names**; only include those present in the provided.
              - **Focus solely on the problem provided do not correct more from the code**
              - **do not include any extra information in the output
              - **return the full code corrected do not miss any line**
              - **if the old implementation is correct do not change it and return exactly the same full code given 
              - Adherence to these guidelines is critical. Any deviation, such as creating non-existent tool names or returning empty result or returning full test code missing some line , will lead to immediate disqualification from the task.
            """
        )
    )
])


def get_code_error(graph, doc_ref):
    res = graph.query(
        query_get_code.format(doc_ref)
    )
    return [record['ce'] for record in res]


@traceable(
    name="repair error code",
)
def repair_error_code(graph, testcode, doc_ref):
    error_nodes = get_code_error(graph, doc_ref)
    corrected_code = testcode
    for error_node in error_nodes:
        model = OpenIA_LLM.get_model(
            OpenIA_LLM.get_version_model("repair_error_code")
        )
        chain = prompt_fix_code | model

        row_res = chain.invoke({

            "error_explanation": error_node['explanation'],
            "potential_error_line": error_node['reference'],
            "full_code": corrected_code
        })
        corrected_code = row_res.content

    print(corrected_code)
    return corrected_code


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('-dBuserName', type=str, default=os.environ['NEO4J_USERNAME'], help='database userName')
        parser.add_argument('-gituserName', type=str, default='MM', help='git userName')
        parser.add_argument('-database', type=str, default='neo4j', help='database name')
        parser.add_argument('-cmd', type=str, default='FULL', help='cmd')
        parser.add_argument('-req_file_path', type=str, default=dataset['req'],
                            help='requirement file path')
        parser.add_argument('-test_code_file_path', type=str, default=dataset['code'],
                            help='test code file path')
        return parser.parse_args()


    with open(os.path.join(os.path.dirname(__file__), "..", "..", "..", 'app-config.json')) as f:
        config = json.load(f)
        dataset = config['dataset']
    args = parse_args()
    graph_connect = create_graph_database_connection(args)
    doc_ref_id = f"{args.req_file_path}--||--{args.test_code_file_path}"
    testcase = read_file(args.test_code_file_path)

    res = repair_error_code(graph_connect, testcase, doc_ref_id)
    print(res)
