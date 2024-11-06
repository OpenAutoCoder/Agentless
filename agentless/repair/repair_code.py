import argparse
import json
import os

from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from apps.helper import read_file
from apps.services.open_ia_llm import OpenIA_LLM



default_prompt_to_fix_code = """
# Code Improvement Instructions for GPT-4

## 1. Overview
You will be provided with the following:
- **Full Code**: The complete Python code that may contain issues.
- **Analysis Context**: A description of the intended functionality and context for the code.

Your task is to analyze the provided code and make improvements or corrections as necessary. Return the full corrected code without any additional information beyond the corrected code itself.

## 2. Instructions
- Analyze the full code for potential issues, inefficiencies, or areas for improvement.
- Apply the necessary corrections to enhance the code while ensuring its intended functionality is preserved.
- Return the entire corrected code.

## 3. Output Requirements
- Return the full code with the applied improvements.
- Provide no additional explanations, comments, or information beyond the corrected code itself.

## 4. Strict Compliance
- Do **not** include any explanations or comments in the output.
- **do not include any extra information in the output
- Focus solely on improving the code without being guided by specific error indications.
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
            Please review and improve the code below:
            --- BEGIN FULL CODE FILE ---
            ```
            {full_code}
            ```
            --- END FILE ---

            ## Strict Guidelines:
            - **Do not invent or fabricate any tool names**; only include those present in the provided code.
            - **Focus solely on improving the provided code without specific error context.**
            - **Do not include any extra information in the output.**
            - **Return the full code corrected; do not miss any line.**
            - **do not include any extra information in the output
            - **If the old implementation is correct, return exactly the same full code given.**
            - Adherence to these guidelines is critical. Any deviation will lead to disqualification from the task.
            """
        )
    )
])

@traceable(
    name="repair error code",
)
def repair_error_code( testcode):
    model = OpenIA_LLM.get_model(
        OpenIA_LLM.get_version_model("repair_error_code")
    )
    chain = prompt_fix_code | model

    row_res = chain.invoke({
        "full_code": testcode
    })
    corrected_code = row_res.content

    print("corrected after review the code error:")
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
    testcase = read_file(args.test_code_file_path)

    res = repair_error_code( testcase)
    print(res)
