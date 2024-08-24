from agentless.repair.syntax_error_fixing.agent_prompts import syntax_error_fixing_w_diffs_prompt, diff_file_format_instructions
from agentless.repair.cintra_diffs_generator.InMemoryDiffApplier import InMemoryDiffApplier
from agentless.repair.cintra_diffs_generator.code_changing_models import DiffFileContents
from agentless.util.model import make_model
from agentless.util.postprocess_data import check_syntax
from agentless.util.utils import setup_logger
import os
import json

class DiffPatchError(Exception):
    """Custom exception raised when a diff patch fails to apply."""
    def __init__(self, message="Failed to apply diff patch"):
        self.message = message
        super().__init__(self.message)


def return_simple_line_numbers_with_code(code: str) -> str:
    code_lines = code.split('\n')
    code_with_line_numbers = [f"Line {i + 1}: {line}" for i, line in enumerate(code_lines)]
    joined_lines = "\n".join(code_with_line_numbers)
    return joined_lines


def fix_syntax_error_diff_files(syntax_error, code_with_error, logger):
    """
    Fixes syntax errors in the provided code by generating and applying diffs.

    This function takes a syntax error and the code containing the error, generates a diff to fix the error,
    and applies the diff to produce the corrected code.

    Args:
        syntax_error (str): The syntax error message.
        code_with_error (str): The code containing the syntax error.
        instance_id (str): The instance ID for logging purposes.

    Returns:
        str: The corrected code after applying the diff.

    Raises:
        ValueError: If there is an issue extracting the diff content from the model output.
    """
    code_with_line_numbers = return_simple_line_numbers_with_code(code_with_error)
    print(f'code: {code_with_error}')
    print(f'code with line numbers: {code_with_line_numbers}')

    prompt = syntax_error_fixing_w_diffs_prompt(
        syntax_error=syntax_error,
        code_file_with_lines=code_with_line_numbers,
        format_instructions=diff_file_format_instructions
    )

    model = make_model(
        model="gpt-4o",
        backend="openai",
        logger=logger,
        max_tokens=1024,
        temperature=0,
        batch_size=1,
    )
    print(prompt)

    output = model.codegen(prompt, num_samples=1)
    response=output[0]['response']
    try:
        extracted_diff_contents = DiffFileContents.extract_diff_content(response)
    except ValueError as e:
        raise e

    diff_file = str(extracted_diff_contents.diff)

    print(f'diff_file: {diff_file}')

    diff_applier = InMemoryDiffApplier(code_with_error, diff_file)
    try:
        applied_diff = diff_applier.apply_diff()
        new_code = applied_diff["updated_content"]
        print(f'new_code: {new_code}')
    except Exception as e:
        print(f'Error applying diff: {e}')
        raise DiffPatchError(f'Error applying diff: {e}')


if __name__ == "__main__":
    with open('agentless/retreival/mock_codefiles.json', 'r') as f:
        mock_codefiles = json.load(f)
    code_with_error = mock_codefiles['single_syntax_error_example.py']
    log_file = os.path.join(
        "logs", "syntax_error_fixing", f"test_syntax_error_fixing.log"
    )
    os.makedirs(os.path.join("logs", "syntax_error_fixing"), exist_ok=True)
    logger = setup_logger(log_file)
    try:
        syntax_error = check_syntax(code_with_error)
    except SyntaxError as e:
        print(f'Error checking syntax: {e}')
        instance_id = "test_instance"
        result = fix_syntax_error_diff_files(e, code_with_error, logger)