import json
import os
import re
import glob
def remove_duplicates(dict_list):
    seen = set()
    new_list = []
    
    for dictionary in dict_list:
        # Convert dictionary to a tuple of its items for hashability
        dict_tuple = frozenset(dictionary.items())
        if dict_tuple not in seen:
            seen.add(dict_tuple)
            new_list.append(dictionary)
            
    return new_list

def get_execution_result_files(folder_path):
    # Construct the search pattern 
    search_pattern = os.path.join(folder_path, '*_ordered_regression_results.jsonl')
    
    # Use glob to find all files that match the search pattern
    result_files = glob.glob(search_pattern)
    
    return result_files

def remove_ansi_sequences(input_string):
    # Define the regex pattern for ANSI escape sequences
    ansi_escape_pattern = r'\x1b\[\d+m'
    # Use re.sub() to replace all occurrences of the pattern with an empty string
    clean_string = re.sub(ansi_escape_pattern, '', input_string)
    
    return clean_string


def create_instance_test_dict(jsonl_file_path):
    instance_test_dict = {}
    
    # Open and read the jsonl file line by line
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            # Parse each line as a JSON object
            json_obj = json.loads(line.strip())
            # Set instance_id as the key and test as the value
            instance_id = json_obj['instance_id']
            test = json_obj['patch']
            instance_test_dict[instance_id] = test
    
    return instance_test_dict

def txt_file_contains_string(path_to_txt, expected_output):
    """
    Check if the given text file contains the specified string.

    :param path_to_txt: Path to the text file.
    :param expected_output: The string to search for in the text file.
    :return: True if the string is found in the text file, otherwise False.
    """
    try:
        with open(path_to_txt, 'r', encoding='utf-8') as file:
            content = file.read()
            return expected_output in remove_ansi_sequences(content).replace("System check identified no issues (0 silenced).\n","")
    except FileNotFoundError:
        print(f"The file at {path_to_txt} was not found.")
    except IOError:
        print(f"An error occurred while reading the file at {path_to_txt}.")
    
    return False



def generate_no_op_patch(patch_input):
    # Create the header for the no-op patch based on the input
    header_lines = []
    for line in patch_input.splitlines():
        if line.startswith('diff --git ') or line.startswith('index ') or line.startswith('--- ') or line.startswith('+++ '):
            header_lines.append(line)
        elif line.startswith('@@ '):
            break
    
    # Create a no-op patch by utilizing the identified header lines
    no_op_patch = '\n'.join(header_lines) + '\n'
    no_op_patch += "@@ -1,1 +1,1 @@\n"
    no_op_patch += " \n"

    return no_op_patch


def create_patch_from_code(python_code: str) -> str:
    patch_header = """diff --git a/reproduce_bug.py b/reproduce_bug.py
new file mode 100644
index 0000000..e69de29
"""
    patch_body = []
    patch_body.append('--- /dev/null')
    patch_body.append('+++ b/reproduce_bug.py')

    code_lines = python_code.split('\n')
    patch_body.append(f'@@ -0,0 +1,{len(code_lines)} @@')

    for line in code_lines:
        patch_body.append(f'+{line}')

    return patch_header + '\n'.join(patch_body) + '\n'

def load_existing_results(jsonl_filename):
    results = {}
    try:
        with open(jsonl_filename, 'r') as jsonl_file:
            for line in jsonl_file:
                record = json.loads(line.strip())
                instance_id = record.pop("instance_id")
                results.setdefault(instance_id, []).append(                        {
                            "normalized_patch": record["normalized_patch"].strip(),
                            "patch": record["patch"],
                            "plausible": record["plausible"],
                        })
    except FileNotFoundError:
        print(f"No existing results file found at {jsonl_filename}. Proceeding with an empty execution_results dictionary.")
    return results