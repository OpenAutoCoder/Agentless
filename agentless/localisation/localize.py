import ast
import json
import logging
import re
from uuid import uuid4



from langchain_community.graphs.graph_document import Node, Relationship
from langsmith import traceable

from apps.services.code_preperation import annotate_code
from apps.services.quality_checkers.quality_check import CheckerFailure
from apps.services.code_skeleton_extractor import  find_by_method, \
    filtered_nodes_by_label
from apps.services.quality_checkers.test_code_qte_check import schema_test_code

FILES_TO_USE = [
    "interface.py",
    "catalog.py",
    "__init__.py",
]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def filter_files(structure: dict, files: list):
    files_level = structure.keys()
    files_to_remove = []
    for file in files_level:
        if file.lower().endswith(".py"):
            if file.lower() not in files:
                files_to_remove.append(file)
        if "." not in file and isinstance(structure[file], dict):
            filter_files(structure[file], files)
    for file in files_to_remove:
        del structure[file]
    return structure




def recursive_filter_files(sequence, methode, obj_loc):
    if len(sequence) == 0:
        temp = obj_loc
        if type(obj_loc) == dict:
            if "methods" in obj_loc:
                temp["methods"] = []
                for method in obj_loc["methods"]:
                    if methode in method["method"]:
                        temp["methods"].append(method)
        elif type(obj_loc) == list:
            temp = []
            for obj in obj_loc:
                if methode in obj["method"]:
                    temp.append(obj)
        return temp
    path = sequence[0]
    if path in obj_loc:
        return recursive_filter_files(sequence[1:], methode, obj_loc[path])


def filter_taf_files(sequence, file_name, files_locs, full_obj, methods):
    if len(sequence) == 0:
        if file_name in full_obj:
            items = full_obj[file_name]
            result = []
            for item in items:
                if item["class"] is None:
                    for method in methods:
                        if method in item["method"]:
                            result.append(item)
                            break
                else:
                    result_class = []
                    class_methods = item["methods"]
                    for method in class_methods:
                        for m in methods:
                            if m in method["method"]:
                                result_class.append(method)
                                break
                    if len(result_class) > 0:
                        item["methods"] = result_class
                        result.append(item)
            files_locs[file_name] = result
        return
    path = sequence[0]
    if path in full_obj:
        if path not in files_locs:
            files_locs[path] = {}
        filter_taf_files(sequence[1:], file_name, files_locs[path], full_obj[path], methods)




@traceable(
    name="7.3.3.check for the 1:1 relation between tools and pseudo code"
)
def verify_number_tools(res):
    if len(res) != 1:
        raise CheckerFailure(f"the number of tools is not equal to 1, result :{res}")
    return True

@traceable(
    name="7.3.1.verification of the skeleton"
)
def verification_skeleton(skeleton):
    if len(skeleton) == 0:
        raise CheckerFailure("skeleton is empty")
    ret_stimulus_count = 0
    reporting = 0
    for line in skeleton:
        if "stimulation" in line["step_explication"].lower() or "retrieval" in line["step_explication"].lower():
            ret_stimulus_count += 1
        if "report" in line["step_explication"].lower():
            reporting += 1
    if ret_stimulus_count == 0:
        raise CheckerFailure("no stimulation or retrieval found in the skeleton")
    if reporting == 0:
        raise CheckerFailure("no reporting found in the skeleton")
    return True

@traceable(
    name="7.3.verification with skeleton to test step"
)
def verification_with_skeleton(locs, fl, graph):
    final_locs = []

    files_taf = filtered_nodes_by_label(graph)
    with open("app-config.json", "r") as f:
        data = json.load(f)
        count = data["reps"]
    passed = False
    skeleton = []
    for i in range(count):
        try:
            skeleton = fl.give_skeleton(files_taf)
            passed = verification_skeleton(skeleton)
            logger.info(f"verification passed for skeleton at iteration {i}")
            break
        except CheckerFailure:
            logging.error(f"verification failed for skeleton at iteration {i}")
            passed = False
    if not passed:
        raise CheckerFailure("verification failed for skeleton")
    i = 0
    for line in skeleton:
        final_locs_agg = set()
        for method in line["methods_used"]:
            final_locs_agg.add(method)
        if not "glue_logic" in line["step_explication"].lower():
            for loc in locs:
                final_locs_agg.add(loc)

        final_locs.append({
            "line": line,
            "number": i,
            "locs_line": list(final_locs_agg),
        })
        i += 1
    return final_locs


def get_related_instructions(instruction, relations, nodes, relation_types, result, processed=None):
    if instruction is None:
        return

    if processed is None:
        processed = set()
    if instruction.id in processed:
        return

    processed.add(instruction.id)
    relations_found = [rel for rel in relations if
                       rel.target.id == instruction.id and rel.type.upper() in relation_types]

    if not relations_found:
        return
    ids = [rel.source.id for rel in relations_found]
    nodes_code = [node for node in nodes if node.id in ids]
    result.extend(nodes_code)
    for line in nodes_code:
        get_related_instructions(line, relations, nodes, relation_types, result, processed)




def find_line_code_nodes(nodes,line):
    node_types = schema_test_code.keys()
    for node in nodes:
        if node.type.upper() in node_types:
            if line in node.properties['reference']:
                return node
    return None

@traceable(
    name = "7.4.3.verify overlap between pseudo code and code"
)
def verify_overlap(code_lines, nodes_taken, nodes):

    for line in code_lines:
        node_code_found = find_line_code_nodes(nodes_taken, line.strip())
        if node_code_found is not None:
            raise CheckerFailure(f"the line {line} is already taken by another pseudo code")
    return True

def extract_method_name(code_string):
    try:
        tree = ast.parse(code_string)
    except SyntaxError:
        return None
    class MethodVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute):
                method_name = node.func.attr
                method_names.append(method_name)
            elif isinstance(node.func, ast.Name):
                method_name = node.func.id
                method_names.append(method_name)
            self.generic_visit(node)

    method_names = []

    visitor = MethodVisitor()
    visitor.visit(tree)

    return method_names[0] if method_names else None


def verify_function_usage(code_lines, function_name):
    for code_string in code_lines:
        correct_pattern = rf'\bself\.\w+\.\s*{function_name}\s*\(.*?\)\s*'  # Matches self.attribute.functionName(params)

        correct_match = re.search(correct_pattern, code_string)

        if correct_match:
            return True
    return False


@traceable(
    name="7.4.2.verify number of api calls in pseudo code"
)
def verification_api_calls  (code_lines, graph):
    methods_names = []
    for line in code_lines:
        method_name = extract_method_name(line)
        if method_name is not None:
            methods_names.append(method_name)
    if len(methods_names) == 0:
        return True

    found_method = 0
    for method in methods_names:
        if 'report' in method:
            continue
        res_query = find_by_method(graph, [method])
        if verify_function_usage(code_lines,method) or len(res_query) != 0:
            found_method += 1

        if found_method > 1 :
            raise CheckerFailure(f"more than one api call found in the pseudo code")

    return True



@traceable(
    name="7.4.generate pseudo code nodes and relations"
)
def verify_used_tools_by_pseudo_code(test_step, tools, nodes, fl, full_code,doc_ref, graph):
    nodes_generated =[]
    relations_generated = []
    nodes_taken = []
    tools = sorted(tools, key=lambda x: x['number'])
    for line_code in tools:
        line = line_code["line"]
        seq = line['step_explication'].split(":")
        label = seq[0].strip()
        step_explication = seq[1].strip()
        node = Node(
            id=f"{label}_pseudo_code_{str(uuid4())}",
            type="Pseudo_Code",
            properties={
                "doc_ref": doc_ref,
                "label": label,
                'number': line_code['number'],
                "step_explication": step_explication,
                "explanation": line['step_explication']
            }
        )

        relation_test_step_code = Relationship(
            id= f"{label}_test_step_{str(uuid4())}",
            type="USE_PSEUDO_CODE",
            source=test_step,
            target=node,
            properties={
                "doc_ref": doc_ref,
            }
        )
        nodes_generated.append(node)
        relations_generated.append(relation_test_step_code)
        locs = line_code["locs_line"]
        for loc in locs:
            segment = loc.split(":")
            node_tool = Node(
                id=f"{uuid4()}--||--{test_step.id}--||--{loc}",
                type="Tool_Suggestion",
                properties={
                    "doc_ref": doc_ref,
                    "function": loc.split(":")[1].strip() if len(segment) > 1 else "",
                    "path": loc.split(":")[0].strip() if len(segment) > 1 else loc,
                }
            )
            relationship = Relationship(
                source=node,
                target=node_tool,
                type="HAS_TOOL_SUGGESTION"
            )
            nodes_generated.append(node_tool)
            relations_generated.append(relationship)
    with open("app-config.json", "r") as f:
        data = json.load(f)
        count = data["reps"]
    prepared_code = annotate_code(full_code)
    code_lines = []
    verification = False
    for i in range(count):
        try:
            code_lines = fl.map_pseudo_code_to_code(prepared_code,[line_code["line"] for line_code in tools ], nodes_taken)
            verification = verify_overlap(code_lines, nodes_taken, nodes)
            if verification:
                break
        except CheckerFailure:
            logging.error(f"verification failed for pseudo code {test_step} at iteration {i}")
    if not verification:
        raise CheckerFailure(f"verification failed for pseudo code {test_step}")
    for line_found in code_lines:
        node_code_found = find_line_code_nodes(nodes,line_found.strip())
        if node_code_found is not None:
            nodes_taken.append(node_code_found)
            relation_code = Relationship(
                id=f"code_{str(uuid4())}",
                type="TEST_STEP_CODE_MAP",
                source=test_step,
                target=node_code_found,
                properties={
                    "doc_ref": doc_ref,
                }
            )
            relations_generated.append(relation_code)

    return nodes_generated, relations_generated



