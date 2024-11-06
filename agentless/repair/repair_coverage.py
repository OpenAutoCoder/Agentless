import argparse
import json
import logging
import os

from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from apps.helper import read_file
from apps.services.neo4jDB.graphDB_dataAccess import create_graph_database_connection
from apps.services.open_ia_llm import OpenIA_LLM
from apps.services.quality_checkers.test_code_qte_check import schema_test_code
from apps.services.code_skeleton_extractor import filtered_methods_by_file_name_function

repair_relevant_file_instruction = """
Below are some code segments, each from a relevant file. One or more of these files may contain bugs.
"""
repair_relevant_file_with_scope_instruction = """
Below are some code segments, each from a relevant file. One or more of these files may contain bugs.
In the file below, "..." refers to some less relevant content being omited for brebity.
"""
with_scope_explanation = """
Note that "..." refers to some omitted content that is not actually in the files. Your *SEARCH/REPLACE* edit must not contain such "...".
"""
repair_relevant_file_with_suspicious_loc_instruction = """
Below are some code segments, each from a relevant file. One or more of these files may contain bugs. Some suspicious locations are provided for closer inspection.
"""

context = """
You are an expert in the testing of embedded software for automotive cars. 
You are currently working on a project to test the embedded software of a car. 
You have been provided with  TAF framework that is used for the testing
You are tasked to analyse requirement and fix the problems in the implementation of the test code
You see suggestions of tools that can be used to fix the problem the suggestion will be provided 
Keep in mind the differing perspectives: the requirements are written from the perspective of the Device Under Test (DUT), while the test framework methods are written from the perspective of the test system interacting with the DUT.
- **do not include any extra information in the output

"""

repair_prompt_combine_topn = """
We are currently solving the following requirement
--- Requirement ---
{test_step}
--- ISSUE ---
{issue}

--- CODE potential issue ---
{content_error}

-- Possible tools to use --
{tools}

--- BEGIN FILE ---
```
{content}
```
--- END FILE ---

## Strict Guidelines:
- **Do not invent or fabricate any tool names**; only include those present in the provided.
- **Focus solely on the test step provided do not correct more from the requirement**
- **do not include any extra information in the output
- **Main action focus: Only implement the code that represent the primary action in the test step. Ignore setup or context actions that occur before or after.
- **use methods and functions only from the tools given do not integrate any other
- **return the full test code corrected do not miss any line**
- **if the old implementation is correct do not change it and return exactly the same full test code given 
- Adherence to these guidelines is critical. Any deviation, such as creating non-existent tool names or returning empty result or returning full test code missing some line , will lead to immediate disqualification from the task.

"""

query_get_cr = """
MATCH (cr:Coverage_Result)-[:{}]->()
WHERE apoc.convert.fromJsonMap(cr.explanation).result = false AND cr.doc_ref = '{}'
RETURN cr
"""
query_getter_coverage = """
MATCH (cr:Coverage_Result)-[:{}]->(child)
WHERE apoc.convert.fromJsonMap(cr.explanation).result = false AND cr.doc_ref = '{}' AND cr.id = '{}'
CALL apoc.path.spanningTree(child, {{
  relationshipFilter: '<'
}}) YIELD path
WITH cr, child, nodes(path) AS pathNodes, relationships(path) AS rels
WITH child, 
     [node IN pathNodes | {{id: node.id, labels: labels(node), explanation: node.explanation, number:node.number, reference: node.reference}}] AS nodes,
     [rel IN rels | {{source: startNode(rel).id, target: endNode(rel).id, type: type(rel), explanation: rel.explanation}}] AS relationships
RETURN nodes, relationships
"""

query_get_cr_tool = """
MATCH (cr:Test_step)-[:{}]->(child:Tool_Suggestion)
WHERE cr.id = '{}' AND cr.doc_ref = '{}' AND child.doc_ref = '{}'
RETURN child
"""


def get_tools(node_step, graph, doc_ref):
    res = graph.query(
        query_get_cr_tool.format("HAS_TOOL_SUGGESTION", node_step['id'], doc_ref, doc_ref)
    )
    tools = [el['child'] for el in res]
    return tools


def verify_element_id(node_find, gra):
    for na in gra['nodes']:
        if na['id'] == node_find['id']:
            return True
    return False


def build_graph(id_cr, doc_ref, rel, graph):
    res = graph.query(
        query_getter_coverage.format(rel, doc_ref, id_cr)
    )
    gra = {
        "nodes": [],
        "edges": [],
    }
    for record in res:
        for nt in record['nodes']:
            if not verify_element_id(nt, gra):
                gra['nodes'].append(nt)
        for edge in record['relationships']:
            gra['edges'].append(edge)
    return gra


def get_next_node(el, gra):
    for edge in gra['edges']:
        if edge['source'] == el['id']:
            for nn in gra['nodes']:
                if nn['id'] == edge['target']:
                    return nn
    return None

@traceable(
    name="iteration in repair coverage error tickets",
)
def iteration_repair(cr ,doc_ref,graph, model, corrected_code):
    graph_req = build_graph(cr['id'], doc_ref, "COVER_TEST_STEP", graph)
    graph_code = build_graph(cr['id'], doc_ref, "COVERED_BY_INSTRUCTION", graph)

    node_code = None
    node_cr = None
    test_step = None
    for n in graph_req['nodes']:
        if n['labels'][0].upper() == "Coverage_Result".upper():
            node_cr = n
        if n['labels'][0].upper() == "TEST_STEP":
            test_step = n

    for n in graph_code['nodes']:
        if n['labels'][0].upper() in schema_test_code.keys():
            if node_code is None:
                node_code = n
            else:
                if n['number'] > node_code['number']:
                    node_code = n
    tools = get_tools(test_step, graph, doc_ref)
    paths = []
    functions_name = []
    for tool in tools:
        path = tool['path']
        seq = path.split(".")
        last = seq[-1]
        while last[0].isupper() or last == "py":
            seq.pop()
            last = seq[-1]

        path = "/".join(seq)
        paths.append(path + ".py")
        functions_name.append(tool['function'])

    code = ""
    node = node_code
    while node is not None:
        code = f"{node['reference']}\n" + code
        node = get_next_node(node, graph_code)
    code = f"```\n{code[:-1]}\n```"
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                context,
            ),
            (
                "human",
                repair_prompt_combine_topn,
            ),
        ]
    )

    chain = prompt | model
    tools_graph = filtered_methods_by_file_name_function(graph, paths, functions_name)
    res = chain.invoke({
        "test_step": test_step['explanation'],
        "issue": json.loads(node_cr['explanation'])['explanation'],
        "tools": tools_graph,
        "content": corrected_code,
        "content_error": code,

    })
    return res.content


@traceable(name="repair coverage error tickets")
def generate_coverage_error_ticket(graph, doc_ref, test_code):
    crs = graph.query(
        query_get_cr.format("COVER_TEST_STEP", doc_ref)
    )
    crs = [cr['cr'] for cr in crs]
    model = OpenIA_LLM.get_model(
        OpenIA_LLM.get_version_model(
            "generate_coverage_error_ticket"
        )
    )
    corrected_code = test_code
    for cr in crs:
        corrected_code = iteration_repair(cr, doc_ref, graph, model, corrected_code)
    print("Corrected code after repair coverage error tickets")
    print(corrected_code)
    return corrected_code


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('-dBuserName', type=str, default=os.environ['NEO4J_USERNAME'], help='database userName')
        parser.add_argument('-gituserName', type=str, default='MM', help='git userName')
        parser.add_argument('-database', type=str, default='neo4j', help='database name')
        parser.add_argument('-cmd', type=str, default='FULL', help='cmd')
        parser.add_argument('-req_path', type=str,
                            default="datasets/datasets/requirements/sensing_powerpath_current.txt",
                            help='requirement file path')
        parser.add_argument('-code_path', type=str,
                            default="datasets/datasets/testcases/broken/sensing_powerpath_current.py",
                            help='test code file path')
        return parser.parse_args()


    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    MODULE_LOGGER = logging.getLogger(__name__)

    graph_connect = create_graph_database_connection(args)
    test_code = read_file(args.code_path)
    generate_coverage_error_ticket(graph_connect,
                                   f"{args.req_path}--||--{args.code_path}", test_code)
    graph_connect._driver.close()