import argparse
import json
import logging
import os

from langsmith import traceable

from Agentless.agentless.util.model import make_model
from apps.helper import read_file
from apps.services.code_skeleton_extractor import filtered_methods_by_file_name_function, filtered_nodes_by_label
from apps.services.neo4jDB.graphDB_dataAccess import create_graph_database_connection
from apps.services.open_ia_llm import OpenIA_LLM

query_get_pseudo_code_nodes = """
Match (n:Pseudo_Code) where n.doc_ref = '{doc_ref}' return n
"""

query_get_tools_from_pseudo_code = """
Match (n:Pseudo_Code)-[:HAS_TOOL_SUGGESTION]->(k:Tool_Suggestion)
 where n.doc_ref = '{doc_ref}' and k.doc_ref = '{doc_ref}' and n.id = '{id_node_code}'
    return k
"""

query_get_related_code = """
Match (n:Pseudo_Code)-[:PSEUDO_CODE_MAP]->(k)
 where n.doc_ref = '{doc_ref}' and k.doc_ref = '{doc_ref}' and n.id = '{id_node_code}'
    return k
"""

prompt_generate_version_code = """
You are an expert in test code implementation within the automotive zone controller domain, using a custom test framework repository called TAF (Test Automotive Framework). 
Your task is to analyse test step of requirement and i will provide you with functions and methods from the TAF to implement it , then create a code that implement the test step of requirement using the functions and methods
For more context i will provide you with the rest of the TAF if needed but you should mainly use the tool given and i will provide you the requirement form more context
Keep in mind the differing perspectives: the requirements are written from the perspective of the Device Under Test (DUT), while the test framework methods are written from the perspective of the test system interacting with the DUT.


### Requirement ###
{requirement}

### Test step ###
{test_step}

### Tools to use ###
{tools}


### TAF code ###
{taf_code}


## Strict Guidelines:
- **Do not invent or fabricate any tool names**; only include those present in the provided.
- **Focus solely on the test step provided do not implement more from the requirement**
- **do not include any extra information in the output
- **Main action focus: Only implement the code that represent the primary action in the test step. Ignore setup or context actions that occur before or after.
- Adherence to these guidelines is critical. Any deviation, such as creating non-existent tool names or returning empty result , will lead to immediate disqualification from the task.

"""


prompt_to_compare_versions_implementation="""
You are an expert in test code implementation within the automotive zone controller domain, using a custom test framework repository called TAF (Test Automotive Framework). 
i have implement a new version of a test a step of requirement using a new methods and functions so i want to analyse my implementation and the old implementation witch in most case better and give me what should pass in the code
I will provide you with the TAF description for more context same for requirement
Keep in mind the differing perspectives: the requirements are written from the perspective of the Device Under Test (DUT), while the test framework methods are written from the perspective of the test system interacting with the DUT.

### Requirement ###
{requirement}

### Test step ###
{test_step}

### my implementation ###
{generated}

### old implementation
{old}

### the full test code ###
{test_code}

### TAF description ###
{taf}

## Strict Guidelines:
- **Do not invent or fabricate any tool names**; only include those present in the provided.
- **Focus solely on the test step provided do not correct more from the requirement**
- **do not include any extra information in the output
- **Main action focus: Only implement the code that represent the primary action in the test step. Ignore setup or context actions that occur before or after.
- **use methods and functions only from the 2 implementation do not integrate any other
- **return the full test code corrected do not miss any line**
- **if the old implementation is correct do not change it and return exactly the same full test code given 
- Adherence to these guidelines is critical. Any deviation, such as creating non-existent tool names or returning empty result or returning full test code missing some line , will lead to immediate disqualification from the task.

"""


def retrieve_pseudo_code(doc_ref, graph):
    res = graph.query(query_get_pseudo_code_nodes.format(doc_ref=doc_ref))
    return [a['n'] for a in res]

def retrieve_code_related(doc_ref, line, graph):
    res = graph.query(query_get_related_code.format(doc_ref=doc_ref,id_node_code=line["id"]))
    return [a['k'] for a in res]

def retrieve_tools_related(doc_ref, line, graph):
    res = graph.query(query_get_tools_from_pseudo_code.format(doc_ref=doc_ref,id_node_code=line["id"]))
    return [a['k'] for a in res]


def treat_path(path):
    path = path.replace(".py","").replace(".","/")
    seq = path.split("/")
    if str(seq[-1][0]).isupper():
        seq = seq[:-1]
    return '/'.join(seq)

@traceable(
    name="generate a version of code implemented with the right tools found"
)
def generate_version_with_tool(requirement_text,tools, line, taf):
    model = make_model(
        model=OpenIA_LLM.get_version_model("generate_version_with_tool"),
        max_tokens=9000,
        temperature=0,
        batch_size=1,
    )
    template_prompt = prompt_generate_version_code.format(
        requirement=requirement_text,
        test_step=line["explanation"],
        tools=json.dumps(tools,indent=4),
        taf_code=json.dumps(taf,indent=4),
    )
    traj = model.codegen(template_prompt, num_samples=1)[0]
    raw_output = traj["response"]
    return raw_output

@traceable(
    name="merge the 2 versions of the code"
)
def merge_two_versions_of_code(line,taf,old_version,new_version,test_code_text, requirement_text):
    template_prompt = prompt_to_compare_versions_implementation.format(
        requirement=requirement_text,
        test_step=line["explanation"],
        generated=new_version,
        old=old_version,
        test_code=test_code_text,
        taf=json.dumps(
            taf,indent=4
        )
    )
    model = make_model(
        model=OpenIA_LLM.get_version_model("merge_two_versions_of_code"),
        max_tokens=9000,
        temperature=0,
        batch_size=1,
    )
    traj = model.codegen(template_prompt, num_samples=1)[0]
    raw_output = traj["response"]
    return raw_output


@traceable(name="iteration of the repair of the code")
def iteration_repair(tools_not_used,graph,line,requirement_text, code,code_corrected):
    tools_name = [tool['function'].split("(")[0] for tool in tools_not_used]
    paths = [treat_path(tool["path"]) for tool in tools_not_used]
    description_tools = filtered_methods_by_file_name_function(graph, paths, tools_name)
    taf_code = filtered_nodes_by_label(graph, line['label'].strip())
    new_version = generate_version_with_tool(requirement_text, description_tools, line, taf_code)
    full_taf = filtered_nodes_by_label(graph)
    return merge_two_versions_of_code(line, full_taf, code, new_version, code_corrected, requirement_text)


@traceable(
    name="repair the code based on the use of the tools used"
)
def repair_code_tools(doc_ref, requirement_text ,test_code_text, graph):
    code_corrected = test_code_text
    lines_code = retrieve_pseudo_code(doc_ref, graph)
    for line in lines_code:
        code_related = retrieve_code_related(doc_ref, line, graph)
        tools_related = retrieve_tools_related(doc_ref, line, graph)
        code = ''
        number = 0
        if len(code_related) > 0:
            code_related.sort(key=lambda x: int(x["number"] if "number" in x.keys() else 0))
            number  = int(code_related[0]['number'])
        for code_line in code_related:
            if int(code_line["number"]) - number > 1:
                code += "\n#other code...   \n\n"
            code += code_line['reference']
            code += '\n'

        tools_not_used = []
        for tools_line in tools_related:
            if tools_line['function'] not in code:
                tools_not_used.append(tools_line)

        if len(tools_not_used) > 0:
            code_corrected = iteration_repair(tools_not_used, graph, line, requirement_text, code, code_corrected)
        print("="*80)
    print(code_corrected)
    return code_corrected





dataset = {'req': 'datasets/datasets/requirements/sensing_powerpath_current.txt',
           'code': 'datasets/datasets/testcases/broken/sensing_powerpath_current.py'}


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('-dBuserName', type=str, default=os.environ['NEO4J_USERNAME'], help='database userName')
        parser.add_argument('-gituserName', type=str, default='MM', help='git userName')
        parser.add_argument('-database', type=str, default='neo4j', help='database name')
        parser.add_argument('-cmd', type=str, default='AA', help='cmd')
        parser.add_argument('-req_file_path', type=str, default=dataset['req'],
                            help='requirement file path')
        parser.add_argument('-test_code_file_path', type=str, default=dataset['code'],
                            help='test code file path')
        return parser.parse_args()


    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    MODULE_LOGGER = logging.getLogger(__name__)

    graph_connect = create_graph_database_connection(args)
    test_code = read_file(args.test_code_file_path)
    requirement = read_file(args.req_file_path)
    doc_ref_id = f"{args.req_file_path}--||--{args.test_code_file_path}"
    repair_code_tools(doc_ref_id, requirement,test_code, graph_connect)

