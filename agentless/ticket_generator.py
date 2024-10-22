import argparse
import json
import logging
import os

from apps.services.neo4jDB.graphDB_dataAccess import create_graph_database_connection
from apps.services.quality_checkers.test_code_qte_check import schema_test_code

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

query_get_cr = """
MATCH (cr:Coverage_Result)-[:{}]->()
WHERE apoc.convert.fromJsonMap(cr.explanation).result = false AND cr.doc_ref = '{}'
RETURN cr
"""

query_getter_logic = """
MATCH (le:Business_logic_error)-[:{}]->(child)
WHERE le.doc_ref = '{}' AND le.id = '{}'
CALL apoc.path.spanningTree(child, {{
  relationshipFilter: '<'
}}) YIELD path
WITH le, child, nodes(path) AS pathNodes, relationships(path) AS rels
WITH child, 
     [node IN pathNodes | {{id: id(node), labels: labels(node), explanation: node.explanation, number:node.number, reference: node.reference}}] AS nodes,
     [rel IN rels | {{source: id(startNode(rel)), target: id(endNode(rel)), type: type(rel), explanation: rel.explanation}}] AS relationships
RETURN nodes, relationships
"""

query_getter_code = """
MATCH (child)-[:{}]->(le:Code_error)
WHERE le.doc_ref = '{}' AND le.id = '{}'
CALL apoc.path.spanningTree(child, {{
  relationshipFilter: '<'
}}) YIELD path
WITH le, child, nodes(path) AS pathNodes, relationships(path) AS rels
WITH child, 
     [node IN pathNodes | {{id: id(node), labels: labels(node), explanation: node.explanation, number:node.number, reference: node.reference}}] AS nodes,
     [rel IN rels | {{source: id(startNode(rel)), target: id(endNode(rel)), type: type(rel), explanation: rel.explanation}}] AS relationships
RETURN nodes, relationships
"""

query_get_code = """
MATCH (le:Code_error)
WHERE le.doc_ref = '{}'
RETURN le
"""


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


def build_graph_logic(id_le, doc_ref, rel, graph):
    res = graph.query(
        query_getter_logic.format(rel, doc_ref, id_le)
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


def build_graph_code(id_le, doc_ref, rel, graph):
    res = graph.query(
        query_getter_code.format(rel, doc_ref, id_le)
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


def generate_coverage_error_ticket(graph, doc_ref):
    crs = graph.query(
        query_get_cr.format("COVER_TEST_STEP", doc_ref)
    )
    crs = [cr['cr'] for cr in crs]
    tickets = []
    for cr in crs:
        graph_req = build_graph(cr['id'], doc_ref, "COVER_TEST_STEP", graph)
        graph_code = build_graph(cr['id'], doc_ref, "COVERED_BY_INSTRUCTION", graph)

        node_code = None
        node_cr = None
        for n in graph_req['nodes']:
            if n['labels'][0].upper() == "Coverage_Result".upper():
                node_cr = n

        for n in graph_code['nodes']:
            if n['labels'][0].upper() in schema_test_code.keys():
                if node_code is None:
                    node_code = n
                else:
                    if n['number'] > node_code['number']:
                        node_code = n
        ticket = ""

        exp = json.loads(node_cr['explanation'])
        ticket += f"the coverage is not met because the following elements are not covered:\n {exp['explanation']}\n"

        ticket += ("\n\n"
                   "the following code can be the problem:\npython\n")
        code = ""
        node = node_code
        while node is not None:
            code = f"{node['reference']}\n" + code
            node = get_next_node(node, graph_code)
        code = f"```\n{code[:-1]}\n```"
        ticket += code
        tickets.append(ticket)
    return tickets


def generate_code_error_ticket(graph, doc_ref):
    les = graph.query(
        query_get_code.format(doc_ref)
    )
    les = [cr['le'] for cr in les]
    tickets = []
    for le in les:
        graph_code = build_graph_code(le['id'], doc_ref, "ERROR", graph)

        node_code = None
        node_le = le

        for n in graph_code['nodes']:
            if n['labels'][0].upper() in schema_test_code.keys():
                if node_code is None:
                    node_code = n
                else:
                    if n['number'] > node_code['number']:
                        node_code = n
        ticket = ""
        exp = node_le['explanation']
        ticket += f"there is code error may be found in the code:\n {exp}\n"

        ticket += ("\n\n"
                   "the following code can be the problem:\npython\n")
        code = ""
        node = node_code
        while node is not None:
            code = f"{node['reference']}\n" + code
            node = get_next_node(node, graph_code)
        code = f"```\n{code[:-1]}\n```"
        ticket += code
        tickets.append(ticket)
    return tickets


def get_all_tickets(graph, doc_ref):
    tickets_cov = generate_coverage_error_ticket(graph, doc_ref)
    tickets_code = generate_code_error_ticket(graph, doc_ref)
    final = []
    final.extend(tickets_cov)
    final.extend(tickets_code)
    return final


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('-dBuserName', type=str, default=os.environ['NEO4J_USERNAME'], help='database userName')
        parser.add_argument('-gituserName', type=str, default='MM', help='git userName')
        parser.add_argument('-database', type=str, default='neo4j', help='database name')
        parser.add_argument('-req_path', type=str, default="datasets/datasets/requirements/sensing_powerpath_current.txt",
                            help='requirement file path')
        parser.add_argument('-code_path', type=str, default="datasets/datasets/testcases/broken/sensing_powerpath_current.py",
                            help='test code file path')
        return parser.parse_args()


    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    MODULE_LOGGER = logging.getLogger(__name__)

    graph_connect = create_graph_database_connection(args)

    tickets_res = get_all_tickets(graph_connect,
                                  f"{args.req_path}--||--{args.code_path}")
    graph_connect._driver.close()
    print(tickets_res)
