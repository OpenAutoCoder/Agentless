import argparse
import json
import os

from Agentless.agentless.localisation.localize import get_test_steps, localize

tools = os.listdir(
    "datasets/datasets/required_tools"
)

output_report = []


def number_functions(obj):
    res = 0
    for k in obj.keys():
        res += len(obj[k])
    return res


for tool in tools:

    path_req = os.path.join(
        "datasets/datasets/requirements", tool.replace(".json", ".txt"))

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('-dBuserName', type=str, default=os.environ['NEO4J_USERNAME'], help='database userName')
        parser.add_argument('-gituserName', type=str, default='MM', help='git userName')
        parser.add_argument('-database', type=str, default='neo4j', help='database name')
        parser.add_argument('-cmd', type=str, default='FULL', help='cmd')
        parser.add_argument('-req_path', type=str, default="datasets/datasets/requirements/sensing_powerpath_current.txt",
                            help='requirement file path')
        parser.add_argument('-instance_id', type=str, default="sensing_powerpath_current", help='instance id')
        parser.add_argument('-output_folder', type=str, default="tests", help='output folder')
        parser.add_argument('-output_file', type=str, default="loc_outputs.jsonl", help='output file')
        parser.add_argument('-doc_ref', type=str, default="datasets/datasets/requirements/sensing_powerpath_current.txt",
                            help='doc ref')
        parser.add_argument('-top_n', type=int, default=25, help='top n')
        parser.add_argument('-temperature', type=float, default=0.0, help='temperature')
        parser.add_argument('-sticky_scroll', type=bool, default=False, help='sticky scroll')
        parser.add_argument('-context_window', type=int, default=20, help='context window')

        return parser.parse_args()
    args = parse_args()
    test_steps = get_test_steps(args.req_path)

    _, found = localize(args, test_steps)
    tools_val = json.loads(open(os.path.join("datasets/datasets/required_tools", tool)).read())

    locs = found["locs"]
    tools_found = {}
    research = {}
    for loc in locs:
        try:
            path = loc.split(":")[0].strip().replace(".__init__.py", "")
            function = loc.split(":")[1].strip()
            seq = path.split(".")
            if seq[-1][0].isupper():
                seq = seq[:-1]
            path = ".".join(seq)
            if path not in tools_found:
                tools_found[path] = []
            tools_found[path].append(function)
        except:
            print("Error in ", loc)
    for p in tools_val.keys():
        elements = p.split(".")
        if elements[-1][0].isupper():
            interface = elements[-1]
            elements = elements[:-1]
        path = ".".join(elements)
        found = False
        for f in tools_found.keys():
            if path in f:
                found = True
                break
        if found:
            research[path] = 0
            res_arr = [a.lower() for a in tools_found[path]]
            for f in tools_val[p]:
                for r in res_arr:
                    if f.lower() in r:
                        research[path] += 1
                        break
    report = {
        "requirement": tool.replace(".json", ""),
        "files_expected": len(tools_val.keys()),
        "expected_functions": number_functions(tools_val),
        "files_found": len(tools_found.keys()),
        "found_functions": number_functions(tools_found),
        "files_found_correct": len(research.keys()),
        "correct_functions": sum(research.values()),
        "percentage_files_found": (len(research.keys()) / len(tools_val.keys()))*100,
        "percentage_functions_found": (sum(research.values()) / number_functions(tools_val)) * 100,
        "final_score": (len(research.keys()) / len(tools_val.keys())) * (sum(research.values()) / number_functions(tools_val)) * 100,
        "found": research,
        "results": tools_found
    }
    output_report.append(report)

    with open("outputs/report.json", "w") as f:
        f.write(json.dumps(output_report))
