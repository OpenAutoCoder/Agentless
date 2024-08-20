import json
import os

from apps.agentless.localisation.localize import get_test_steps, localize

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
    args = {
        "req_path": path_req,
        "instance_id": tool.replace(".json", ""),
        "output_folder": "outputs",
        "output_file": "loc_outputs.jsonl",
        "doc_ref": path_req,
        "top_n": 25,
        "temperature": 0.0,
        "sticky_scroll": False,
        "context_window": 20,

    }
    test_steps = get_test_steps(args["req_path"])

    _, found = localize(args, test_steps)
    tools_val = json.loads(open(os.path.join("datasets/datasets/required_tools", tool)).read())

    locs = found["locs"]
    tools_found = {}
    research = {}
    for loc in locs:
        try:
            path = loc.split(":")[0].strip().replace(".__init__.py", "")
            function = loc.split(":")[1].strip()
            if path not in tools_found:
                tools_found[path] = []
            tools_found[path].append(function)
        except:
            print("Error in ", loc)
    for p in tools_val.keys():
        if p in tools_found.keys():
            research[p] = 0
            res_arr = [a.lower() for a in tools_found[p]]
            for f in tools_val[p]:
                if f.lower() in res_arr:
                    research[p] += 1
    report = {
        "requirement": tool.replace(".json", ""),
        "files_expected": len(tools_val.keys()),
        "expected_functions": number_functions(tools_val),
        "files_found": len(tools_found.keys()),
        "found_functions": number_functions(tools_found),
        "files_found_correct": len(research.keys()),
        "correct_functions": sum(research.values()),
        "found": research,
        "results": tools_found
    }
    output_report.append(report)

    with open("outputs/report.json", "w") as f:
        f.write(json.dumps(output_report))
