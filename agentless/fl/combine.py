import argparse
import json
import os
from collections import Counter

from tqdm import tqdm

from agentless.util.utils import load_jsonl


def combine_file_level(args):

    embed_used_locs = load_jsonl(args.retrieval_loc_file)
    model_used_locs = load_jsonl(args.model_loc_file)

    with open(f"{args.output_folder}/embed_used_locs.jsonl", "w") as f:
        for loc in embed_used_locs:
            f.write(json.dumps(loc) + "\n")

    with open(f"{args.output_folder}/model_used_locs.jsonl", "w") as f:
        for loc in model_used_locs:
            f.write(json.dumps(loc) + "\n")

    for pred in tqdm(model_used_locs, colour="MAGENTA"):
        instance_id = pred["instance_id"]

        model_loc = pred["found_files"]
        retrieve_loc = [x for x in embed_used_locs if x["instance_id"] == instance_id][
            0
        ]["found_files"]

        combined_loc_counter = Counter()
        combined_locs = []

        for loc in model_loc[: args.top_n]:
            combined_loc_counter[loc] += 1

        for loc in retrieve_loc[: args.top_n]:
            combined_loc_counter[loc] += 1

        combined_locs = [loc for loc, _ in combined_loc_counter.most_common()]

        with open(args.output_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "found_files": combined_locs,
                        "additional_artifact_loc_file": {},
                        "file_traj": {},
                    }
                )
                + "\n"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="combined_locs.jsonl")
    parser.add_argument("--retrieval_loc_file", type=str, required=True)
    parser.add_argument("--model_loc_file", type=str, required=True)
    # supports file level (step-1) combination
    parser.add_argument("--top_n", type=int, required=True)

    args = parser.parse_args()

    args.output_file = os.path.join(args.output_folder, args.output_file)
    assert not os.path.exists(args.output_file), "Output file already exists"

    os.makedirs(args.output_folder, exist_ok=True)

    # dump argument
    with open(os.path.join(args.output_folder, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    combine_file_level(args)


if __name__ == "__main__":
    main()
