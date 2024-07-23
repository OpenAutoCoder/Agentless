def load_classification_results():
    import csv

    instance_id_to_type = {}

    with open("swebench_lite_classifications.csv", "r") as f:
        reader = csv.reader(f)
        # skip first row
        next(reader)
        for row in reader:
            instance_id = row[0]

            instance_id_to_type[instance_id] = {
                "problem_statement": row[1],
                "gt_patch": row[2],
                "test_patch": row[3],
                "description_classification": row[4],
                "solution_classification": row[5],
                "line_location": row[6],
                "function_location": row[7],
                "file_location": row[8],
            }

    return instance_id_to_type


def load_swebench_s_problems():

    instance_id_to_type = load_classification_results()
    description_to_instance_id = {}
    description_patch_to_instance_id = {}

    for instance_id, instance_type in instance_id_to_type.items():
        if (
            instance_type["description_classification"]
            not in description_to_instance_id
        ):
            description_to_instance_id[instance_type["description_classification"]] = []

        description_to_instance_id[instance_type["description_classification"]].append(
            instance_id
        )

        if (
            instance_type["solution_classification"]
            not in description_patch_to_instance_id
        ):
            description_patch_to_instance_id[
                instance_type["solution_classification"]
            ] = []

        description_patch_to_instance_id[
            instance_type["solution_classification"]
        ].append(instance_id)

    filtered_set = set([instance_id for instance_id in instance_id_to_type.keys()])

    filtered_set = (
        filtered_set
        - set(description_to_instance_id["Not enough info"])
        - set(description_patch_to_instance_id["Misleading"])
        - set(description_patch_to_instance_id["Exact patch"])
    )

    with open("swebench_lite_s_problems.txt", "w") as f:
        for instance_id in filtered_set:
            f.write(instance_id + "\n")


if __name__ == "__main__":
    load_swebench_s_problems()
