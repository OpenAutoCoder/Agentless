import matplotlib as mpl
from matplotlib import pyplot as plt

plt.style.use(
    "https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pacoty.mplstyle"
)
plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.weight"] = "bold"
plt.rcParams.update({"font.size": 10})

plt.rcParams["axes.facecolor"] = "#FFFFFF"
plt.rcParams["axes.edgecolor"] = "black"

mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=[
        "#9fc5e8",
        "#b4a7d6",
        "#b6d7a8",
        "#ea9999",
        "#d5a6bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
)

from load_filtered_benchmark import load_classification_results


def graph_benchmark_classification():

    instance_id_to_type = load_classification_results()

    description_to_count = {}
    description_patch_to_count = {}
    location_to_count = {}
    location_function_to_count = {}
    location_file_to_count = {}
    location_line_to_count = {}

    for instance_id, classification in instance_id_to_type.items():
        description = classification["description_classification"]
        description_patch = classification["solution_classification"]
        location_line = classification["line_location"]
        location_function = classification["function_location"]
        location_file = classification["file_location"]
        location = classification["line_location"]

        if description in description_to_count:
            description_to_count[description] += 1
        else:
            description_to_count[description] = 1

        if description_patch in description_patch_to_count:
            description_patch_to_count[description_patch] += 1
        else:
            description_patch_to_count[description_patch] = 1

        if (
            location_line == "Natural language"
            or location_function == "Natural language"
            or location_file == "Natural language"
        ):
            location = "Natural language"
        elif (
            location_line == "Stacktrace"
            or location_function == "Stacktrace"
            or location_file == "Stacktrace"
        ):
            location = "Stacktrace"
        elif (
            location_line == "Keywords"
            or location_function == "Keywords"
            or location_file == "Keywords"
        ):
            location = "Keywords"
        else:
            location = "None"

        if location in location_to_count:
            location_to_count[location] += 1
        else:
            location_to_count[location] = 1

        if location_line in location_line_to_count:
            location_line_to_count[location_line] += 1
        else:
            location_line_to_count[location_line] = 1

        if location_function in location_function_to_count:
            location_function_to_count[location_function] += 1
        else:
            location_function_to_count[location_function] = 1

        if location_file in location_file_to_count:
            location_file_to_count[location_file] += 1
        else:
            location_file_to_count[location_file] = 1

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    wdges, labels, autopct = ax.pie(
        description_to_count.values(),
        labels=[x.replace(" ", "\n") for x in description_to_count.keys()],
        explode=[
            0.1 if x == "Not enough info" else 0.02 for x in description_to_count.keys()
        ],
        autopct="%1.1f%%",
        shadow={"ox": -0.04, "edgecolor": "none", "shade": 0.5},
        textprops={"fontsize": 18},
    )
    plt.setp(labels, fontsize=12)
    fig.tight_layout()
    plt.savefig("benchmark_pie_description.pdf")

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    wdges, labels, autopct = ax.pie(
        description_patch_to_count.values(),
        labels=[x.replace(" ", "\n") for x in description_patch_to_count.keys()],
        explode=[
            0.1 if x == "Exact patch" else 0.02
            for x in description_patch_to_count.keys()
        ],
        autopct="%1.1f%%",
        shadow={"ox": -0.04, "edgecolor": "none", "shade": 0.5},
        startangle=90,
        textprops={"fontsize": 18},
    )
    plt.setp(labels, fontsize=12)

    fig.tight_layout()
    plt.savefig("benchmark_pie_patch.pdf")

    # 4 stack bar plots
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    bottom = [0, 0, 0, 0]
    counts = {}

    width = 0.5

    for loc in ["None", "Stacktrace", "Natural language", "Keywords"]:
        if loc == "Stacktrace":
            counts["Stack trace"] = []
            if loc in location_line_to_count:
                counts["Stack trace"].append(location_line_to_count[loc] / 3)
            else:
                counts["Stack trace"].append(0)

            counts["Stack trace"].append(location_function_to_count[loc] / 3)
            counts["Stack trace"].append(location_file_to_count[loc] / 3)
            counts["Stack trace"].append(location_to_count[loc] / 3)
        else:
            counts[loc] = []

            if loc in location_line_to_count:
                counts[loc].append(location_line_to_count[loc] / 3)
            else:
                counts[loc].append(0)

            counts[loc].append(location_function_to_count[loc] / 3)
            counts[loc].append(location_file_to_count[loc] / 3)
            counts[loc].append(location_to_count[loc] / 3)

    for label, count in counts.items():
        ax.bar(
            ["Line", "Function", "File", "Overall"],
            count,
            bottom=bottom,
            label=label,
            width=width,
        )
        bottom = [sum(x) for x in zip(bottom, count)]

        # write text
        for i, v in enumerate(count):
            if v == 0:
                continue
            ax.text(
                i,
                bottom[i] - v / 1.5,
                str(round(v, 1)) + "%",
                color="black",
                fontweight="bold",
                ha="center",
                fontsize=15,
            )

    ax.set_ylabel("Percentage")
    ax.legend(fancybox=True, shadow=True, prop={"size": 10}, frameon=True)
    fig.tight_layout()
    plt.savefig("benchmark_bar_location.pdf")


if __name__ == "__main__":
    graph_benchmark_classification()
