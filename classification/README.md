# SWE-bench Lite-*S* and problem classifications

Here we provide our manual classifications of the problems in SWE-bench lite

> [!NOTE]
>
> The classification results differ slightly with the paper as we have performed further manual checking and modified the classifications of few problems.

## Quick Setup

First install the neccessary graphing utilizies:

```shell
cd classification
pip install -r requirements_graph.txt
```

## Graphing

Run the following command to produce Figure 5 in the paper:

```shell
python graph_classification.py
```

This will produce various graphs: `benchmark_bar_location.pdf`, `benchmark_pie_description.pdf`, and `benchmark_pie_patch.pdf`

For the detailed classifications, we have released our raw classifications for each problem and can be access in `swebench_lite_classifications.csv`

## SWE-bench Lite-*S* problems

To obtain the problems in SWE-bench Lite-*S*, run the following command:

```shell
python load_filtered_benchmark.py
```

This produces the `swebench_lite_s_problems.txt` which contains the instance_ids of the subset of problems

Note: we filter out the problems that contain the exact patch in the problem description, misleading solutions, or do not provide enough information in the original issue description.
