# Agentless experiments on SWE-Bench lite 

In this document, we will go through the steps to generate the patches on SWE-bench lite

## 🐈 Setup

First create the environment 

```shell
git clone https://github.com/OpenAutoCoder/Agentless.git
cd Agentless

conda create -n agentless python=3.11 
conda activate agentless
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

Then export your OpenAI API key 
```shell
export OPENAI_API_KEY={key_here}
```

## 🙀 Localization 

> [!TIP]
> 
> For localization, you can use `--target_id` to specific a particular bug you want to target. 
> 
> For example `--target_id=django__django-11039`

In localization, the goal is find the locations in source code where we need to edit to fix the issues. 
**Agentless** uses a 3-stage localization step to first localize to specific files, then to relevant code elements, and finally to fine-grained edit locations.

> [!TIP]
> 
> Since for each issue in the benchmark we need to checkout the repository and process the files, you might want to save some time by downloading the preprocessed data here: [swebench_lite_repo_structure.zip](https://github.com/OpenAutoCoder/Agentless/releases/tag/v0.1.0)
>
> After downloading, please unzip and export the location as such `export PROJECT_FILE_LOC={folder which you saved}`

Run the following command to generate the edit locations: 

```shell
mkdir results # where we will save our results
python agentless/fl/localize.py --file_level --related_level --fine_grain_line_level \
                                --output_folder results/location --top_n 3 \
                                --compress \
                                --context_window=10 \
                                --temperature 0.8 \
                                --num_samples 4
```

This will save all the localized locations in  `results/location/loc_outputs.jsonl` with the logs saved in `results/location/localize.log`

Note that in the last stage of our localization we perform sampling (i.e., with `--temperature 0.8` and `--num_samples 4`) to obtain 4 sets of edit locations.

Following the steps described in our paper, we then perform merging to form a bigger list of edit locations.

Run the following command to merge:

```shell
python agentless/fl/localize.py --merge \
                                --output_folder results/location_merged \
                                --start_file results/location/loc_outputs.jsonl \
                                --num_samples 4
```

This will perform pair-wise merging of samples (i.e., sample 0 and 1 will be merged and sample 2 and 3 will be merged). Furthermore it will also merge all samples together. 

The merged location files can be found in `results/location_merged/loc_merged_{st_id}-{en_id}_outputs.jsonl` where `st_id` and `en_id` indicates the samples that are being merged. The location file with all samples merged together can be found as `results/location_merged/loc_all_merged_outputs.jsonl`. Furthermore, we also include the location of each individual sample for completeness within the folder. 

For our experiments on SWE-bench lite we use the two pair-wised merged locations (i.e., `loc_merged_0-1_outputs.jsonl` and `loc_merged_2-3_outputs.jsonl`) and run two repair runs

## 😼 Repair

Using the two sets of edit locations from before, we now perform repair. 

**Agentless** generates multiple patches per issue (controllable via parameters) and then perform majority voting to select the final patch for submission 

Run the following command to generate the patches:

```shell
python agentless/repair/repair_coverage.py --loc_file results/location_merged/loc_merged_0-1_outputs.jsonl \
                                  --output_folder results/repair_run_1 \
                                  --loc_interval --top_n=3 --context_window=10 \
                                  --max_samples 21  --cot --diff_format \
                                  --gen_and_process 
```

```shell
python agentless/repair/repair_coverage.py --loc_file results/location_merged/loc_merged_2-3_outputs.jsonl \
                                  --output_folder results/repair_run_2 \
                                  --loc_interval --top_n=3 --context_window=10 \
                                  --max_samples 21  --cot --diff_format \
                                  --gen_and_process 
```

These commands generate 21 samples each (1 greedy and 20 via temperature sampling) as defined `--max_samples 21`. The `--context_window` indicates the amount of code lines before and after each localized edit location we provide to the model for repair. The repair results is saved in `results/repair_run_{i}/output.jsonl`, which contains the raw output of each sample as well as the any trajectory information (e.g., number of tokens). The complete logs are also saved in `results/repair_run_{i}/repair.log` 

The above two commands will combine to generate 42 samples in total for each bug. 

Finally, we perform majority voting to select the final patch to solve each issue (from the 42 samples). Run the following command:

```shell
python agentless/repair/rerank.py --patch_folder results/repair_run_1,results/repair_run_2 --num_samples 42 --deduplicate --plausible
```

In this case, we use `--num_samples 42` to pick from the 42 samples we generated previously, `--deduplicate` to apply normalization to each patch for better voting, and `--plausible` to select patches that can pass the previous regression tests (*warning: this feature is not yet implemented*)

This command will produced the `all_preds.jsonl` that contains the final selected patch for each instance_id which you can then directly use your favorite way of testing SWE-bench for evaluation!
