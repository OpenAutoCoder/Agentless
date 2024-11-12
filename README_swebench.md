# Agentless experiments on SWE-Bench

<p align="center">
    <big><a href="#-setup">🐈Setup</a></big> |
    <big><a href="#-localization">🙀Localization</a></big> | 
    <big><a href="#-repair">😼Repair</a></big> |
    <big><a href="#-patch-validation-and-selection">😸Patch Validation and Selection</a></big> |
    <big><a href="#-cost">💰Cost</a></big>
</p>

In this document, we will go through the steps to generate the patches on SWE-bench. 
Currently, we support both [SWE-Bench Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite) and [SWE-Bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)

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

Create the folder to save our results
```shell
mkdir results # where we will save our results
```

> [!TIP]
> 
> We currently support SWE-Bench Lite and SWE-Bench Verified benchmarks, you can use `--dataset` to select the benchmark (by default it will be SWE-Bench Lite)
> 
> For example `--dataset=princeton-nlp/SWE-bench_Verified`

> [!TIP]
> 
> Since for each issue in the benchmark (both SWE-Bench Lite and SWE-Bench Verified) we need to checkout the repository and process the files, you might want to save some time by downloading the preprocessed data here: [swebench_lite_repo_structure.zip](https://github.com/OpenAutoCoder/Agentless/releases/tag/v1.5.0)
>
> After downloading, please unzip and export the location as such `export PROJECT_FILE_LOC={folder which you saved}`

> [!TIP]
> 
> You can use `--target_id` to specific a particular bug you want to target. 
> 
> For example `--target_id=django__django-10914`

> [!TIP]
> 
> We use multiple threads (controllable via `--num_threads`) to speed up the Agentless process 

## 🙀 Localization 

In localization, the goal is to find the locations in source code where we need to edit to fix the issues. 
At a high-level, **Agentless** uses a 3-stage localization step to first localize to specific files, then to relevant code elements, and finally to fine-grained edit locations. 
We will now take you through the step-by-step procedure of Agentless in each of the localization steps.

#### 1. Localize to suspicious files

First, we localize to suspicious files. This is done in a multi-step process where we combine both LLM localized files with retrieval files.

Run the following command to generate the LLM-predicted suspicious files:

```shell
python agentless/fl/localize.py --file_level \
                                --output_folder results/swe-bench-lite/file_level \
                                --num_threads 10 \
                                --skip_existing 
```

This will save all the LLM-predicted suspicious file locations in  `results/swe-bench-lite/file_level/loc_outputs.jsonl` with the logs saved in `results/swe-bench-lite/file_level/localization_logs`

We then complement the previous suspicious files with a simple embedding-based retrieval method to identify additional suspicious files.

This is done by first filtering out irrelevant folders by using LLM to produce a list of irrelevant folders that do not need to be retrieved from with the following command:

```shell
python agentless/fl/localize.py --file_level \
                                --irrelevant \
                                --output_folder results/swe-bench-lite/file_level_irrelevant \
                                --num_threads 10 \
                                --skip_existing 
```

This will save the identified irrelevant folders in `results/swe-bench-lite/file_level_irrelevant/loc_outputs.jsonl` with the logs saved in `results/swe-bench-lite/file_level_irrelevant/localization_logs`

We then perform the retrieval (note the embedding is done with OpenAI `text-embedding-3-small` model) by passing in the irrelevant folders and running the following command: 

```shell
python agentless/fl/retrieve.py --index_type simple \
                                --filter_type given_files \
                                --filter_file results/swe-bench-lite/file_level_irrelevant/loc_outputs.jsonl \
                                --output_folder results/swe-bench-lite/retrievel_embedding \
                                --persist_dir embedding/swe-bench_simple \
                                --num_threads 10 
```

This will save the retrieved files in `results/swe-bench-lite/retrievel_embedding/retrieve_locs.jsonl` with the logs saved in `results/swe-bench-lite/retrievel_embedding/retrieval_logs`

Finally we merge the LLM-predicted suspicious file locations with the embedding-based retrieved files to obtain a final list of relevant files:

```shell
python agentless/fl/combine.py  --retrieval_loc_file results/swe-bench-lite/retrievel_embedding/retrieve_locs.jsonl \
                                --model_loc_file results/swe-bench-lite/file_level/loc_outputs.jsonl \
                                --top_n 3 \
                                --output_folder results/swe-bench-lite/file_level_combined 
```

`results/swe-bench-lite/file_level_combined/combined_locs.jsonl` contains the final list of suspicious files identified by Agentless.

#### 2. localize to related elements

Next, we move on to localizing the related elements.

Run the following command to provide the suspicious files from the first stage as input:

```shell
python agentless/fl/localize.py --related_level \
                                --output_folder results/swe-bench-lite/related_elements \
                                --top_n 3 \
                                --compress_assign \
                                --compress \
                                --start_file results/swe-bench-lite/file_level_combined/combined_locs.jsonl \
                                --num_threads 10 \
                                --skip_existing 
```

This will save the related elements in `results/swe-bench-lite/related_elements/loc_outputs.jsonl` with the logs saved in `results/swe-bench-lite/related_elements/localization_logs`

#### 3. localize to edit locations

Finally, using the related elements, we then localize to the edit locations. This is done via sampling to obtain multiple different sets of edit locations:


```shell
python agentless/fl/localize.py --fine_grain_line_level \
                                --output_folder results/swe-bench-lite/edit_location_samples \
                                --top_n 3 \
                                --compress \
                                --temperature 0.8 \
                                --num_samples 4 \
                                --start_file results/swe-bench-lite/related_elements/loc_outputs.jsonl \
                                --num_threads 10 \
                                --skip_existing 
```

This will save the edit locations in `results/swe-bench-lite/edit_location_samples/loc_outputs.jsonl` with the logs saved in `results/swe-bench-lite/edit_location_samples/localization_logs`

<details><summary>⏬ Structure of `loc_outputs.jsonl` <i>:: click to expand ::</i> </summary>
<div>

- `instance_id`: task ID of the issue
- `found_files`: list of files localized by the model
- `additional_artifact_loc_file`: raw output of the model during file-level localization
- `file_traj`: trajectory of the model during file-level localization (e.g., \# of tokens)
- `found_related_locs`: dict of relevant code elements localized by the model
- `additional_artifact_loc_related`: raw output of the model during relevant-code-level localization 
- `related_loc_traj`: trajectory of the model during relevant-code-level localization
- `found_edit_locs`: dict of edit locations localized by the model
- `additional_artifact_loc_edit_location`: raw output of the model during edit-location-level localization 
- `edit_loc_traj`: trajectory of the model during edit-location-level localization

</div>
</details>

Run the following command to separate the individual sets of edit locations:

```shell
python agentless/fl/localize.py --merge \
                                --output_folder results/swe-bench-lite/edit_location_individual \
                                --top_n 3 \
                                --num_samples 4 \
                                --start_file results/swe-bench-lite/edit_location_samples/loc_outputs.jsonl 
```

The separate sets of edit locations can be found in `results/swe-bench-lite/edit_location_individual`. The location files will be named `loc_merged_{x}-{x}_outputs.jsonl` where `x` indicates the individual samples. For our experiments on SWE-bench, we will use all 4 sets of edit locations and perform repairs on them individually to generate 4 different repair runs.

## 😼 Repair

Using the 4 sets of edit locations from before, we now perform repair. 

**Agentless** generates multiple patches per issue (controllable via parameters) and then perform majority voting with patch validation to select the final patch for submission 

Run the following command to generate the patches:

```shell
python agentless/repair/repair.py --loc_file results/swe-bench-lite/edit_location_individual/loc_merged_0-0_outputs.jsonl \
                                  --output_folder results/swe-bench-lite/repair_sample_1 \
                                  --loc_interval \
                                  --top_n=3 \
                                  --context_window=10 \
                                  --max_samples 10  \
                                  --cot \
                                  --diff_format \
                                  --gen_and_process \
                                  --num_threads 2 
```

<details><summary>Additional Repair Commands</summary>
<div>

```shell
for i in {1..3}; do
    python agentless/repair/repair.py --loc_file results/swe-bench-lite/edit_location_individual/loc_merged_${i}-${i}_outputs.jsonl \
                                    --output_folder results/swe-bench-lite/repair_sample_$((i+1)) \
                                    --loc_interval \
                                    --top_n=3 \
                                    --context_window=10 \
                                    --max_samples 10  \
                                    --cot \
                                    --diff_format \
                                    --gen_and_process \
                                    --num_threads 2 
done

```
</div>
</details>

These commands generate 10 samples each (1 greedy and 9 via temperature sampling) as defined `--max_samples 10`. The `--context_window` indicates the amount of code lines before and after each localized edit location we provide to the model for repair. The patches are saved in `results/swe-bench-lite/repair_sample_{i}/output.jsonl`, which contains the raw output of each sample as well as any trajectory information (e.g., number of tokens). The complete logs are also saved in `results/swe-bench-lite/repair_sample_{i}/repair_logs/` 

The above commands will combine to generate 40 samples in total for each bug. 

## 😸 Patch Validation and Selection 

Since Agentless generates multiple candidate patches per issue, we need a way to select a final patch for submission.

To do this, Agentless leverages both regression tests that exist in the codebase as well as generating new reproduction tests that can verify if the patch can solve the original issue.

#### Regression test selection

We first select a set of regression tests (tests that already exist in the repository and pass on the original codebase) to run.

Run the following command to get a list of passing tests in the original codebase:

```shell
python agentless/test/run_regression_tests.py --run_id generate_regression_tests \
                                              --output_file results/swe-bench-lite/passing_tests.jsonl 
```

This will generate a list of passing tests at `results/swe-bench-lite/passing_tests.jsonl`

> [!NOTE]
> 
> We do not use any of provided PASS_TO_PASS field in the SWE-bench benchmark
> 
> We select tests from the complete list of tests which can pass in the original repository

Next, we ask the LLM to remove any tests which should not be ran with the following command:

```shell
python agentless/test/select_regression_tests.py --passing_tests results/swe-bench-lite/passing_tests.jsonl \
                                                 --output_folder results/swe-bench-lite/select_regression 
```

This will produce a list of final regression tests in `results/swe-bench-lite/select_regression/output.jsonl` with the logs at `results/swe-bench-lite/select_regression/select_test_logs`

We can run this on all the patches generate, repeated for each repair run (i.e., by changing `folder`):

```shell
folder=results/swe-bench-lite/repair_sample_1
for num in {0..9..1}; do
    run_id_prefix=$(basename $folder); 
    python agentless/test/run_regression_tests.py --regression_tests results/swe-bench-lite/select_regression/output.jsonl \
                                                  --predictions_path="${folder}/output_${num}_processed.jsonl" \
                                                  --run_id="${run_id_prefix}_regression_${num}" --num_workers 10;
done
```

This will output the regression test results in the same folder as the repair results. `results/swe-bench-lite/repair_sample_1/output_{i}_regression_test_results.jsonl` contains the regression test results for each patch number (`i`). 

> [!NOTE]
> 
> We also perform post-processing to generate the complete git-diff patch for each repair sample.
> 
> You can find the individual patch in `results/repair/output_{i}_processed.jsonl` where `i` is the sample number. 

#### Reproduction test generation

In addition to the regression tests, Agentless also generates a reproduction test that attempt to check if the patch can solve the original issue.

Similar to patch generation, Agentless also generates multiple samples of reproduction tests, and then perform selection:
```shell
python agentless/test/generate_reproduction_tests.py --max_samples 40 \
                                                     --output_folder results/swe-bench-lite/reproduction_test_samples \
                                                     --num_threads 10 
```

This will generate 40 samples (1 greedy + 39 temperature sampling) per issue. The generated reproduction tests can be found in `results/swe-bench-lite/reproduction_test_samples/output.jsonl`. The corresponding logs can be found in `results/swe-bench-lite/reproduction_test_samples/generating_test_logs/`.

Now we will execute each of these generated tests on the original repository to see if they can reproduce the original issue.

```shell
for st in {0..36..4}; do   en=$((st + 3));   
        echo "Processing ${st} to ${en}";   
        for num in $(seq $st $en); do     
            echo "Processing ${num}";     
            python agentless/test/run_reproduction_tests.py --run_id="reproduction_test_generation_filter_sample_${num}" \
                                                            --test_jsonl="results/swe-bench-lite/reproduction_test_samples/output_${num}_processed_reproduction_test.jsonl" \
                                                            --num_workers 6 \
                                                            --testing;
done & done
```

> [!WARNING]
> 
> In the above command we execute multiple SWE-bench evaluations in parallel, please ensure that your machine is able to handle that
>
> If not, you may want to reduce the amount of parallelization 

This produces verification results for each tests in the same folder: `results/swe-bench-lite/reproduction_test_samples/`

We then perform majority voting to select one reproduction test per issue:

```shell
python agentless/test/generate_reproduction_tests.py --max_samples 40 \
                                                     --output_folder results/swe-bench-lite/reproduction_test_samples \
                                                     --output_file reproduction_tests.jsonl \
                                                     --select
```

This will generate the reproduction test file at: `results/swe-bench-lite/reproduction_test_samples/reproduction_tests.jsonl`

Finally, we evaluate the generated patches on the selected reproduction test. Similar to regression test execution, this is repeated for each repair run (i.e., by changing `folder`):

```shell
folder=results/swe-bench-lite/repair_sample_1
for num in {0..9..1}; do
    run_id_prefix=$(basename $folder); 
    python agentless/test/run_reproduction_tests.py --test_jsonl results/swe-bench-lite/reproduction_test_samples/reproduction_tests.jsonl \
                                                    --predictions_path="${folder}/output_${num}_processed.jsonl" \
                                                    --run_id="${run_id_prefix}_reproduction_${num}" --num_workers 10;
done
```

 This will output the reproduction test results in the same folder as the repair results. `results/swe-bench-lite/repair_sample_1/output_{i}_reproduction_test_results.jsonl` contains the reproduction test results for each patch number (`i`). 


#### Reranking and patch selection 

Finally, using the regression and reproduction test results, Agentless performs reranking to select the final patch for submission.

Run the following command (`--regression` indicates we are using regression tests for selection `--reproduction` indicates we are using the reproduction tests for selection)

```shell
python agentless/repair/rerank.py --patch_folder results/swe-bench-lite/repair_sample_1/,results/swe-bench-lite/repair_sample_2/,results/swe-bench-lite/repair_sample_3/,results/swe-bench-lite/repair_sample_4/ \
                                  --num_samples 40 \
                                  --deduplicate \
                                  --regression \
                                  --reproduction
```

This command will produced the `all_preds.jsonl` that contains the final selected patch for each instance_id which you can then directly use your favorite way of testing SWE-bench for evaluation!


## 💰 Cost 

To measure the cost of running Agentless, we have provided helpful utilities. 

For each of the `output.jsonl` files produced for each of the steps (including substeps), run the following command:

```shell
python dev/util/cost.py --output_file example_step/output.jsonl 
```

This will output the dollar cost as well as the number of tokens. `--embedding_cost` can be used to compute the cost of the embedding step. 
