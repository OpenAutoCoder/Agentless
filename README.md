# 😺 Agentless

<p align="center">
    <a href="https://arxiv.org/abs/TODO"><img src="https://img.shields.io/badge/📃-Arxiv-b31b1b?style=for-the-badge"></a>
    <a href="https://github.com/OpenAutoCoder/Agentless/blob/master/LICENSE"><img src="https://forthebadge.com/images/badges/cc-by.svg" style="height: 28px"></a>
</p>

<p align="center">
    <big><a href="#-setup">🐈Setup</a></big> |
    <big><a href="#-localization">🙀Localization</a></big> |
    <big><a href="#-repair">😼Repair</a></big> |
    <big><a href="#-artifacts">🐈‍⬛Artifacts</a></big> |
    <big><a href="#-citation">📝Citation</a></big> |
    <big><a href="#-acknowledgement">😽Acknowledgement</a></big>
</p>

## 😺 About 

**Agentless** is an *agentless* approach to automatically solve software development problems. To solve each issue, **Agentless** follows a simple two phase process: localization and repair.
- 🙀 Localization: **Agentless** employs a hierarchical process to first localize the fault to specific files, then to relevant classes or functions, and finally to fine-grained edit locations
- 😼 Repair : **Agentless** takes the edit locations and generates multiple candidate patches in a simple diff format, performs test filtering, and re-ranks all remaining patches to selects one to submit

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

<details><summary>⏬ Developer Setup</summary>
<div>

```shell
# for contribution, please install the pre-commit hook.
pre-commit install  # this allows a more standardized code style
```

</div>
</details>

Then export your OpenAI API key 
```shell
export OPENAI_API_KEY={key_here}
```

Now you are ready to run **Agentless** on the problems in SWE-bench! We now go through a step-by-step example of how to run **Agentless** 

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
                                --context_window=10 
```

This will save all the localized locations in  `results/location/loc_outputs.jsonl` with the logs saved in `results/location/localize.log`


<details><summary>⏬ Structure of `loc_outputs.jsonl` <i>:: click to expand ::</i> </summary>
<div>

- `instance_id`: task ID of the issue
- `found_files`: list of files localized by the model
- `additional_artifact_loc_file`: raw output of the model during file-level localization
- `file_traj`: trajectory of the model during file-level localization (e.g., \# of tokens)
- `found_related_locs`: list of relevant code elements localized by the model
- `additional_artifact_loc_related`: raw output of the model during relevant-code-level localization 
- `related_loc_traj`: trajectory of the model during relevant-code-level localization
- `found_edit_locs`: list of edit locations localized by the model
- `additional_artifact_loc_edit_location`: raw output of the model during edit-location-level localization 
- `edit_loc_traj`: trajectory of the model during edit-location-level localization

</div>
</details>

<details><summary>🙀 Individual localization steps <i>:: click to perform the individual localization step ::</i> </summary>
<div>

#### Localize to files

We first start by localization to specific files

```shell
mkdir results # where we will save our results
python agentless/fl/localize.py --file_level --output_folder results/file_level
```

This command saves the file-level localization in `results/file_level/loc_outputs.jsonl`, you can also check `results/file_level/localize.log` for detailed logs

#### Localize to related elements

Next, we localize to related elements within each of the files we localize

```shell
python agentless/fl/localize.py --related_level \
                                --output_folder results/related_level \
                                --start_file results/file_level/loc_outputs.jsonl \
                                --top_n 3 --compress
```

Here the `--start_file` refers to the previous file-level localization. `--top_n` argument indicates the number of files we want to consider.

Similar to the previous stage, this command saves the related-element localization in `results/related_level/loc_outputs.jsonl`, with logs in `results/related_level/localize.log`

#### Localize to edit locations

Finally, we take the related elements from the previous step and localize to the edit locations we want the LLM to generate patches for

```shell
python agentless/fl/localize.py --fine_grain_line_level \
                                --output_folder results/edit_location \
                                --start_file results/related_level/loc_outputs.jsonl \
                                --top_n 3 --context_window=10 
```

Here the `--start_file` refers to the previous related-element localization. `--context_window` indicates the amount of lines before and after we provide to the LLM.

The final edit locations **Agentless** will perform repair on is saved in `results/edit_location/loc_outputs.jsonl`, with logs in `results/edit_location/localize.log`

> you can use `--temperature` and `--num_samples` to sample multiple of the third step and merge them 

</div>
</details>

## 😼 Repair

Using the edit locations (i.e., `found_edit_locs`) from before, we now perform repair. 

**Agentless** generates multiple patches per issue (controllable via parameters) and then perform majority voting to select the final patch for submission 

Run the following command to generate the patches:

```shell
python agentless/repair/repair.py --loc_file results/location/loc_outputs.jsonl \
                                  --output_folder results/repair \
                                  --loc_interval --top_n=3 --context_window=10 \
                                  --max_samples 10  --cot --diff_format \
                                  --gen_and_process 
```

This command generates 10 samples (1 greedy and 9 via temperature sampling) as defined `--max_samples 10`. The `--context_window` indicates the amount of code lines before and after each localized edit location we provide to the model for repair. The repair results is saved in `results/repair/output.jsonl`, which contains the raw output of each sample as well as the any trajectory information (e.g., number of tokens). The complete logs are also saved in `results/repair/repair.log`

> [!NOTE]
> 
> We also perform post-processing to generate the complete git-diff patch for each repair samples.
> 
> You can find the individual patch in `results/repair/output_{i}_processed.jsonl` where `i` is the sample number. 

Finally, we perform majority voting to select the final patch to solve each issue. Run the following command:

```shell
python agentless/repair/rerank.py --patch_folder results/repair --num_samples 10 --deduplicate --plausible
```

In this case, we use `--num_samples 10` to pick from the 10 samples we generated previously, `--deduplicate` to apply normalization to each patch for better voting, and `--plausible` to select patches that can pass the previous regression tests (*warning: this feature is not yet implemented*)

This command will produced the `all_preds.jsonl` that contains the final selected patch for each instance_id which you can then directly use your favorite way of testing SWE-bench for evaluation!

## 🐈‍⬛ Artifacts

You can download the complete artifacts of **Agentless** in our [v0.1.0 release](https://github.com/OpenAutoCoder/Agentless/releases/tag/v0.1.0):
- 🐈‍⬛ agentless_logs: raw logs and trajectory information
- 🐈‍⬛ swebench_lite_repo_structure**: preprocessed structure information for each SWE-Bench-lite problem
- 🐈‍⬛ 20240630_agentless_gpt4o: evaluated run of **Agentless** used in our paper

## 📝 Citation

```bibtex
@article{agentless,
  author    = {Xia, Chunqiu Steven and Deng, Yinlin and Dunn, Soren and Zhang, Lingming},
  title     = {Agentless: Demystifying LLM-based Software Engineering Agents},
  year      = {2024},
  journal   = {arXiv preprint},
}
```

> [!NOTE]
> 
> The first two authors contributed equally to this work, with author order determined via [*Nigiri*](https://senseis.xmp.net/?Nigiri)

## 😽 Acknowledgement 

* [SWE-bench](https://www.swebench.com/)
* [Aider](https://github.com/paul-gauthier/aider)
* [SWE-bench-docker](https://github.com/aorwall/SWE-bench-docker)