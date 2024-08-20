export PYTHONPATH="$PYTHONPATH:/srv/home/soren/Agentless/agentless/test/SWE_bench"


**Apply test patch, don't gt_patch:**
python agentless/repair/rerank.py --patch_folder /scratch/agentless/data/cleaned_swe_lite/corrected_gt_patches --num_samples 1 --output_file results/apply_test_patch/apply_test_patch.jsonl --testing --plausible --instance_ids astropy__astropy-14365 --apply_test_patch

**Get subset of tests to run:**
python agentless/repair/rerank.py --patch_folder /scratch/agentless/data/cleaned_swe_lite/corrected_gt_patches --num_samples 1 --output_file results/get_passing_tests/get_passing_tests.jsonl  --plausible --testing --run_all_tests

**Get the PASS_TO_PASS test subset:**
python agentless/util/get_instances_to_eval.py

**Don't apply test patch, don't apply gt_patch:**
python agentless/repair/rerank.py --patch_folder /scratch/agentless/data/cleaned_swe_lite/corrected_gt_patches --num_samples 1 --output_file results/no_test_no_model/no_test_no_model.jsonl --testing --plausible --instance_ids astropy__astropy-14365

**Don't apply test patch, do apply gt_patch:**

python agentless/repair/rerank.py --patch_folder /scratch/agentless/data/cleaned_swe_lite/corrected_gt_patches --num_samples 1 --output_file results/no_test_model/no_test_model.jsonl --plausible --instance_ids astropy__astropy-14365

**Try previously selected patches:**
python agentless/repair/rerank.py --patch_folder /srv/home/soren/20240630_agentless_gpt4o/ --num_samples 1 --output_file results/no_test_with_model_generated/no_test_model_generated.jsonl --plausible

**Trying selected patches for new run:**
python agentless/repair/rerank.py --patch_folder /scratch/agentless/data/results/repair_run_embedding_combined_run --num_samples 1 --output_file results/new_model_generated/new_model_generated.jsonl --plausible


**Reproduce previous Agentless results:**

python agentless/repair/rerank.py --patch_folder /scratch/agentless/data/results/original_top3_add_globvar_0627_0_1/,/scratch/agentless/data/results/original_top3_add_globvar_0627_2_3/ --num_samples 42 --deduplicate --plausible --output_file results/reproduce_original_agentless/reproduce_original_agentless.jsonl

Patch analysis
Scaling analysis for patches
Regenerate tests