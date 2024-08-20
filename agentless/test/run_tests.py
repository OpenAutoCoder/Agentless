import docker
import json
import platform
import resource
import traceback

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Any

from Agentless.agentless.test.SWE_bench.swebench.harness.constants import (
    SWEbenchInstance,
    KEY_INSTANCE_ID,
    FAIL_TO_PASS,
    PASS_TO_PASS,
    MAP_REPO_VERSION_TO_SPECS,
    USE_X86,
)
from Agentless.agentless.test.SWE_bench.swebench.harness.docker_build import build_env_images
from Agentless.agentless.test.SWE_bench.swebench.harness.docker_utils import (
    should_remove,
    list_images,
    clean_images,
)
from Agentless.agentless.test.SWE_bench.swebench.harness.run_evaluation import (
    run_instance,
    get_dataset_from_preds,
)
from Agentless.agentless.test.SWE_bench.swebench.harness.test_spec import (
    TestSpec,
    make_env_script_list,
    make_repo_script_list,
    DIFF_MODIFIED_FILE_REGEX
)
from Agentless.agentless.test.SWE_bench.swebench.harness.utils import get_test_directives

OPEN_FILE_LIMIT = 4096

NOOP_PATCH = (
    "diff --git a/invisible.py b/invisible.py\n"
    "new file mode 100644\n"
    "index 0000000..e69de29\n"
)

NOOP_PATCH = '''diff --git a/this_is_invisible.py b/this_is_invisible.py
new file mode 100644
index 0000000..e69de29
--- /dev/null
+++ b/this_is_invisible.py
@@ -0,0 +1 @@
+# This is a commented out line
'''

NOOP_PATCH_2 = '''diff --git a/this_is_invisible_2.py b/this_is_invisible_2.py
new file mode 100644
index 0000000..e69de29
--- /dev/null
+++ b/this_is_invisible_2.py
@@ -0,0 +1 @@
+# This is a commented out line
'''

def make_regression_spec(instance: SWEbenchInstance) -> TestSpec:
    if isinstance(instance, TestSpec):
        return instance
    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance["repo"]
    version = instance["version"]
    base_commit = instance["base_commit"]
    test_patch = instance["test_patch"]
    apply_test_patch = instance["apply_test_patch"]

    def _from_json_or_obj(key: str) -> Any:
        """If key points to string, load with json"""
        if isinstance(instance[key], str):
            return json.loads(instance[key])
        return instance[key]

    pass_to_pass = _from_json_or_obj(PASS_TO_PASS)
    fail_to_pass = _from_json_or_obj(FAIL_TO_PASS)

    env_name = "testbed"
    repo_directory = f"/{env_name}"
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]

    repo_script_list = make_repo_script_list(specs, repo, repo_directory, base_commit, env_name)
    env_script_list = make_env_script_list(instance, specs, env_name)
    eval_script_list = make_regression_script_list(
        instance, specs, env_name, repo_directory, base_commit, test_patch, apply_test_patch
    )
    if platform.machine() in {"aarch64", "arm64"}:
        # use arm64 unless explicitly specified
        arch = "arm64" if instance_id not in USE_X86 else "x86_64"
    else:
        arch = "x86_64"

    return TestSpec(
        instance_id=instance_id,
        repo=repo,
        env_script_list=env_script_list,
        repo_script_list=repo_script_list,
        eval_script_list=eval_script_list,
        version=version,
        arch=arch,
        FAIL_TO_PASS=fail_to_pass, # Remove the fail to pass cases
        PASS_TO_PASS=pass_to_pass,
    )

def make_regression_script_list(instance, specs, env_name, repo_directory, base_commit, test_patch, apply_test_patch):
    """
    Applies the test patch and runs the tests.
    """
    # test_files = re.findall(DIFF_MODIFIED_FILE_REGEX, test_patch)

    # Reset test files to the state they should be in before the patch.
    reset_tests_command = f"git checkout {base_commit}"# {' '.join(test_files)}" 

    HEREDOC_DELIMITER = "EOF_114329324912"
    if apply_test_patch:
        apply_test_patch_command = (
            f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
        )
    else:
        apply_test_patch_command =  f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{NOOP_PATCH_2}\n{HEREDOC_DELIMITER}"

    test_command = " ".join(
        [
            MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]["test_cmd"],
            *get_test_directives(instance),
        ]
    )
    eval_commands = [
        f"source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
    ]
    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]
    eval_commands += [
        f"git config --global --add safe.directory {repo_directory}",  # for nonroot user
        f"cd {repo_directory}",
        # This is just informational, so we have a record
        f"git status",
        f"git show",
        f"git diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
    ]
    if "install" in specs:
        eval_commands.append(specs["install"])
    eval_commands += [
        reset_tests_command,
        apply_test_patch_command, # If not explictly set to apply the test patch this command doesn't do anything
        # "git add this_is_invisible.py",
        test_command,
        reset_tests_command,
    ]
    return eval_commands

def run_tests(
        instance_ids : list, 
        model_patches : list,
        max_workers : int,
        run_id : str,        
        regression_test_file : str,
        instances_to_run : list,
        timeout : int,
        apply_model_patch = True,
        apply_test_patch = False,
        run_all_tests = False
    ):

    assert len(instance_ids) == len(model_patches), "There must be the same number of instance_ids as model patches"
    resource.setrlimit(resource.RLIMIT_NOFILE, (OPEN_FILE_LIMIT, OPEN_FILE_LIMIT))

    print(f"Using run_id: {run_id}")
    dataset_name = "princeton-nlp/SWE-bench_Lite"
    split = "test"
    client = docker.from_env()
    force_rebuild = False
    clean = False

    predictions = {}

    for idx, one_instance_id in enumerate(instance_ids):
            if not apply_model_patch:
                patch_to_apply = NOOP_PATCH
            else:
                patch_to_apply = model_patches[idx]
            predictions[one_instance_id] = {"model_name_or_path" : "test", 
                                        "model_patch": patch_to_apply,
                                        "instance_id" : one_instance_id}

    dataset = get_dataset_from_preds(dataset_name, split, instance_ids, predictions, run_id)
    existing_images = list_images(client)

    print(f"Running {len(dataset)} unevaluated instances...")
    if not dataset:
        print("No instances to run.")
    else:
        build_env_images(client, dataset, force_rebuild, max_workers)

    instances = get_dataset_from_preds(
        dataset_name, 
        split, 
        instance_ids, 
        predictions, 
        run_id)
    
    instance_test_dict = {}
    
    # Open and read the jsonl file line by line
    with open(regression_test_file, 'r') as file:
        for line in file:
            # Parse each line as a JSON object
            json_obj = json.loads(line.strip())
            # Set instance_id as the key and test as the value
            instance_id = json_obj['instance_id']
            test = json_obj['tests_passing_in_original_repo']
            instance_test_dict[instance_id] = test

    no_f2p_instances = []
    for instance in instances:
        revised_instance = instance
        revised_instance["FAIL_TO_PASS"] = '[]'
        if not run_all_tests:
            revised_instance["PASS_TO_PASS"] = instance_test_dict[instance["instance_id"]]
        revised_instance["apply_test_patch"] = apply_test_patch
        no_f2p_instances.append(revised_instance)
    
    test_specs = list(map(make_regression_spec, no_f2p_instances))


    instance_image_ids = {x.instance_image_key for x in test_specs}
    existing_images = {
        tag for i in client.images.list(all=True)
        for tag in i.tags if tag in instance_image_ids
    }

    if instances_to_run:
        ids = instances_to_run
    else:
        ids = [test_spec.instance_id for test_spec in test_specs]

    should_remove_val_by_instance = {test_spec.instance_id : should_remove(test_spec.instance_id, "env", clean, existing_images) for test_spec in test_specs}

    results = {}
    resolved_dict = {}

    with tqdm(total=len(ids), smoothing=0) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a future for running each instance
            futures = {
                executor.submit(
                    run_instance,
                    test_spec,
                    predictions[test_spec.instance_id],
                    should_remove_val_by_instance[test_spec.instance_id],
                    force_rebuild,
                    client,
                    run_id,
                    timeout,
                ): None
                for test_spec in test_specs if test_spec.instance_id in ids
            }
            # Wait for each future to complete
            for future in as_completed(futures):
                pbar.update(1)
                result = future.result()  
                if result:
                    instance_id = result[0]
                    resolved = result[1][instance_id]["resolved"]

                    resolved_dict[instance_id] = resolved
                try:
                    # Update progress bar, check if instance ran successfully
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    results[instance_id] = False  # Or handle the error case as needed
                    resolved_dict[instance_id] = False
                    continue

    print("All instances run.")
    clean_images(client, existing_images, "env", clean)
    return resolved_dict