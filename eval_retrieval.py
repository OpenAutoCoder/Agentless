"""
Runs retrieval evaluation on the given dataset.
"""
import pandas as pd
import re
import random
import argparse
import json
from multiprocessing import Pool
import os
import sys
from collections import defaultdict
from datasets import load_dataset
from typing import List

from agentless.retreival.VectorDB import QdrantDB
from agentless.retreival.rerank_files import rerank_files
from agentless.retreival.upload_swe_bench_to_qdrant import init_repo


def load_jsonl(filepath):
    """
    Load a JSONL file from the given filepath.

    Arguments:
    filepath -- the path to the JSONL file to load

    Returns:
    A list of dictionaries representing the data in each line of the JSONL file.
    """
    with open(filepath, "r") as file:
        return [json.loads(line) for line in file]


def get_affected_files(patch_string):
    pattern = r'diff --git a/(.*?) b/(.*?)$'
    matches = re.findall(pattern, patch_string, re.MULTILINE)
    affected_files = set()
    for match in matches:
        affected_files.add(match[0])  # 'a' path
        affected_files.add(match[1])  # 'b' path
    
    return list(affected_files)


def find_relevant_files_using_vector_search(name:str, code_dict:dict, problem_statement: str, top_n: int = 15):
    # filter out all non python files
    code_dict = {k: v for k, v in code_dict.items() if k.endswith('.py')}

    # filter out test files
    code_dict = {k: v for k, v in code_dict.items() if not k.startswith('test')}

    
    # Initialize vectordb
    vectordb = QdrantDB(
        repo_name=name,
        codebase_dict=code_dict,
        include_chunk_content=True,
        regenerate_embeddings=False,
    )
    
    # Search for relevant files
    vector_search_results = vectordb.search_code(problem_statement, top_n=top_n)

    vector_search_results_files = [result['file_path'] for result in vector_search_results]

    code_dict_for_vector_search = {}
    for file in vector_search_results_files:
        code_dict_for_vector_search[file] = code_dict[file]
        
    reranked_files = rerank_files(problem_statement, code_dict_for_vector_search)
    
    return reranked_files

def create_vector_search_predictions(swe_bench_data, pred_jsonl_path, top_n=10, instance_ids: List[str] = None):
    """
    Creates or updates the predictions for the given dataset and prediction jsonl file.
    """
    # Read existing predictions if file exists
    existing_predictions = {}
    if os.path.exists(pred_jsonl_path):
        with open(pred_jsonl_path, "r") as f:
            for line in f:
                pred = json.loads(line)
                existing_predictions[pred["instance_id"]] = pred

    if instance_ids is not None:
        filtered_swe_bench_data = swe_bench_data.filter(lambda x: x["instance_id"] in instance_ids)
    else:
        filtered_swe_bench_data = swe_bench_data

    for instance in filtered_swe_bench_data:
        instance_id = instance["instance_id"]
        problem_statement = instance["problem_statement"]
        owner, repo_name = instance["repo"].split('/')

        # Use custom_init_repo to initialize the repository
        fs = init_repo(owner, repo_name, instance['base_commit'])

        code_dict = fs.get_code_dict()
        
        # Find relevant files using vector search
        relevant_files = find_relevant_files_using_vector_search(
            name=repo_name,
            code_dict=code_dict,
            problem_statement=problem_statement,
            top_n=top_n
        )

        relevant_files_paths = [file for file in relevant_files]
        
        prediction = {
            "instance_id": instance_id,
            "found_files": relevant_files_paths
        }
        existing_predictions[instance_id] = prediction

    # Save updated predictions to jsonl file
    with open(pred_jsonl_path, "w") as f:
        for pred in existing_predictions.values():
            json.dump(pred, f)
            f.write("\n")
    
    print(f"Predictions saved to {pred_jsonl_path}")


def get_retrieval_eval_results(swe_bench_data, pred_jsonl_path, top_k_list: List[int] = [1,3,5,7,10]):
    """
    Runs retrieval evaluation on the given dataset and returns the results along with actual files changed.
    """
    pred_data = load_jsonl(pred_jsonl_path)
    avg_recalls = {k: 0 for k in top_k_list}
    count = 0
    actual_files_changed = {}
    instance_recalls = defaultdict(dict)

    for pred in pred_data:
        instance_id = pred["instance_id"]
        lite_dataset = swe_bench_data.filter(lambda x: x["instance_id"] == instance_id)
        if len(lite_dataset) == 0:
            continue
        
        gt_patch = lite_dataset[0]["patch"]
        pred_files = pred["found_files"]
        gt_files = get_affected_files(gt_patch)
        
        actual_files_changed[instance_id] = gt_files

        for _k in top_k_list:
            pred_files_k = pred_files[:_k]
            
            intersection = set(pred_files_k) & set(gt_files)
            recall_at_k = len(intersection) / len(set(gt_files)) if gt_files else 0
            avg_recalls[_k] += recall_at_k
            instance_recalls[instance_id][f"recall@{_k}"] = recall_at_k
        count += 1

    if count == 0:
        print("No instances found in the dataset.")
        return {k: 0 for k in top_k_list}, 0, {}

    avg_recalls = {k: recall / count for k, recall in avg_recalls.items()}

    # Add actual_files_changed and recalls to the predictions
    updated_predictions = []
    with open(pred_jsonl_path, "r") as f:
        for line in f:
            pred = json.loads(line)
            instance_id = pred["instance_id"]
            if instance_id in actual_files_changed:
                # Create a new dictionary with the desired order
                updated_pred = {
                    "instance_id": pred["instance_id"],
                    "actual_files_changed": list(actual_files_changed[instance_id]),
                    **instance_recalls[instance_id],
                    **{k: v for k, v in pred.items() if k != "instance_id"}
                }
            else:
                updated_pred = pred
            updated_predictions.append(updated_pred)

    # Write updated predictions back to the file
    with open(pred_jsonl_path, "w") as f:
        for pred in updated_predictions:
            json.dump(pred, f)
            f.write("\n")

    print(f"Updated predictions with actual_files_changed and recalls saved to {pred_jsonl_path}")
    
    return avg_recalls, count


if __name__ == "__main__":
    # use parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, default="exploiter345/SWE-bench_Verified_50")
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--preds_path", type=str, default="")
    
    args = parser.parse_args()

    assert args.preds_path != ""

    all_instance_ids = ['astropy__astropy-12907', 'astropy__astropy-13033', 'astropy__astropy-13236', 'astropy__astropy-13398', 'astropy__astropy-13453', 'astropy__astropy-13579', 'astropy__astropy-13977', 'astropy__astropy-14096', 'astropy__astropy-14182', 'astropy__astropy-14309', 'astropy__astropy-14365', 'astropy__astropy-14369', 'astropy__astropy-14508', 'astropy__astropy-14539', 'astropy__astropy-14598', 'astropy__astropy-14995', 'astropy__astropy-7166', 'astropy__astropy-7336', 'astropy__astropy-7606', 'astropy__astropy-7671', 'astropy__astropy-8707', 'astropy__astropy-8872', 'django__django-10097', 'django__django-10554', 'django__django-10880', 'django__django-10914', 'django__django-10973', 'django__django-10999', 'django__django-11066', 'django__django-11087', 'django__django-11095', 'django__django-11099', 'django__django-11119', 'django__django-11133', 'django__django-11138', 'django__django-11141', 'django__django-11149', 'django__django-11163', 'django__django-11179', 'django__django-11206', 'django__django-11211', 'django__django-11239', 'django__django-11265', 'django__django-11276', 'django__django-11292', 'django__django-11299', 'django__django-11333', 'django__django-11400', 'django__django-11433', 'django__django-11451', 'django__django-11477', 'django__django-11490', 'django__django-11532', 'django__django-11551', 'django__django-11555', 'django__django-11603', 'django__django-11728', 'django__django-11734', 'django__django-11740', 'django__django-11749', 'django__django-11790', 'django__django-11815', 'django__django-11820', 'django__django-11848', 'django__django-11880', 'django__django-11885', 'django__django-11951', 'django__django-11964', 'django__django-11999', 'django__django-12039', 'django__django-12050', 'django__django-12125', 'django__django-12143', 'django__django-12155', 'django__django-12193', 'django__django-12209', 'django__django-12262', 'django__django-12273', 'django__django-12276', 'django__django-12304', 'django__django-12308', 'django__django-12325', 'django__django-12406', 'django__django-12419', 'django__django-12663', 'django__django-12708', 'django__django-12713', 'django__django-12741', 'django__django-12754', 'django__django-12774', 'django__django-12858', 'django__django-12965', 'django__django-13012', 'django__django-13023', 'django__django-13028', 'django__django-13033', 'django__django-13089', 'django__django-13109', 'django__django-13112', 'django__django-13121', 'django__django-13128', 'django__django-13158', 'django__django-13195', 'django__django-13212', 'django__django-13279', 'django__django-13297', 'django__django-13315', 'django__django-13343', 'django__django-13344', 'django__django-13346', 'django__django-13363', 'django__django-13401', 'django__django-13406', 'django__django-13410', 'django__django-13417', 'django__django-13449', 'django__django-13512', 'django__django-13513', 'django__django-13516', 'django__django-13551', 'django__django-13568', 'django__django-13569', 'django__django-13590', 'django__django-13658', 'django__django-13670', 'django__django-13741', 'django__django-13786', 'django__django-13794', 'django__django-13807', 'django__django-13809', 'django__django-13810', 'django__django-13820', 'django__django-13821', 'django__django-13837', 'django__django-13925', 'django__django-13933', 'django__django-13964', 'django__django-14007', 'django__django-14011', 'django__django-14017', 'django__django-14034', 'django__django-14053', 'django__django-14089', 'django__django-14122', 'django__django-14140', 'django__django-14155', 'django__django-14170', 'django__django-14238', 'django__django-14311', 'django__django-14315', 'django__django-14349', 'django__django-14351', 'django__django-14373', 'django__django-14376', 'django__django-14404', 'django__django-14434', 'django__django-14493', 'django__django-14500', 'django__django-14534', 'django__django-14539', 'django__django-14559', 'django__django-14580', 'django__django-14608', 'django__django-14631', 'django__django-14672', 'django__django-14725', 'django__django-14752', 'django__django-14765', 'django__django-14771', 'django__django-14787', 'django__django-14792', 'django__django-14855', 'django__django-14915', 'django__django-14999', 'django__django-15022', 'django__django-15037', 'django__django-15098', 'django__django-15103', 'django__django-15104', 'django__django-15127', 'django__django-15128', 'django__django-15161', 'django__django-15252', 'django__django-15268', 'django__django-15277', 'django__django-15278', 'django__django-15280', 'django__django-15315', 'django__django-15368', 'django__django-15375', 'django__django-15380', 'django__django-15382', 'django__django-15467', 'django__django-15499', 'django__django-15503', 'django__django-15525', 'django__django-15554', 'django__django-15561', 'django__django-15563', 'django__django-15569', 'django__django-15572', 'django__django-15629', 'django__django-15695', 'django__django-15731', 'django__django-15732', 'django__django-15741', 'django__django-15814', 'django__django-15851', 'django__django-15863', 'django__django-15916', 'django__django-15930', 'django__django-15957', 'django__django-15973', 'django__django-15987', 'django__django-16032', 'django__django-16082', 'django__django-16100', 'django__django-16116', 'django__django-16136', 'django__django-16139', 'django__django-16145', 'django__django-16255', 'django__django-16256', 'django__django-16263', 'django__django-16315', 'django__django-16333', 'django__django-16429', 'django__django-16454', 'django__django-16485', 'django__django-16493', 'django__django-16502', 'django__django-16527', 'django__django-16560', 'django__django-16569', 'django__django-16595', 'django__django-16612', 'django__django-16631', 'django__django-16642', 'django__django-16661', 'django__django-16662', 'django__django-16667', 'django__django-16801', 'django__django-16819', 'django__django-16877', 'django__django-16899', 'django__django-16901', 'django__django-16938', 'django__django-16950', 'django__django-17029', 'django__django-17084', 'django__django-17087', 'django__django-7530', 'django__django-9296', 'matplotlib__matplotlib-13989', 'matplotlib__matplotlib-14623', 'matplotlib__matplotlib-20488', 'matplotlib__matplotlib-20676', 'matplotlib__matplotlib-20826', 'matplotlib__matplotlib-20859', 'matplotlib__matplotlib-21568', 'matplotlib__matplotlib-22719', 'matplotlib__matplotlib-22865', 'matplotlib__matplotlib-22871', 'matplotlib__matplotlib-23299', 'matplotlib__matplotlib-23314', 'matplotlib__matplotlib-23412', 'matplotlib__matplotlib-23476', 'matplotlib__matplotlib-24026', 'matplotlib__matplotlib-24149', 'matplotlib__matplotlib-24177', 'matplotlib__matplotlib-24570', 'matplotlib__matplotlib-24627', 'matplotlib__matplotlib-24637', 'matplotlib__matplotlib-24870', 'matplotlib__matplotlib-24970', 'matplotlib__matplotlib-25122', 'matplotlib__matplotlib-25287', 'matplotlib__matplotlib-25311', 'matplotlib__matplotlib-25332', 'matplotlib__matplotlib-25479', 'matplotlib__matplotlib-25775', 'matplotlib__matplotlib-25960', 'matplotlib__matplotlib-26113', 'matplotlib__matplotlib-26208', 'matplotlib__matplotlib-26291', 'matplotlib__matplotlib-26342', 'matplotlib__matplotlib-26466', 'mwaskom__seaborn-3069', 'mwaskom__seaborn-3187', 'pallets__flask-5014', 'psf__requests-1142', 'psf__requests-1724', 'psf__requests-1766', 'psf__requests-1921', 'psf__requests-2317', 'psf__requests-2931', 'psf__requests-5414', 'psf__requests-6028', 'pydata__xarray-2905', 'pydata__xarray-3095', 'pydata__xarray-3151', 'pydata__xarray-3305', 'pydata__xarray-3677', 'pydata__xarray-3993', 'pydata__xarray-4075', 'pydata__xarray-4094', 'pydata__xarray-4356', 'pydata__xarray-4629', 'pydata__xarray-4687', 'pydata__xarray-4695', 'pydata__xarray-4966', 'pydata__xarray-6461', 'pydata__xarray-6599', 'pydata__xarray-6721', 'pydata__xarray-6744', 'pydata__xarray-6938', 'pydata__xarray-6992', 'pydata__xarray-7229', 'pydata__xarray-7233', 'pydata__xarray-7393', 'pylint-dev__pylint-4551', 'pylint-dev__pylint-4604', 'pylint-dev__pylint-4661', 'pylint-dev__pylint-4970', 'pylint-dev__pylint-6386', 'pylint-dev__pylint-6528', 'pylint-dev__pylint-6903', 'pylint-dev__pylint-7080', 'pylint-dev__pylint-7277', 'pylint-dev__pylint-8898', 'pytest-dev__pytest-10051', 'pytest-dev__pytest-10081', 'pytest-dev__pytest-10356', 'pytest-dev__pytest-5262', 'pytest-dev__pytest-5631', 'pytest-dev__pytest-5787', 'pytest-dev__pytest-5809', 'pytest-dev__pytest-5840', 'pytest-dev__pytest-6197', 'pytest-dev__pytest-6202', 'pytest-dev__pytest-7205', 'pytest-dev__pytest-7236', 'pytest-dev__pytest-7324', 'pytest-dev__pytest-7432', 'pytest-dev__pytest-7490', 'pytest-dev__pytest-7521', 'pytest-dev__pytest-7571', 'pytest-dev__pytest-7982', 'pytest-dev__pytest-8399', 'scikit-learn__scikit-learn-10297', 'scikit-learn__scikit-learn-10844', 'scikit-learn__scikit-learn-10908', 'scikit-learn__scikit-learn-11310', 'scikit-learn__scikit-learn-11578', 'scikit-learn__scikit-learn-12585', 'scikit-learn__scikit-learn-12682', 'scikit-learn__scikit-learn-12973', 'scikit-learn__scikit-learn-13124', 'scikit-learn__scikit-learn-13135', 'scikit-learn__scikit-learn-13142', 'scikit-learn__scikit-learn-13328', 'scikit-learn__scikit-learn-13439', 'scikit-learn__scikit-learn-13496', 'scikit-learn__scikit-learn-13779', 'scikit-learn__scikit-learn-14053', 'scikit-learn__scikit-learn-14087', 'scikit-learn__scikit-learn-14141', 'scikit-learn__scikit-learn-14496', 'scikit-learn__scikit-learn-14629', 'scikit-learn__scikit-learn-14710', 'scikit-learn__scikit-learn-14894', 'scikit-learn__scikit-learn-14983', 'scikit-learn__scikit-learn-15100', 'scikit-learn__scikit-learn-25102', 'scikit-learn__scikit-learn-25232', 'scikit-learn__scikit-learn-25747', 'scikit-learn__scikit-learn-25931', 'scikit-learn__scikit-learn-25973', 'scikit-learn__scikit-learn-26194', 'scikit-learn__scikit-learn-26323', 'scikit-learn__scikit-learn-9288', 'sphinx-doc__sphinx-10323', 'sphinx-doc__sphinx-10435', 'sphinx-doc__sphinx-10449', 'sphinx-doc__sphinx-10466', 'sphinx-doc__sphinx-10614', 'sphinx-doc__sphinx-10673', 'sphinx-doc__sphinx-11445', 'sphinx-doc__sphinx-11510', 'sphinx-doc__sphinx-7440', 'sphinx-doc__sphinx-7454', 'sphinx-doc__sphinx-7462', 'sphinx-doc__sphinx-7590', 'sphinx-doc__sphinx-7748', 'sphinx-doc__sphinx-7757', 'sphinx-doc__sphinx-7889', 'sphinx-doc__sphinx-7910', 'sphinx-doc__sphinx-7985', 'sphinx-doc__sphinx-8035', 'sphinx-doc__sphinx-8056', 'sphinx-doc__sphinx-8120', 'sphinx-doc__sphinx-8265', 'sphinx-doc__sphinx-8269', 'sphinx-doc__sphinx-8459', 'sphinx-doc__sphinx-8475', 'sphinx-doc__sphinx-8548', 'sphinx-doc__sphinx-8551', 'sphinx-doc__sphinx-8593', 'sphinx-doc__sphinx-8595', 'sphinx-doc__sphinx-8621', 'sphinx-doc__sphinx-8638', 'sphinx-doc__sphinx-8721', 'sphinx-doc__sphinx-9229', 'sphinx-doc__sphinx-9230', 'sphinx-doc__sphinx-9258', 'sphinx-doc__sphinx-9281', 'sphinx-doc__sphinx-9320', 'sphinx-doc__sphinx-9367', 'sphinx-doc__sphinx-9461', 'sphinx-doc__sphinx-9591', 'sphinx-doc__sphinx-9602', 'sphinx-doc__sphinx-9658', 'sphinx-doc__sphinx-9673', 'sphinx-doc__sphinx-9698', 'sphinx-doc__sphinx-9711', 'sympy__sympy-11618', 'sympy__sympy-12096', 'sympy__sympy-12419', 'sympy__sympy-12481', 'sympy__sympy-12489', 'sympy__sympy-13031', 'sympy__sympy-13091', 'sympy__sympy-13372', 'sympy__sympy-13480', 'sympy__sympy-13551', 'sympy__sympy-13615', 'sympy__sympy-13647', 'sympy__sympy-13757', 'sympy__sympy-13798', 'sympy__sympy-13852', 'sympy__sympy-13877', 'sympy__sympy-13878', 'sympy__sympy-13974', 'sympy__sympy-14248', 'sympy__sympy-14531', 'sympy__sympy-14711', 'sympy__sympy-14976', 'sympy__sympy-15017', 'sympy__sympy-15345', 'sympy__sympy-15349', 'sympy__sympy-15599', 'sympy__sympy-15809', 'sympy__sympy-15875', 'sympy__sympy-15976', 'sympy__sympy-16450', 'sympy__sympy-16597', 'sympy__sympy-16766', 'sympy__sympy-16792', 'sympy__sympy-16886', 'sympy__sympy-17139', 'sympy__sympy-17318', 'sympy__sympy-17630', 'sympy__sympy-17655', 'sympy__sympy-18189', 'sympy__sympy-18199', 'sympy__sympy-18211', 'sympy__sympy-18698', 'sympy__sympy-18763', 'sympy__sympy-19040', 'sympy__sympy-19346', 'sympy__sympy-19495', 'sympy__sympy-19637', 'sympy__sympy-19783', 'sympy__sympy-19954', 'sympy__sympy-20154', 'sympy__sympy-20428', 'sympy__sympy-20438', 'sympy__sympy-20590', 'sympy__sympy-20801', 'sympy__sympy-20916', 'sympy__sympy-21379', 'sympy__sympy-21596', 'sympy__sympy-21612', 'sympy__sympy-21847', 'sympy__sympy-21930', 'sympy__sympy-22080', 'sympy__sympy-22456', 'sympy__sympy-22714', 'sympy__sympy-22914', 'sympy__sympy-23262', 'sympy__sympy-23413', 'sympy__sympy-23534', 'sympy__sympy-23824', 'sympy__sympy-23950', 'sympy__sympy-24066', 'sympy__sympy-24213', 'sympy__sympy-24443', 'sympy__sympy-24539', 'sympy__sympy-24562', 'sympy__sympy-24661']

    # Extract pylint-dev instance IDs
    pylint_instance_ids = [instance_id for instance_id in all_instance_ids if instance_id.startswith('pylint-dev__pylint-')]
    print(f"Number of pylint-dev instances: {len(pylint_instance_ids)}")
    print("Pylint-dev instance IDs:", pylint_instance_ids)
    # load the dataset 
    swe_bench_data = load_dataset(args.dataset_id, split=args.split_name)

    create_vector_search_predictions(swe_bench_data, args.preds_path, instance_ids=pylint_instance_ids)
    avg_recalls, count = get_retrieval_eval_results(swe_bench_data, args.preds_path)
    for k, recall in avg_recalls.items():
        print(f"Average recall@{k}: {recall:.4f}")