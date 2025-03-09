import os
import tarfile
import json
import pandas as pd
from collections import defaultdict
import shutil

def untar_directory(tar_path, extract_path):
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"Tar file '{tar_path}' not found.")
    
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_path)

def parse_ost_json_file(json_file, SCORES_OF_INTEREST):
    """
    Parses the OST json file and extracts specified scores.

    Args:
        json_file (str): Path to the json file.
        SCORES_OF_INTEREST (list): List of keys representing scores to extract.

    Returns:
        dict: A dictionary mapping score names to their values.
              If a score is not found in the json file, it is set to None.

    Raises:
        FileNotFoundError: If the json file does not exist.
        json.JSONDecodeError: If the file is not a valid json.
    """
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"json file '{json_file}' not found.")
    
    with open(json_file, 'r') as f:
        data = json.load(f)

    scores = {key: data.get(key, None) for key in SCORES_OF_INTEREST}
    return scores

def parse_and_rank_models(target_dir, SCORES_OF_INTEREST):
    """
    Processes all json files in the target directory and ranks models based on each score.

    Args:
        target_dir (str): Path to the directory containing json score files.
        SCORES_OF_INTEREST (list): List of scores to extract and rank.

    Returns:
        dict: A dictionary where each key is a score type and the value is a sorted list of 
              tuples (model_name, score) in descending order.

    Notes:
        - If a score key is missing from a json file, a warning is printed.
        - If all scores for a particular key are None, another warning is printed.
        - Assumes json filenames follow the pattern 'scores_<model_name>.json'.
    
    """
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Target directory '{target_dir}' not found.")
    
    scores_dict = defaultdict(list)
    missing_key_warnings = set()

    json_files = [f for f in os.listdir(target_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"Warning: No json files found in '{target_dir}'.")
    
    for json_file in json_files:
        model_name = json_file.replace('scores_', '').replace('.json', '')
        json_path = os.path.join(target_dir, json_file)
        scores = parse_ost_json_file(json_path, SCORES_OF_INTEREST)
        
        for key, value in scores.items():
            if value is not None:
                scores_dict[key].append((model_name, value))
            else:
                if key not in missing_key_warnings:
                    missing_key_warnings.add(key)
                    print(f"Warning: Found None for key '{key}' in one or more models in '{target_dir}'")

    # Ensure no None values are present before sorting
    ranked_scores = {}
    for key, models in scores_dict.items():
        valid_models = [model for model in models if model[1] is not None]
        if valid_models:
            ranked_scores[key] = sorted(valid_models, key=lambda x: x[1], reverse=True)
        else:
            print(f"Warning: No valid scores to rank for key '{key}' in '{target_dir}'")

    return ranked_scores

def store_rankings(target_id, ranked_scores, output_dir):
    """
    Stores model rankings in CSV files.

    Args:
        target_id (str): Target ID for which rankings are stored.
        ranked_scores (dict): Dictionary where each key is a score type and the value is 
                              a list of tuples (model_name, score).
        output_dir (str): Directory where ranking CSV files will be saved.
    """
    target_output_dir = os.path.join(output_dir, target_id)
    os.makedirs(target_output_dir, exist_ok=True)

    for key, models in ranked_scores.items():
        output_file = os.path.join(target_output_dir, f"{target_id}_{key}_ranking.csv")
        df = pd.DataFrame(models, columns=["model_name", key])
        df.to_csv(output_file, index=False)

def clean_up_directory(target_dir):
    """
    Deletes a directory and all its contents to free up space.

    Args:
        target_dir (str): Path to the directory to be deleted.

    Returns:
        None

    Raises:
        FileNotFoundError: If the directory does not exist.
        OSError: If deletion fails.
    """
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    else:
        raise FileNotFoundError(f"Directory '{target_dir}' not found.")

def extract_scores_for_all_targets(targets_file, base_dir, output_dir, SCORES_OF_INTEREST):
    """
    Extracts score data for target list, ranks models, and cleans up.

    Args:
        targets_file (str): Path to a file containing a list of target IDs.
        base_dir (str): Directory containing score files from OST (.tgz).
        output_dir (str): Directory where ranked scores should be stored.
        SCORES_OF_INTEREST (list): List of score keys to extract and rank.

    Notes:
        - Each target must have a corresponding .tgz file in `base_dir`.
        - If a tar file for a target is missing, the function skips it with a warning.
    """
    if not os.path.exists(targets_file):
        raise FileNotFoundError(f"Targets file '{targets_file}' not found.")

    with open(targets_file, 'r') as f:
        target_ids = [line.strip() for line in f]

    for target_id in target_ids:
        tar_path = os.path.join(base_dir, f"{target_id}_scores.tgz")
        target_dir = os.path.join(base_dir, f"{target_id}_scores")
        
        if not os.path.exists(tar_path):
            print(f"Warning: Tar file for target '{target_id}' not found. Skipping...")
            continue
        
        print(f"Processing target '{target_id}'...")

        # Step 1: Extract files
        untar_directory(tar_path, target_dir)
        
        # Step 2: Parse and rank models
        ranked_scores = parse_and_rank_models(target_dir, SCORES_OF_INTEREST)
        
        # Step 3: Store rankings
        store_rankings(target_id, ranked_scores, output_dir)
        
        # Step 4: Clean up extracted directory
        clean_up_directory(target_dir)
        
        print(f"Finished processing target '{target_id}'.")

if __name__ == "__main__":

    targets_file = "./data/all_targets.txt"
    base_dir = "./ost_scores/"  
    output_dir = "./data/per_score_rankings/"

    SCORES_OF_INTEREST = [
        'lddt', 'ilddt', 'tm_score', 'ics', 'ips', 'qs_global', 
        'qs_best', 'dockq_ave', 'dockq_wave', 'oligo_gdtts', 
        'oligo_gdtha', 'rmsd'
    ] # Includes oligomer scores, for monomers, if a score is not found, the corresponding key will simply not be added

    extract_scores_for_all_targets(targets_file, base_dir, output_dir, SCORES_OF_INTEREST)
