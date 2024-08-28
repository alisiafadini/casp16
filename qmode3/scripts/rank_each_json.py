import os
import tarfile
import json
import pandas as pd
from collections import defaultdict
import shutil

def untar_directory(tar_path, extract_path):
    """Untar the tgz file to the scores directory."""
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_path)

def parse_json_file(json_file, SCORES_OF_INTEREST):
    """Parse the scores JSON file and return the scores we are interested in."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    scores = {}
    for key in SCORES_OF_INTEREST:
        scores[key] = data.get(key, None)
    return scores

def process_target_directory(target_dir, SCORES_OF_INTEREST):
    """Process all JSON files in the target directory and rank models based on each score."""
    scores_dict = defaultdict(list)
    missing_key_warnings = set()
    
    json_files = [f for f in os.listdir(target_dir + "/" + target_dir) if f.endswith('.json')]
    for json_file in json_files:
        model_name = json_file.replace('scores_', '').replace('.json', '')
        json_path = os.path.join(target_dir, target_dir, json_file)
        scores = parse_json_file(json_path, SCORES_OF_INTEREST)
        
        for key, value in scores.items():
            if value is not None:
                scores_dict[key].append((model_name, value))
            else:
                if key not in missing_key_warnings:
                    missing_key_warnings.add(key)
                    print(f"Warning: Found None for key '{key}' in one or more models in target directory '{target_dir}'")
    
    # Ensure no None values are present before sorting
    ranked_scores = {}
    for key, models in scores_dict.items():
        # Filter out any None values just in case
        valid_models = [model for model in models if model[1] is not None]
        if valid_models:
            ranked_scores[key] = sorted(valid_models, key=lambda x: x[1], reverse=True)
        else:
            print(f"Warning: No valid scores to rank for key '{key}' in target directory '{target_dir}'")

    return ranked_scores

def store_rankings(target_id, ranked_scores, output_dir):
    """Store the rankings in a CSV file."""
    os.makedirs(os.path.join(output_dir, target_id), exist_ok=True)
    for key, models in ranked_scores.items():
        output_file = os.path.join(output_dir, target_id, f"{target_id}_{key}_ranking.csv")
        df = pd.DataFrame(models, columns=["model_name", key])
        df.to_csv(output_file, index=False)

def clean_up_directory(target_dir):
    """Remove the extracted directory to save space."""
    shutil.rmtree(target_dir)

def process_all_targets(targets_file, base_dir, output_dir, SCORES_OF_INTEREST):
    """Main function to process all targets listed in the targets list."""
    with open(targets_file, 'r') as f:
        target_ids = [line.strip() for line in f]

    for target_id in target_ids:
        tar_path = os.path.join(base_dir, f"{target_id}_scores.tgz")
        target_dir = os.path.join(base_dir, f"{target_id}_scores")
        
        if not os.path.exists(tar_path):
            print(f"Tar file for target {target_id} not found. Skipping...")
            continue
        
        print(f"Processing target {target_id}...")
        
        # Step 1: Untar the directory
        untar_directory(tar_path, target_dir)
        
        # Step 2: Parse JSON files and rank models
        ranked_scores = process_target_directory(target_dir, SCORES_OF_INTEREST)
        
        # Step 3: Store rankings
        store_rankings(target_id, ranked_scores, output_dir)
        
        # Step 4: Clean up extracted directory
        clean_up_directory(target_dir)
        
        print(f"Finished processing target {target_id}")

if __name__ == "__main__":
    targets_file = "all_targets.txt"
    base_dir = "./"  
    output_dir = "./per_score_rankings/"
    
    SCORES_OF_INTEREST = ['lddt', 'ilddt', 'local_lddt', 'tm_score', 'ics', 'ips', 'qs_global', 'qs_best', 'dockq_ave', 'dockq_wave', 'oligo_gdtts', 'oligo_gdtha', 'rmsd']
    
    process_all_targets(targets_file, base_dir, output_dir, SCORES_OF_INTEREST)
