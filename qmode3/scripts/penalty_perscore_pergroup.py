import os
import pandas as pd
import numpy as np
import re  # Import the regular expression module

def clean_model_name(model_name):
    """Remove 'ranked_*_', 'unrelaxed_', and extra '.pdb' from the model name if they exist."""
    # Remove 'ranked_*_' pattern
    model_name = re.sub(r'ranked_\d+_', '', model_name)
    # Remove 'unrelaxed_' if present anywhere in the string
    model_name = model_name.replace('unrelaxed_', '')
    # Remove duplicate '.pdb' if present
    if model_name.endswith('.pdb.pdb'):
        model_name = model_name[:-4]  # Trim the extra '.pdb'
    return model_name


def parse_group_prediction(prediction_file):
    """Parse the group's prediction file and extract the model names."""
    with open(prediction_file, 'r') as f:
        lines = f.readlines()

    # Look for the line after "QMODE 3" that contains the model names
    model_lines = []
    capture_models = False
    for i, line in enumerate(lines):
        if re.match(r"^QMODE\s+3", line):
            capture_models = True
            continue

        if capture_models:
            stripped_line = line.strip()
            if stripped_line == "END":
                break
            model_lines.append(stripped_line)

    # Combine all lines of models into a single list
    model_line_combined = ' '.join(model_lines)
    predicted_models = model_line_combined.split()

    # Ensure there are exactly 5 model names
    if len(predicted_models) == 5:
        return [clean_model_name(model) for model in predicted_models]
    else:
        raise ValueError(f"Error: Expected 5 model names in {prediction_file}, but found {len(predicted_models)}.")

def process_score(score, score_name):
    """Convert the score to a float, averaging if necessary."""
    try:
        # Try to convert directly to float (single number case)
        return float(score)
    except ValueError:
        # If conversion fails, assume it's a list and average the numbers
        try:
            score_list = eval(score)  # Convert string representation of list to a list
            if isinstance(score_list, list) and all(isinstance(x, (int, float)) for x in score_list):
                avg_score = np.mean(score_list)
                print(f"Warning: Averaging list of scores for {score_name}")
                return avg_score
            else:
                raise ValueError
        except (SyntaxError, ValueError):
            raise ValueError(f"Could not process score: {score}")

def compute_penalty(true_rankings, predicted_models, score_name):
    """Compute the penalty for the predicted rankings."""
    penalty = 0.0
    for i in range(min(5, len(predicted_models))):
        predicted_model = clean_model_name(predicted_models[i])  # Clean model name
        true_score = process_score(true_rankings[i][1], score_name)  # Process the true score
        predicted_score = next(
            (process_score(score, score_name) 
             for model, score in true_rankings 
             if clean_model_name(model) == predicted_model), 
            None
        )
        
        if predicted_score is not None:
            penalty += (true_score - predicted_score) ** 2
        else:
            print(f"Warning: Predicted model {predicted_model} not found in true rankings.")
            penalty += true_score ** 2  # Max penalty if predicted model is not in the true rankings

    return penalty

def process_group_predictions(target_id, group_predictions_dir, true_rankings_dir, output_file):
    """Process all group predictions and compute penalties for a specific target."""
    # Load true rankings for each score
    true_rankings = {}
    for score_file in os.listdir(true_rankings_dir):
        if score_file.startswith(target_id):
            # Extract the score name more robustly
            score_name = '_'.join(score_file.split('_')[-3:-1])
            score_path = os.path.join(true_rankings_dir, score_file)
            #print(f"Processing score file: {score_path}")
            df = pd.read_csv(score_path)
            true_rankings[score_name] = df.values.tolist()

    # Prepare to collect penalties
    penalties = []
    
    # Define the target-specific directory
    target_prediction_dir = os.path.join(group_predictions_dir, target_id)
    
    if not os.path.exists(target_prediction_dir):
        print(f"Error: Directory {target_prediction_dir} does not exist.")
        return
    
    # List all group prediction files in the target-specific directory
    group_files = [f for f in os.listdir(target_prediction_dir) if f.startswith(target_id)]
    
    if not group_files:
        print(f"Error: No predicted model files found for target {target_id} in {target_prediction_dir}")
        return

    for group_file in group_files:
        group_number = group_file[len(target_id):].replace('.txt', '')  # Extract the group name
        predicted_models = parse_group_prediction(os.path.join(target_prediction_dir, group_file))
        #print(f"Predicted models for group {group_number}: {predicted_models}")
        
        group_penalties = {'group': group_number}
        for score_name, rankings in true_rankings.items():
            penalty = compute_penalty(rankings, predicted_models, score_name)
            group_penalties[score_name] = penalty
        
        penalties.append(group_penalties)

    # Save penalties to CSV
    df_penalties = pd.DataFrame(penalties)
    df_penalties.to_csv(output_file, index=False)
    #print(f"Saved penalties to {output_file}")

def process_all_targets(targets_file, group_predictions_dir, true_rankings_base_dir, output_base_dir):
    """Process all targets listed in the targets file."""
    with open(targets_file, 'r') as f:
        target_ids = [line.strip() for line in f]

    for target_id in target_ids:
        print(f"Processing target {target_id}...")
        true_rankings_dir = os.path.join(true_rankings_base_dir, target_id)
        output_file = os.path.join(output_base_dir, f"{target_id}_group_penalties.csv")
        
        process_group_predictions(target_id, group_predictions_dir, true_rankings_dir, output_file)

if __name__ == "__main__":
    targets_file = "all_targets.txt"
    group_predictions_dir = "./group_predictions/"
    true_rankings_base_dir = "./per_score_rankings/"
    output_base_dir = "./penalty_outputs/"

    os.makedirs(output_base_dir, exist_ok=True)  # Ensure the output directory exists
    process_all_targets(targets_file, group_predictions_dir, true_rankings_base_dir, output_base_dir)
