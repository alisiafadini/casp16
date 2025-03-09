import os
import pandas as pd
import numpy as np
import re  # Import the regular expression module

def clean_model_name(model_name):
    """
    Cleans a model name by removing unnecessary prefixes and suffixes.

    This function:
    - Removes 'ranked_*_' pattern (where * is a digit).
    - Removes 'unrelaxed_' if present.
    - Fixes redundant '.pdb.pdb' to just '.pdb'.

    Args:
        model_name (str): The raw model filename.

    Returns:
        str: The cleaned model name.
    """
    model_name = re.sub(r'ranked_\d+_', '', model_name)  # Remove 'ranked_*_'
    model_name = model_name.replace('unrelaxed_', '')  # Remove 'unrelaxed_'
    if model_name.endswith('.pdb.pdb'):
        model_name = model_name[:-4]  # Trim the extra '.pdb'
    return model_name


def parse_group_prediction(prediction_file):
    """
    Parses a group's prediction file to extract the predicted model names.

    The function looks for the section that follows "QMODE 3" and extracts exactly
    5 model names. It also ensures that the extracted names are cleaned.

    Args:
        prediction_file (str): Path to the group's prediction file.

    Returns:
        tuple:
            - list[str]: Cleaned model names.
            - list[str]: Original model names.

    Raises:
        ValueError: If the number of extracted model names is not exactly 5.
    """
    with open(prediction_file, 'r') as f:
        lines = f.readlines()

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

    # Extract model names
    model_line_combined = ' '.join(model_lines)
    predicted_models = model_line_combined.split()

    if len(predicted_models) == 5:
        return [clean_model_name(model) for model in predicted_models], predicted_models
    else:
        raise ValueError(f"Error: Expected 5 model names in {prediction_file}, but found {len(predicted_models)}.")


def process_score(score, score_name):
    """
    Handles cases where the score from OST is a list of values
    If the score is a list, it computes the average.

    Args:
        score (str): The score value as a string (may be a single value or list).
        score_name (str): The name of the score for error messages.

    Returns:
        float: The processed score.

    Raises:
        ValueError: If the score cannot be converted to a float or a valid list.
    """
    try:
        return float(score)  # Try direct conversion
    except ValueError:
        try:
            score_list = eval(score)  # Convert string representation of list to a list
            if isinstance(score_list, list) and all(isinstance(x, (int, float)) for x in score_list):
                print(f"Warning: Averaging list of scores for {score_name}")
                return np.mean(score_list)
            else:
                raise ValueError
        except (SyntaxError, ValueError):
            raise ValueError(f"Could not process score: {score}")


def compute_ranking_error(true_rankings, predicted_models, score_name, group_name, og_model_names):
    """
    Computes the ranking error for the predicted rankings by comparing them with true rankings.

    The ranking error is computed as the squared difference between the scores for the predicted 
    top 5 and the true top 5 models.

    Args:
        true_rankings (list[tuple]): List of (model_name, score) tuples.
        predicted_models (list[str]): List of predicted model names.
        score_name (str): Name of the score being evaluated.
        group_name (str): Name of the group whose predictions are being evaluated.
        og_model_names (list[str]): Original model names before cleaning.

    Returns:
        float: The computed ranking error
    """
    ranking_error = 0.0
    needs_sorting = 'rmsd' in score_name.lower()

    if needs_sorting:
        #if RMSD is being evaluated, the rankings should be flipped in order (lower is better)
        true_rankings = sorted(true_rankings, key=lambda x: float(x[1]) if isinstance(x[1], (int, float)) else process_score(x[1], score_name))

    for i in range(min(5, len(predicted_models))):
        predicted_model = predicted_models[i]
        true_score = process_score(true_rankings[i][1], score_name)

        predicted_score = next(
            (process_score(score, score_name) for model, score in true_rankings if clean_model_name(model) == predicted_model),
            None
        )

        if predicted_score is not None:
            ranking_error += (true_score - predicted_score) ** 2
        else:
            print(f"Warning: Predicted model {predicted_model} from group {group_name} not found in true rankings.")
            print(f"Original model name was {og_model_names[i]}")
            ranking_error += true_score ** 2  # Max ranking_error if the model is not in the rankings

    return ranking_error


def process_group_predictions(target_id, group_predictions_dir, true_rankings_dir, output_file):
    """
    Processes all group predictions for a specific target and computes ranking error.
    """
    true_rankings = {}

    for score_file in os.listdir(true_rankings_dir):
        if score_file.startswith(target_id):
            score_name = '_'.join(score_file.split('_')[-3:-1])
            score_path = os.path.join(true_rankings_dir, score_file)
            df = pd.read_csv(score_path)
            true_rankings[score_name] = df.values.tolist()

    ranking_errors = []
    target_prediction_dir = os.path.join(group_predictions_dir, target_id)

    if not os.path.exists(target_prediction_dir):
        print(f"Error: Directory {target_prediction_dir} does not exist.")
        return

    group_files = [f for f in os.listdir(target_prediction_dir) if f.startswith(target_id)]

    if not group_files:
        print(f"Error: No predicted model files found for target {target_id} in {target_prediction_dir}")
        return

    for group_file in group_files:
        group_number = group_file[len(target_id):].replace('.txt', '')
        predicted_models, og_model_names = parse_group_prediction(os.path.join(target_prediction_dir, group_file))

        group_ranking_errors = {'group': group_number}
        for score_name, rankings in true_rankings.items():
            ranking_error = compute_ranking_error(rankings, predicted_models, score_name, group_number, og_model_names)
            group_ranking_errors[score_name] = ranking_error

        ranking_errors.append(group_ranking_errors)
        
    df_ranking_errors = pd.DataFrame(ranking_errors)
    df_ranking_errors.to_csv(output_file, index=False)


def compute_per_target_ranking_error(targets_file, group_predictions_dir, true_rankings_base_dir, output_base_dir):
    """
    Processes all targets listed in the targets file and computes ranking errors.

    Args:
        targets_file (str): Path to a file containing a list of target IDs.
        group_predictions_dir (str): Directory containing all group predictions.
        true_rankings_base_dir (str): Directory containing per-score rankings.
        output_base_dir (str): Directory where ranking_error outputs should be stored.
    """
    with open(targets_file, 'r') as f:
        target_ids = [line.strip() for line in f]

    for target_id in target_ids:
        print(f"Processing target {target_id}...")
        true_rankings_dir = os.path.join(true_rankings_base_dir, target_id)
        output_file = os.path.join(output_base_dir, f"{target_id}_group_ranking_errors.csv")

        process_group_predictions(target_id, group_predictions_dir, true_rankings_dir, output_file)


if __name__ == "__main__":
    targets_file = "./data/all_targets.txt"
    group_predictions_dir = "./data/all_group_predictions/"
    true_rankings_base_dir = "./data/per_score_rankings/"
    output_base_dir = "./data/all_ranking_errors/"

    os.makedirs(output_base_dir, exist_ok=True) 
    compute_per_target_ranking_error(targets_file, group_predictions_dir, true_rankings_base_dir, output_base_dir)
