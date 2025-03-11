import numpy as np
import pandas as pd
from collections import Counter
import argparse
import copy
import os

"""
Script to compute rankings based on mean values of reLLG, best model only.
Like rank_reLLG_by_zscore_best.py, but using mean values instead of Z-scores.
"""

def parse_arguments():
    """
    Parse command-line arguments for the script.

    :return: Parsed argument object.
    """
    parser = argparse.ArgumentParser(description="Mean value computation for group reLLG scores across targets.")
    parser.add_argument('--base_directory', required=True, type=str,
                        help="Base directory containing CSV files.")
    parser.add_argument('--targets', required=True, type=str,
                        help="Name of file containing list of target names.")
    parser.add_argument('--score_type',
                        choices=['as_lddt','as_lddt_coarser','bfactor_constant',
                                 'per_atom_ratio','lddt_ratio','per_atom_delta','lddt_delta'],
                        help="Desired score type for this comparison")
    parser.add_argument('--missing_zero', action='store_true',
                        help = "Score of zero for missing predictions")
    parser.add_argument('--top_only', action='store_true',
                        help = "Use first prediction rather than best")
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug mode for additional output.")
    return parser.parse_args()

def load_csv_files(file_paths, targets, score_type, top_only=False, debug=False):
    """
    Reads multiple CSV files into a dictionary of DataFrames, setting 'group' as the index.
    This ensures that each predictor group is identified uniquely in the dataframe.

    :param file_paths: List of file paths for CSV files.
    :param debug: Whether to print debug information.
    :return: Dictionary of DataFrames, where each key is the file path, and the value is the DataFrame.
    """
    target_dataframes = {}
    for target, file_path in zip(targets, file_paths):
        # Ensure the file exists before attempting to read
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Check if the DataFrame contains the necessary 'Model' column
        if 'Model' not in df.columns:
            raise ValueError(f"Missing 'Model' column in {file_path}.")

        # Parse model name into group and prediction# columns
        df['group'] = df['Model'].str.extract(r'(TS\d{3})')
        df['prednum'] = df['Model'].str.extract(r'_(\d+)').astype(int)
        if top_only:
            df = df[df['prednum'] < 2] # Choose first rather than best prediction.
        else:
            df = df[df['prednum'] < 6] # Only choose from top 5 predictions.

        # Could use status flags, but corresponding values are #N/A or 0 as seems appropriate
        df.drop('status_as_lddt', axis=1, inplace=True)
        df.drop('status_as_lddt_coarser', axis=1, inplace=True)
        df.drop('status_bfactor_constant', axis=1, inplace=True)

        # Just keep the best prediction by the chosen criterion.
        # First get down to one relevant column in the dataframe
        # For ratios and differences, just keep the ones where the compared
        # columns differ.
        if score_type == 'as_lddt':
          # df.drop('reLLG_as_lddt', axis=1, inplace=True)
            df.drop('reLLG_as_lddt_coarser', axis=1, inplace=True)
            df.drop('reLLG_bfactor_constant', axis=1, inplace=True)
            this_label = 'reLLG_as_lddt'
        elif score_type == 'as_lddt_coarser':
            df.drop('reLLG_as_lddt', axis=1, inplace=True)
            # df.drop('reLLG_as_lddt_coarser', axis=1, inplace=True)
            df.drop('reLLG_bfactor_constant', axis=1, inplace=True)
            this_label = 'reLLG_as_lddt_coarser'
        elif score_type == 'bfactor_constant':
            df.drop('reLLG_as_lddt', axis=1, inplace=True)
            df.drop('reLLG_as_lddt_coarser', axis=1, inplace=True)
            # df.drop('reLLG_bfactor_constant', axis=1, inplace=True)
            this_label = 'reLLG_bfactor_constant'
        elif score_type == 'per_atom_ratio':
            df['per_atom_ratio'] = df['reLLG_as_lddt'] / df['reLLG_as_lddt_coarser']
            df.drop('reLLG_as_lddt', axis=1, inplace=True)
            df.drop('reLLG_as_lddt_coarser', axis=1, inplace=True)
            df.drop('reLLG_bfactor_constant', axis=1, inplace=True)
            this_label = 'per_atom_ratio'
            df.loc[df[this_label] == 1.0] = np.nan
        elif score_type == 'lddt_ratio':
            df['lddt_ratio'] = df['reLLG_as_lddt'] / df['reLLG_bfactor_constant']
            df.drop('reLLG_as_lddt', axis=1, inplace=True)
            df.drop('reLLG_as_lddt_coarser', axis=1, inplace=True)
            df.drop('reLLG_bfactor_constant', axis=1, inplace=True)
            this_label = 'lddt_ratio'
            df.loc[df[this_label] == 1.0] = np.nan
        elif score_type == 'per_atom_delta':
            df['per_atom_delta'] = df['reLLG_as_lddt'] - df['reLLG_as_lddt_coarser']
            df.drop('reLLG_as_lddt', axis=1, inplace=True)
            df.drop('reLLG_as_lddt_coarser', axis=1, inplace=True)
            df.drop('reLLG_bfactor_constant', axis=1, inplace=True)
            this_label = 'per_atom_delta'
            df.loc[df[this_label] == 0.0] = np.nan
        elif score_type == 'lddt_delta':
            df['lddt_delta'] = df['reLLG_as_lddt'] - df['reLLG_bfactor_constant']
            df.drop('reLLG_as_lddt', axis=1, inplace=True)
            df.drop('reLLG_as_lddt_coarser', axis=1, inplace=True)
            df.drop('reLLG_bfactor_constant', axis=1, inplace=True)
            this_label = 'lddt_delta'
            df.loc[df[this_label] == 0.0] = np.nan

        df.dropna(inplace=True)

        if not top_only:
            # Keep only the best value for each group, marked as prednum 0.
            groups_in_df = list(set(df['group'].tolist()))
            for group in groups_in_df:
                group_subset = df.loc[df['group'] == group]
                group_max = group_subset[this_label].max(numeric_only=True)
                df.loc[(df['group'] == group) & (df[this_label] == group_max), 'prednum'] = 0
                df.drop_duplicates(['group','prednum'],inplace=True) # Deal with duplicate models

            df = df[df['prednum'] < 1] # Only keep the best prediction for each group.
        df.drop('prednum', axis=1, inplace=True)

        target_dataframes[target] = df

        if debug:
            # Print debugging information about the DataFrame shape
            print(f"Loaded {target} with shape {df.shape}")

    return target_dataframes

def filter_groups_with_few_models(target_dataframes, fraction = 0.8):

    group_counter = Counter()
    # Iterate over each target (DataFrame), counting each group only once
    for target, df in target_dataframes.items():
        df_grouped_mean = df.groupby('group').mean(numeric_only=True)
        group_counter.update(df_grouped_mean.index)
    most_found = group_counter.most_common(1)[0][1] # This is too awkward...
    target = int(fraction * most_found)
    groups_with_target_fraction = [group for group, count in group_counter.items()
                                      if count >= target]

    return groups_with_target_fraction

def compute_average_scores_across_targets(target_dataframes, missing_zero=False, debug=False):
    """
    Computes average scores for each group across multiple target dataframes.
    No filtering of outliers, but filter groups with too few models.

    :param target_dataframes: Dictionary of DataFrames (one per target).
    :param debug: Whether to print debug information.
    :return: Dictionary with average scores for each group.
    """
    groups_with_target_fraction = filter_groups_with_few_models(target_dataframes)
    if debug:
        print("Number of groups to keep after target fraction filter: ",
              len(groups_with_target_fraction))

    # Dictionary to store the final scores, initialized with empty lists for each group
    average_scores_by_group = {group: [] for group in groups_with_target_fraction}

    # Iterate over each target (DataFrame) after setting NaN values to zero
    for target, target_df in target_dataframes.items():
        target_df.replace(to_replace=np.nan, value=0.0, inplace=True)
        if debug:
            print(f"Processing target: {target} with shape {target_df.shape}")

        for group in groups_with_target_fraction:
            group_subset = target_df.loc[target_df['group'] == group]
            if len(group_subset.index) > 0:
                group_mean = group_subset.mean(numeric_only=True)
                average_scores_by_group[group].append(group_mean)

        if debug:
            print("Average Final scores by group:\n",average_scores_by_group)

    # Step 4: Compute the average scores across targets for each group
    if missing_zero:
        num_targets = len(target_dataframes.items())
        final_avg_scores = {group: np.sum(scores)/num_targets
                              for group, scores in average_scores_by_group.items() if scores}
    else:
        final_avg_scores = {group: np.mean(scores) for group, scores in average_scores_by_group.items() if scores}

    return final_avg_scores

def add_target_prefix_to_columns(dataframe, target_name):
    """
    Adds a target-specific prefix to the columns of a DataFrame.
    This helps in distinguishing the same columns across different targets.

    :param dataframe: DataFrame to modify.
    :param target_name: Target name to use as a prefix.
    :return: Modified DataFrame.
    """
    # Add prefix to all columns if they don't already have it
    new_columns = [f"{target_name}_{col}" if not col.startswith(target_name) else col for col in dataframe.columns]
    dataframe.columns = new_columns
    return dataframe

def merge_and_rank_dataframes(target_dataframes, final_avg_scores, debug=False):
    """
    Combines multiple dataframes with prefixed columns and computes the rank based on the final mean scores.

    :param target_dataframes: Dictionary of DataFrames.
    :param target_names: List of target names.
    :param final_avg_scores: Final scores to include.
    :return: Combined and ranked DataFrame.
    """
    # List to store DataFrames with prefixed columns
    modified_dataframes = []

    for target_name, target_df in target_dataframes.items():
        df_grouped_mean = target_df.groupby('group').mean(numeric_only=True)
        # Add target-specific prefix to the columns
        prefixed_df = add_target_prefix_to_columns(df_grouped_mean, target_name)
        prefixed_df.reset_index(inplace=True)  # Ensure 'group' is a column, not an index
        modified_dataframes.append(prefixed_df)

    # Combine all DataFrames into one by merging on the 'group' column
    combined_dataframe = modified_dataframes[0]
    for df in modified_dataframes[1:]:
        combined_dataframe = pd.merge(combined_dataframe, df, on='group', how='outer')

    # Map the final average scores to the combined DataFrame
    combined_dataframe['Mean_score'] = combined_dataframe['group'].map(final_avg_scores)

    # Identify the rows where 'Mean_score' is NaN (groups that will be dropped)
    groups_with_nan_z_scores = combined_dataframe[combined_dataframe['Mean_score'].isna()]['group'].tolist()

    # If there are any groups with NaN scores, issue a warning
    if debug and groups_with_nan_z_scores:
        print("Warning: The following groups are being dropped due to missing 'Mean_score' values:\n")
        print("\n".join(groups_with_nan_z_scores))

    # Now drop rows where 'Mean_score' is NaN
    combined_dataframe.dropna(subset=['Mean_score'], inplace=True)

    # Rank the groups based on their mean scores (here using dense ranking)
    combined_dataframe['Mean_Rank'] = combined_dataframe['Mean_score'].rank(method='dense', ascending=False)

    # Sort the DataFrame by the ranking
    combined_dataframe.sort_values(by='Mean_Rank', inplace=True)

    return combined_dataframe

def rank_groups(target_dataframes, missing_zero=False, debug=False):
    """
    Rank groups based on the chosen score.

    :param target_dataframes_overall: Dictionary of DataFrames.
    :param debug: controls level of printing.
    :return: ranked combined dataframe.
    """

    # Compute and average the scores
    final_avg_scores = compute_average_scores_across_targets(
                            target_dataframes, missing_zero=missing_zero, debug=debug)

    # Combine the dataframes and rank based on the scores
    ranked_combined_dataframe = merge_and_rank_dataframes(target_dataframes, final_avg_scores, debug)

    # Display output depending on the debug mode
    if debug:
        print("Top of final dataframe")
        print(ranked_combined_dataframe.head(2))  # Print a sample in debug mode

    return ranked_combined_dataframe

def main():
    """
    - Parse command-line arguments
    - Load CSV files
    - Compute scores
    - Merge and rank data
    """
    args = parse_arguments()

    # Step 1: Create file paths for all target files
    target_list_file = args.targets
    with open(target_list_file, 'r') as file:
      targets = [line.strip() for line in file]
    target_file_paths = ([os.path.join(args.base_directory,
              f"{target}_{target}_table_rellg.csv") for target in targets])
    score_type = args.score_type
    missing_zero = args.missing_zero

    # Step 2: Load DataFrames from specified CSV files, keeping best of 5 for chosen target
    target_dataframes = load_csv_files(target_file_paths, targets=targets, score_type=score_type,
                                       top_only=args.top_only, debug=args.debug)

    # Step 3: Compute the desired ranking
    if args.top_only:
        file_spec = f"mean_reLLG_{score_type}_top1"
    else:
        file_spec = f"mean_reLLG_{score_type}_best"
    if missing_zero:
      file_spec = f"{file_spec}_zero"
    ranked_reLLG = rank_groups(target_dataframes, missing_zero=missing_zero, debug=args.debug)
    ranked_reLLG.to_csv(f"{file_spec}.csv")

if __name__ == '__main__':
    main()
