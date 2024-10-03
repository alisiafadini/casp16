import numpy as np
import pandas as pd
from collections import Counter
import argparse
import copy
import os

def parse_arguments():
    """
    Parse command-line arguments for the script.

    :return: Parsed argument object.
    """
    parser = argparse.ArgumentParser(description="Z-score computation for group reLLG scores across targets.")
    parser.add_argument('--base_directory', required=True, type=str,
                        help="Base directory containing CSV files.")
    parser.add_argument('--targets', required=True, type=str,
                        help="Name of file containing list of target names.")
    parser.add_argument('--score_type',
                        choices=['as_lddt','as_lddt_coarser','bfactor_constant',
                                 'per_atom_ratio','lddt_ratio','per_atom_delta','lddt_delta'],
                        help="Desired score type for this comparison")
    parser.add_argument('--missing_zero', action='store_true',
                        help = "Z-score of zero for missing predictions")
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug mode for additional output.")
    return parser.parse_args()

def load_csv_files(file_paths, targets, score_type, debug=False):
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
            df['per_atom_ratio'] = df['reLLG_as_lddt'] - df['reLLG_as_lddt_coarser']
            df.drop('reLLG_as_lddt', axis=1, inplace=True)
            df.drop('reLLG_as_lddt_coarser', axis=1, inplace=True)
            df.drop('reLLG_bfactor_constant', axis=1, inplace=True)
            this_label = 'per_atom_delta'
            df.loc[df[this_label] == 0.0] = np.nan
        elif score_type == 'lddt_delta':
            df['lddt_ratio'] = df['reLLG_as_lddt'] - df['reLLG_bfactor_constant']
            df.drop('reLLG_as_lddt', axis=1, inplace=True)
            df.drop('reLLG_as_lddt_coarser', axis=1, inplace=True)
            df.drop('reLLG_bfactor_constant', axis=1, inplace=True)
            this_label = 'lddt_delta'
            df.loc[df[this_label] == 0.0] = np.nan

        df.dropna(inplace=True)
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

def calculate_z_scores(target_dataframe, groups_to_include=None):
    """
    Calculates Z-scores for each group in the DataFrame. If groups_to_include is
    provided, the function will calculate Z-scores only for those groups.

    Z-scores are calculated as (group score - mean of column) / standard deviation of column.

    :param target_dataframe: DataFrame to compute Z-scores from.
    :param groups_to_include: Optional list of groups to filter by.
    :return: Dictionary with Z-scores for each group.

    """
    if groups_to_include is not None:
        # Filter the DataFrame to include only selected groups
        target_dataframe = target_dataframe[target_dataframe['group'].isin(groups_to_include)]

    groups_in_filtered_df = list(set(target_dataframe['group'].tolist()))

    # Calculate the means and standard deviations of each column (target scores)
    # Note there may only be one numeric column if combined scores are not being calculated
    target_means = target_dataframe.mean(axis=0, numeric_only=True, skipna=True)
    target_stds = target_dataframe.std(axis=0, numeric_only=True, skipna=True)

    # Initialize a dictionary to store Z-scores
    z_scores_by_group = {}
    for group in groups_in_filtered_df:
        group_subset = target_dataframe.loc[target_dataframe['group'] == group]
        if len(group_subset.index) > 0:
            group_mean = group_subset.mean(numeric_only=True)
            # Z-score formula: (group score - mean) / standard deviation
            z_scores = (group_mean - target_means) / target_stds
            z_scores_by_group[group] = z_scores.mean()  # We take the mean Z-score across all columns for simplicity

    return z_scores_by_group

def filter_groups_by_z_score(z_scores_by_group, threshold=-2):
    """
    Filters out groups whose Z-score is below a certain threshold.
    In this case, we are filtering out groups whose Z-score is, by default,
    worse than 2 standard deviations below the mean.

    :param z_scores_by_group: Dictionary of Z-scores for each group.
    :param threshold: Threshold below which groups are filtered out.
    :return: Lists of groups to keep and ignore.
    """
    # Return groups whose Z-scores are above the threshold
    groups_above = [group for group, score in z_scores_by_group.items() if score >= threshold]
    groups_below = [group for group, score in z_scores_by_group.items() if score < threshold]
    return [groups_above, groups_below]

def compute_average_z_scores_across_targets(target_dataframes, missing_zero=False, debug=False):
    """
    Computes average Z-scores for each group across multiple target dataframes.
    It first computes initial Z-scores, filters out outliers, recalculates Z-scores, and averages them.

    :param target_dataframes: Dictionary of DataFrames (one per target).
    :param debug: Whether to print debug information.
    :return: Dictionary with average Z-scores for each group.
    """
    groups_with_target_fraction = filter_groups_with_few_models(target_dataframes)
    if debug:
        print("Number of groups to keep after target fraction filter: ",
              len(groups_with_target_fraction))

    # Dictionary to store the final Z-scores, initialized with empty lists for each group
    average_z_scores_by_group = {group: [] for group in groups_with_target_fraction}

    # Iterate over each target (DataFrame)
    for target, target_df in target_dataframes.items():
        if debug:
            print(f"Processing target: {target} with shape {target_df.shape}")

        # Step 1: Compute initial Z-scores
        initial_z_scores = calculate_z_scores(target_df,
                                              groups_to_include = groups_with_target_fraction)

        if debug:
            print("Number of groups in initial scoring: ", len(initial_z_scores))
            print(f"Initial Z-scores for {target}: {initial_z_scores}")

        # Step 2: Filter out outliers based on the threshold
        [groups_above, groups_below] = filter_groups_by_z_score(initial_z_scores)
        if debug:
            print("Number of groups to keep after z filter: ",len(groups_above))

        # Step 3: Recalculate Z-scores for remaining groups
        recalculated_z_scores = calculate_z_scores(target_df,
                                                   groups_to_include = groups_above)

        if debug:
            print(f"Recalculated Z-scores for {target}: {recalculated_z_scores}")

        # Store recalculated Z-scores for each group, with minimum of zero
        for group, score in recalculated_z_scores.items():
            this_score = max(0.,score)
            average_z_scores_by_group[group].append(this_score)
        for group in groups_below:
            average_z_scores_by_group[group].append(0.)

        if debug:
            print("Average Final Z-scores by group:\n",average_z_scores_by_group)

    # Step 4: Compute the average Z-scores across targets for each group
    if missing_zero:
        num_targets = len(target_dataframes.items())
        final_avg_z_scores = {group: np.sum(scores)/num_targets
                              for group, scores in average_z_scores_by_group.items() if scores}
    else:
        final_avg_z_scores = {group: np.mean(scores) for group, scores in average_z_scores_by_group.items() if scores}


    return final_avg_z_scores

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

def merge_and_rank_dataframes(target_dataframes, final_avg_z_scores, debug=False):
    """
    Combines multiple dataframes with prefixed columns and computes the rank based on the final mean Z-scores.

    :param target_dataframes: Dictionary of DataFrames.
    :param target_names: List of target names.
    :param final_avg_z_scores: Final Z-scores to include.
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

    # Map the final average Z-scores to the combined DataFrame
    combined_dataframe['Mean_Z'] = combined_dataframe['group'].map(final_avg_z_scores)

    # Identify the rows where 'Mean_Z' is NaN (groups that will be dropped)
    groups_with_nan_z_scores = combined_dataframe[combined_dataframe['Mean_Z'].isna()]['group'].tolist()

    # If there are any groups with NaN Z-scores, issue a warning
    if debug and groups_with_nan_z_scores:
        print("Warning: The following groups are being dropped due to missing 'Mean_Z' values:\n")
        print("\n".join(groups_with_nan_z_scores))

    # Now drop rows where 'Mean_Z' is NaN
    combined_dataframe.dropna(subset=['Mean_Z'], inplace=True)

    # Rank the groups based on their mean Z-scores (here using dense ranking)
    combined_dataframe['Mean_Z_Rank'] = combined_dataframe['Mean_Z'].rank(method='dense', ascending=False)

    # Sort the DataFrame by the ranking
    combined_dataframe.sort_values(by='Mean_Z_Rank', inplace=True)

    return combined_dataframe

def rank_groups(target_dataframes, missing_zero=False, debug=False):
    """
    Rank groups based on the chosen score.

    :param target_dataframes_overall: Dictionary of DataFrames.
    :param debug: controls level of printing.
    :return: ranked combined dataframe.
    """

    # Compute and average the Z-scores
    final_avg_z_scores = compute_average_z_scores_across_targets(
                            target_dataframes, missing_zero=missing_zero, debug=debug)

    # Reset missing values to a Z-score if zero if requested
    # if missing_zero:
      # final_avg_z_scores.to_csv("before.csv")
      # final_avg_z_scores.replace(to_replace=np.nan, value=0.0, inplace=True)
      # final_avg_z_scores.to_csv("after.csv")

    # Combine the dataframes and rank based on the Z-scores
    ranked_combined_dataframe = merge_and_rank_dataframes(target_dataframes, final_avg_z_scores, debug)

    # Display output depending on the debug mode
    if debug:
        print(ranked_combined_dataframe.head(2))  # Print a sample in debug mode

    return ranked_combined_dataframe

def main():
    """
    - Parse command-line arguments
    - Load CSV files
    - Compute Z-scores
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
                                       debug=args.debug)

    # Step 3: Compute the desired ranking
    file_spec = f"ranked_reLLG_{score_type}_best"
    if missing_zero:
      file_spec = f"{file_spec}_zero"
    ranked_reLLG = rank_groups(target_dataframes, missing_zero=missing_zero, debug=args.debug)
    ranked_reLLG.to_csv(f"{file_spec}.csv")

if __name__ == '__main__':
    main()
