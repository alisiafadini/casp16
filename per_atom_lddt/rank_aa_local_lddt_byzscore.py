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
    parser = argparse.ArgumentParser(description="Z-score computation for group atomic pLDDT scores across targets.")
    parser.add_argument('--base_directory', required=True, type=str,
                        help="Base directory containing CSV files.")
    parser.add_argument('--targets', required=True, type=str,
                        help="Name of file containing list of target names.")
    parser.add_argument('--top_only', action='store_true',
                        help="Use only the top prediction for each target")
    parser.add_argument('--missing_zero', action='store_true',
                        help = "Z-score of zero for missing predictions")
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug mode for additional output.")
    return parser.parse_args()

def load_csv_files(file_paths, targets, debug=False, top_only=False):
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

        # Choose whether to use only the top or up to 5 predictions.
        if top_only:
          df = df[df['prednum'] < 2] # Just use top prediction.
        else:
          df = df[df['prednum'] < 6] # Use up to 5 predictions.
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

def calculate_z_scores(target_dataframe, bigger_is_better=True, groups_to_include=None):
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
        grouped_mean = group_subset.mean(numeric_only=True)

        # for "bigger_is_better":
        #    Z-score formula: (group score - mean) / standard deviation
        # for not "bigger_is_better":
        #    Z-score formula: (mean - group score) / standard deviation
        if bigger_is_better:
          z_scores = (grouped_mean - target_means) / target_stds
        else:
          z_scores = (target_means - grouped_mean) / target_stds

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

def compute_average_z_scores_across_targets(target_dataframes, bigger_is_better=True,
                                            missing_zero=False, debug=False):
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
                                              bigger_is_better = bigger_is_better,
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
                                                   bigger_is_better = bigger_is_better,
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

def rank_groups_CC_per_atom(target_dataframes_overall, missing_zero=False, debug=False):
    """
    Rank groups based on the CC of per-atom pLDDT.

    :param target_dataframes_overall: Dictionary of DataFrames.
    :param debug: controls level of printing.
    :return: ranked combined dataframe.
    """
    # First delete unwanted scores from overall target_dataframes
    target_dataframes = copy.deepcopy(target_dataframes_overall)
    for target, df in target_dataframes.items():
        # df.drop('CC_per_atom', axis=1, inplace=True)
        df.drop('RMSD_per_atom', axis=1, inplace=True)
        df.drop('CC_per_residue', axis=1, inplace=True)
        df.drop('RMSD_per_residue', axis=1, inplace=True)
        df.dropna(inplace=True)

    # Compute and average the Z-scores across the targets
    final_avg_z_scores = compute_average_z_scores_across_targets(target_dataframes,
                                                                 missing_zero=missing_zero, debug=debug)

    # Combine the dataframes and rank based on the Z-scores
    ranked_combined_dataframe = merge_and_rank_dataframes(target_dataframes, final_avg_z_scores, debug)

    # Display output depending on the debug mode
    if debug:
        print(ranked_combined_dataframe.head(2))  # Print a sample in debug mode

    return ranked_combined_dataframe

def rank_groups_RMSD_per_atom(target_dataframes_overall, missing_zero=False, debug=False):
    """
    Rank groups based on the RMSD of per-atom pLDDT.

    :param target_dataframes_overall: Dictionary of DataFrames.
    :param debug: controls level of printing.
    :return: ranked combined dataframe.
    """
    # First delete unwanted scores from overall target_dataframes
    target_dataframes = copy.deepcopy(target_dataframes_overall)
    for target, df in target_dataframes.items():
        df.drop('CC_per_atom', axis=1, inplace=True)
        # df.drop('RMSD_per_atom', axis=1, inplace=True)
        df.drop('CC_per_residue', axis=1, inplace=True)
        df.drop('RMSD_per_residue', axis=1, inplace=True)
        df.dropna(inplace=True)

    # Compute and average the Z-scores across the targets
    final_avg_z_scores = compute_average_z_scores_across_targets(target_dataframes,
                                                                 bigger_is_better=False,
                                                                 missing_zero=missing_zero,
                                                                 debug=debug)

    # Combine the dataframes and rank based on the Z-scores
    ranked_combined_dataframe = merge_and_rank_dataframes(target_dataframes, final_avg_z_scores, debug)

    # Display output depending on the debug mode
    if debug:
        print(ranked_combined_dataframe.head(2))  # Print a sample in debug mode

    return ranked_combined_dataframe

def rank_groups_CC_per_residue(target_dataframes_overall, missing_zero=False, debug=False):
    """
    Rank groups based on the CC of per-residue pLDDT.

    :param target_dataframes_overall: Dictionary of DataFrames.
    :param debug: controls level of printing.
    :return: ranked combined dataframe.
    """
    # First delete unwanted scores from overall target_dataframes
    target_dataframes = copy.deepcopy(target_dataframes_overall)
    for target, df in target_dataframes.items():
        df.drop('CC_per_atom', axis=1, inplace=True)
        df.drop('RMSD_per_atom', axis=1, inplace=True)
        # df.drop('CC_per_residue', axis=1, inplace=True)
        df.drop('RMSD_per_residue', axis=1, inplace=True)
        df.dropna(inplace=True)

    # Compute and average the Z-scores across the targets
    final_avg_z_scores = compute_average_z_scores_across_targets(target_dataframes,
                                                                 missing_zero=missing_zero, debug=debug)

    # Combine the dataframes and rank based on the Z-scores
    ranked_combined_dataframe = merge_and_rank_dataframes(target_dataframes, final_avg_z_scores, debug)

    # Display output depending on the debug mode
    if debug:
        print(ranked_combined_dataframe.head(2))  # Print a sample in debug mode

    return ranked_combined_dataframe

def rank_groups_RMSD_per_residue(target_dataframes_overall, missing_zero=False, debug=False):
    """
    Rank groups based on the RMSD of per-residue pLDDT.

    :param target_dataframes_overall: Dictionary of DataFrames.
    :param debug: controls level of printing.
    :return: ranked combined dataframe.
    """
    # First delete unwanted scores from overall target_dataframes
    target_dataframes = copy.deepcopy(target_dataframes_overall)
    for target, df in target_dataframes.items():
        df.drop('CC_per_atom', axis=1, inplace=True)
        df.drop('RMSD_per_atom', axis=1, inplace=True)
        df.drop('CC_per_residue', axis=1, inplace=True)
        # df.drop('RMSD_per_residue', axis=1, inplace=True)
        df.dropna(inplace=True)

    # Compute and average the Z-scores across the targets
    final_avg_z_scores = compute_average_z_scores_across_targets(target_dataframes,
                                                                 bigger_is_better=False,
                                                                 missing_zero=missing_zero,
                                                                 debug=debug)

    # Combine the dataframes and rank based on the Z-scores
    ranked_combined_dataframe = merge_and_rank_dataframes(target_dataframes, final_avg_z_scores, debug)

    # Display output depending on the debug mode
    if debug:
        print(ranked_combined_dataframe.head(2))  # Print a sample in debug mode

    return ranked_combined_dataframe

def rank_groups_CC_ratio(target_dataframes_overall, missing_zero=False, debug=False):
    """
    Rank groups based on the ratio of CCs per-atom to per-residue pLDDT.

    :param target_dataframes_overall: Dictionary of DataFrames.
    :param debug: controls level of printing.
    :return: ranked combined dataframe.
    """
    # First delete unwanted scores from overall target_dataframes
    target_dataframes = copy.deepcopy(target_dataframes_overall)
    for target, df in target_dataframes.items():
        df['CC_ratio'] = df['CC_per_atom'] / df['CC_per_residue']
        # df = df[df['CC_ratio'] != 1.0]
        df.drop('CC_per_atom', axis=1, inplace=True)
        df.drop('RMSD_per_atom', axis=1, inplace=True)
        df.drop('CC_per_residue', axis=1, inplace=True)
        df.drop('RMSD_per_residue', axis=1, inplace=True)
        df.replace(to_replace=1.0, value=np.nan, inplace=True)
        df.dropna(inplace=True)

    # Compute and average the Z-scores across the targets
    final_avg_z_scores = compute_average_z_scores_across_targets(target_dataframes,
                                                                 missing_zero=missing_zero, debug=debug)

    # Combine the dataframes and rank based on the Z-scores
    ranked_combined_dataframe = merge_and_rank_dataframes(target_dataframes, final_avg_z_scores, debug)

    # Display output depending on the debug mode
    if debug:
        print(ranked_combined_dataframe.head(2))  # Print a sample in debug mode

    return ranked_combined_dataframe

def rank_groups_RMSD_ratio(target_dataframes_overall, missing_zero=False, debug=False):
    """
    Rank groups based on the ratio of per-residue RMSD to per-atom RMSD.

    :param target_dataframes_overall: Dictionary of DataFrames.
    :param debug: controls level of printing.
    :return: ranked combined dataframe.
    """
    # First delete unwanted scores from overall target_dataframes
    target_dataframes = copy.deepcopy(target_dataframes_overall)
    for target, df in target_dataframes.items():
        df['RMSD_ratio'] = df['RMSD_per_residue'] / df['RMSD_per_atom']
        # df = df[df['RMSD_ratio'] != 1.0]
        df.drop('CC_per_atom', axis=1, inplace=True)
        df.drop('RMSD_per_atom', axis=1, inplace=True)
        df.drop('CC_per_residue', axis=1, inplace=True)
        df.drop('RMSD_per_residue', axis=1, inplace=True)
        df.replace(to_replace=1.0, value=np.nan, inplace=True)
        df.dropna(inplace=True)

    # Compute and average the Z-scores across the targets
    final_avg_z_scores = compute_average_z_scores_across_targets(target_dataframes,
                                                                 missing_zero=missing_zero, debug=debug)

    # Combine the dataframes and rank based on the Z-scores
    ranked_combined_dataframe = merge_and_rank_dataframes(target_dataframes, final_avg_z_scores, debug)

    # Display output depending on the debug mode
    if debug:
        print(ranked_combined_dataframe.head(2))  # Print a sample in debug mode

    return ranked_combined_dataframe

def rank_groups_CC_delta(target_dataframes_overall, missing_zero=False, debug=False):
    """
    Rank groups based on the ratio of CCs per-atom to per-residue pLDDT.

    :param target_dataframes_overall: Dictionary of DataFrames.
    :param debug: controls level of printing.
    :return: ranked combined dataframe.
    """
    # First delete unwanted scores from overall target_dataframes
    target_dataframes = copy.deepcopy(target_dataframes_overall)
    for target, df in target_dataframes.items():
        df['CC_delta'] = df['CC_per_atom'] - df['CC_per_residue']
        # df = df[df['CC_delta'] != 1.0]
        df.drop('CC_per_atom', axis=1, inplace=True)
        df.drop('RMSD_per_atom', axis=1, inplace=True)
        df.drop('CC_per_residue', axis=1, inplace=True)
        df.drop('RMSD_per_residue', axis=1, inplace=True)
        df.replace(to_replace=0.0, value=np.nan, inplace=True)
        df.dropna(inplace=True)

    # Compute and average the Z-scores across the targets
    final_avg_z_scores = compute_average_z_scores_across_targets(target_dataframes,
                                                                 missing_zero=missing_zero, debug=debug)

    # Combine the dataframes and rank based on the Z-scores
    ranked_combined_dataframe = merge_and_rank_dataframes(target_dataframes, final_avg_z_scores, debug)

    # Display output depending on the debug mode
    if debug:
        print(ranked_combined_dataframe.head(2))  # Print a sample in debug mode

    return ranked_combined_dataframe

def rank_groups_RMSD_delta(target_dataframes_overall, missing_zero=False, debug=False):
    """
    Rank groups based on the ratio of per-residue RMSD to per-atom RMSD.

    :param target_dataframes_overall: Dictionary of DataFrames.
    :param debug: controls level of printing.
    :return: ranked combined dataframe.
    """
    # First delete unwanted scores from overall target_dataframes
    target_dataframes = copy.deepcopy(target_dataframes_overall)
    for target, df in target_dataframes.items():
        df['RMSD_delta'] = df['RMSD_per_residue'] - df['RMSD_per_atom']
        # df = df[df['RMSD_delta'] != 1.0]
        df.drop('CC_per_atom', axis=1, inplace=True)
        df.drop('RMSD_per_atom', axis=1, inplace=True)
        df.drop('CC_per_residue', axis=1, inplace=True)
        df.drop('RMSD_per_residue', axis=1, inplace=True)
        df.replace(to_replace=0.0, value=np.nan, inplace=True)
        df.dropna(inplace=True)

    # Compute and average the Z-scores across the targets
    final_avg_z_scores = compute_average_z_scores_across_targets(target_dataframes,
                                                                 missing_zero=missing_zero, debug=debug)

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
              f"{target}_scores_per_atom.csv") for target in targets])

    # Step 2: Load DataFrames from specified CSV files, keeping either 1 or up to
    # 5 predictions from each group for each target
    target_dataframes = load_csv_files(target_file_paths, targets=targets,
                                       debug=args.debug, top_only=args.top_only)

    # Step 3: Compute the desired rankings, each with its own .csv file output
    if args.top_only:
        suffix = "_top1"
    else:
        suffix = "_top5"
    if args.missing_zero:
        suffix = f"{suffix}_zero"

    ranked_CC_per_atom = rank_groups_CC_per_atom(target_dataframes,
                                                 missing_zero=args.missing_zero, debug=args.debug)
    ranked_CC_per_atom.to_csv(f"ranked_CC_per_atom{suffix}.csv")

    ranked_RMSD_per_atom = rank_groups_RMSD_per_atom(target_dataframes,
                                                 missing_zero=args.missing_zero, debug=args.debug)
    ranked_RMSD_per_atom.to_csv(f"ranked_RMSD_per_atom{suffix}.csv")

    ranked_CC_per_residue = rank_groups_CC_per_residue(target_dataframes,
                                                 missing_zero=args.missing_zero, debug=args.debug)
    ranked_CC_per_residue.to_csv(f"ranked_CC_per_residue{suffix}.csv")

    ranked_RMSD_per_residue = rank_groups_RMSD_per_residue(target_dataframes,
                                                 missing_zero=args.missing_zero, debug=args.debug)
    ranked_RMSD_per_residue.to_csv(f"ranked_RMSD_per_residue{suffix}.csv")

    ranked_CC_ratio = rank_groups_CC_ratio(target_dataframes,
                                                 missing_zero=args.missing_zero, debug=args.debug)
    ranked_CC_ratio.to_csv(f"ranked_CC_ratio{suffix}.csv")

    ranked_RMSD_ratio = rank_groups_RMSD_ratio(target_dataframes,
                                                 missing_zero=args.missing_zero, debug=args.debug)
    ranked_RMSD_ratio.to_csv(f"ranked_RMSD_ratio{suffix}.csv")

    ranked_CC_delta = rank_groups_CC_delta(target_dataframes,
                                                 missing_zero=args.missing_zero, debug=args.debug)
    ranked_CC_delta.to_csv(f"ranked_CC_delta{suffix}.csv")

    ranked_RMSD_delta = rank_groups_RMSD_delta(target_dataframes,
                                                 missing_zero=args.missing_zero, debug=args.debug)
    ranked_RMSD_delta.to_csv(f"ranked_RMSD_delta{suffix}.csv")

if __name__ == '__main__':
    main()
