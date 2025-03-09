import numpy as np
import pandas as pd
import os
import argparse

def parse_arguments():
    """
    This script calculates weighted penalties for different predictor groups based on 
    ranking errors and the score covariance matrix for a set of MassiveFold models. 
    The results can be optionally saved.
    """
    parser = argparse.ArgumentParser(
        description="Compute weighted penalties for groups using covariance-based weighting of ranking errors."
    )
    parser.add_argument('--error-directory', required=True, type=str, 
                        help="Directory containing CSV files with group ranking error data.")
    parser.add_argument('--target', required=True, nargs='+',
                        help="Target to be analyzed")
    parser.add_argument('--covariances-directory', required=True, type=str, 
                        help="Directory containing score covariance matrix CSV files.")
    parser.add_argument('--target_category', required=True, type=str, 
                        help="monomer, homo_oligomer, or hetero_oligomer")
    parser.add_argument('--valid-groups-file', required=True, type=str, 
                        help="File containing valid group names.")
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug mode for additional output.")
    parser.add_argument('--output-file', type=str,
                        help="Optional output file path to save computed weighted penalties.")
    return parser.parse_args()

def normalize_penalties(penalties):
    """
    Normalizes an array of penalties to a scale between 0 and 1.

    :param penalties: NumPy array of penalty values.
    :return: Normalized penalty values.
    """
    min_penalty, max_penalty = np.min(penalties), np.max(penalties)
    return (penalties - min_penalty) / (max_penalty - min_penalty)

def load_valid_group_names(file_path):
    """
    Loads valid group names from a specified text file.

    :param file_path: Path to the text file containing valid group names.
    :return: A set of valid group names.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    
    with open(file_path, 'r') as file:
        return {line.strip() for line in file}

def add_prefix_to_columns(df, target_name):
    """
    Ensures all columns in a DataFrame have a target-specific prefix.
    
    :param df: DataFrame containing group penalty scores.
    :param target_name: Prefix to add to column names if missing.
    :return: Modified DataFrame with consistent column names.
    """
    df.columns = [f"{target_name}_{col}" if not col.startswith(target_name) else col for col in df.columns]
    return df

def load_csv_files(file_paths, valid_groups, debug=False):
    """
    Reads multiple CSV files into a dictionary of DataFrames, filtering out invalid groups.

    :param file_paths: List of CSV file paths containing penalty scores.
    :param valid_groups: Set of valid group names.
    :param debug: Flag to enable debug mode.
    :return: Dictionary mapping file paths to DataFrames.
    """
    dataframes = {}
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")
        
        df = pd.read_csv(file_path)
        if 'group' not in df.columns:
            raise ValueError(f"Missing 'group' column in {file_path}.")
        
        df = df[df['group'].isin(valid_groups)]
        df.set_index('group', inplace=True)
        dataframes[file_path] = df
        
        if debug:
            print(f"Loaded {file_path} with shape {df.shape}")
    
    return dataframes

def load_covariance_matrices(directory, target_category, debug=False):
    """
    Reads covariance matrices from CSV files into a dictionary.

    :param file_paths: List of file paths for covariance matrices.
    :param target_names: Corresponding target names for the covariance files.
    :param debug: Flag to enable debug mode.
    :return: Dictionary mapping target names to NumPy covariance matrices.
    """
        
    df = pd.read_csv(f"{directory}/common_covariances_{target_category}.csv", index_col=0)

    df.drop(columns=[col for col in ["rmsd", "ips", "ics"] if col in df.columns], errors='ignore', inplace=True)
    df.drop(index=[row for row in ["rmsd", "ips", "ics"] if row in df.index], errors='ignore', inplace=True)
        
    covariance_matrix = df.values.astype(float)
        
    if debug:
        print(f"Loaded covariance matrix with shape {df.shape}")
    
    return covariance_matrix

def compute_weighted_penalties(dataframes_dict, cov_matrix, debug=False):
    """
    Computes weighted penalties for each group using covariance-weighted distance.

    :param dataframes_dict: Dictionary of DataFrames (penalty scores per target).
    :param covariances_dict: Dictionary of covariance matrices (per target).
    :param debug: Flag to enable debug mode.
    :return: Dictionary of computed penalties per group.
    """
    penalties_per_group = {}
    
    def calculate_weighted_distance(y, cov):
        try:
            inv_covmat = np.linalg.pinv(cov)  # Use pseudo-inverse in case of singular matrix
        except np.linalg.LinAlgError:
            print("Covariance matrix is not invertible")
            return None

        left_term = np.dot(y, inv_covmat)
        randy_weight = np.dot(left_term, y.T)
        return np.sqrt(randy_weight.diagonal())
    
    for file_path, df in dataframes_dict.items():
        target_name = os.path.basename(file_path).split('_')[0]
        df = add_prefix_to_columns(df, target_name)
        columns_to_drop = [f"{target_name}_rmsd", f"{target_name}_ips", f"{target_name}_ics"]
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

        distances = calculate_weighted_distance(df.values, cov_matrix)
        if distances is not None:
            distances = normalize_penalties(distances)
            for group, penalty in zip(df.index, distances):
                penalties_per_group[group] = penalties_per_group.get(group, 0) + penalty
    
    return penalties_per_group


def main():
    args = parse_arguments()
    
    # Define the base output directory as "weighted_penalties/"
    base_output_dir = "weighted_penalties"
    
    # Create category-specific subdirectory inside "weighted_penalties/"
    category_dir = os.path.join(base_output_dir, args.target_category)
    os.makedirs(category_dir, exist_ok=True)  # Creates the directory if it doesn't exist
    
    valid_groups = load_valid_group_names(args.valid_groups_file)
    ranking_error_file_paths = [os.path.join(args.error_directory, f"{t}_group_ranking_errors.csv") for t in args.target]
    group_dataframes = load_csv_files(ranking_error_file_paths, valid_groups, debug=args.debug)
    covariance_matrices = load_covariance_matrices(args.covariances_directory, args.target_category, debug=args.debug)
    weighted_penalties = compute_weighted_penalties(group_dataframes, covariance_matrices, debug=args.debug)
    
    if args.output_file:
        output_path = os.path.join(category_dir, args.output_file)  # Save inside "weighted_penalties/category/"
        pd.DataFrame(list(weighted_penalties.items()), columns=['group', 'weighted_penalty']).to_csv(output_path, index=False)
        print(f"Weighted penalties saved to {output_path}")

if __name__ == '__main__':
    main()

