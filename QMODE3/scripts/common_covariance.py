import pandas as pd
import numpy as np
import os
import argparse


"""
This script processes csv files containing OpenStructure scores for target MF models, calculates 
Mahalanobis distances to identify anomalies (such as "yarn" ball models from multimer_v1), 
and filters out such outliers based on a Z-score threshold.

1. Reads score csv files from specified list of targets and aggregates their data.
2. Computes Mahalanobis distances among scores to detect anomalies 
3. Ranks models based on their Mahalanobis distances.
4. Filters models based on a specified Z-score threshold to remove potential outliers.
5. Computes and optionally saves the covariance matrix of the filtered data.


### Arguments:
--directory: Path to the root directory containing target subdirectories with OST score csv files (eg ./data/per_score_rankings)
--target_names: List of target names to process
--z_threshold: Z-score threshold for filtering Mahalanobis distance outliers (default: 3).
--debug: Enables debug mode 
--save_cov: Path to save the computed covariance matrix.

### Example Usage:
python common_MF_covariance.py --directory ./data/per_score_rankings --target_names H1208 H1215 --z_threshold 2.5 --debug --save_cov cov_matrix.csv

"""

def debug_print(debug, *args):
    if debug:
        print(*args)

def add_prefix_to_columns(df, prefix):
    """
    Adds a prefix to the columns of a DataFrame that don't already have it.
    
    Args:
        df (pd.DataFrame): DataFrame to modify.
        prefix (str): the prefix to add to columns.
    
    Returns:
        pd.DataFrame: modified DataFrame with prefixed columns.
    """
    df.columns = [f"{prefix}_{col}" if not col.startswith(prefix) else col for col in df.columns]
    return df


def read_csv_files(directory, target_names, debug=False):
    """
    Reads score csv files from subdirectories (one per target) and returns a concatenated DataFrame.
    
    Args:
        directory (str): main directory containing subdirectories for each target name.
        target_names (list of str): list of target names to process
        debug (bool): flag for debug mode.

    Returns:
        pd.DataFrame: concatenated DataFrame with model names and target names stacked.
    """
    all_data = []

    for target_name in target_names:
        target_dir = os.path.join(directory, target_name)  # Subdirectory for the target
        if not os.path.exists(target_dir):
            if debug:
                print(f"Target directory {target_dir} does not exist, skipping.")
            continue

        data = {}
        for filename in os.listdir(target_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(target_dir, filename) 
                if debug:
                    print(f"Reading file: {filepath}")
                
                try:
                    df = pd.read_csv(filepath)
                except FileNotFoundError:
                    print(f"File not found: {filepath}")
                    continue
                
                score_name = df.columns[-1]  # Last column is the score in our files
                df.set_index('model_name', inplace=True)
                
                # Append target_name to each model name to differentiate between targets
                df.index = df.index + f"_{target_name}"
                data[score_name] = df[score_name]
                
                if debug:
                    print(f"Read file: {filename} with shape: {df.shape}")
        
        # Combine all the data for the current target
        if data:
            combined_df = pd.concat(data, axis=1)
            all_data.append(combined_df)
            if debug:
                print(f"Concatenated DataFrame for target '{target_name}' shape: {combined_df.shape}")

    # Stack all the DataFrames for each target
    if all_data:
        final_df = pd.concat(all_data)
    else:
        final_df = pd.DataFrame()  # In case no data was found
    
    if debug:
        print(f"Final stacked DataFrame shape: {final_df.shape}")

    return final_df




def calculate_mahalanobis_distance(df, debug=False):
    """
    Calculates Mahalanobis distance for each observation in df.
    
    Args:
        df (pd.DataFrame): DataFrame with observations to calculate distances for.
        debug (bool): Flag for debug mode.

    Returns:
        pd.DataFrame: df with an additional Mahalanobis distance column.
    """
    def mahalanobis(y, data, cov):
        y_mu = y - np.mean(data, axis=0)
        inv_covmat = np.linalg.pinv(cov)
        left = np.dot(y_mu, inv_covmat)
        return np.sqrt(np.dot(left, y_mu.T).diagonal())
    
    cov_matrix = np.cov(df.values.T)
    distances = mahalanobis(df, df, cov_matrix)
    
    df['Mahalanobis_distance'] = distances
    debug_print(debug, f"Calculated Mahalanobis distances with covariance matrix shape: {cov_matrix.shape}")
    
    return df


def rank_mahalanobis_distances(df, debug=False):
    """
    Ranks Mahalanobis distances in df in ascending order and resorts df based on this rank
    
    Args:
        df (pd.DataFrame): DataFrame with a Mahalanobis distance column.
        debug (bool): Flag for debug mode.

    Returns:
        pd.DataFrame: DataFrame with an additional rank column.
    """
    df['Mahalanobis_Rank'] = df['Mahalanobis_distance'].rank(ascending=True)
    df.sort_values(by='Mahalanobis_Rank', inplace=True)
    debug_print(debug, f"Ranked Mahalanobis distances. Top 5: {df.head()}")
    
    return df


def filter_by_zscore(df, z_threshold=3, debug=False):
    """
    Filters rows based on a z-score threshold of Mahalanobis distance.
    
    Args:
        df (pd.DataFrame): DataFrame with Mahalanobis distance column.
        z_threshold (float): Z-score threshold to filter outliers.
        debug (bool): Flag for debug mode.
    
    Returns:
        pd.DataFrame: Filtered DataFrame.

    Note:
        The Z-score thresholding assumes that the Mahalanobis distances are approximately normally distributed. 
        This is not quite the case, but in practice this assumption makes no difference for the goal of removing 
        contributions from the "yarn" ball models.
    """
    mean_distance = df['Mahalanobis_distance'].mean()
    std_distance = df['Mahalanobis_distance'].std()
    df_filtered = df[df['Mahalanobis_distance'] <= (mean_distance + z_threshold * std_distance)]
    
    debug_print(debug, f"Filtered data to {df_filtered.shape[0]} rows using Z-score threshold of {z_threshold}.")
    return df_filtered


def compute_covariance_matrix(df, debug=False):
    """
    Computes the covariance matrix for the original numeric (score-related)
    columns in df (excluding 'Mahalanobis_distance' and 'Mahalanobis_Rank').
    
    Args:
        df (pd.DataFrame): DataFrame from which to compute the covariance matrix.
        debug (bool): Flag for debug mode.

    Returns:
        pd.DataFrame: Covariance matrix of the numeric columns.
    """
    # Drop columns that are not part of the original feature set
    df_numeric = df.drop(columns=['Mahalanobis_distance', 'Mahalanobis_Rank'], errors='ignore')
    
    # Compute the covariance matrix
    covariance_matrix = df_numeric.cov()

    # Print the covariance matrix if in debug mode
    debug_print(debug, f"Covariance matrix:\n{covariance_matrix}\nShape: {covariance_matrix.shape}")
    
    return covariance_matrix


def main():
    """
    Main function to handle the execution of the script.
    """
    parser = argparse.ArgumentParser(description="Process CSV files to compute Mahalanobis distances.")
    parser.add_argument('--directory', type=str, required=True, help='Directory containing CSV files.')
    parser.add_argument('--target_names', nargs='+', required=True, help='List of prefixes for target files.')
    parser.add_argument('--z_threshold', type=float, default=3, help='Z-score threshold for filtering.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode.')
    parser.add_argument('--save_cov', type=str, help='Path to save the filtered covariance matrix.')

    args = parser.parse_args()

    print(f"Reading score CSV files from directory: {args.directory}.")
    combined_df = read_csv_files(os.path.join(args.directory), args.target_names, args.debug)
    
    print("Calculating Mahalanobis distances.")
    result_df = calculate_mahalanobis_distance(combined_df, args.debug)
    
    print("Ranking via Mahalanobis distances.")
    result_df = rank_mahalanobis_distances(result_df, args.debug)
    
    print(f"Filtering distances using Z-score threshold of {args.z_threshold}.")
    result_df_filtered = filter_by_zscore(result_df, args.z_threshold, args.debug)
    
    # Compute the covariance matrix (and print it if in debug mode)
    covariance_matrix = compute_covariance_matrix(result_df_filtered, args.debug)
    
    # Save the covariance matrix to a file if specified
    if args.save_cov:
        covariance_matrix.to_csv(args.save_cov)
        print(f"Covariance matrix saved to {args.save_cov} with shape {covariance_matrix.shape}")




if __name__ == "__main__":
    main()


