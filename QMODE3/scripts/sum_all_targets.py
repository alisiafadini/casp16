import os
import pandas as pd

# Define base directory
base_directory = "weighted_penalties"

# Define category directories inside "weighted_penalties"
directories = [
    'hetero_oligomers',
    'homo_oligomers',
    'monomers'
]

for directory in directories:
    category_path = os.path.join(base_directory, directory)  # Full path to category directory

    # Initialize an empty DataFrame to hold the summed penalties
    combined_data = pd.DataFrame()

    # Check if the category directory exists
    if not os.path.exists(category_path):
        print(f"Skipping {directory} as it does not exist.")
        continue

    # Iterate through all CSV files in the category directory
    for filename in os.listdir(category_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(category_path, filename)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Ensure 'group' and 'weighted_penalty' columns exist
            if 'group' not in df.columns or 'weighted_penalty' not in df.columns:
                print(f"Skipping {file_path} due to missing required columns.")
                continue

            # If combined_data is empty, initialize it with the current df
            if combined_data.empty:
                combined_data = df
            else:
                # Merge the data, summing penalties for common groups
                combined_data = pd.concat([combined_data, df])

    # If no valid data was found, continue to the next category
    if combined_data.empty:
        print(f"No valid data found in {directory}. Skipping processing.")
        continue

    # Group by 'group' and compute the mean of the weighted penalties
    avg_data = combined_data.groupby('group')['weighted_penalty'].mean().reset_index()

    # Sort by weighted_penalty (lowest to highest)
    avg_data = avg_data.sort_values(by='weighted_penalty', ascending=True)

    # Create the output CSV filename
    output_filename = os.path.join(base_directory, f"ranked_weighted_penalties_{directory}.csv")

    # Save the final DataFrame to a CSV file
    avg_data.to_csv(output_filename, index=False)

    print(f"Processed {directory} and saved {output_filename}")

