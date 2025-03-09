import os
import pandas as pd

input_dir = "./per_score_rankings"
output_dir = "./concatenated_rankings"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for target_name in os.listdir(input_dir):
    target_dir = os.path.join(input_dir, target_name)

    if os.path.isdir(target_dir):
        # Dictionary to store dataframe for each score_type
        data_frames = {}

        # Loop through each csv file in the target directory
        for file_name in os.listdir(target_dir):
            if file_name.endswith("_ranking.csv") and file_name.startswith(target_name):
                # Extract the score_type from the file name
                score_type = file_name.replace(f"{target_name}_", "").replace(
                    "_ranking.csv", ""
                )

                # Read the csv file and rename the score_type column to its actual name
                file_path = os.path.join(target_dir, file_name)
                df = pd.read_csv(file_path)
                df.rename(columns={df.columns[1]: score_type}, inplace=True)

                # Store dataframe with model_name as the key
                data_frames[score_type] = df.set_index("model_name")

        # Combine all score_type dataframes into one dataframes
        combined_df = pd.concat(data_frames.values(), axis=1)
        combined_df.reset_index(
            inplace=True
        )  # Reset the index to include model_name as a column

        # Save the combined dataframe
        output_file = os.path.join(output_dir, f"{target_name}_combined_scores.csv")
        combined_df.to_csv(output_file, index=False)

