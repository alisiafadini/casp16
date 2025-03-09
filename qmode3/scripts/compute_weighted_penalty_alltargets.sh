#!/bin/bash

# Set the base directories
base_directory="./data_test/all_ranking_errors"
script_directory="code/casp16/qmode3/scripts"
covariances_directory="./data_test/covariances_massivefold"
valid_groups_file="./data_test/valid_group_names_withMF.txt"

# Define targets for each category
monomers_targets=('T1210' 'T1278' 'T1212' 'T1279' 'T1272s2' 'T1280' 'T1226' 'T1266'
                  'T1272s3' 'T1284' 'T1231' 'T1272s4' 'T1272s5' 'T1272s6' 'T1272s7'
                  'T1272s8' 'T1243' 'T1272s9' 'T1246' 'T1274' 'T1207' 'T1276')

homo_oligomers_targets=('T1249v2' 'T1257' 'T1218' 'T1259' 'T1292' 'T1234' 'T1270' 'T1295'
                         'T1235' 'T1298' 'T1237' 'T1240' 'T1201' 'T1206' 'T1249v1')

hetero_oligomers_targets=('H1202' 'H1230' 'H1204' 'H1232' 'H1208' 'H1233' 'H1213' 'H1236'
                           'H1215' 'H1244' 'H1217' 'H1245' 'H1220' 'H1258' 'H1222' 'H1265'
                           'H1223' 'H1267' 'H1225' 'H1227' 'H1229')

# Function to run groups_lineup_with_weightedpenalty.py for a given target list and category
run_penalty_script () {
    local output_suffix="${!#}"  # Get the last argument as the category (monomers, homo_oligomers, hetero_oligomers)
    local targets=("${@:1:$#-1}")  # Get all arguments except the last one (actual targets)
    local output_directory="weighted_penalties/${output_suffix}"  # Set correct output directory

    # Ensure the output directory exists
    mkdir -p "$output_directory"

    for target in "${targets[@]}"; do
        echo "Processing $target ($output_suffix)..."

        # Define the correct output file path
        output_file="${target}_weightedpenalty_${output_suffix}.csv"

        python3 "$script_directory/groups_lineup_with_weightedpenalty.py" \
            --error-directory "$base_directory" \
            --target "$target" \
            --valid-groups-file "$valid_groups_file" \
            --covariances-directory "$covariances_directory" \
            --target-category "$output_suffix" \
            --output-file "$output_file"

        echo "$target ($output_suffix) processing complete."
    done
}

# Run the analysis for each category
echo "Starting monomers processing..."
run_penalty_script "${monomers_targets[@]}" "monomers"
echo "Monomers processing completed."

echo "Starting homo-oligomers processing..."
run_penalty_script "${homo_oligomers_targets[@]}" "homo_oligomers"
echo "Homo-oligomers processing completed."

echo "Starting hetero-oligomers processing..."
run_penalty_script "${hetero_oligomers_targets[@]}" "hetero_oligomers"
echo "Hetero-oligomers processing completed."

# Run sum_all_targets.py script
echo "Running summary script..."
python3 "$script_directory/sum_all_targets.py"
echo "Summary script completed."
