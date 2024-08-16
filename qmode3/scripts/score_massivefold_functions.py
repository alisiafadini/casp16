import os
import subprocess
from multiprocessing import Pool
import shutil
import glob
from tqdm import tqdm


"""
Script to compare the structural similarity between MassiveFold monomer models and reference CASP16 structures using Open Structure (ost-2.8.0) within a Docker container.

Executes a Docker command to compare the model and reference structures, generating a JSON file with the comparison scores.

Usage:
    Replace hard-coded:
            * num_proc = 16
            * target_id_list = "massivefold_target_ids_oligomers.txt"
            * target_folder = "references_oligo"
    at bottom of file

"""
def modify_reference_chain(model_pdb, reference_pdb):

    """
    Modifies the chain ID in a reference PDB file to match the chain ID found in a model PDB file.

    The function reads the chain ID from the first ATOM or HETATM line in the model PDB file
    and uses this chain ID to modify all ATOM and HETATM lines in the reference PDB file.
    The modified reference PDB file is saved with a new name prefixed by 'modified_'.

    Args:
        model_pdb (str): Path to the model PDB file.
        reference_pdb (str): Path to the reference PDB file.

    Returns:
        str: Path to the modified reference PDB file.

    Raises:
        ValueError: If the model PDB file does not contain any ATOM or HETATM lines with a chain ID.
    """
        
    # Extract the chain ID from the model PDB
    model_chain_id = None
    with open(model_pdb, 'r') as model_file:
        for line in model_file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                model_chain_id = line[21]
                break
    
    if model_chain_id is None:
        raise ValueError(f"Could not find a chain ID in model PDB: {model_pdb}")
    
    # Modify the reference PDB to match the chain ID
    modified_reference_pdb = f"modified_{os.path.basename(reference_pdb)}"
    with open(reference_pdb, 'r') as ref_file:
        ref_lines = ref_file.readlines()

    with open(modified_reference_pdb, 'w') as ref_file_modified:
        for line in ref_lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                line = line[:21] + model_chain_id + line[22:]
            ref_file_modified.write(line)
    
    return modified_reference_pdb

# Function to run the ost command
def compare_structures_oligomer(args):

    """
    Compares the structure of a model PDB file against a reference PDB file using the Open Structure Tool (ost-2.8.0).

    The function first modifies the chain ID of the reference PDB file to match the model PDB file by
    calling `modify_reference_chain()`. It then runs a Docker command to execute the `ost-2.8.0 compare-structures`
    tool, comparing the model against the modified reference. The comparison scores are saved as a JSON file in the specified output directory.

    Args:
        args (tuple): A tuple containing:
            - model (str): Path to the model PDB file.
            - reference (str): Path to the reference PDB file.
            - output_dir (str): Directory where the output JSON file should be saved.

    """

    model, reference, output_dir = args
    command = (
        f"docker run --rm -v $(pwd):/home ost-2.8.0 compare-structures "
        f"--model {model} "
        f"--reference {reference} "
        f"--output {output_dir}/scores_{os.path.basename(model)}.json "
        f"--lddt "
        f"--illdt "
        f"--tm-score "
        f"--rigid-scores "
        f"--ics "
        f"--ips "
        f"--qs-score "
        f"--dockq "
        f"--patch-scores "
    )
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Completed comparison for {model}")
    except subprocess.CalledProcessError as e:
        print(f"Error comparing {model}: {e}")

def compare_structures_monomer(args):

    """
    Compares the structure of a model PDB file against a reference PDB file using the Open Structure Tool (ost-2.8.0).

    The function first modifies the chain ID of the reference PDB file to match the model PDB file by
    calling `modify_reference_chain()`. It then runs a Docker command to execute the `ost-2.8.0 compare-structures`
    tool, comparing the model against the modified reference. The comparison scores are saved as a JSON file in the specified output directory.

    Args:
        args (tuple): A tuple containing:
            - model (str): Path to the model PDB file.
            - reference (str): Path to the reference PDB file.
            - output_dir (str): Directory where the output JSON file should be saved.

    """

    model, reference, output_dir = args
    reference_modified = modify_reference_chain(model, reference)
    command = (
        f"docker run --rm -v $(pwd):/home ost-2.8.0 compare-structures "
        f"--model {model} "
        f"--reference {reference_modified} "
        f"--output {output_dir}/scores_{os.path.basename(model)}.json "
        f"--lddt "
        f"--local-lddt "
        f"--tm-score "
        f"--rigid-scores "
    )
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Completed comparison for {model}")
    except subprocess.CalledProcessError as e:
        print(f"Error comparing {model}: {e}")

# Main script
def process_target(target_id_og, ref_file, num_proc, scoring_type):
    
    target_id = target_id_og.split('_')[0].strip()
    
    # Step 2: Download the corresponding massivefold target dataset
    download_url = f"https://casp-capri.sinbios.plbs.fr/index.php/s/TTqScLKZM5W6ZFi/download?path=%2Fgathered_runs&files={target_id_og.strip()}_MassiveFold_all_pdbs.tar.gz"
    output_filename = f"{target_id}_MassiveFold_all_pdbs.tar.gz"
    subprocess.run(['wget', download_url, '-O', output_filename])

    # Extract the tar.gz file
    subprocess.run(['tar', '-xzf', output_filename])

    extracted_dirs = [d for d in os.listdir() if os.path.isdir(d) and d.startswith(target_id)]
    if not extracted_dirs:
        print(f"No directory found matching {target_id} after extraction.")
        return

    downloaded_dir = extracted_dirs[0]
    print(f"Using extracted directory: {downloaded_dir}")

    # Step 3: Create a target_id_scores directory
    scores_dir = f"{target_id}_scores"
    os.makedirs(scores_dir, exist_ok=True)
    print(scores_dir)

    # Confirm that the scores directory was created
    if not os.path.exists(scores_dir):
        print(f"Failed to create directory: {scores_dir}")
        return
    else:
        print(f"Scores directory {scores_dir} successfully created.")

    # Step 6: Run compare_structures using multiprocessing
    # Get a list of all PDB files in the directory
    pdb_files = [
        os.path.join(downloaded_dir, f)
        for f in os.listdir(downloaded_dir)
        if f.endswith(".pdb")
    ]

    if not pdb_files:
        print(f"No PDB files found in directory {downloaded_dir}.")
        return
    
    # Create a list of tuples for the pool map
    tasks = [(model, ref_file, scores_dir) for model in pdb_files]

    # Create a pool of workers
    with Pool(num_proc) as pool:  # Use 16 processors
        if scoring_type == "monomer":
            pool.map(compare_structures_monomer, tasks)
        elif scoring_type == "oligomer":
            pool.map(compare_structures_oligomer, tasks)

    print(f"All comparisons completed for {target_id}.")

    # Step 5: Check that there are 8040 json files in target_id_scores
    json_files = glob.glob(os.path.join(scores_dir, '*.json'))
    if len(json_files) == 8040:
        print(f"All 8040 json files found for {target_id}")
    else:
        print(f"Expected 8040 json files but found {len(json_files)} for {target_id}")

    # Step 6: Delete the downloaded target_id folder and tar.gz
    shutil.rmtree(downloaded_dir)
    if os.path.exists(output_filename):
        os.remove(output_filename)

    print(f"Processing of {target_id} completed.\n")


