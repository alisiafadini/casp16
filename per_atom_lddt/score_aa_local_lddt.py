import os
import traceback
from tqdm import tqdm
import subprocess
from multiprocessing import Pool

"""
Script to compute per-atom lDDT between monomer models and reference CASP16 evaluation units using Open Structure (OST) within a Singularity container.

Executes a Singularity command to compare the model and reference structures, generating a JSON file with the comparison scores.

Usage:
    This is hard-wired at the moment

    Run from a directory containing:
      targets.list: list of CASP domain target names
      domain_targets: directory containing target structures of domains
      domain_model_dirs: directory of directories containing predictions

    The number of processors is hardwired as num_proc at the bottom

"""

# Function to run the ost command
def compare_structures_aa_local_lddt(args):

    """
    Compares the structure of a model PDB file against a reference PDB file using the dev version of Open Structure Tool (ost).

    Args:
        args (tuple): A tuple containing:
            - model (str): Path to the model PDB file.
            - reference (str): Path to the reference PDB file.
            - output_dir (str): Directory where the output JSON file should be saved.

    """

    model, reference, output_dir = args
    command = (
        f"singularity run --app OST /home/rjr27/OpenStructure/ost_dev_09_17.sif compare-structures "
        f"-m {model} -mf pdb "
        f"-r {reference} "
        f"--output {output_dir}/scores_{os.path.basename(model)}.json "
        f"--aa-local-lddt "
    )
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Completed comparison for {model}")
    except subprocess.CalledProcessError as e:
        print(f"Error comparing {model}: {e}")

# Main script
def process_target(target_id, ref_file, num_proc):

    models_dir = f"domain_model_dirs/{target_id}"
    if not os.path.isdir(models_dir):
        print(f"Directory named {models_dir} not found.")

    # Create a target_id_scores directory
    scores_dir = f"{target_id}_scores"
    os.makedirs(scores_dir, exist_ok=True)
    print(scores_dir)

    # Confirm that the scores directory was created
    if not os.path.exists(scores_dir):
        print(f"Failed to create directory: {scores_dir}")
        return
    else:
        print(f"Scores directory {scores_dir} successfully created.")

    # Run compare_structures using multiprocessing
    # Get a list of all PDB files in the directory
    pdb_files = [
        os.path.join(models_dir, f)
        for f in os.listdir(models_dir)
        if f.endswith(".pdb")
    ]

    if not pdb_files:
        print(f"No PDB files found in directory {models_dir}.")
        return

    # Create a list of tuples for the pool map
    tasks = [(model, ref_file, scores_dir) for model in pdb_files]

    # Create a pool of workers
    with Pool(num_proc) as pool:  # Use suitable number of processors
        pool.map(compare_structures_aa_local_lddt, tasks)

    print(f"All comparisons completed for {target_id}.")

def main(target_id, failed_ids, num_proc):

    try:
        process_target(target_id, f"domain_targets/{target_id}.pdb", num_proc)
    except Exception as e:
        print(f"Error processing {target_id}: {e}")
        traceback.print_exc()
        failed_ids.append(target_id)

if __name__ == "__main__":
    num_proc = 10
    target_id_list = "targets.list"
    failed_ids = []

    # Load target_ids from the targets.list file
    with open(target_id_list, 'r') as file:
        target_ids = [line.strip() for line in file]

    # Process each target_id
    for i, target_id in enumerate(tqdm(target_ids, desc="Processing Targets")):
        print("%%%%%%%%%%")
        print(f"Processing target number {i}: {target_id}")
        main(target_id, failed_ids, num_proc)
        print("%%%%%%%%%%")

    # Write failed IDs to a file if there are any
    if failed_ids:
        with open("failed_ids.txt", "w") as f:
            for failed_id in failed_ids:
                f.write(f"{failed_id}\n")
        print(f"Failed IDs written to failed_ids.txt")
