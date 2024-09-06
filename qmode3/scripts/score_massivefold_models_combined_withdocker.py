import os
import traceback
from tqdm import tqdm
from score_massivefold_functions import process_target



def check_and_run_comparison(massivefold_file, target_id, specific_suffix, mono_condition_met, oligo_condition_met, num_proc):
    # Check for specific versions of oligomer and monomer references
    for version in range(4):  # This covers version 0 (default) to 3
        if version == 0:
            monomer_reference = f"{target_id}{specific_suffix}.pdb"
            oligomer_reference = f"{target_id}o.pdb"
        else:
            monomer_reference = f"{target_id}v{version}{specific_suffix}.pdb"
            oligomer_reference = f"{target_id}v{version}o.pdb"

        if os.path.exists(f"targets_oligo/{oligomer_reference}"):
            print("PROCESSING", oligomer_reference)
            process_target(massivefold_file, oligomer_reference, f"targets_oligo/{oligomer_reference}", num_proc, scoring_type="oligomer")
            oligo_condition_met = True

        elif os.path.exists(f"targets_mono/{monomer_reference}") and not oligo_condition_met:
            process_target(massivefold_file, monomer_reference, f"targets_mono/{monomer_reference}", num_proc, scoring_type="monomer")
            mono_condition_met = True

    return mono_condition_met, oligo_condition_met

def main(massivefold_file, failed_ids, num_proc):
    base_name = os.path.basename(massivefold_file)
    target_name = base_name.split(".")[0]
    mono_condition_met = False
    oligo_condition_met = False

    try:
        if target_name.startswith("T"):
            if "s" in target_name:
                target_id, specific_suffix = target_name.split("s")
                specific_suffix = f"s{target_name.split('s')[-1]}"
            else:
                target_id = target_name.split("_")[0][:5]
                specific_suffix = ""

            mono_condition_met, oligo_condition_met = check_and_run_comparison(
                massivefold_file, target_id, specific_suffix, mono_condition_met, oligo_condition_met, num_proc
            )

        elif target_name.startswith("H"):
            target_id = target_name[:5]
            oligomer_reference = f"{target_id}.pdb"
            if os.path.exists(f"targets_oligo/{oligomer_reference}"):
                process_target(massivefold_file, f"targets_oligo/{oligomer_reference}", num_proc, scoring_type="oligomer")
                oligo_condition_met = True

        if not mono_condition_met and not oligo_condition_met:
            print(f"Warning: No valid comparison found for target {target_name}")
            failed_ids.append(target_name)
    
    except Exception as e:
        print(f"Error processing {target_name}: {e}")
        traceback.print_exc()
        failed_ids.append(target_name)

if __name__ == "__main__":
    num_proc = 20
    target_id_list = "massivefold_target_ids.txt"
    failed_ids = []

    # Load target_ids from the massivefold_target_ids.txt file
    with open(target_id_list, 'r') as file:
        target_ids = [line.strip() for line in file]
    
    # Process each target_id
    for i, target in enumerate(tqdm(target_ids, desc="Processing Targets")):
        print("%%%%%%%%%%")
        print(f"Processing target number {i}: {target}")
        main(target, failed_ids, num_proc)
        print("%%%%%%%%%%")

    # Write failed IDs to a file if there are any
    if failed_ids:
        with open("failed_ids.txt", "w") as f:
            for failed_id in failed_ids:
                f.write(f"{failed_id}\n")
        print(f"Failed IDs written to failed_ids.txt")
