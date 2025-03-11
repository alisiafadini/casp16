import os
import json
import numpy as np
import pandas as pd

from ost import io
from ost import mol

"""
Script to compare actual LDDT values computed by OST, using the known target
structure, with the predicted (pLDDT) values in the B-factor column of the model.
Four scores are computed per model: correlation coefficient (CC) computed on a
per-atom basis, RMSD computed per-atom, CC using the per-residue pLDDT average,
RMSD using the per-residue pLDDT average.

Usage:
    This is hard-wired at the moment

    Run from a directory containing:
      targets.list: list of CASP domain target names
      csv files with lDDT scores from score_aa_local_lddt.py
    which in turn is inside a directory containing
      domain_model_dirs: directory of directories containing predictions

"""

def compare_lddt_values(mdl, scoring_data):

    bfactors = list()
    bfactors_per_residue = list()
    aa_lddt = list()

    for atom_desc, score in scoring_data["aa_local_lddt"].items():

        if score is None:
            # reasons for score being None:
            # 1 - Atom is not resolved in reference
            # 2 - Atom is in a residue in reference that has spectacularly bad
            #     stereochemistry and has thus been removed from analysis
            continue

        split_atom_desc = atom_desc.split('.')
        assert(len(split_atom_desc) == 4)
        cname = split_atom_desc[0]
        rnum = int(split_atom_desc[1])
        ins_code = split_atom_desc[2]
        aname = split_atom_desc[3]

        # The ost way of finding a residue
        residue = mdl.FindResidue(cname, mol.ResNum(rnum, ins_code))
        assert(residue.IsValid())

        # First get mean B for this residue (probably not the fastest way)
        bmean = 0.
        for atom in residue.atoms:
            bmean += atom.GetBFactor()
        bmean /= len(residue.atoms)

        # and the atom
        atom = residue.FindAtom(aname)
        assert(atom.IsValid())

        this_atom_b = atom.GetBFactor()
        bfactors.append(this_atom_b)
        bfactors_per_residue.append(bmean)
        aa_lddt.append(score*100.) # Put on same scale as pLDDT in PDB file

    if len(bfactors) == 0: # E.g. from major problem interpreting PDB file
        return [None,None,None,None]
    bfactor_array = np.array(bfactors)
    if bfactor_array.max() == bfactor_array.min():
        return [None,None,None,None]

    bfactor_per_residue_array = np.array(bfactors_per_residue)
    aa_lddt_array = np.array(aa_lddt)
    cc = np.corrcoef(bfactor_array, aa_lddt_array)
    rmsd = np.sqrt(((bfactor_array-aa_lddt_array)**2).mean())
    cc_per_residue = np.corrcoef(bfactor_per_residue_array, aa_lddt_array)
    rmsd_per_residue = np.sqrt(((bfactor_per_residue_array-aa_lddt_array)**2).mean())
    return [cc[0,1],rmsd,cc_per_residue[0,1],rmsd_per_residue]

def score_models_for_target(models_path, scores_path):
    group_scores = {}
    pdblist = [file for file in os.listdir(models_path) if file.endswith('.pdb')]
    nsaved = 0
    for pdb in pdblist:
        lddt_scores_file = os.path.join(scores_path,f"scores_{pdb}.json")
        if not os.path.exists(lddt_scores_file):
            print(f"lddt_scores_file {lddt_scores_file} not found. Skipping...")
            continue
        pdb_file = os.path.join(models_path, pdb)
        # print(f"Ready to score {pdb_file} and {lddt_scores_file}")
        model = io.LoadPDB(pdb_file)
        with open(lddt_scores_file, 'r') as fh:
            scoring_data = json.load(fh)

        result = compare_lddt_values(model, scoring_data)
        if result[0] is None:
            continue
        # group_name = pdb.str.extract(r'(TS\d{3})')
        nsaved += 1
        group_scores[nsaved] = [pdb]+result

    dataframe_scores = pd.DataFrame.from_dict(group_scores, orient='index',
                          columns = ['Model','CC_per_atom', 'RMSD_per_atom', 'CC_per_residue', 'RMSD_per_residue'] )
    dataframe_scores.to_csv(f"{scores_path}_per_atom.csv", index=False)
    return dataframe_scores


def main(target_ids, model_dirs, lddt_scores_dir):
    for target_id in target_ids:
        print(f"Calculating for {target_id}")
        scores_path = os.path.join(lddt_scores_dir, f"{target_id}_scores")
        if not os.path.exists(scores_path):
            print(f"Directory for scores {scores_path} not found. Skipping...")
            continue
        models_path = os.path.join(model_dirs, f"{target_id}")
        if not os.path.exists(models_path):
            print(f"ERROR: directory for scores {scores_path} not found. Skipping...")
            continue
        df_scores = score_models_for_target(models_path, scores_path)

if __name__ == "__main__":
    target_id_list = "targets.list"
    # Load target_ids from the targets.list file
    with open(target_id_list, 'r') as file:
        target_ids = [line.strip() for line in file]
    model_dirs = "../domain_model_dirs"
    lddt_scores_dir = "./"

    # Start processing
    main(target_ids, model_dirs, lddt_scores_dir)
