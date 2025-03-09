# CASP16 QMODE3 MassiveFold Model Scoring and Ranking

This folder contains scripts for **scoring, ranking, and analyzing** MassiveFold predictions from CASP16 QMODE3. The analysis compares MassiveFold models against experimental references using OpenStructure (OST) and generates penalties for different target categories based on group predictions.

---

## 📂 Repository Structure
```
QMODE3/
│── data/
│   ├── per_score_rankings/        # Per-score rankings from OST
│   ├── all_ranking_errors/        # Ranking errors per group
│   ├── covariances_massivefold/   # Covariances for penalty computation
│   ├── concatenated_rankings/     # Merged per-score rankings
│── weighted_penalties/            # Final weighted penalties per target type
│── scripts/
│   ├── score_massivefold_models_combined_withdocker.py   # Calls OST for scoring
│   ├── score_massivefold_functions.py                    # Scoring helper functions
│   ├── rank_each_json.py                                 # Ranks MF models per score
│   ├── ranking_error_perscore_pergroup.py               # Computes ranking errors 
│   ├── common_covariance.py                           # Generates covariance matrices
│   ├── groups_lineup_with_weightedpenalty.py            # Computes weighted penalties per targets
│   ├── sum_all_targets.py                                # Summarizes penalties across targets
│   ├── concatenate_score_rankings.py                     # Merges per-score rankings (extra script)
│── compute_weighted_penalty_alltargets.sh               # Runs penalty computations
│── README.md
```

---

## Installation
1. Install dependencies:
   ```bash
   pip install numpy pandas scipy
   ```
2. Ensure OpenStructure (OST) is installed and accessible:
   ```bash
   ost -h
   ```
3. Clone this repo and:
   ```bash
   cd QMODE3
   ```

---

## 📊 Workflow
### **Step 1: Score MassiveFold Models**
```bash
python scripts/score_massivefold_models_combined_withdocker.py
```

### **Step 2: Rank Models**
```bash
python scripts/rank_each_json.py
```

### **Step 3: Compute Ranking Errors**
```bash
python scripts/ranking_error_perscore_pergroup.py
```

### **Step 4: Generate Covariance Matrices**
```bash
python scripts/common_MF_covariance.py
```

### **Step 5: Compute Weighted Penalties**
```bash
sh compute_weighted_penalty_alltargets.sh
```

### Calls, under the hood, 
```bash
scripts/groups_lineup_with_weightedpenalty.py, scripts/sum_all_targets.py
```

---

## Contact
For questions, reach out at **af840@cam.ac.uk**.


