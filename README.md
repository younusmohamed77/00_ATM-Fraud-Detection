# Fraud Detection Dataset README

This repository contains data for a **Fraud Detection** problem where only a small percentage of transactions are fraudulent. Below are the details of each file and instructions to combine them.

---

## File Descriptions

1. **train.csv**  
   - **Columns** (28 total):  
     - `id` (int) — Unique identifier (masked).  
     - `Group` (string) — Grouping label (masked).  
     - `Per1` to `Per9` (float) — Numeric features (masked).  
     - `Dem1` to `Dem9` (float) — Additional numeric features (masked).  
     - `Cred1` to `Cred6` (float) — Credit/risk features (masked).  
     - `Normalised_FNT` (float) — A numeric field (masked).  
     - `Target` (int) — Fraud indicator (1 = Fraud, 0 = Clean).  

2. **test_share.csv**  
   - **Columns** (27 total): Same as `train.csv` except **no** `Target` column.  
   - Used for final predictions or model evaluation.

3. **Geo_scores.csv**  
   - **Columns**:  
     - `id` (int)  
     - `geo_score` (float)  
   - Contains geospatial location scores related to transactions.

4. **Lambda_wts.csv**  
   - **Columns**:  
     - `Group` (string)  
     - `lambda_wt` (float)  
   - Proprietary weight/score for each group.

5. **Qset_tats.csv**  
   - **Columns**:  
     - `id` (int)  
     - `qsets_normalized_tat` (float)  
   - Network turn-around times (TAT) for each transaction.

6. **instance_scores.csv**  
   - **Columns**:  
     - `id` (int)  
     - `instance_scores` (float)  
   - Vulnerability or risk qualification scores.

---

## Usage

1. **Combine Files**  
   - Merge `Geo_scores.csv`, `Qset_tats.csv`, `instance_scores.csv` on `id`.  
   - Merge `Lambda_wts.csv` on `Group`.  

2. **Model Building**  
   - Explore the imbalance (majority transactions are clean).  
   - Use oversampling (RandomOverSampler) or undersampling or SMOTE to address the minority class.  
   - Train classifiers such as RandomForest, XGBoost, or LightGBM.

3. **Feature Engineering**  
   - Check which additional scores (`geo_score`, `lambda_wt`, etc.) add predictive value.  
   - Consider outlier treatment or scaling if needed.

4. **Evaluation**  
   - Since data is highly imbalanced, use **Precision**, **Recall**, **F1-score**, or **ROC-AUC**.  
   - Do not rely solely on accuracy.

5. **Predicting**  
   - After training, make predictions on `test_share.csv` (which lacks the `Target` column).  
   - Submit results or finalize them as needed.

---

## Example Python Workflow

```python
import pandas as pd

# Read main files
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_share.csv')

# Read extra data
geo_df = pd.read_csv('Geo_scores.csv')
lambda_df = pd.read_csv('Lambda_wts.csv')
tat_df = pd.read_csv('Qset_tats.csv')
inst_df = pd.read_csv('instance_scores.csv')

# Merge additional data to train_df if needed
train_merged = pd.merge(train_df, geo_df, on='id', how='left')
train_merged = pd.merge(train_merged, tat_df, on='id', how='left')
train_merged = pd.merge(train_merged, inst_df, on='id', how='left')
# Merge lambda on Group
train_merged = pd.merge(train_merged, lambda_df, on='Group', how='left')

# Proceed with preprocessing, modeling, etc.
