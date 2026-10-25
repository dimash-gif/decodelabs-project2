# DecodeLabs Project 2 — Fraud Detection Pipeline

Supervised learning pipeline for detecting fraudulent credit-card transactions in a highly imbalanced dataset (284,807 transactions, 0.17% fraud).

SMOTE runs inside an `imblearn` pipeline, so resampling never touches the validation fold. Logistic Regression and Random Forest are tuned with `GridSearchCV` and judged on precision, recall, F1 and ROC-AUC — accuracy is excluded by design.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) into this folder.

## Run

```bash
python fraud_detection.py --data creditcard.csv
```

Metrics, plots and the fitted model are written to `outputs/`.
