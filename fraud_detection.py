#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SEARCH_SUBSAMPLE = 60_000

SCORING = {
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "avg_precision": "average_precision",
}
REFIT_ON = "f1"

GRID_LR = {
    "smote__k_neighbors": [3, 5],
    "classifier__C": [0.01, 0.1, 1.0],
}
GRID_RF = {
    "smote__k_neighbors": [3, 5],
    "classifier__max_depth": [10, None],
}


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(
            f"ERROR: {path} not found.\n"
            "Download creditcard.csv from Kaggle (mlg-ulb/creditcardfraud) "
            "and pass it with --data."
        )

    df = pd.read_csv(path)

    if "Class" not in df.columns:
        sys.exit(f"ERROR: no 'Class' column in {path}. Columns: {list(df.columns)[:10]}")

    print(f"Loaded {path}  ->  {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def explore(df: pd.DataFrame) -> None:
    banner("1. THE REALITY OF THE DATA")

    counts = df["Class"].value_counts().sort_index()
    legit, fraud = int(counts.loc[0]), int(counts.loc[1])
    total = legit + fraud

    print(f"Legitimate : {legit:>7,}  ({legit / total:.4%})")
    print(f"Fraudulent : {fraud:>7,}  ({fraud / total:.4%})")
    print(f"Imbalance  : 1 fraud per {legit / fraud:.0f} legitimate transactions")
    print(f"\nMissing values : {int(df.isna().sum().sum())}")
    print(f"Duplicate rows : {int(df.duplicated().sum())}")

    naive = np.zeros(len(df), dtype=int)
    print("\n--- Baseline: predict 'legitimate' every time ---")
    print(f"Accuracy  : {accuracy_score(df['Class'], naive):.4%}   <- looks superb")
    print(f"Recall    : {recall_score(df['Class'], naive, zero_division=0):.4%}   <- catches nothing")
    print(f"Precision : {precision_score(df['Class'], naive, zero_division=0):.4%}")
    if "Amount" in df.columns:
        lost = df.loc[df["Class"] == 1, "Amount"].sum()
        print(f"Fraud missed: {fraud:,} transactions worth {lost:,.2f}")


def split_data(df: pd.DataFrame):
    banner("2. STRATIFIED TRAIN/TEST SPLIT")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    print(f"train: {len(X_train):>7,} rows, {int(y_train.sum()):>4,} fraud ({y_train.mean():.4%})")
    print(f"test : {len(X_test):>7,} rows, {int(y_test.sum()):>4,} fraud ({y_test.mean():.4%})")
    return X_train, X_test, y_train, y_test


def build_pipelines():
    pipe_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    pipe_rf = Pipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("classifier", RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])

    return pipe_lr, pipe_rf


def tune(pipe, grid, X, y, name):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    n_combos = int(np.prod([len(v) for v in grid.values()]))

    print(f"\n--- {name}: {n_combos} combinations x {CV_FOLDS} folds "
          f"= {n_combos * CV_FOLDS} fits ---")

    search = GridSearchCV(
        pipe, grid, cv=cv, scoring=SCORING, refit=REFIT_ON, n_jobs=-1, verbose=1
    )
    search.fit(X, y)

    print(f"best params : {search.best_params_}")
    print(f"best CV {REFIT_ON:<4}: {search.best_score_:.4f}")
    return search


def cv_table(search, name) -> pd.DataFrame:
    res = pd.DataFrame(search.cv_results_)
    cols = [c for c in res.columns if c.startswith("param_")]
    cols += [f"mean_test_{m}" for m in SCORING]
    out = res[cols].copy()
    out.insert(0, "model", name)
    return out.sort_values(f"mean_test_{REFIT_ON}", ascending=False)


def evaluate(model, name, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "model": name,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "avg_precision": average_precision_score(y_test, y_prob),
        "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
        "_y_pred": y_pred,
        "_y_prob": y_prob,
    }


def report(results, y_test) -> None:
    for r in results:
        banner(r["model"])
        print(classification_report(
            y_test, r["_y_pred"],
            target_names=["Legitimate", "Fraudulent"], digits=4, zero_division=0,
        ))
        print(f"ROC-AUC               : {r['roc_auc']:.4f}")
        print(f"Average precision (PR): {r['avg_precision']:.4f}")
        print(f"Fraud caught          : {r['TP']} of {r['TP'] + r['FN']}")
        print(f"Fraud missed          : {r['FN']}")
        print(f"False alarms          : {r['FP']} legitimate transactions flagged")


def threshold_sweep(best, y_test) -> pd.DataFrame:
    banner(f"THRESHOLD TUNING - {best['model']}")

    rows = []
    for t in np.arange(0.1, 0.95, 0.1):
        pred = (best["_y_prob"] >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        rows.append({
            "threshold": round(float(t), 2),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "fraud_caught": int(tp),
            "fraud_missed": int(fn),
            "false_alarms": int(fp),
        })

    table = pd.DataFrame(rows).round(4)
    print(table.to_string(index=False))
    return table


def make_plots(results, y_test, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4.6))
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, results):
        cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred legit", "Pred fraud"],
                    yticklabels=["True legit", "True fraud"])
        ax.set_title(f"{r['model']}\nrecall={r['recall']:.3f}  precision={r['precision']:.3f}")
    plt.tight_layout()
    plt.savefig(outdir / "confusion_matrices.png", dpi=130)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["_y_prob"])
        axes[0].plot(fpr, tpr, lw=2, label=f"{r['model']} (AUC={r['roc_auc']:.4f})")
        prec, rec, _ = precision_recall_curve(y_test, r["_y_prob"])
        axes[1].plot(rec, prec, lw=2, label=f"{r['model']} (AP={r['avg_precision']:.4f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[0].legend(loc="lower right")

    axes[1].axhline(y_test.mean(), color="k", ls="--", lw=1,
                    label=f"Chance (={y_test.mean():.4f})")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(outdir / "roc_pr_curves.png", dpi=130)
    plt.close()

    print(f"\nPlots saved to {outdir}/confusion_matrices.png and {outdir}/roc_pr_curves.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="DecodeLabs Project 2 - Fraud Detection Pipeline")
    ap.add_argument("--data", type=Path, default=Path("creditcard.csv"),
                    help="path to creditcard.csv (default: ./creditcard.csv)")
    ap.add_argument("--outdir", type=Path, default=Path("outputs"),
                    help="where to write model, tables and plots")
    ap.add_argument("--full", action="store_true",
                    help="grid search on the full training set instead of a subsample")
    ap.add_argument("--no-plots", action="store_true", help="skip figure generation")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    explore(df)
    X_train, X_test, y_train, y_test = split_data(df)

    banner("3. HYPERPARAMETER TUNING")
    if not args.full and len(X_train) > SEARCH_SUBSAMPLE:
        X_search, _, y_search, _ = train_test_split(
            X_train, y_train, train_size=SEARCH_SUBSAMPLE,
            stratify=y_train, random_state=RANDOM_STATE,
        )
        print(f"Searching on a {len(X_search):,}-row stratified subsample "
              f"({int(y_search.sum())} fraud). Final fit uses all {len(X_train):,} rows.")
    else:
        X_search, y_search = X_train, y_train
        print(f"Searching on the full {len(X_search):,}-row training set.")

    pipe_lr, pipe_rf = build_pipelines()
    search_lr = tune(pipe_lr, GRID_LR, X_search, y_search, "LogisticRegression")
    search_rf = tune(pipe_rf, GRID_RF, X_search, y_search, "RandomForest")

    cv_results = pd.concat(
        [cv_table(search_lr, "LogisticRegression"), cv_table(search_rf, "RandomForest")],
        ignore_index=True,
    ).round(4)
    print("\n--- Cross-validation results ---")
    print(cv_results.to_string(index=False))
    cv_results.to_csv(args.outdir / "cv_results.csv", index=False)

    best_lr, best_rf = search_lr.best_estimator_, search_rf.best_estimator_
    if len(X_search) < len(X_train):
        banner("4. REFIT ON THE FULL TRAINING SET")
        best_lr.fit(X_train, y_train)
        best_rf.fit(X_train, y_train)
        print("Refitted both pipelines on every training row.")

    banner("5. FINAL EVALUATION ON THE SEALED TEST SET")
    results = [
        evaluate(best_lr, "LogisticRegression", X_test, y_test),
        evaluate(best_rf, "RandomForest", X_test, y_test),
    ]

    summary = pd.DataFrame(results).drop(columns=["_y_pred", "_y_prob"]).round(4)
    print(summary.to_string(index=False))
    summary.to_csv(args.outdir / "test_metrics.csv", index=False)

    report(results, y_test)

    winner = max(results, key=lambda r: r["avg_precision"])
    threshold_sweep(winner, y_test).to_csv(args.outdir / "threshold_sweep.csv", index=False)

    if not args.no_plots:
        make_plots(results, y_test, args.outdir)

    banner("6. RECOMMENDATION")
    print(f"Model: {winner['model']}")
    print(f"  precision     {winner['precision']:.4f}")
    print(f"  recall        {winner['recall']:.4f}")
    print(f"  F1            {winner['f1']:.4f}")
    print(f"  ROC-AUC       {winner['roc_auc']:.4f}")
    print(f"  avg precision {winner['avg_precision']:.4f}")
    print(f"  catches {winner['TP']}/{winner['TP'] + winner['FN']} fraud, "
          f"{winner['FP']} false alarms")

    best_model = best_lr if winner["model"] == "LogisticRegression" else best_rf
    model_path = args.outdir / "fraud_pipeline.joblib"
    joblib.dump(best_model, model_path)
    print(f"\nSaved model -> {model_path}")
    print(f"All artifacts in {args.outdir}/")


if __name__ == "__main__":
    main()
