"""
Decision Tree (CART) classifier for mouse context-switching behavior.

Predicts `target_response` from 6 boolean features using scikit-learn
DecisionTreeClassifier. Cross-validates over max_depth to find the optimal
tree depth, then fits a final model and writes results to REPORT.md.

Run from project root:
    uv run python decision_tree/fit.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_PATH = DATA_DIR / "training_set.parquet"
TEST_PATH = DATA_DIR / "test_set.parquet"

REPORT_PATH = SCRIPT_DIR / "REPORT.md"
TREE_PLOT_PATH = SCRIPT_DIR / "tree_plot.png"

FEATURES = [
    "previous_response",
    "previous_reward",
    "is_vis_target",
    "is_vis_non_target",
    "is_aud_target",
    "is_aud_non_target",
]
TARGET = "target_response"

MAX_DEPTH_CANDIDATES = [2, 3, 4, 5, 6, 8, 10, None]
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
train_df = pd.read_parquet(TRAIN_PATH)
test_df = pd.read_parquet(TEST_PATH)

X_train = train_df[FEATURES].values
y_train = train_df[TARGET].values.astype(int)
X_test = test_df[FEATURES].values
y_test = test_df[TARGET].values.astype(int)

print(f"  Training set: {X_train.shape[0]:,} trials, {X_train.shape[1]} features")
print(f"  Test set:     {X_test.shape[0]:,} trials, {X_test.shape[1]} features")
print(f"  Positive class rate (train): {y_train.mean():.3f}")
print(f"  Positive class rate (test):  {y_test.mean():.3f}")

# ---------------------------------------------------------------------------
# Cross-validation over max_depth
# ---------------------------------------------------------------------------
print("\nCross-validating over max_depth candidates...")
cv_results: dict[str | int, float] = {}
for depth in MAX_DEPTH_CANDIDATES:
    label = depth if depth is not None else "None"
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=CV_FOLDS, scoring="balanced_accuracy")
    mean_score = scores.mean()
    std_score = scores.std()
    cv_results[depth] = mean_score
    print(f"  max_depth={str(label):>4s}:  balanced_accuracy = {mean_score:.4f} +/- {std_score:.4f}")

best_depth = max(cv_results, key=cv_results.get)  # type: ignore[arg-type]
best_depth_label = best_depth if best_depth is not None else "None"
print(f"\nBest max_depth: {best_depth_label}  (balanced_accuracy = {cv_results[best_depth]:.4f})")

# ---------------------------------------------------------------------------
# Fit final model
# ---------------------------------------------------------------------------
print("\nFitting final DecisionTreeClassifier...")
final_clf = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
final_clf.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# Evaluate on test set
# ---------------------------------------------------------------------------
print("Evaluating on test set...")
y_pred = final_clf.predict(X_test)
y_prob = final_clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
bal_accuracy = balanced_accuracy_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_prob)
logloss = log_loss(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print(f"  Accuracy:          {accuracy:.4f}")
print(f"  Balanced accuracy: {bal_accuracy:.4f}")
print(f"  AUC-ROC:           {auc_roc:.4f}")
print(f"  Log-loss:          {logloss:.4f}")
print(f"  Confusion matrix:\n{cm}")

# ---------------------------------------------------------------------------
# Tree rules and feature importances
# ---------------------------------------------------------------------------
tree_rules = export_text(final_clf, feature_names=FEATURES)
print(f"\nDecision tree rules:\n{tree_rules}")

importances = final_clf.feature_importances_
importance_order = np.argsort(importances)[::-1]
print("Feature importances:")
for idx in importance_order:
    print(f"  {FEATURES[idx]:>25s}: {importances[idx]:.4f}")

# ---------------------------------------------------------------------------
# Save tree plot
# ---------------------------------------------------------------------------
print(f"\nSaving tree visualization to {TREE_PLOT_PATH}...")
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    final_clf,
    feature_names=FEATURES,
    class_names=["no_response", "response"],
    filled=True,
    rounded=True,
    ax=ax,
    fontsize=8,
)
ax.set_title(f"Decision Tree (max_depth={best_depth_label})", fontsize=14)
fig.tight_layout()
fig.savefig(TREE_PLOT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Done.")

# ---------------------------------------------------------------------------
# Write REPORT.md
# ---------------------------------------------------------------------------
print(f"Writing report to {REPORT_PATH}...")

# Build cross-validation results table
cv_table_rows = []
for depth in MAX_DEPTH_CANDIDATES:
    label = depth if depth is not None else "None"
    marker = " *" if depth == best_depth else ""
    cv_table_rows.append(f"| {label} | {cv_results[depth]:.4f} |{marker}")
cv_table = "\n".join(cv_table_rows)

# Build feature importance table
fi_rows = []
for idx in importance_order:
    fi_rows.append(f"| {FEATURES[idx]} | {importances[idx]:.4f} |")
fi_table = "\n".join(fi_rows)

report = f"""\
# Decision Tree -- Results

## Cross-Validation (max_depth selection)

{CV_FOLDS}-fold cross-validation on the training set using balanced accuracy.

| max_depth | balanced_accuracy |
|---|---|
{cv_table}

Best max_depth: **{best_depth_label}**

## Test Set Metrics

| Metric | Value |
|---|---|
| Accuracy | {accuracy:.4f} |
| Balanced accuracy | {bal_accuracy:.4f} |
| AUC-ROC | {auc_roc:.4f} |
| Log-loss | {logloss:.4f} |

## Confusion Matrix

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| Actual 0 | {cm[0, 0]:,} | {cm[0, 1]:,} |
| Actual 1 | {cm[1, 0]:,} | {cm[1, 1]:,} |

## Feature Importances

| Feature | Importance |
|---|---|
{fi_table}

## Decision Rules

```
{tree_rules}```

## Tree Visualization

![Decision Tree](tree_plot.png)
"""

REPORT_PATH.write_text(report, encoding="utf-8")
print("  Done.")
print("\nAll finished.")
