"""
GEE (population-averaged GLMM) for mouse context-switching behavior.

Fits a binomial GEE with logit link grouped by subject_id, then evaluates
predictions on a held-out test set.

Run:  uv run python glmm/fit.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GLMM_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent.parent / "data"
REPORT_PATH = GLMM_DIR / "REPORT.md"

FEATURES = [
    "previous_response",
    "previous_reward",
    "is_vis_target",
    "is_vis_non_target",
    "is_aud_target",
    "is_aud_non_target",
]
TARGET = "target_response"
GROUP = "subject_id"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    """Load a parquet file and cast booleans to int for statsmodels."""
    df = pd.read_parquet(path)
    for col in FEATURES + [TARGET]:
        df[col] = df[col].astype(int)
    # GEE requires the data to be sorted by group
    df = df.sort_values(GROUP).reset_index(drop=True)
    return df


def build_report(
    result,
    feature_names: list[str],
    y_test: np.ndarray,
    y_pred_prob: np.ndarray,
) -> str:
    """Build a Markdown report string from GEE results and test-set metrics."""
    lines: list[str] = []
    lines.append("# GEE (Population-Averaged GLMM) -- Results\n")

    # ----- Fixed effects table -----
    lines.append("## Fixed effects\n")

    params = result.params
    bse = result.bse
    pvalues = result.pvalues
    conf_int = result.conf_int()
    odds_ratios = np.exp(params)
    or_ci_lo = np.exp(conf_int[0])
    or_ci_hi = np.exp(conf_int[1])

    lines.append(
        "| Feature | Coef | SE | z | p-value | OR | OR 95% CI |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|"
    )
    for name in feature_names:
        coef = params[name]
        se = bse[name]
        z = coef / se
        p = pvalues[name]
        o = odds_ratios[name]
        lo = or_ci_lo[name]
        hi = or_ci_hi[name]
        p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
        lines.append(
            f"| {name} | {coef:+.4f} | {se:.4f} | {z:+.2f} | {p_str} "
            f"| {o:.4f} | [{lo:.4f}, {hi:.4f}] |"
        )

    # Intercept (sm.add_constant names it "const")
    name = "const"
    coef = params[name]
    se = bse[name]
    z = coef / se
    p = pvalues[name]
    o = odds_ratios[name]
    lo = or_ci_lo[name]
    hi = or_ci_hi[name]
    p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
    lines.append(
        f"| Intercept | {coef:+.4f} | {se:.4f} | {z:+.2f} | {p_str} "
        f"| {o:.4f} | [{lo:.4f}, {hi:.4f}] |"
    )

    lines.append("")

    # ----- Model info -----
    lines.append("## Model summary\n")
    lines.append(f"- **Family**: Binomial (logit link)")
    lines.append(f"- **Correlation structure**: Exchangeable")
    lines.append(f"- **Number of groups (subjects)**: {result.fit_history['n_groups']}")
    lines.append(f"- **Scale**: {result.scale:.4f}")
    lines.append("")

    # ----- Test-set metrics -----
    lines.append("## Test-set evaluation\n")

    y_pred_class = (y_pred_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred_class)
    bal_acc = balanced_accuracy_score(y_test, y_pred_class)
    auc = roc_auc_score(y_test, y_pred_prob)
    ll = log_loss(y_test, y_pred_prob)

    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Accuracy | {acc:.4f} |")
    lines.append(f"| Balanced accuracy | {bal_acc:.4f} |")
    lines.append(f"| AUC-ROC | {auc:.4f} |")
    lines.append(f"| Log-loss | {ll:.4f} |")
    lines.append("")

    # ----- Confusion matrix -----
    lines.append("## Confusion matrix\n")
    cm = confusion_matrix(y_test, y_pred_class)
    lines.append("```")
    lines.append("              Predicted 0  Predicted 1")
    lines.append(f"  Actual 0      {cm[0, 0]:>7d}      {cm[0, 1]:>7d}")
    lines.append(f"  Actual 1      {cm[1, 0]:>7d}      {cm[1, 1]:>7d}")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("GEE (Population-Averaged GLMM) for context-switching behavior")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/4] Loading training data ...")
    train_df = load_data(DATA_DIR / "training_set.parquet")
    print(f"       Training set: {len(train_df):,} trials, "
          f"{train_df[GROUP].nunique()} subjects")

    print("[2/4] Loading test data ...")
    test_df = load_data(DATA_DIR / "test_set.parquet")
    print(f"       Test set:     {len(test_df):,} trials, "
          f"{test_df[GROUP].nunique()} subjects")

    # --- Prepare design matrices ---
    X_train = sm.add_constant(train_df[FEATURES], prepend=False)
    y_train = train_df[TARGET]
    groups_train = train_df[GROUP]

    X_test = sm.add_constant(test_df[FEATURES], prepend=False)
    y_test = test_df[TARGET].values

    feature_names = FEATURES  # for report (excludes Intercept which is added separately)

    # --- Fit GEE ---
    print("[3/4] Fitting GEE (binomial, logit, exchangeable correlation) ...")
    family = Binomial(link=Logit())
    cov_struct = Exchangeable()

    model = GEE(
        endog=y_train,
        exog=X_train,
        groups=groups_train,
        family=family,
        cov_struct=cov_struct,
    )
    result = model.fit()

    print("\n--- GEE coefficient summary ---")
    print(result.summary())
    print()

    # Store n_groups in fit_history so the report can access it
    result.fit_history["n_groups"] = train_df[GROUP].nunique()

    # --- Predict on test set ---
    print("[4/4] Evaluating on test set ...")
    y_pred_prob = result.predict(X_test)
    y_pred_class = (y_pred_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred_class)
    bal_acc = balanced_accuracy_score(y_test, y_pred_class)
    auc = roc_auc_score(y_test, y_pred_prob)
    ll = log_loss(y_test, y_pred_prob)

    print(f"       Accuracy:          {acc:.4f}")
    print(f"       Balanced accuracy: {bal_acc:.4f}")
    print(f"       AUC-ROC:           {auc:.4f}")
    print(f"       Log-loss:          {ll:.4f}")

    # --- Write report ---
    report = build_report(result, feature_names, y_test, y_pred_prob)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
