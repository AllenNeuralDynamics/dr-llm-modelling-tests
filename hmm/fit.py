"""
Hidden Markov Model for mouse context-switching behavior.

Fits a GaussianHMM to trial-level features, selects the best number of
hidden states via BIC, and evaluates prediction on the held-out test set.

Run from project root:
    uv run python hmm/fit.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"
REPORT_PATH = HERE / "REPORT.md"

FEATURE_COLS = [
    "previous_response",
    "previous_reward",
    "is_vis_target",
    "is_vis_non_target",
    "is_aud_target",
    "is_aud_non_target",
]

TARGET_COL = "target_response"

N_COMPONENTS_CANDIDATES = [2, 3, 4]
N_ITER = 200
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    """Load a parquet file and return a DataFrame."""
    df = pd.read_parquet(path)
    print(f"  Loaded {path.name}: {len(df):,} trials, {df['subject_id'].nunique()} subjects")
    return df


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """Extract feature matrix (N, 6) as float64."""
    return df[FEATURE_COLS].astype(np.float64).values


def get_lengths(df: pd.DataFrame) -> list[int]:
    """Return the number of trials per subject, preserving row order."""
    # Subjects appear in contiguous blocks sorted by (subject_id, trial_index)
    # in the parquet files.  We group by subject_id in the order they appear.
    lengths: list[int] = []
    current_subject = None
    current_count = 0
    for sid in df["subject_id"].values:
        if sid != current_subject:
            if current_subject is not None:
                lengths.append(current_count)
            current_subject = sid
            current_count = 1
        else:
            current_count += 1
    if current_count > 0:
        lengths.append(current_count)
    return lengths


def fit_hmm(X: np.ndarray, lengths: list[int], n_components: int) -> GaussianHMM:
    """Fit a GaussianHMM and return the model."""
    model = GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=N_ITER,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    model.fit(X, lengths)
    return model


def compute_bic(model: GaussianHMM, X: np.ndarray, lengths: list[int]) -> float:
    """
    Compute BIC = -2 * log_likelihood + k * ln(N).

    k = free parameters:
        - startprob: (n - 1)
        - transmat: n * (n - 1)
        - means: n * d
        - covars (diag): n * d
    """
    n = model.n_components
    d = X.shape[1]
    k = (n - 1) + n * (n - 1) + n * d + n * d
    log_likelihood = model.score(X, lengths)
    n_samples = X.shape[0]
    return -2 * log_likelihood + k * np.log(n_samples)


def state_response_probs(
    df: pd.DataFrame,
    states: np.ndarray,
    n_components: int,
) -> dict[int, float]:
    """Compute P(target_response=True | state) from training data."""
    probs: dict[int, float] = {}
    for s in range(n_components):
        mask = states == s
        if mask.sum() == 0:
            probs[s] = 0.5  # fallback
        else:
            probs[s] = df.loc[mask, TARGET_COL].mean()
    return probs


def format_matrix(mat: np.ndarray, decimals: int = 4) -> str:
    """Format a 2D numpy array as a markdown table."""
    n = mat.shape[0]
    header = "| | " + " | ".join(f"State {j}" for j in range(n)) + " |"
    separator = "|---|" + "|".join("---" for _ in range(n)) + "|"
    rows = []
    for i in range(mat.shape[0]):
        row = f"| **State {i}** | " + " | ".join(
            f"{mat[i, j]:.{decimals}f}" for j in range(n)
        ) + " |"
        rows.append(row)
    return "\n".join([header, separator] + rows)


def format_means(means: np.ndarray, feature_names: list[str]) -> str:
    """Format emission means as a markdown table."""
    n = means.shape[0]
    header = "| Feature | " + " | ".join(f"State {j}" for j in range(n)) + " |"
    separator = "|---|" + "|".join("---" for _ in range(n)) + "|"
    rows = []
    for i, feat in enumerate(feature_names):
        row = f"| {feat} | " + " | ".join(
            f"{means[j, i]:.4f}" for j in range(n)
        ) + " |"
        rows.append(row)
    return "\n".join([header, separator] + rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("HMM: Hidden Markov Model for context-switching behavior")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data...")
    train_df = load_data(DATA_DIR / "training_set.parquet")
    test_df = load_data(DATA_DIR / "test_set.parquet")

    X_train = prepare_features(train_df)
    X_test = prepare_features(test_df)
    lengths_train = get_lengths(train_df)
    lengths_test = get_lengths(test_df)

    print(f"  Training sequences: {len(lengths_train)}, "
          f"total trials: {sum(lengths_train):,}")
    print(f"  Test sequences: {len(lengths_test)}, "
          f"total trials: {sum(lengths_test):,}")

    # ------------------------------------------------------------------
    # 2. Model selection
    # ------------------------------------------------------------------
    print("\n[2/5] Fitting HMMs and selecting best model by BIC...")
    results: dict[int, dict] = {}

    for n_comp in N_COMPONENTS_CANDIDATES:
        print(f"  Fitting n_components={n_comp} ...", end=" ", flush=True)
        model = fit_hmm(X_train, lengths_train, n_comp)
        ll = model.score(X_train, lengths_train)
        bic = compute_bic(model, X_train, lengths_train)
        converged = model.monitor_.converged
        print(f"log-likelihood={ll:,.1f}  BIC={bic:,.1f}  converged={converged}")
        results[n_comp] = {"model": model, "ll": ll, "bic": bic, "converged": converged}

    best_n = min(results, key=lambda k: results[k]["bic"])
    best = results[best_n]
    model: GaussianHMM = best["model"]
    print(f"\n  Best model: n_components={best_n} (BIC={best['bic']:,.1f})")

    # ------------------------------------------------------------------
    # 3. Decode training states and compute state-conditional response rates
    # ------------------------------------------------------------------
    print("\n[3/5] Decoding hidden states on training data...")
    train_states = model.predict(X_train, lengths_train)
    state_probs = state_response_probs(train_df, train_states, best_n)

    for s in range(best_n):
        count = (train_states == s).sum()
        print(f"  State {s}: {count:,} trials, "
              f"P(response|state)={state_probs[s]:.4f}")

    # ------------------------------------------------------------------
    # 4. Predict on test set
    # ------------------------------------------------------------------
    print("\n[4/5] Predicting on test data...")
    test_states = model.predict(X_test, lengths_test)
    y_true = test_df[TARGET_COL].values.astype(int)

    # Predicted probability of target_response=True
    y_prob = np.array([state_probs[s] for s in test_states])
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    ll_score = log_loss(y_true, y_prob)

    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced accuracy: {bal_acc:.4f}")
    print(f"  AUC-ROC:           {auc:.4f}")
    print(f"  Log-loss:          {ll_score:.4f}")

    # ------------------------------------------------------------------
    # 5. Write report
    # ------------------------------------------------------------------
    print("\n[5/5] Writing report to hmm/REPORT.md...")

    # Build BIC comparison table
    bic_header = "| n_components | Log-likelihood | BIC | Converged |"
    bic_sep = "|---|---|---|---|"
    bic_rows = []
    for n_comp in N_COMPONENTS_CANDIDATES:
        r = results[n_comp]
        marker = " *" if n_comp == best_n else ""
        bic_rows.append(
            f"| {n_comp}{marker} | {r['ll']:,.1f} | {r['bic']:,.1f} | {r['converged']} |"
        )
    bic_table = "\n".join([bic_header, bic_sep] + bic_rows)

    # State-conditional response rates table
    sr_header = "| State | N trials (train) | P(target_response=True) |"
    sr_sep = "|---|---|---|"
    sr_rows = []
    for s in range(best_n):
        count = int((train_states == s).sum())
        sr_rows.append(f"| {s} | {count:,} | {state_probs[s]:.4f} |")
    sr_table = "\n".join([sr_header, sr_sep] + sr_rows)

    report = f"""\
# HMM Results

## Model selection

{bic_table}

\\* = selected model

## Selected model: {best_n} hidden states

### Transition matrix

{format_matrix(model.transmat_)}

### Emission means per state

{format_means(model.means_, FEATURE_COLS)}

### State-conditional response rates (training data)

{sr_table}

## Test-set evaluation

| Metric | Value |
|---|---|
| Accuracy | {acc:.4f} |
| Balanced accuracy | {bal_acc:.4f} |
| AUC-ROC | {auc:.4f} |
| Log-loss | {ll_score:.4f} |

## Interpretation

The HMM discovers {best_n} latent states from the trial sequence data. Each state
corresponds to a distinct behavioral pattern characterized by different emission
means across the 6 features. The transition matrix captures how mice switch between
these modes over the course of a session, reflecting the block-based structure of
the context-switching task.

Prediction is performed by decoding the most likely state sequence on test data
and mapping each state to its training-set response probability. This approach
leverages the temporal structure of the task rather than treating trials as
independent observations.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"  Report written to {REPORT_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
