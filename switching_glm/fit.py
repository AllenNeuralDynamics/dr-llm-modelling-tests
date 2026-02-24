"""
Switching GLM: regime-switching logistic regression for context-switching task.

Two latent states (visual context, auditory context), each with its own full
logistic regression over all 6 features.  State transitions governed by a
single symmetric p_switch parameter.  Fit via EM (forward-backward E-step,
weighted logistic regression M-step).

Run from project root:
    uv run python switching_glm/fit.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
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
HERE = Path(__file__).parent
DATA_DIR = HERE.parent / "data"
REPORT_PATH = HERE / "REPORT.md"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "previous_response",
    "previous_reward",
    "is_vis_target",
    "is_vis_non_target",
    "is_aud_target",
    "is_aud_non_target",
]
TARGET_COL = "target_response"
STATE_NAMES = ["Visual context", "Auditory context"]

N_STATES = 2
N_FEATURES = len(FEATURE_COLS)  # 6
N_COEFS = N_FEATURES + 1  # 7 (with intercept)
N_PARAMS = 1 + N_STATES * N_COEFS  # 15

N_RESTARTS = 3
MAX_EM_ITER = 50
EM_TOL = 1e-4
CLIP_EPS = 1e-12

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    """Load a parquet file, sort by subject_id, return DataFrame."""
    df = pd.read_parquet(path)
    df = df.sort_values("subject_id").reset_index(drop=True)
    print(f"  Loaded {path.name}: {len(df):,} trials, {df['subject_id'].nunique()} subjects")
    return df


def prepare_arrays(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Build feature matrix X (with intercept), response vector y, and
    per-subject lengths array.

    Returns
    -------
    X : array (N, 7)  -- column 0 is intercept (=1)
    y : array (N,)     -- binary response
    lengths : list[int] -- trials per subject (in row order)
    """
    raw = df[FEATURE_COLS].values.astype(float)
    X = np.column_stack([np.ones(len(raw)), raw])  # (N, 7)
    y = df[TARGET_COL].values.astype(int)

    # Per-subject lengths (data is already sorted by subject_id)
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

    return X, y, lengths


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

def _logsumexp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp over a 1-D array."""
    c = x.max()
    if np.isinf(c):
        return -np.inf
    return float(c + np.log(np.sum(np.exp(x - c))))


def _log_bernoulli(y: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """
    Log-probability of Bernoulli outcomes given logits.

    log P(y | logits) = y * log(sigma) + (1-y) * log(1-sigma)

    Uses the identity  log sigma(z) = -log(1+exp(-z))  and
    log(1 - sigma(z)) = -log(1+exp(z))  for numerical stability.
    """
    # Stable log-sigmoid: -softplus(-z) = z - softplus(z)
    # log sigma(z)     = -softplus(-z)
    # log(1-sigma(z))  = -softplus(z)
    def _softplus(z: np.ndarray) -> np.ndarray:
        # softplus(z) = log(1+exp(z)), stable version
        return np.where(z > 20, z, np.log1p(np.exp(np.clip(z, -500, 20))))

    log_sig = -_softplus(-logits)
    log_1msig = -_softplus(logits)
    return y * log_sig + (1 - y) * log_1msig


# ---------------------------------------------------------------------------
# Forward-backward
# ---------------------------------------------------------------------------

def forward_backward(
    X: np.ndarray,
    y: np.ndarray,
    lengths: list[int],
    betas: np.ndarray,
    p_switch: float,
) -> tuple[np.ndarray, float, float]:
    """
    Forward-backward algorithm in log-space.

    Parameters
    ----------
    X : (N, 7) feature matrix (with intercept)
    y : (N,)   binary responses
    lengths : per-subject sequence lengths
    betas : (2, 7) coefficient matrices
    p_switch : transition probability

    Returns
    -------
    gamma : (N, 2) posterior state probabilities
    total_ll : total log-likelihood
    expected_switches : expected number of state switches (for M-step)
    """
    N = len(y)
    log_trans = np.log(np.clip(
        np.array([[1 - p_switch, p_switch],
                  [p_switch, 1 - p_switch]]),
        CLIP_EPS, None,
    ))
    log_pi = np.log(np.array([0.5, 0.5]))

    # Precompute log emission for all trials and states: (N, 2)
    logits = X @ betas.T  # (N, 2)
    log_emit = np.column_stack([
        _log_bernoulli(y.astype(float), logits[:, k])
        for k in range(N_STATES)
    ])  # (N, 2)

    gamma = np.zeros((N, N_STATES))
    total_ll = 0.0
    expected_switches = 0.0

    idx = 0
    for length in lengths:
        end = idx + length

        # --- Forward pass ---
        log_alpha = np.zeros((length, N_STATES))
        log_alpha[0] = log_pi + log_emit[idx]
        for t in range(1, length):
            for k in range(N_STATES):
                log_alpha[t, k] = (
                    _logsumexp(log_alpha[t - 1] + log_trans[:, k])
                    + log_emit[idx + t, k]
                )

        # --- Backward pass ---
        log_beta = np.zeros((length, N_STATES))
        # log_beta[-1] = 0  (log(1))
        for t in range(length - 2, -1, -1):
            for k in range(N_STATES):
                log_beta[t, k] = _logsumexp(
                    log_trans[k, :] + log_emit[idx + t + 1, :] + log_beta[t + 1]
                )

        # --- Posterior gamma ---
        log_gamma = log_alpha + log_beta  # (length, 2)
        for t in range(length):
            log_norm = _logsumexp(log_gamma[t])
            gamma[idx + t] = np.exp(log_gamma[t] - log_norm)

        # --- Log-likelihood contribution ---
        total_ll += _logsumexp(log_alpha[-1])

        # --- Expected transitions (xi) for p_switch update ---
        # xi[t, j, k] = alpha[t,j] * trans[j,k] * emit[t+1,k] * beta[t+1,k] / P(obs)
        # We only need sum over t of xi where j != k
        log_evidence = _logsumexp(log_alpha[-1])
        for t in range(length - 1):
            for j in range(N_STATES):
                for k in range(N_STATES):
                    if j != k:
                        log_xi_tjk = (
                            log_alpha[t, j]
                            + log_trans[j, k]
                            + log_emit[idx + t + 1, k]
                            + log_beta[t + 1, k]
                            - log_evidence
                        )
                        expected_switches += np.exp(log_xi_tjk)

        idx += length

    return gamma, total_ll, expected_switches


# ---------------------------------------------------------------------------
# Forward filtering (causal, for prediction)
# ---------------------------------------------------------------------------

def forward_filter(
    X: np.ndarray,
    y: np.ndarray,
    lengths: list[int],
    betas: np.ndarray,
    p_switch: float,
) -> np.ndarray:
    """
    Forward filtering: P(state_t=k | obs_1..t) for each trial.

    Returns
    -------
    filtered : (N, 2)
    """
    N = len(y)
    log_trans = np.log(np.clip(
        np.array([[1 - p_switch, p_switch],
                  [p_switch, 1 - p_switch]]),
        CLIP_EPS, None,
    ))
    log_pi = np.log(np.array([0.5, 0.5]))

    logits = X @ betas.T  # (N, 2)
    log_emit = np.column_stack([
        _log_bernoulli(y.astype(float), logits[:, k])
        for k in range(N_STATES)
    ])

    filtered = np.zeros((N, N_STATES))
    idx = 0
    for length in lengths:
        log_alpha = log_pi + log_emit[idx]
        norm = _logsumexp(log_alpha)
        filtered[idx] = np.exp(log_alpha - norm)

        for t in range(1, length):
            log_alpha_new = np.empty(N_STATES)
            for k in range(N_STATES):
                log_alpha_new[k] = (
                    _logsumexp(log_alpha + log_trans[:, k])
                    + log_emit[idx + t, k]
                )
            log_alpha = log_alpha_new
            norm = _logsumexp(log_alpha)
            filtered[idx + t] = np.exp(log_alpha - norm)

        idx += length

    return filtered


# ---------------------------------------------------------------------------
# EM algorithm
# ---------------------------------------------------------------------------

def fit_em(
    X: np.ndarray,
    y: np.ndarray,
    lengths: list[int],
    betas_init: np.ndarray,
    p_switch_init: float,
    verbose: bool = True,
) -> tuple[np.ndarray, float, float, int]:
    """
    Fit the switching GLM via EM.

    Returns
    -------
    betas : (2, 7) fitted coefficients
    p_switch : fitted switch probability
    best_ll : final log-likelihood
    n_iter : number of EM iterations
    """
    betas = betas_init.copy()
    p_switch = p_switch_init
    prev_ll = -np.inf
    total_transitions = sum(L - 1 for L in lengths)

    for iteration in range(1, MAX_EM_ITER + 1):
        # ----- E-step -----
        gamma, ll, expected_switches = forward_backward(
            X, y, lengths, betas, p_switch,
        )

        if verbose:
            print(f"    EM iter {iteration:3d}:  LL = {ll:,.2f}")

        # Check convergence
        if iteration > 1 and abs(ll - prev_ll) < EM_TOL:
            if verbose:
                print(f"    Converged (delta LL = {abs(ll - prev_ll):.2e})")
            break
        prev_ll = ll

        # ----- M-step: p_switch -----
        p_switch = np.clip(expected_switches / total_transitions, CLIP_EPS, 1 - CLIP_EPS)

        # ----- M-step: betas via weighted logistic regression -----
        for k in range(N_STATES):
            weights = gamma[:, k]
            # Skip if weights are effectively zero
            if weights.sum() < CLIP_EPS:
                continue
            clf = LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                C=1e6,  # very weak regularization (near-unregularized)
                fit_intercept=False,  # intercept is already in X
            )
            clf.fit(X, y, sample_weight=weights)
            betas[k] = clf.coef_[0]

    return betas, p_switch, ll, iteration


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_probs(
    X: np.ndarray,
    y: np.ndarray,
    lengths: list[int],
    betas: np.ndarray,
    p_switch: float,
) -> np.ndarray:
    """
    Predict P(lick_t) using forward filtering and per-state logistic models.

    P(lick_t) = sum_k P(state_t=k | obs_1..t) * sigmoid(X_t @ beta_k)
    """
    filtered = forward_filter(X, y, lengths, betas, p_switch)
    logits = X @ betas.T  # (N, 2)
    probs_per_state = expit(logits)  # (N, 2)
    y_prob = np.sum(filtered * probs_per_state, axis=1)
    return np.clip(y_prob, CLIP_EPS, 1 - CLIP_EPS)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Compute classification metrics."""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    p_switch: float,
    betas: np.ndarray,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    train_cm: np.ndarray,
    test_cm: np.ndarray,
    n_train: int,
    n_test: int,
    n_train_subjects: int,
    n_test_subjects: int,
    n_em_iter: int,
) -> None:
    """Write REPORT.md with full results."""
    feature_names = ["intercept"] + FEATURE_COLS

    lines: list[str] = []
    w = lines.append

    w("# Switching GLM (Regime-Switching Logistic Regression) -- Results")
    w("")
    w("*Auto-generated by `switching_glm/fit.py`.*")
    w("")

    # --- Model summary ---
    w("## Model summary")
    w("")
    w("- **Hidden states**: 2 (visual context, auditory context)")
    w("- **Emission**: Full logistic regression per state (7 coefficients each)")
    w(f"- **Total parameters**: {N_PARAMS}")
    w(f"- **Training data**: {n_train:,} trials from {n_train_subjects} subjects")
    w(f"- **Test data**: {n_test:,} trials from {n_test_subjects} subjects")
    w(f"- **EM iterations to convergence**: {n_em_iter}")
    w("")

    # --- p_switch ---
    w("## Switch probability")
    w("")
    expected_block = 1.0 / p_switch if p_switch > 0 else float("inf")
    w(f"- `p_switch = {p_switch:.6f}`")
    w(f"- Expected block length: {expected_block:.1f} trials (= 1 / p_switch)")
    w("")

    # --- Coefficient tables ---
    w("## Coefficient tables")
    w("")
    for k in range(N_STATES):
        w(f"### {STATE_NAMES[k]} (State {k})")
        w("")
        w("| Feature | Coefficient | Odds Ratio |")
        w("|---|---|---|")
        for j, name in enumerate(feature_names):
            coef = betas[k, j]
            odds = np.exp(coef)
            w(f"| {name} | {coef:+.4f} | {odds:.4f} |")
        w("")

    # --- Coefficient comparison ---
    w("## Coefficient comparison across states")
    w("")
    w("The difference `beta_visual - beta_auditory` highlights how feature effects")
    w("change between contexts.")
    w("")
    w("| Feature | Visual ctx | Auditory ctx | Difference |")
    w("|---|---|---|---|")
    for j, name in enumerate(feature_names):
        diff = betas[0, j] - betas[1, j]
        w(f"| {name} | {betas[0, j]:+.4f} | {betas[1, j]:+.4f} | {diff:+.4f} |")
    w("")

    # --- Training metrics ---
    w("## Training-set metrics")
    w("")
    w("| Metric | Value |")
    w("|---|---|")
    w(f"| Accuracy | {train_metrics['accuracy']:.4f} |")
    w(f"| Balanced accuracy | {train_metrics['balanced_accuracy']:.4f} |")
    w(f"| AUC-ROC | {train_metrics['auc_roc']:.4f} |")
    w(f"| Log-loss | {train_metrics['log_loss']:.4f} |")
    w("")

    tn, fp, fn, tp = train_cm.ravel()
    w("### Training confusion matrix")
    w("")
    w("|  | Predicted Negative | Predicted Positive |")
    w("|---|---|---|")
    w(f"| **Actual Negative** | {tn:,} | {fp:,} |")
    w(f"| **Actual Positive** | {fn:,} | {tp:,} |")
    w("")

    # --- Test metrics ---
    w("## Test-set metrics")
    w("")
    w("| Metric | Value |")
    w("|---|---|")
    w(f"| Accuracy | {test_metrics['accuracy']:.4f} |")
    w(f"| Balanced accuracy | {test_metrics['balanced_accuracy']:.4f} |")
    w(f"| AUC-ROC | {test_metrics['auc_roc']:.4f} |")
    w(f"| Log-loss | {test_metrics['log_loss']:.4f} |")
    w("")

    tn, fp, fn, tp = test_cm.ravel()
    w("### Test confusion matrix")
    w("")
    w("|  | Predicted Negative | Predicted Positive |")
    w("|---|---|---|")
    w(f"| **Actual Negative** | {tn:,} | {fp:,} |")
    w(f"| **Actual Positive** | {fn:,} | {tp:,} |")
    w("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report written to {REPORT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Switching GLM: regime-switching logistic regression")
    print("=" * 70)

    rng = np.random.RandomState(RANDOM_SEED)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data...")
    train_df = load_data(DATA_DIR / "training_set.parquet")
    test_df = load_data(DATA_DIR / "test_set.parquet")

    X_train, y_train, train_lengths = prepare_arrays(train_df)
    X_test, y_test, test_lengths = prepare_arrays(test_df)

    print(f"  Training: {len(train_lengths)} subjects, {len(y_train):,} trials")
    print(f"  Test:     {len(test_lengths)} subjects, {len(y_test):,} trials")

    # ------------------------------------------------------------------
    # 2. Initial logistic regression for beta initialization
    # ------------------------------------------------------------------
    print("\n[2/5] Fitting baseline logistic regression for initialization...")
    baseline_clf = LogisticRegression(
        solver="lbfgs", max_iter=1000, C=1e6, fit_intercept=False,
    )
    baseline_clf.fit(X_train, y_train)
    beta_base = baseline_clf.coef_[0]  # (7,)
    print(f"  Baseline coefficients: {beta_base}")

    # ------------------------------------------------------------------
    # 3. Multi-start EM
    # ------------------------------------------------------------------
    print("\n[3/5] Running EM with multi-start...")

    best_ll = -np.inf
    best_betas = None
    best_p_switch = None
    best_n_iter = None

    for restart in range(N_RESTARTS):
        print(f"\n  --- Restart {restart + 1}/{N_RESTARTS} ---")

        if restart == 0:
            # Informed start: perturb baseline slightly
            p_switch_init = 0.02
            perturbation = rng.normal(0, 0.1, size=(N_STATES, N_COEFS))
            betas_init = np.stack([beta_base, beta_base]) + perturbation
        else:
            # Random start
            p_switch_init = rng.uniform(0.005, 0.1)
            perturbation = rng.normal(0, 0.5, size=(N_STATES, N_COEFS))
            betas_init = np.stack([beta_base, beta_base]) + perturbation

        betas_fit, p_switch_fit, ll_fit, n_iter_fit = fit_em(
            X_train, y_train, train_lengths,
            betas_init, p_switch_init,
            verbose=True,
        )

        print(f"  Result: LL = {ll_fit:,.2f}, p_switch = {p_switch_fit:.6f}, "
              f"iters = {n_iter_fit}")

        if ll_fit > best_ll:
            best_ll = ll_fit
            best_betas = betas_fit.copy()
            best_p_switch = p_switch_fit
            best_n_iter = n_iter_fit

    print(f"\n  Best LL across restarts: {best_ll:,.2f}")
    print(f"  Best p_switch: {best_p_switch:.6f} "
          f"(expected block length: {1.0 / best_p_switch:.1f} trials)")

    # ------------------------------------------------------------------
    # 4. Display fitted coefficients
    # ------------------------------------------------------------------
    print("\n[4/5] Fitted coefficients:")
    feature_names = ["intercept"] + FEATURE_COLS
    print(f"  {'Feature':<22s}  {'Visual ctx':>12s}  {'Auditory ctx':>12s}  {'Diff':>10s}")
    print(f"  {'-' * 22}  {'-' * 12}  {'-' * 12}  {'-' * 10}")
    for j, name in enumerate(feature_names):
        diff = best_betas[0, j] - best_betas[1, j]
        print(f"  {name:<22s}  {best_betas[0, j]:+12.4f}  {best_betas[1, j]:+12.4f}  {diff:+10.4f}")

    # ------------------------------------------------------------------
    # 5. Evaluate and write report
    # ------------------------------------------------------------------
    print("\n[5/5] Evaluating and writing report...")

    # Training metrics
    print("  Training set:")
    train_probs = predict_probs(
        X_train, y_train, train_lengths, best_betas, best_p_switch,
    )
    train_metrics = compute_metrics(y_train, train_probs)
    train_pred = (train_probs >= 0.5).astype(int)
    train_cm = confusion_matrix(y_train, train_pred)
    for name, val in train_metrics.items():
        print(f"    {name:<20s}: {val:.4f}")

    # Test metrics
    print("  Test set:")
    test_probs = predict_probs(
        X_test, y_test, test_lengths, best_betas, best_p_switch,
    )
    test_metrics = compute_metrics(y_test, test_probs)
    test_pred = (test_probs >= 0.5).astype(int)
    test_cm = confusion_matrix(y_test, test_pred)
    for name, val in test_metrics.items():
        print(f"    {name:<20s}: {val:.4f}")

    # Write report
    write_report(
        p_switch=best_p_switch,
        betas=best_betas,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        train_cm=train_cm,
        test_cm=test_cm,
        n_train=len(y_train),
        n_test=len(y_test),
        n_train_subjects=len(train_lengths),
        n_test_subjects=len(test_lengths),
        n_em_iter=best_n_iter,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
