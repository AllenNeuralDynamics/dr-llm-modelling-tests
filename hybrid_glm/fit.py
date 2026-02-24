"""
Hybrid GLM: Context HMM posterior + Logistic Regression.

Combines the context HMM's inferred context posterior with a logistic
regression. The HMM is re-fit on training data to obtain P(visual context |
observations up to t) via forward filtering, then this continuous feature
(plus interactions with stimulus indicators) is added to the standard boolean
features in a logistic regression.

Run from project root:
    uv run python hybrid_glm/fit.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
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
STIMULUS_COLS = [
    "is_vis_target",
    "is_vis_non_target",
    "is_aud_target",
    "is_aud_non_target",
]
STIMULUS_NAMES = ["vis_target", "vis_non_target", "aud_target", "aud_non_target"]
TARGET_COL = "target_response"

N_STATES = 2
N_STIMULI = 4
N_HMM_PARAMS = 1 + N_STATES * N_STIMULI  # 9 total
N_RESTARTS = 3
CLIP_EPS = 1e-12

RANDOM_SEED = 42


# ===========================================================================
# Context HMM (self-contained reimplementation)
# ===========================================================================


def _logsumexp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    c = x.max()
    if np.isinf(c):
        return -np.inf
    return c + np.log(np.sum(np.exp(x - c)))


# ---------------------------------------------------------------------------
# Data encoding
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    """Load a parquet file and return a DataFrame."""
    df = pd.read_parquet(path)
    print(f"  Loaded {path.name}: {len(df):,} trials, {df['subject_id'].nunique()} subjects")
    return df


def encode_trials(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Encode trial data into arrays for the HMM.

    Returns
    -------
    stimuli : int array (N,)
        Stimulus index 0-3 for each trial.
    responses : int array (N,)
        1 if the mouse licked, 0 otherwise.
    lengths : list[int]
        Number of trials per subject (preserving row order).
    """
    stim_matrix = df[STIMULUS_COLS].values.astype(int)  # (N, 4)
    stimuli = np.argmax(stim_matrix, axis=1)  # (N,)

    responses = df[TARGET_COL].values.astype(int)  # (N,)

    # Compute per-subject lengths
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

    return stimuli, responses, lengths


# ---------------------------------------------------------------------------
# Parameter packing / unpacking (logit space for unconstrained optimization)
# ---------------------------------------------------------------------------

def pack_params(p_switch: float, resp_probs: np.ndarray) -> np.ndarray:
    """
    Pack parameters into a 1D array in logit (unconstrained) space.

    Parameters
    ----------
    p_switch : float
        Context switch probability (0, 1).
    resp_probs : array (2, 4)
        Response probabilities for each state x stimulus.

    Returns
    -------
    params : array (9,)
    """
    p_switch_clipped = np.clip(p_switch, CLIP_EPS, 1 - CLIP_EPS)
    resp_clipped = np.clip(resp_probs, CLIP_EPS, 1 - CLIP_EPS)
    return np.concatenate([
        [logit(p_switch_clipped)],
        logit(resp_clipped).ravel(),
    ])


def unpack_params(params: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Unpack a 1D parameter vector from logit space.

    Returns
    -------
    p_switch : float
    resp_probs : array (2, 4)
    """
    p_switch = float(expit(params[0]))
    resp_probs = expit(params[1:]).reshape(N_STATES, N_STIMULI)
    return p_switch, resp_probs


def get_transition_matrix(p_switch: float) -> np.ndarray:
    """Build the 2x2 symmetric transition matrix."""
    return np.array([
        [1 - p_switch, p_switch],
        [p_switch, 1 - p_switch],
    ])


# ---------------------------------------------------------------------------
# Forward algorithm (log-space for numerical stability)
# ---------------------------------------------------------------------------

def forward_log_likelihood(
    params: np.ndarray,
    stimuli: np.ndarray,
    responses: np.ndarray,
    lengths: list[int],
) -> float:
    """
    Compute total log-likelihood across all subjects using the forward algorithm.

    Uses log-space arithmetic throughout for numerical stability.
    """
    p_switch, resp_probs = unpack_params(params)
    log_trans = np.log(np.clip(get_transition_matrix(p_switch), CLIP_EPS, None))

    # Precompute log emission probabilities for all trials
    rp_clipped = np.clip(resp_probs, CLIP_EPS, 1 - CLIP_EPS)
    log_rp = np.log(rp_clipped)        # (2, 4)
    log_1mrp = np.log(1 - rp_clipped)  # (2, 4)

    # For each trial: log_emit[t, k] = log P(obs_t | state_k)
    log_emit = (
        responses[:, None] * log_rp[:, stimuli].T
        + (1 - responses[:, None]) * log_1mrp[:, stimuli].T
    )  # (N, 2)

    # Uniform initial state distribution
    log_pi = np.log(np.array([0.5, 0.5]))

    total_ll = 0.0
    idx = 0
    for length in lengths:
        # Forward pass for this subject
        alpha = log_pi + log_emit[idx]  # (2,)

        for t in range(idx + 1, idx + length):
            alpha_new = np.empty(N_STATES)
            for k in range(N_STATES):
                alpha_new[k] = _logsumexp(alpha + log_trans[:, k]) + log_emit[t, k]
            alpha = alpha_new

        total_ll += _logsumexp(alpha)
        idx += length

    return total_ll


def neg_log_likelihood(
    params: np.ndarray,
    stimuli: np.ndarray,
    responses: np.ndarray,
    lengths: list[int],
) -> float:
    """Negative log-likelihood for minimization."""
    return -forward_log_likelihood(params, stimuli, responses, lengths)


# ---------------------------------------------------------------------------
# Forward filtering for online prediction
# ---------------------------------------------------------------------------

def forward_filter(
    p_switch: float,
    resp_probs: np.ndarray,
    stimuli: np.ndarray,
    responses: np.ndarray,
    lengths: list[int],
) -> np.ndarray:
    """
    Forward filtering: compute P(state_t=k | obs_1..t) for each trial.

    This is the causal posterior used for prediction (no future information).

    Returns
    -------
    filtered : array (N, 2)
        Filtered state probabilities.
    """
    trans = get_transition_matrix(p_switch)
    log_trans = np.log(np.clip(trans, CLIP_EPS, None))

    rp_clipped = np.clip(resp_probs, CLIP_EPS, 1 - CLIP_EPS)
    log_rp = np.log(rp_clipped)
    log_1mrp = np.log(1 - rp_clipped)

    log_emit = (
        responses[:, None] * log_rp[:, stimuli].T
        + (1 - responses[:, None]) * log_1mrp[:, stimuli].T
    )  # (N, 2)

    log_pi = np.log(np.array([0.5, 0.5]))
    N = len(stimuli)
    filtered = np.zeros((N, N_STATES))

    idx = 0
    for length in lengths:
        # Forward pass, normalizing at each step to get filtered probabilities
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


# ===========================================================================
# Feature construction
# ===========================================================================

def build_features(
    df: pd.DataFrame,
    p_vis_context: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    Build the feature matrix for the logistic regression.

    Features:
        - 6 original boolean columns
        - p_vis_context (continuous, from HMM forward filtering)
        - 4 interaction terms: p_vis_context x each stimulus indicator

    Returns
    -------
    X : array (N, 11)
    feature_names : list[str]
    """
    X_base = df[FEATURE_COLS].values.astype(float)  # (N, 6)

    # Stimulus indicators for interactions
    stim_indicators = df[STIMULUS_COLS].values.astype(float)  # (N, 4)

    # Interaction terms: p_vis_context * each stimulus indicator
    interactions = p_vis_context[:, None] * stim_indicators  # (N, 4)

    X = np.column_stack([X_base, p_vis_context, interactions])

    feature_names = list(FEATURE_COLS) + ["p_vis_context"] + [
        f"p_vis_context x {name}" for name in STIMULUS_NAMES
    ]

    return X, feature_names


# ===========================================================================
# Report writing
# ===========================================================================

def write_report(
    p_switch: float,
    resp_probs: np.ndarray,
    model: LogisticRegression,
    feature_names: list[str],
    train_metrics: dict[str, object],
    test_metrics: dict[str, object],
    n_train: int,
    n_test: int,
    n_train_subjects: int,
    n_test_subjects: int,
) -> None:
    """Write REPORT.md with all results."""
    lines: list[str] = []
    w = lines.append

    w("# Hybrid GLM (Context HMM + Logistic Regression) -- Results")
    w("")
    w("*Auto-generated by `hybrid_glm/fit.py`.*")
    w("")

    # ----- Data summary -----
    w("## Data")
    w("")
    w(f"- Training trials: {n_train:,} ({n_train_subjects} subjects)")
    w(f"- Test trials: {n_test:,} ({n_test_subjects} subjects)")
    w(f"- Features: {len(feature_names)} (6 boolean + 1 continuous + 4 interactions)")
    w("")

    # ----- HMM parameters -----
    w("## Context HMM fitted parameters")
    w("")
    w(f"- `p_switch = {p_switch:.6f}`")
    w(f"- Expected block length: {1 / p_switch:.1f} trials (= 1 / p_switch)")
    w("")

    # Transition matrix
    trans = get_transition_matrix(p_switch)
    w("### Transition matrix")
    w("")
    w("| | Visual context | Auditory context |")
    w("|---|---|---|")
    w(f"| **Visual context** | {trans[0, 0]:.6f} | {trans[0, 1]:.6f} |")
    w(f"| **Auditory context** | {trans[1, 0]:.6f} | {trans[1, 1]:.6f} |")
    w("")

    # Response probabilities
    w("### Response probabilities: P(lick | state, stimulus)")
    w("")
    w("| Stimulus | Visual context (State 0) | Auditory context (State 1) |")
    w("|---|---|---|")
    for s_idx, s_name in enumerate(STIMULUS_NAMES):
        w(f"| {s_name} | {resp_probs[0, s_idx]:.4f} | {resp_probs[1, s_idx]:.4f} |")
    w("")

    # ----- Logistic regression coefficients -----
    w("## Logistic regression coefficients")
    w("")
    intercept = model.intercept_[0]
    coefficients = model.coef_[0]
    odds_ratios = np.exp(coefficients)

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "odds_ratio": odds_ratios,
    }).sort_values("coefficient", key=abs, ascending=False)

    w("Sorted by absolute coefficient value (largest effect first).")
    w("")
    w("| Feature | Coefficient | Odds Ratio |")
    w("|---|---|---|")
    for _, row in coef_df.iterrows():
        w(f"| {row['feature']} | {row['coefficient']:+.4f} | {row['odds_ratio']:.4f} |")
    w(f"| *(intercept)* | {intercept:+.4f} | -- |")
    w("")

    # ----- Training-set metrics -----
    w("## Training-set metrics")
    w("")
    w("| Metric | Value |")
    w("|---|---|")
    w(f"| Accuracy | {train_metrics['accuracy']:.4f} |")
    w(f"| Balanced accuracy | {train_metrics['balanced_accuracy']:.4f} |")
    w(f"| AUC-ROC | {train_metrics['auc_roc']:.4f} |")
    w(f"| Log-loss | {train_metrics['log_loss']:.4f} |")
    w("")

    # Training confusion matrix
    cm_train = train_metrics["confusion_matrix"]
    tn, fp, fn, tp = cm_train.ravel()
    w("### Training confusion matrix")
    w("")
    w("|  | Predicted Negative | Predicted Positive |")
    w("|---|---|---|")
    w(f"| **Actual Negative** | {tn:,} | {fp:,} |")
    w(f"| **Actual Positive** | {fn:,} | {tp:,} |")
    w("")

    # ----- Test-set metrics -----
    w("## Test-set metrics")
    w("")
    w("| Metric | Value |")
    w("|---|---|")
    w(f"| Accuracy | {test_metrics['accuracy']:.4f} |")
    w(f"| Balanced accuracy | {test_metrics['balanced_accuracy']:.4f} |")
    w(f"| AUC-ROC | {test_metrics['auc_roc']:.4f} |")
    w(f"| Log-loss | {test_metrics['log_loss']:.4f} |")
    w("")

    # Test confusion matrix
    cm_test = test_metrics["confusion_matrix"]
    tn, fp, fn, tp = cm_test.ravel()
    w("### Test confusion matrix")
    w("")
    w("|  | Predicted Negative | Predicted Positive |")
    w("|---|---|---|")
    w(f"| **Actual Negative** | {tn:,} | {fp:,} |")
    w(f"| **Actual Positive** | {fn:,} | {tp:,} |")
    w("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report written to {REPORT_PATH}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("=" * 65)
    print("Hybrid GLM: Context HMM posterior + Logistic Regression")
    print("=" * 65)

    rng = np.random.RandomState(RANDOM_SEED)

    # ------------------------------------------------------------------
    # 1. Load and encode data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading data...")
    train_df = load_data(DATA_DIR / "training_set.parquet")
    test_df = load_data(DATA_DIR / "test_set.parquet")

    train_stimuli, train_responses, train_lengths = encode_trials(train_df)
    test_stimuli, test_responses, test_lengths = encode_trials(test_df)

    n_train = len(train_stimuli)
    n_test = len(test_stimuli)
    n_train_subjects = len(train_lengths)
    n_test_subjects = len(test_lengths)
    print(f"  Training: {n_train_subjects} subjects, {n_train:,} trials")
    print(f"  Test: {n_test_subjects} subjects, {n_test:,} trials")

    # ------------------------------------------------------------------
    # 2. Fit context HMM on training data
    # ------------------------------------------------------------------
    print("\n[2/6] Fitting context HMM on training data (multi-start, L-BFGS-B)...")

    best_nll = np.inf
    best_params = None

    for restart in range(N_RESTARTS):
        # Initialization
        if restart == 0:
            # Informed start: small switch prob, sensible response probs
            p_switch_init = 0.02
            resp_init = np.array([
                [0.7, 0.1, 0.3, 0.1],  # Visual context
                [0.3, 0.1, 0.7, 0.1],  # Auditory context
            ])
        else:
            # Random start
            p_switch_init = rng.uniform(0.005, 0.1)
            resp_init = rng.uniform(0.1, 0.9, size=(N_STATES, N_STIMULI))

        params_init = pack_params(p_switch_init, resp_init)

        print(f"  Restart {restart + 1}/{N_RESTARTS}: ", end="", flush=True)

        result = minimize(
            neg_log_likelihood,
            params_init,
            args=(train_stimuli, train_responses, train_lengths),
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-10, "disp": False},
        )

        nll = result.fun
        p_sw, rp = unpack_params(result.x)
        print(f"NLL={nll:,.2f}, p_switch={p_sw:.6f}, converged={result.success}")

        if nll < best_nll:
            best_nll = nll
            best_params = result.x.copy()

    # Unpack best parameters
    p_switch, resp_probs = unpack_params(best_params)
    print(f"\n  Best NLL: {best_nll:,.2f}")
    print(f"  p_switch: {p_switch:.6f} (expected block length: {1 / p_switch:.1f} trials)")

    # Display fitted response probabilities
    print("\n  Fitted response probabilities P(lick | state, stimulus):")
    print(f"  {'Stimulus':<18s}  {'Visual ctx':>12s}  {'Auditory ctx':>12s}")
    print(f"  {'-' * 18}  {'-' * 12}  {'-' * 12}")
    for s_idx, s_name in enumerate(STIMULUS_NAMES):
        print(f"  {s_name:<18s}  {resp_probs[0, s_idx]:12.4f}  {resp_probs[1, s_idx]:12.4f}")

    # ------------------------------------------------------------------
    # 3. Forward filter on train and test data
    # ------------------------------------------------------------------
    print("\n[3/6] Running forward filtering on train and test data...")

    train_filtered = forward_filter(
        p_switch, resp_probs, train_stimuli, train_responses, train_lengths
    )
    test_filtered = forward_filter(
        p_switch, resp_probs, test_stimuli, test_responses, test_lengths
    )

    # P(visual context) = filtered[:, 0]
    train_p_vis = train_filtered[:, 0]
    test_p_vis = test_filtered[:, 0]

    print(f"  Train p_vis_context: mean={train_p_vis.mean():.4f}, "
          f"std={train_p_vis.std():.4f}")
    print(f"  Test  p_vis_context: mean={test_p_vis.mean():.4f}, "
          f"std={test_p_vis.std():.4f}")

    # ------------------------------------------------------------------
    # 4. Build features and fit logistic regression
    # ------------------------------------------------------------------
    print("\n[4/6] Building features and fitting logistic regression...")

    X_train, feature_names = build_features(train_df, train_p_vis)
    X_test, _ = build_features(test_df, test_p_vis)

    y_train = train_df[TARGET_COL].values.astype(int)
    y_test = test_df[TARGET_COL].values.astype(int)

    print(f"  Feature matrix shape: {X_train.shape}")
    print(f"  Features: {feature_names}")

    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X_train, y_train)
    print("  Logistic regression fitted.")

    # Print coefficients
    intercept = model.intercept_[0]
    coefficients = model.coef_[0]
    odds_ratios = np.exp(coefficients)

    print(f"\n  Coefficients (intercept={intercept:+.4f}):")
    for name, coef, odds in zip(feature_names, coefficients, odds_ratios):
        print(f"    {name:35s}  coef={coef:+.4f}  OR={odds:.4f}")

    # ------------------------------------------------------------------
    # 5. Evaluate on training set
    # ------------------------------------------------------------------
    print("\n[5/6] Evaluating on training set...")

    train_y_pred = model.predict(X_train)
    train_y_prob = model.predict_proba(X_train)[:, 1]

    train_metrics = {
        "accuracy": accuracy_score(y_train, train_y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_train, train_y_pred),
        "auc_roc": roc_auc_score(y_train, train_y_prob),
        "log_loss": log_loss(y_train, train_y_prob),
        "confusion_matrix": confusion_matrix(y_train, train_y_pred),
    }

    for name in ["accuracy", "balanced_accuracy", "auc_roc", "log_loss"]:
        print(f"  {name:<20s}: {train_metrics[name]:.4f}")
    print(f"  Confusion matrix:\n{train_metrics['confusion_matrix']}")

    # ------------------------------------------------------------------
    # 6. Evaluate on test set and write report
    # ------------------------------------------------------------------
    print("\n[6/6] Evaluating on test set and writing report...")

    test_y_pred = model.predict(X_test)
    test_y_prob = model.predict_proba(X_test)[:, 1]

    test_metrics = {
        "accuracy": accuracy_score(y_test, test_y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, test_y_pred),
        "auc_roc": roc_auc_score(y_test, test_y_prob),
        "log_loss": log_loss(y_test, test_y_prob),
        "confusion_matrix": confusion_matrix(y_test, test_y_pred),
    }

    for name in ["accuracy", "balanced_accuracy", "auc_roc", "log_loss"]:
        print(f"  {name:<20s}: {test_metrics[name]:.4f}")
    print(f"  Confusion matrix:\n{test_metrics['confusion_matrix']}")

    # Write report
    write_report(
        p_switch=p_switch,
        resp_probs=resp_probs,
        model=model,
        feature_names=feature_names,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        n_train=n_train,
        n_test=n_test,
        n_train_subjects=n_train_subjects,
        n_test_subjects=n_test_subjects,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
