"""
Bayesian context-inference RL model v2 with FIXED reward likelihoods.

The previous model (context_rl) estimated p_reward_correct and p_reward_incorrect as
free parameters, but they collapsed to boundary values (0.5 and 0.0) making the
Bayesian update barely functional.

This v2 model fixes the reward likelihoods to their true task values:
  - P(reward | responded to target in correct context) = 1.0
  - P(reward | responded to target in wrong context) = 0.0

This makes the Bayesian update maximally sharp: a single rewarded target trial
provides definitive evidence about context.  The model has 4 free parameters per
subject (beta, bias, gamma, v_nontgt) instead of 6.

Run from project root:
    uv run python context_rl_v2/fit.py
"""

from __future__ import annotations

import time as time_mod
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid
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
DATA_DIR = HERE.parent.parent / "data"
REPORT_PATH = HERE / "REPORT.md"

STIMULUS_COLS = [
    "is_vis_target",
    "is_vis_non_target",
    "is_aud_target",
    "is_aud_non_target",
]

# Stimulus index mapping:
#   0 = vis_target, 1 = vis_non_target, 2 = aud_target, 3 = aud_non_target
STIM_VIS_TGT = 0
STIM_VIS_NONTGT = 1
STIM_AUD_TGT = 2
STIM_AUD_NONTGT = 3

# Parameter order: beta, bias, gamma, v_nontgt
PARAM_NAMES = ["beta", "bias", "gamma", "v_nontgt"]

BOUNDS = [
    (0.1, 20.0),    # beta
    (-10.0, 10.0),  # bias
    (0.001, 0.5),   # gamma
    (-5.0, 0.0),    # v_nontgt
]

# Multi-start initial guesses (4 parameters each)
INIT_GUESSES = [
    [5.0, -1.0, 0.05, -2.0],
    [3.0, -2.0, 0.1, -1.0],
    [8.0, 0.0, 0.02, -3.0],
    [2.0, -1.0, 0.2, -0.5],
]

P_VIS_INIT = 0.5  # initial context belief: maximum uncertainty

# Fixed reward likelihoods (the key change from v1)
# Small epsilon to avoid log(0) in numerical computations
LIK_EPSILON = 0.001
LIK_HIGH = 1.0 - LIK_EPSILON  # 0.999
LIK_LOW = LIK_EPSILON          # 0.001

# Belief clipping bounds
BELIEF_MIN = 0.001
BELIEF_MAX = 0.999


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

def _derive_reward(df: pd.DataFrame) -> np.ndarray:
    """Derive per-trial reward from the *next* trial's ``previous_reward``.

    For the last trial in each subject-session we cannot know the reward, so we
    set it to NaN (the belief update step will be skipped for that trial).
    """
    reward = np.full(len(df), np.nan)
    reward[:-1] = df["previous_reward"].values[1:].astype(float)
    # Boundary between subjects: where subject_id changes, the mapping is
    # invalid -- mark as NaN so we skip the update.
    subj = df["subject_id"].values
    boundary = np.where(subj[:-1] != subj[1:])[0]
    reward[boundary] = np.nan
    return reward


def run_model_forward(
    params: np.ndarray,
    stim_idx: np.ndarray,
    responses: np.ndarray,
    rewards: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the Bayesian context-inference model forward with fixed likelihoods.

    Parameters
    ----------
    params : array of shape (4,)
        [beta, bias, gamma, v_nontgt]
    stim_idx : int array of shape (n_trials,)
        Index (0-3) indicating which stimulus was presented.
    responses : bool/float array of shape (n_trials,)
        Whether the mouse licked (1.0 = lick, 0.0 = no lick).
    rewards : float array of shape (n_trials,)
        1.0 = rewarded, 0.0 = not rewarded, NaN = unknown (skip update).

    Returns
    -------
    p_lick : array of shape (n_trials,)
        Model-predicted probability of licking on each trial.
    p_vis_trace : array of shape (n_trials,)
        Context belief p_vis at each trial (before decision, after prior leak).
    """
    beta, bias, gamma, v_nontgt = params
    n = len(stim_idx)
    p_lick = np.empty(n, dtype=np.float64)
    p_vis_trace = np.empty(n, dtype=np.float64)

    p_vis = P_VIS_INIT  # belief: P(current context = visual-rewarded)

    for t in range(n):
        s = stim_idx[t]

        # --- Prior leak: account for possible context switch ---
        p_vis_prior = gamma * 0.5 + (1.0 - gamma) * p_vis

        # Record the belief used for this trial's decision
        p_vis_trace[t] = p_vis_prior

        # --- Compute value of current stimulus given context belief ---
        if s == STIM_VIS_TGT:
            value = p_vis_prior
        elif s == STIM_AUD_TGT:
            value = 1.0 - p_vis_prior
        else:
            # Non-target stimuli: fixed low value
            value = v_nontgt

        # --- Decision ---
        p_lick[t] = expit(beta * value + bias)

        # --- Bayesian belief update (only if mouse responded AND reward is known) ---
        if not np.isnan(rewards[t]) and responses[t]:
            # Mouse responded to a stimulus -- reward is informative about context
            rewarded = rewards[t] == 1.0

            if s == STIM_VIS_TGT:
                # Responded to visual target
                if rewarded:
                    # Reward confirms visual context
                    lik_vis = LIK_HIGH
                    lik_aud = LIK_LOW
                else:
                    # No reward contradicts visual context
                    lik_vis = LIK_LOW
                    lik_aud = LIK_HIGH

            elif s == STIM_AUD_TGT:
                # Responded to auditory target
                if rewarded:
                    # Reward confirms auditory context
                    lik_vis = LIK_LOW
                    lik_aud = LIK_HIGH
                else:
                    # No reward contradicts auditory context
                    lik_vis = LIK_HIGH
                    lik_aud = LIK_LOW

            else:
                # Responded to non-target: reward is always 0, uninformative
                # (non-targets are never rewarded regardless of context)
                p_vis = p_vis_prior
                continue

            # Bayes' rule
            numerator = p_vis_prior * lik_vis
            denominator = p_vis_prior * lik_vis + (1.0 - p_vis_prior) * lik_aud

            # Clip for numerical stability
            denominator = max(denominator, 1e-12)
            p_vis = numerator / denominator
            p_vis = np.clip(p_vis, BELIEF_MIN, BELIEF_MAX)

        else:
            # No response or unknown reward: belief carries forward with leak
            p_vis = p_vis_prior

    return p_lick, p_vis_trace


def neg_log_likelihood(
    params: np.ndarray,
    stim_idx: np.ndarray,
    responses: np.ndarray,
    rewards: np.ndarray,
) -> float:
    """Negative log-likelihood of observed responses under the model."""
    p_lick, _ = run_model_forward(params, stim_idx, responses, rewards)
    # Clip for numerical stability
    eps = 1e-12
    p_lick = np.clip(p_lick, eps, 1.0 - eps)
    ll = responses * np.log(p_lick) + (1.0 - responses) * np.log(1.0 - p_lick)
    return -np.sum(ll)


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def _stim_index(df: pd.DataFrame) -> np.ndarray:
    """Convert one-hot stimulus columns to a single integer index (0-3)."""
    idx = np.zeros(len(df), dtype=np.int32)
    for i, col in enumerate(STIMULUS_COLS):
        idx[df[col].values] = i
    return idx


def fit_subject(
    stim_idx: np.ndarray,
    responses: np.ndarray,
    rewards: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit model parameters for one subject via multi-start L-BFGS-B.

    Returns
    -------
    best_params : array of shape (4,)
    best_nll : float
    """
    best_nll = np.inf
    best_params = np.array(INIT_GUESSES[0])

    for x0 in INIT_GUESSES:
        try:
            res = minimize(
                neg_log_likelihood,
                x0=x0,
                args=(stim_idx, responses, rewards),
                method="L-BFGS-B",
                bounds=BOUNDS,
                options={"maxiter": 2000, "ftol": 1e-10},
            )
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x
        except Exception:
            continue

    return np.asarray(best_params), best_nll


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, p_pred: np.ndarray) -> dict[str, float]:
    """Compute classification metrics from true labels and predicted probabilities."""
    y_pred = (p_pred >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, p_pred),
        "log_loss": log_loss(y_true, p_pred),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_report(
    param_df: pd.DataFrame,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    example_subject_id: int,
    example_traces: dict[str, np.ndarray],
    n_subjects: int,
) -> None:
    """Write a markdown report to REPORT_PATH."""
    lines: list[str] = []
    lines.append("# Bayesian Context-Inference RL Model v2 (Fixed Likelihoods) -- Results\n")
    lines.append("*Auto-generated by `context_rl_v2/fit.py`*\n")

    lines.append(
        "This model fixes reward likelihoods to their true task values "
        "(P(reward|correct context) = 1.0, P(reward|wrong context) = 0.0) "
        "instead of estimating them as free parameters. The v1 model's "
        "`p_reward_correct` and `p_reward_incorrect` collapsed to boundary "
        "values (0.5 and 0.0), making the Bayesian update barely functional. "
        "With fixed likelihoods, the p_vis trace should now show sharp context "
        "switches after rewarded target trials.\n"
    )

    # -- Fitted parameters summary --
    lines.append("## Fitted parameters (across all subjects)\n")
    lines.append("| Parameter | Median | IQR (25th) | IQR (75th) | Min | Max |")
    lines.append("|-----------|--------|------------|------------|-----|-----|")
    for param in PARAM_NAMES:
        vals = param_df[param]
        q25, q50, q75 = vals.quantile([0.25, 0.5, 0.75])
        lines.append(
            f"| `{param}` | {q50:.4f} | {q25:.4f} | {q75:.4f} "
            f"| {vals.min():.4f} | {vals.max():.4f} |"
        )
    lines.append(f"\nNumber of subjects: {n_subjects}\n")
    lines.append(
        "Fixed (not estimated): `p_reward_correct` = 1.0, "
        "`p_reward_incorrect` = 0.0 (with epsilon = 0.001)\n"
    )

    # -- Training metrics --
    lines.append("## Training-set metrics (in-sample)\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k, v in train_metrics.items():
        lines.append(f"| {k} | {v:.4f} |")

    # -- Test metrics --
    lines.append("\n## Test-set metrics (out-of-sample)\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k, v in test_metrics.items():
        lines.append(f"| {k} | {v:.4f} |")

    # -- Example subject --
    lines.append(f"\n## Example subject trace (subject_id={example_subject_id})\n")
    lines.append(
        "First 80 trials showing predicted P(lick), context belief p_vis, "
        "alongside the actual response, stimulus type, and reward.\n"
    )
    lines.append(
        "The `p_vis` column is the key interpretable output -- it should now show "
        "**sharp context switches** after rewarded target trials, unlike the v1 model "
        "where p_vis drifted gradually.\n"
    )
    n_show = min(80, len(example_traces["p_lick"]))
    lines.append(
        "| Trial | Stimulus | Response | Reward | P(lick) | p_vis |"
    )
    lines.append("|-------|----------|----------|--------|---------|-------|")
    stim_names = ["vis_tgt", "vis_nontgt", "aud_tgt", "aud_nontgt"]
    for t in range(n_show):
        s = stim_names[example_traces["stim_idx"][t]]
        r = "lick" if example_traces["responses"][t] else "---"
        rew_val = example_traces["rewards"][t]
        rew = "yes" if rew_val == 1.0 else ("no" if rew_val == 0.0 else "?")
        p = example_traces["p_lick"][t]
        pv = example_traces["p_vis"][t]
        lines.append(f"| {t} | {s} | {r} | {rew} | {p:.3f} | {pv:.3f} |")

    lines.append("\n---\n")
    lines.append(
        "Model: Bayesian context-inference RL v2 with fixed reward likelihoods "
        "and transition leak, fitted per subject via maximum-likelihood "
        "(L-BFGS-B, 4 random starts).\n"
    )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time_mod.perf_counter()

    # -- Load data --
    print("Loading data ...")
    train_df = pd.read_parquet(DATA_DIR / "training_set.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test_set.parquet")

    # Sort by subject then trial order so the sequential model is valid
    train_df = train_df.sort_values(["subject_id", "trial_index"]).reset_index(drop=True)
    test_df = test_df.sort_values(["subject_id", "trial_index"]).reset_index(drop=True)

    # Derive current-trial reward
    train_rewards = _derive_reward(train_df)
    test_rewards = _derive_reward(test_df)

    # Pre-compute arrays
    train_stim = _stim_index(train_df)
    train_resp = train_df["target_response"].values.astype(np.float64)
    test_stim = _stim_index(test_df)
    test_resp = test_df["target_response"].values.astype(np.float64)

    subjects = train_df["subject_id"].unique()
    n_subjects = len(subjects)
    print(f"Fitting {n_subjects} subjects ...\n")

    # -- Fit per subject --
    param_records: list[dict] = []
    train_p_all = np.empty(len(train_df), dtype=np.float64)
    test_p_all = np.empty(len(test_df), dtype=np.float64)

    # Build index lookups once
    train_subj_arr = train_df["subject_id"].values
    test_subj_arr = test_df["subject_id"].values

    example_subject_id = subjects[0]
    example_traces: dict[str, np.ndarray] = {}

    for i, sid in enumerate(subjects):
        t0 = time_mod.perf_counter()

        # Training data for this subject
        tr_mask = train_subj_arr == sid
        tr_stim = train_stim[tr_mask]
        tr_resp = train_resp[tr_mask]
        tr_rew = train_rewards[tr_mask]

        best_params, best_nll = fit_subject(tr_stim, tr_resp, tr_rew)
        beta, bias_val, gamma, v_nontgt = best_params

        param_records.append({
            "subject_id": sid,
            "beta": beta,
            "bias": bias_val,
            "gamma": gamma,
            "v_nontgt": v_nontgt,
            "nll": best_nll,
        })

        # In-sample predictions
        tr_p, _ = run_model_forward(best_params, tr_stim, tr_resp, tr_rew)
        train_p_all[tr_mask] = tr_p

        # Test-set predictions (run forward with fitted params)
        te_mask = test_subj_arr == sid
        te_stim = test_stim[te_mask]
        te_resp = test_resp[te_mask]
        te_rew = test_rewards[te_mask]
        te_p, te_pvis = run_model_forward(best_params, te_stim, te_resp, te_rew)
        test_p_all[te_mask] = te_p

        # Save example traces for the first subject
        if sid == example_subject_id:
            example_traces = {
                "stim_idx": te_stim,
                "responses": te_resp,
                "rewards": te_rew,
                "p_lick": te_p,
                "p_vis": te_pvis,
            }

        elapsed = time_mod.perf_counter() - t0
        if (i + 1) % 10 == 0 or (i + 1) == n_subjects:
            print(
                f"  [{i + 1:>3}/{n_subjects}] subject {sid}: "
                f"beta={beta:.2f} bias={bias_val:.2f} gamma={gamma:.4f} "
                f"v_nt={v_nontgt:.2f}  "
                f"NLL={best_nll:.1f}  ({elapsed:.1f}s)"
            )

    param_df = pd.DataFrame(param_records)

    # -- Metrics --
    print("\nComputing metrics ...")
    train_metrics = evaluate(train_resp, train_p_all)
    test_metrics = evaluate(test_resp, test_p_all)

    print("\n=== Training-set metrics ===")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Test-set metrics ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # -- Parameter summary --
    print("\n=== Fitted parameters (median [IQR]) ===")
    for param in PARAM_NAMES:
        vals = param_df[param]
        q25, q50, q75 = vals.quantile([0.25, 0.5, 0.75])
        print(f"  {param}: {q50:.4f} [{q25:.4f} - {q75:.4f}]")

    # -- Write report --
    write_report(
        param_df=param_df,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        example_subject_id=example_subject_id,
        example_traces=example_traces,
        n_subjects=n_subjects,
    )

    total = time_mod.perf_counter() - t_start
    print(f"\nDone in {total:.1f}s")


if __name__ == "__main__":
    main()
