"""
Context-aware 2-state HMM for mouse context-switching behavior.

The two hidden states represent the visual-rewarded and auditory-rewarded
contexts. Emission probabilities are Bernoulli: P(lick | state, stimulus).
The transition matrix is symmetric with a single p_switch parameter.

Parameters are fit globally across all training subjects via maximum
likelihood (forward algorithm + L-BFGS-B). Prediction uses forward
filtering to compute posterior state probabilities at each trial.

Run from project root:
    uv run python context_hmm/fit.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
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
STIMULUS_COLS = ["is_vis_target", "is_vis_non_target", "is_aud_target", "is_aud_non_target"]
STIMULUS_NAMES = ["vis_target", "vis_non_target", "aud_target", "aud_non_target"]
STATE_NAMES = ["Visual context", "Auditory context"]
TARGET_COL = "target_response"

N_STATES = 2
N_STIMULI = 4
N_PARAMS = 1 + N_STATES * N_STIMULI  # 9 total
N_RESTARTS = 3
CLIP_EPS = 1e-12

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data loading and encoding
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
    # Determine stimulus type for each trial
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
    # P(obs_t | state_k) = resp_probs[k, stim_t]^resp_t * (1 - resp_probs[k, stim_t])^(1 - resp_t)
    # log P = resp_t * log(resp_probs[k, stim_t]) + (1 - resp_t) * log(1 - resp_probs[k, stim_t])
    rp_clipped = np.clip(resp_probs, CLIP_EPS, 1 - CLIP_EPS)
    log_rp = np.log(rp_clipped)        # (2, 4)
    log_1mrp = np.log(1 - rp_clipped)  # (2, 4)

    # For each trial: log_emit[t, k] = log P(obs_t | state_k)
    log_emit = (
        responses[:, None] * log_rp[:, stimuli].T +
        (1 - responses[:, None]) * log_1mrp[:, stimuli].T
    )  # (N, 2)

    # Uniform initial state distribution (no prior on which context starts)
    log_pi = np.log(np.array([0.5, 0.5]))

    total_ll = 0.0
    idx = 0
    for length in lengths:
        # Forward pass for this subject
        # alpha[k] = log P(obs_1..t, state_t=k)
        alpha = log_pi + log_emit[idx]  # (2,)

        for t in range(idx + 1, idx + length):
            # alpha_new[k] = log(sum_j exp(alpha[j] + log_trans[j,k])) + log_emit[t, k]
            # Use logsumexp for stability
            alpha_new = np.empty(N_STATES)
            for k in range(N_STATES):
                alpha_new[k] = _logsumexp(alpha + log_trans[:, k]) + log_emit[t, k]
            alpha = alpha_new

        # Total log-likelihood for this subject
        total_ll += _logsumexp(alpha)
        idx += length

    return total_ll


def _logsumexp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    c = x.max()
    if np.isinf(c):
        return -np.inf
    return c + np.log(np.sum(np.exp(x - c)))


def neg_log_likelihood(
    params: np.ndarray,
    stimuli: np.ndarray,
    responses: np.ndarray,
    lengths: list[int],
) -> float:
    """Negative log-likelihood for minimization."""
    ll = forward_log_likelihood(params, stimuli, responses, lengths)
    return -ll


# ---------------------------------------------------------------------------
# Forward-backward algorithm for posterior state probabilities
# ---------------------------------------------------------------------------

def forward_backward(
    p_switch: float,
    resp_probs: np.ndarray,
    stimuli: np.ndarray,
    responses: np.ndarray,
    lengths: list[int],
) -> np.ndarray:
    """
    Compute posterior state probabilities P(state_t=k | all observations)
    for each trial using the forward-backward algorithm.

    Returns
    -------
    gamma : array (N, 2)
        Posterior state probabilities.
    """
    trans = get_transition_matrix(p_switch)
    log_trans = np.log(np.clip(trans, CLIP_EPS, None))

    rp_clipped = np.clip(resp_probs, CLIP_EPS, 1 - CLIP_EPS)
    log_rp = np.log(rp_clipped)
    log_1mrp = np.log(1 - rp_clipped)

    log_emit = (
        responses[:, None] * log_rp[:, stimuli].T +
        (1 - responses[:, None]) * log_1mrp[:, stimuli].T
    )  # (N, 2)

    log_pi = np.log(np.array([0.5, 0.5]))
    N = len(stimuli)
    gamma = np.zeros((N, N_STATES))

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
        # log_beta[-1] = 0 (log(1))
        for t in range(length - 2, -1, -1):
            for k in range(N_STATES):
                log_beta[t, k] = _logsumexp(
                    log_trans[k, :] + log_emit[idx + t + 1, :] + log_beta[t + 1]
                )

        # --- Posterior ---
        log_gamma = log_alpha + log_beta  # (length, 2)
        for t in range(length):
            log_norm = _logsumexp(log_gamma[t])
            gamma[idx + t] = np.exp(log_gamma[t] - log_norm)

        idx += length

    return gamma


# ---------------------------------------------------------------------------
# Viterbi decoding
# ---------------------------------------------------------------------------

def viterbi_decode(
    p_switch: float,
    resp_probs: np.ndarray,
    stimuli: np.ndarray,
    responses: np.ndarray,
    lengths: list[int],
) -> np.ndarray:
    """
    Viterbi decoding to find the most likely state sequence.

    Returns
    -------
    states : int array (N,)
    """
    trans = get_transition_matrix(p_switch)
    log_trans = np.log(np.clip(trans, CLIP_EPS, None))

    rp_clipped = np.clip(resp_probs, CLIP_EPS, 1 - CLIP_EPS)
    log_rp = np.log(rp_clipped)
    log_1mrp = np.log(1 - rp_clipped)

    log_emit = (
        responses[:, None] * log_rp[:, stimuli].T +
        (1 - responses[:, None]) * log_1mrp[:, stimuli].T
    )  # (N, 2)

    log_pi = np.log(np.array([0.5, 0.5]))
    N = len(stimuli)
    states = np.zeros(N, dtype=int)

    idx = 0
    for length in lengths:
        # Viterbi for this subject
        delta = np.zeros((length, N_STATES))  # log-space
        psi = np.zeros((length, N_STATES), dtype=int)

        delta[0] = log_pi + log_emit[idx]

        for t in range(1, length):
            for k in range(N_STATES):
                scores = delta[t - 1] + log_trans[:, k]
                psi[t, k] = int(np.argmax(scores))
                delta[t, k] = scores[psi[t, k]] + log_emit[idx + t, k]

        # Backtrace
        path = np.zeros(length, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(length - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        states[idx:idx + length] = path
        idx += length

    return states


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
        responses[:, None] * log_rp[:, stimuli].T +
        (1 - responses[:, None]) * log_1mrp[:, stimuli].T
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


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_probs(
    p_switch: float,
    resp_probs: np.ndarray,
    stimuli: np.ndarray,
    responses: np.ndarray,
    lengths: list[int],
) -> np.ndarray:
    """
    Predict P(lick_t) by marginalizing over states using forward filtering.

    P(lick_t) = sum_k P(state_t=k | obs_1..t) * resp_probs[k, stimulus_t]
    """
    filtered = forward_filter(p_switch, resp_probs, stimuli, responses, lengths)
    # For each trial, marginalize over states
    y_prob = np.sum(filtered * resp_probs[:, stimuli].T, axis=1)
    return np.clip(y_prob, CLIP_EPS, 1 - CLIP_EPS)


# ---------------------------------------------------------------------------
# Evaluation metrics
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
# Plotting
# ---------------------------------------------------------------------------

def plot_example_subject(
    df: pd.DataFrame,
    stimuli: np.ndarray,
    responses: np.ndarray,
    gamma: np.ndarray,
    viterbi_states: np.ndarray,
    lengths: list[int],
    subject_ids: np.ndarray,
    save_path: Path,
) -> str | None:
    """
    Plot decoded context states for an example subject.

    Picks the subject with the most trials (likely to show multiple block
    switches) and creates a figure showing:
    - Top: posterior probability of visual-rewarded state
    - Middle: Viterbi-decoded state
    - Bottom: trial-by-trial responses colored by stimulus type

    Returns the filename if saved successfully, None otherwise.
    """
    # Find subject with most trials
    best_idx = int(np.argmax(lengths))
    start = sum(lengths[:best_idx])
    end = start + lengths[best_idx]

    # Get unique subject IDs in order
    unique_subjects = []
    seen = set()
    for sid in subject_ids:
        if sid not in seen:
            unique_subjects.append(sid)
            seen.add(sid)
    subject_id = unique_subjects[best_idx]

    subj_trials = np.arange(lengths[best_idx])
    subj_gamma = gamma[start:end, 0]  # P(visual context)
    subj_viterbi = viterbi_states[start:end]
    subj_responses = responses[start:end]
    subj_stimuli = stimuli[start:end]

    stim_colors = {
        0: "#1f77b4",  # vis target - blue
        1: "#aec7e8",  # vis non-target - light blue
        2: "#ff7f0e",  # aud target - orange
        3: "#ffbb78",  # aud non-target - light orange
    }

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 2]})

    # Top: posterior P(visual context)
    ax = axes[0]
    ax.plot(subj_trials, subj_gamma, color="black", linewidth=0.8, alpha=0.8)
    ax.fill_between(subj_trials, subj_gamma, alpha=0.3, color="#1f77b4", label="P(visual context)")
    ax.fill_between(subj_trials, subj_gamma, 1.0, alpha=0.3, color="#ff7f0e", label="P(auditory context)")
    ax.set_ylabel("P(visual context)")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Context HMM: decoded states for subject {subject_id} ({lengths[best_idx]} trials)")

    # Middle: Viterbi state
    ax = axes[1]
    # Color background by Viterbi state
    for t in range(len(subj_viterbi)):
        color = "#1f77b4" if subj_viterbi[t] == 0 else "#ff7f0e"
        ax.axvspan(t - 0.5, t + 0.5, color=color, alpha=0.4)
    ax.set_ylabel("Decoded\nstate")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Visual", "Auditory"], fontsize=8)
    ax.set_ylim(-0.5, 1.5)

    # Bottom: responses colored by stimulus
    ax = axes[2]
    for s_idx, s_name in enumerate(STIMULUS_NAMES):
        mask = subj_stimuli == s_idx
        if mask.any():
            trials_s = subj_trials[mask]
            resp_s = subj_responses[mask]
            ax.scatter(
                trials_s, resp_s + (s_idx - 1.5) * 0.05,  # slight vertical jitter
                c=stim_colors[s_idx], s=8, alpha=0.6, label=s_name,
                edgecolors="none",
            )
    ax.set_ylabel("Response (lick)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No", "Yes"])
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel("Trial number")
    ax.legend(loc="upper right", fontsize=7, ncol=2, markerscale=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved example subject figure to {save_path}")
    return save_path.name


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_report(
    p_switch: float,
    resp_probs: np.ndarray,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    n_train_trials: int,
    n_test_trials: int,
    n_train_subjects: int,
    n_test_subjects: int,
    figure_filename: str | None,
) -> None:
    """Write results to REPORT.md."""
    trans = get_transition_matrix(p_switch)

    # Transition matrix table
    trans_header = "| | Visual context | Auditory context |"
    trans_sep = "|---|---|---|"
    trans_rows = [
        f"| **Visual context** | {trans[0, 0]:.6f} | {trans[0, 1]:.6f} |",
        f"| **Auditory context** | {trans[1, 0]:.6f} | {trans[1, 1]:.6f} |",
    ]
    trans_table = "\n".join([trans_header, trans_sep] + trans_rows)

    # Response probability table
    resp_header = "| Stimulus | Visual context (State 0) | Auditory context (State 1) |"
    resp_sep = "|---|---|---|"
    resp_rows = []
    for s_idx, s_name in enumerate(STIMULUS_NAMES):
        resp_rows.append(
            f"| {s_name} | {resp_probs[0, s_idx]:.4f} | {resp_probs[1, s_idx]:.4f} |"
        )
    resp_table = "\n".join([resp_header, resp_sep] + resp_rows)

    # Metrics tables
    def metric_table(metrics: dict[str, float]) -> str:
        header = "| Metric | Value |"
        sep = "|---|---|"
        rows = [
            f"| Accuracy | {metrics['accuracy']:.4f} |",
            f"| Balanced accuracy | {metrics['balanced_accuracy']:.4f} |",
            f"| AUC-ROC | {metrics['auc_roc']:.4f} |",
            f"| Log-loss | {metrics['log_loss']:.4f} |",
        ]
        return "\n".join([header, sep] + rows)

    figure_section = ""
    if figure_filename:
        figure_section = f"""
## Example subject: decoded context states

![Example subject]({figure_filename})

The figure shows the posterior probability of the visual-rewarded context (top),
the Viterbi-decoded state sequence (middle), and trial-by-trial responses colored
by stimulus type (bottom) for the training subject with the most trials. Transitions
between blue (visual context) and orange (auditory context) regions correspond to
the model's inferred block switches.
"""

    report = f"""\
# Context-Aware HMM Results

## Model specification

- **Hidden states**: 2 (visual-rewarded context, auditory-rewarded context)
- **Parameters**: 9 (1 transition + 8 emission)
- **Training data**: {n_train_trials:,} trials from {n_train_subjects} subjects
- **Test data**: {n_test_trials:,} trials from {n_test_subjects} subjects

## Fitted parameters

### Switch probability

`p_switch = {p_switch:.6f}`

Expected block length: {1 / p_switch:.1f} trials (= 1 / p_switch)

### Transition matrix

{trans_table}

### Response probabilities: P(lick | state, stimulus)

{resp_table}

## Per-state response profiles

**Visual-rewarded context (State 0):**
- Responds to visual targets at rate {resp_probs[0, 0]:.4f}
- Responds to auditory targets at rate {resp_probs[0, 2]:.4f}
- Visual non-target false alarm rate: {resp_probs[0, 1]:.4f}
- Auditory non-target false alarm rate: {resp_probs[0, 3]:.4f}

**Auditory-rewarded context (State 1):**
- Responds to auditory targets at rate {resp_probs[1, 2]:.4f}
- Responds to visual targets at rate {resp_probs[1, 0]:.4f}
- Visual non-target false alarm rate: {resp_probs[1, 1]:.4f}
- Auditory non-target false alarm rate: {resp_probs[1, 3]:.4f}

## Training-set evaluation

{metric_table(train_metrics)}

## Test-set evaluation

{metric_table(test_metrics)}
{figure_section}
## Interpretation

This 2-state context HMM explicitly models the key latent variable in the
context-switching task: which modality is currently rewarded. Unlike the standard
HMM (which discovered 4 states corresponding to the 4 stimulus types), this model
uses stimulus identity as an observed covariate and infers the hidden reward context
from the pattern of responses.

The fitted switch probability of {p_switch:.4f} implies an expected block length
of ~{1 / p_switch:.0f} trials, which {"is consistent with" if 50 < 1 / p_switch < 150 else "deviates from"} the experimental design
(~80-100 trials per block).

The response probability matrix reveals the key behavioral signature: mice respond
at higher rates to the target stimulus of the currently rewarded modality and at
lower rates to the non-rewarded target, with low false alarm rates to non-targets
in both contexts.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"  Report written to {REPORT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("Context-Aware HMM: 2-state model for reward-context switching")
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
    print(f"  Training: {len(train_lengths)} subjects, {n_train:,} trials")
    print(f"  Test: {len(test_lengths)} subjects, {n_test:,} trials")

    # ------------------------------------------------------------------
    # 2. Fit model via multi-start optimization
    # ------------------------------------------------------------------
    print("\n[2/6] Fitting model (multi-start optimization, L-BFGS-B)...")

    best_nll = np.inf
    best_params = None

    for restart in range(N_RESTARTS):
        # Random initialization
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
    print(f"  p_switch: {p_switch:.6f} (expected block length: {1/p_switch:.1f} trials)")

    # ------------------------------------------------------------------
    # 3. Display fitted response probabilities
    # ------------------------------------------------------------------
    print("\n[3/6] Fitted response probabilities P(lick | state, stimulus):")
    print(f"  {'Stimulus':<18s}  {'Visual ctx':>12s}  {'Auditory ctx':>12s}")
    print(f"  {'-'*18}  {'-'*12}  {'-'*12}")
    for s_idx, s_name in enumerate(STIMULUS_NAMES):
        print(f"  {s_name:<18s}  {resp_probs[0, s_idx]:12.4f}  {resp_probs[1, s_idx]:12.4f}")

    # ------------------------------------------------------------------
    # 4. Evaluate on training set
    # ------------------------------------------------------------------
    print("\n[4/6] Evaluating on training set...")
    train_y_prob = predict_probs(p_switch, resp_probs, train_stimuli, train_responses, train_lengths)
    train_metrics = compute_metrics(train_responses, train_y_prob)

    for name, val in train_metrics.items():
        print(f"  {name:<20s}: {val:.4f}")

    # ------------------------------------------------------------------
    # 5. Evaluate on test set
    # ------------------------------------------------------------------
    print("\n[5/6] Evaluating on test set...")
    test_y_prob = predict_probs(p_switch, resp_probs, test_stimuli, test_responses, test_lengths)
    test_metrics = compute_metrics(test_responses, test_y_prob)

    for name, val in test_metrics.items():
        print(f"  {name:<20s}: {val:.4f}")

    # ------------------------------------------------------------------
    # 6. Decode states and generate report
    # ------------------------------------------------------------------
    print("\n[6/6] Decoding states and writing report...")

    # Full forward-backward for posterior state probabilities (training data)
    train_gamma = forward_backward(
        p_switch, resp_probs, train_stimuli, train_responses, train_lengths
    )
    train_viterbi = viterbi_decode(
        p_switch, resp_probs, train_stimuli, train_responses, train_lengths
    )

    # State distribution in training data
    for k in range(N_STATES):
        n_k = (train_viterbi == k).sum()
        print(f"  {STATE_NAMES[k]}: {n_k:,} trials ({100 * n_k / n_train:.1f}%)")

    # Plot example subject
    figure_filename = plot_example_subject(
        train_df,
        train_stimuli,
        train_responses,
        train_gamma,
        train_viterbi,
        train_lengths,
        train_df["subject_id"].values,
        HERE / "example_subject.png",
    )

    # Write report
    write_report(
        p_switch=p_switch,
        resp_probs=resp_probs,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        n_train_trials=n_train,
        n_test_trials=n_test,
        n_train_subjects=len(train_lengths),
        n_test_subjects=len(test_lengths),
        figure_filename=figure_filename,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
