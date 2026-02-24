# Switching GLM (Regime-Switching Logistic Regression)

## Motivation

The context HMM (`context_hmm/fit.py`) infers the latent reward context from
stimulus-response patterns but uses only per-state per-stimulus response rates
(8 emission parameters). The logistic regression (`logistic_regression/fit.py`)
uses all 6 trial-level features but assumes a single set of coefficients across
the entire session. The hybrid GLM (`hybrid_glm/fit.py`) bolts the HMM's context
posterior onto the logistic regression as an extra feature, but the two stages are
fit independently.

The **switching GLM** is the most flexible context-aware model: it places a **full
logistic regression** inside each of two latent states and fits everything jointly
via EM. This subsumes both the context HMM (per-state per-stimulus response rates
are a special case of per-state logistic coefficients) and the logistic regression
(feature coefficients), while allowing the relationship between features and
responses to differ across contexts.

## Model

### Hidden states: 2

- **State 0**: Visual-rewarded context
- **State 1**: Auditory-rewarded context

### Transition structure

Symmetric transition matrix with a single free parameter `p_switch`:

```
[[1 - p_switch, p_switch],
 [p_switch,     1 - p_switch]]
```

### Emission structure

In each state k, the response probability is a full logistic regression over all
trial-level features:

```
P(lick | features, state=k) = sigmoid(X @ beta_k)
```

where the design matrix X includes an intercept plus the 6 boolean features:

| Column | Feature |
|---|---|
| 0 | intercept (= 1) |
| 1 | previous_response |
| 2 | previous_reward |
| 3 | is_vis_target |
| 4 | is_vis_non_target |
| 5 | is_aud_target |
| 6 | is_aud_non_target |

Each state has its own 7-dimensional coefficient vector `beta_k`.

### Total parameters: 15

- 1 transition parameter (`p_switch`)
- 2 x 7 = 14 logistic regression coefficients (intercept + 6 features per state)

## Fitting procedure

Parameters are fit via the **Expectation-Maximization (EM)** algorithm:

1. **Initialize**: `p_switch = 0.02`; `beta_0` and `beta_1` from a sklearn
   `LogisticRegression` fit on all training data (perturbed slightly for
   asymmetry between states).
2. **E-step**: Forward-backward algorithm (log-space) computes posterior state
   probabilities `gamma[t, k] = P(state_t = k | all observations)` and expected
   transition counts `xi[t, j, k]`.
3. **M-step**: Update `p_switch` from expected transition counts; update each
   `beta_k` via weighted logistic regression (`sklearn.LogisticRegression` with
   `sample_weight = gamma[:, k]`).
4. **Iterate** until log-likelihood change < 1e-4 or 50 iterations.
5. **Multi-start**: 3 random initializations; keep the best.

## Prediction

For test data, forward filtering (causal -- no future information) computes
`P(state_t = k | obs_1..t)`, then marginalizes over states:

```
P(lick_t) = sum_k P(state_t = k | obs_1..t) * sigmoid(X_t @ beta_k)
```

## Key output

- **Per-state coefficient tables**: Show how each feature's influence on
  responding differs between the visual and auditory contexts. This is the main
  interpretable output.
- **Switch probability and expected block length**: Quantifies the inferred
  temporal structure.
- **Classification metrics**: Accuracy, balanced accuracy, AUC-ROC, log-loss on
  both training and test sets.

## Usage

```bash
uv run python switching_glm/fit.py
```

Results are written to `switching_glm/REPORT.md`.
