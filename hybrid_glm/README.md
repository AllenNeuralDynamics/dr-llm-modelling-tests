# Hybrid GLM (Context HMM + Logistic Regression)

## Motivation

The context HMM (`context_hmm/fit.py`) infers the latent reward context (visual vs
auditory) from the sequence of stimulus-response pairs. The logistic regression
(`logistic_regression/fit.py`) uses trial-level boolean features to predict responses.
Each approach captures different information:

- **Context HMM**: Learns the hidden block structure and computes a trial-by-trial
  belief about which modality is currently rewarded. This belief integrates all prior
  observations in the session but does not directly predict responses.
- **Logistic regression**: Uses immediate predictors (previous response, previous
  reward, stimulus type) but has no memory of the session history beyond one trial back.

The hybrid GLM combines both by feeding the HMM's context posterior as an additional
feature into the logistic regression. This lets the model learn how context belief
modulates stimulus-driven responding.

## Model

1. **Stage 1 -- Context HMM fitting**: Fit the 2-state context HMM on training data
   (same procedure as `context_hmm/fit.py`: 9 parameters, 3 random restarts, L-BFGS-B).
2. **Stage 2 -- Forward filtering**: Run the forward algorithm on both training and
   test data to compute `p_vis_context = P(visual context | observations up to t)` for
   every trial.
3. **Stage 3 -- Logistic regression**: Fit a logistic regression with 11 features:
   - 6 original boolean features: `previous_response`, `previous_reward`,
     `is_vis_target`, `is_vis_non_target`, `is_aud_target`, `is_aud_non_target`
   - 1 continuous feature: `p_vis_context`
   - 4 interaction terms: `p_vis_context` x each stimulus indicator

The interaction terms allow the model to learn that context belief has different
effects depending on stimulus type. For example, a high P(visual context) should
increase responding to visual targets but decrease responding to auditory targets.

## Features

| Feature | Type | Description |
|---|---|---|
| `previous_response` | bool | Whether the mouse licked on the previous trial |
| `previous_reward` | bool | Whether the previous trial was rewarded |
| `is_vis_target` | bool | Visual target stimulus presented |
| `is_vis_non_target` | bool | Visual non-target stimulus presented |
| `is_aud_target` | bool | Auditory target stimulus presented |
| `is_aud_non_target` | bool | Auditory non-target stimulus presented |
| `p_vis_context` | float | HMM forward-filtered P(visual context) |
| `p_vis_context x is_vis_target` | float | Interaction term |
| `p_vis_context x is_vis_non_target` | float | Interaction term |
| `p_vis_context x is_aud_target` | float | Interaction term |
| `p_vis_context x is_aud_non_target` | float | Interaction term |

## Usage

```bash
uv run python hybrid_glm/fit.py
```

Results are written to `hybrid_glm/REPORT.md`.
