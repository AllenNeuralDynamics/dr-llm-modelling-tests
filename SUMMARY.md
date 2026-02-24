# Summary: Interpretable Models for Mouse Context-Switching Behavior

## Task overview

113 mice were trained on an audio-visual context-switching task. In each session, blocks alternate between rewarding responses to visual targets and auditory targets. The mouse must infer which context it is in from reward feedback and adjust its behavior accordingly. We fit 10 interpretable models to predict trial-by-trial lick responses (`target_response`) from 6 boolean features: the current stimulus type (`is_vis_target`, `is_vis_non_target`, `is_aud_target`, `is_aud_non_target`) and the previous trial's outcome (`previous_response`, `previous_reward`).

## Results

| Model | Accuracy | Bal. Acc. | AUC-ROC | Log-loss | # Params |
|---|---|---|---|---|---|
| **Hybrid GLM** | **0.914** | **0.903** | **0.963** | **0.234** | 9 + 12 |
| Switching GLM | 0.886 | 0.879 | 0.947 | 0.297 | 15 |
| Context HMM | 0.825 | 0.793 | 0.917 | 0.370 | 9 |
| Logistic Regression | 0.817 | 0.834 | 0.871 | 0.409 | 22 |
| Decision Tree | 0.817 | 0.834 | 0.873 | 0.413 | depth 5 |
| GLMM (GEE) | 0.817 | 0.834 | 0.871 | 0.411 | 7 |
| Context RL v2 | 0.796 | 0.777 | 0.854 | 0.456 | 4/subj |
| Context RL v1 | 0.796 | 0.778 | 0.855 | 0.455 | 6/subj |
| HMM (4-state) | 0.768 | 0.800 | 0.827 | 0.459 | auto |
| RL (Rescorla-Wagner) | 0.761 | 0.757 | 0.798 | 0.524 | 4/subj |

## Key findings

### 1. Context is the critical missing variable

The simplest models (logistic regression, decision tree, GLMM) all converge on ~82% accuracy and ~0.87 AUC. They treat each trial independently and rely on stimulus type as the dominant predictor. The decision tree reveals this clearly: the primary split is just "is it a target stimulus?" These models cannot distinguish *which context* the mouse is in, capping their performance.

### 2. Explicitly modelling context gives a large boost

The **context HMM** (2 hidden states for visual/auditory context) jumps to 0.917 AUC by tracking the latent block structure. Its fitted parameters are directly interpretable:
- `p_switch = 0.019` implies an expected block length of ~53 trials, consistent with the ~80-100 trial blocks in the experimental design
- The emission matrix reveals an asymmetry: mice respond to auditory targets at 98% in auditory context (near-perfect) but only 68% to visual targets in visual context, suggesting auditory discrimination is stronger

### 3. Combining context inference with feature sensitivity is best

The **hybrid GLM** — which uses the context HMM's forward-filtered posterior P(visual context) as a feature in a logistic regression alongside the original 6 features — dominates all metrics (91.4% accuracy, 0.963 AUC, 0.234 log-loss). Its coefficients reveal how context modulates stimulus processing:

| Feature | Coefficient | Interpretation |
|---|---|---|
| `is_aud_target` | +16.0 | Strong baseline drive to respond to auditory targets |
| `p_vis_context x aud_target` | -13.3 | ...but massively suppressed in visual context |
| `p_vis_context x vis_target` | +5.6 | Visual targets get extra drive in visual context |
| `p_vis_context x aud_non_target` | -7.9 | Auditory non-target false alarms suppressed in visual context |
| `previous_response` | +0.5 | Persistence: mice tend to repeat their last action |
| `previous_reward` | +0.4 | Win-stay: reward increases licking on the next trial |

### 4. The switching GLM reveals context-dependent reward processing

The **switching GLM** (regime-switching logistic regression, 15 parameters) is the runner-up at 0.947 AUC. Its most striking finding is that `previous_reward` has *opposite* effects in the two contexts:
- **Visual context**: previous_reward = -0.33 (reward *suppresses* next lick)
- **Auditory context**: previous_reward = +0.32 (reward *promotes* next lick)

This may reflect the asymmetric task structure: in auditory blocks, reward confirms the mouse should keep licking, while in visual blocks (where auditory targets are also present but unrewarded), recent reward may signal that the mouse just responded to the correct target and can afford to be more selective.

### 5. Per-subject RL models underperform

Both Rescorla-Wagner variants and the Bayesian context-inference RL models (~0.80-0.85 AUC) lag behind the global models. This is likely because:
- Per-subject fitting with 4-6 parameters on ~500 trials is statistically limited
- The RL models process trials sequentially and must "re-learn" context each session, while the HMM-based models use the global emission structure to decode context immediately
- The Rescorla-Wagner model's Q-values conflate stimulus value with context, while the task separates them

## Model recommendations

- **Best overall**: Hybrid GLM (`hybrid_glm/`) — highest performance, fully interpretable coefficients, principled two-stage approach
- **Most mechanistic**: Context HMM (`context_hmm/`) — directly models the latent context-switching process, fewest assumptions, 9 parameters
- **Richest behavioral insights**: Switching GLM (`switching_glm/`) — reveals how every feature has context-dependent effects, including the reversal of reward influence
- **Simplest baseline**: Logistic regression (`logistic_regression/`) — context-free, but coefficients are immediately interpretable

## Investigations

Each subfolder contains `README.md` (model description), `fit.py` (standalone script), and `REPORT.md` (full results):

| Folder | Model |
|---|---|
| `logistic_regression/` | Logistic regression with interaction terms |
| `decision_tree/` | CART with cross-validated depth |
| `glmm/` | GEE with subject clustering |
| `hmm/` | 4-state Gaussian HMM |
| `rl_model/` | Rescorla-Wagner per subject |
| `context_hmm/` | 2-state context HMM |
| `context_rl/` | Bayesian context RL (v1) |
| `context_rl_v2/` | Fixed-likelihood context RL (v2) |
| `hybrid_glm/` | Context HMM posterior + logistic regression |
| `switching_glm/` | Regime-switching logistic regression (EM) |
