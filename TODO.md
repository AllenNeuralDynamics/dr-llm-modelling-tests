# TODO

## Setup
- [x] Explore data (shape, features, class balance)
- [x] Install dependencies (scikit-learn, statsmodels, hmmlearn, scipy, matplotlib)
- [x] Create investigation subdirectories

## Round 1: Baseline models
- [x] Logistic Regression — `logistic_regression/`
- [x] Decision Tree — `decision_tree/`
- [x] GLMM (GEE) — `glmm/`
- [x] HMM (4-state) — `hmm/`
- [x] RL (Rescorla-Wagner) — `rl_model/`

## Round 2: Context-aware models
- [x] Context HMM (2-state) — `context_hmm/`
- [x] Context RL (Bayesian) — `context_rl/` (p_reward params collapsed)

## Round 3: Combining context + features
- [x] Hybrid GLM (HMM posterior + logistic regression) — `hybrid_glm/`
- [x] Context RL v2 (fixed likelihoods) — `context_rl_v2/`
- [x] Switching GLM (regime-switching logistic regression) — `switching_glm/`

## Cross-model comparison (test set)

| Model | Accuracy | Bal. Acc. | AUC-ROC | Log-loss |
|---|---|---|---|---|
| **Hybrid GLM** | **0.9142** | **0.9033** | **0.9630** | **0.2338** |
| Switching GLM | 0.8855 | 0.8787 | 0.9474 | 0.2973 |
| Context HMM | 0.8250 | 0.7932 | 0.9168 | 0.3696 |
| Logistic Regression | 0.8169 | 0.8335 | 0.8707 | 0.4090 |
| Decision Tree | 0.8168 | 0.8335 | 0.8733 | 0.4131 |
| GLMM (GEE) | 0.8169 | 0.8335 | 0.8709 | 0.4113 |
| Context RL | 0.7963 | 0.7775 | 0.8552 | 0.4551 |
| Context RL v2 | 0.7958 | 0.7770 | 0.8544 | 0.4560 |
| HMM (4-state) | 0.7676 | 0.7999 | 0.8269 | 0.4592 |
| RL (Rescorla-Wagner) | 0.7607 | 0.7570 | 0.7975 | 0.5244 |

## Next steps
- [ ] Per-subject analysis of hybrid GLM performance
- [ ] Investigate context RL v2 failure (similar to v1 despite fix)
