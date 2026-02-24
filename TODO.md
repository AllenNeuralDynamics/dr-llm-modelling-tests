# TODO

## Setup
- [x] Explore data (shape, features, class balance)
- [x] Install dependencies (scikit-learn, statsmodels, hmmlearn, scipy, matplotlib)
- [x] Create investigation subdirectories

## Investigations
- [x] Logistic Regression — `logistic_regression/README.md`, `fit.py`, `REPORT.md`
- [x] Decision Tree — `decision_tree/README.md`, `fit.py`, `REPORT.md`
- [x] GLMM (GEE) — `glmm/README.md`, `fit.py`, `REPORT.md`
- [x] HMM — `hmm/README.md`, `fit.py`, `REPORT.md`
- [x] RL Model (Rescorla-Wagner) — `rl_model/README.md`, `fit.py`, `REPORT.md`

## Verification
- [x] Run all scripts, confirm REPORT.md generated for each
- [x] Compare predictive performance across models (see below)

## Cross-model comparison (test set)

| Model | Accuracy | Bal. Accuracy | AUC-ROC | Log-loss |
|---|---|---|---|---|
| Logistic Regression | 0.8169 | 0.8335 | 0.8707 | 0.4090 |
| Decision Tree | 0.8168 | 0.8335 | 0.8733 | 0.4131 |
| GLMM (GEE) | 0.8169 | 0.8335 | 0.8709 | 0.4113 |
| HMM (4 states) | 0.7676 | 0.7999 | 0.8269 | 0.4592 |
| RL (Rescorla-Wagner) | 0.7607 | 0.7570 | 0.7975 | 0.5244 |

## Next steps
- [ ] Deeper analysis: per-subject performance, block-level patterns
- [ ] Investigate why HMM/RL underperform (may need block context info)
- [ ] Consider ensemble or hybrid approaches
