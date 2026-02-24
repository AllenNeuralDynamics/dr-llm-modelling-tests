# Plan: Interpretable Models for Mouse Context-Switching Behavior

## Context
Behavioral data from 113 mice in an audio-visual context-switching task (see EXPERIMENT_DESCRIPTION.md). Predict `target_response` (lick/no-lick) from 6 boolean features. ~60K training trials, ~60K test trials. Models must be **interpretable** and **predictive** on held-out data.

## Data summary
- Training: `data/training_set.parquet` — 60,538 trials, 113 subjects, ~38% positive class
- Test: `data/test_set.parquet` — 60,507 trials, same 113 subjects, ~37% positive class
- Allowed features: `previous_response`, `previous_reward`, `is_vis_target`, `is_vis_non_target`, `is_aud_target`, `is_aud_non_target`
- Metadata (not for modelling, but useful for grouping): `subject_id`, `date`, `time`, `trial_index`

## Five parallel investigations

### 1. `logistic_regression/` — Logistic Regression
- **Rationale**: Coefficients directly show feature importance and direction of effect. Odds ratios are intuitive. Gold standard baseline for interpretability.
- **Approach**: scikit-learn `LogisticRegression`. Include interaction terms (e.g., stimulus x previous_reward) to capture context-dependent effects.

### 2. `decision_tree/` — Decision Tree (CART)
- **Rationale**: Produces human-readable if/then rules. Can be visualized as a flowchart. Captures non-linear interactions naturally.
- **Approach**: scikit-learn `DecisionTreeClassifier`. Cross-validate to find optimal depth that balances interpretability and performance.

### 3. `glmm/` — Generalized Linear Mixed Model
- **Rationale**: Accounts for subject-level variability via random intercepts while keeping fixed effects interpretable. Most statistically appropriate for grouped behavioral data.
- **Approach**: statsmodels `BinomialBayesMixedGLM` or `GEE`. Fixed effects for the 6 features, random intercept per subject.

### 4. `hmm/` — Hidden Markov Model
- **Rationale**: Latent states can represent the mouse's internal context mode (e.g., "attending to visual" vs "attending to auditory"). Transition matrix shows context-switching dynamics. Directly models the sequential nature of the task.
- **Approach**: hmmlearn library. Encode trial features as observations, fit per-subject or pooled.

### 5. `rl_model/` — Reinforcement Learning (Rescorla-Wagner)
- **Rationale**: Classic computational neuroscience model for reward-driven behavior. Parameters (learning rate, inverse temperature) are directly interpretable in cognitive terms. Models how mice update stimulus-response values based on reward history.
- **Approach**: Custom implementation with scipy.optimize for parameter fitting via maximum likelihood.

## Subfolder structure (each investigation)
- `README.md` — describes the model and rationale
- `fit.py` — standalone script (run via `uv run python <folder>/fit.py`)
- `REPORT.md` — auto-generated results (metrics, parameters, plots)

## Dependencies
scikit-learn, statsmodels, hmmlearn, scipy, matplotlib, pandas, pyarrow (managed via `uv add`)

## Shared evaluation metrics
- Accuracy, balanced accuracy
- AUC-ROC
- Log-loss
- Confusion matrix

## How to run
Each script is self-contained and can be run independently:
```
uv run python logistic_regression/fit.py
uv run python decision_tree/fit.py
uv run python glmm/fit.py
uv run python hmm/fit.py
uv run python rl_model/fit.py
```
