# Generalized Linear Mixed Model (via GEE)

## Model
Generalized Estimating Equations (GEE) with a binomial family and logit link,
fitted using `statsmodels.genmod.generalized_estimating_equations.GEE`. Observations
are grouped by `subject_id` with an exchangeable within-subject correlation structure.

## Rationale
Standard logistic regression treats all trials as independent, ignoring the fact
that trials are nested within subjects. A full GLMM with random intercepts per
subject would be ideal but is prone to convergence issues with 113 subjects and
~60K trials. GEE provides a practical alternative: it yields population-averaged
fixed-effect estimates that account for within-subject correlation, with robust
(sandwich) standard errors that remain valid even if the assumed correlation
structure is misspecified. The resulting coefficients are interpretable as log-odds
(and, exponentiated, as odds ratios) for the population, while properly adjusting
for the clustered data structure.

## Features
Six boolean predictors: `previous_response`, `previous_reward`, `is_vis_target`,
`is_vis_non_target`, `is_aud_target`, `is_aud_non_target`.

## Usage
```bash
uv run python glmm/fit.py
```

Results are written to `glmm/REPORT.md`.
