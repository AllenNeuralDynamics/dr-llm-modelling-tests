# Bayesian Context-Inference RL Model v2 (Fixed Reward Likelihoods)

## What this model fixes

The previous `context_rl` model had `p_reward_correct` and `p_reward_incorrect` as
free parameters, but they collapsed to boundary values (median 0.5 and 0.0
respectively) making the Bayesian update barely functional. When
`p_reward_correct = 0.5`, the likelihood ratio after a rewarded trial is 1:0 -- the
update is technically sharp, but only because the parameter hit its bound. When
`p_reward_incorrect = 0.0`, the likelihood ratio after an unrewarded trial is
0.5:1.0 -- a weak, sluggish update. The model was essentially unable to learn the
reward structure from data.

This v2 model **fixes the reward likelihoods** to their true task values instead of
estimating them:

- P(reward | responded to target in correct context) = 1.0
- P(reward | responded to target in wrong context) = 0.0

This makes the Bayesian update maximally sharp: a single rewarded target trial
provides definitive evidence about which context is active. An unrewarded target
trial is equally informative in the opposite direction. The result is that `p_vis`
should show rapid, decisive context switches after informative trials -- exactly the
behavior we expect from an ideal observer in this task.

By removing 2 free parameters (from 6 down to 4), we also reduce overfitting risk
and make the remaining parameters (beta, bias, gamma, v_nontgt) more identifiable.

## Model

The mouse maintains a belief `p_vis` = P(current context = visual-rewarded) and
updates it trial-by-trial via Bayes' rule with fixed, deterministic reward
likelihoods. Decisions are made via a sigmoid decision rule where the value of each
stimulus depends on the context belief.

### Context belief update (Bayesian with fixed likelihoods)

Before each trial, the prior belief is "leaked" toward uncertainty to account for
possible context switches:

    p_vis_prior = gamma * 0.5 + (1 - gamma) * p_vis

where `gamma` is the transition/leak rate.

After observing the trial outcome (response and reward), the belief is updated via
Bayes' rule. The likelihoods are fixed (not estimated), with a small epsilon (0.001)
to avoid log(0):

| Condition | lik_vis | lik_aud |
|---|---|---|
| Responded to vis_target, rewarded | 1.0 - eps | eps |
| Responded to vis_target, NOT rewarded | eps | 1.0 - eps |
| Responded to aud_target, rewarded | eps | 1.0 - eps |
| Responded to aud_target, NOT rewarded | 1.0 - eps | eps |
| Responded to non-target | no update (uninformative) |
| Did not respond | no update (uninformative) |

Bayes' rule:

    p_vis_new = (p_vis_prior * lik_vis) / (p_vis_prior * lik_vis + (1 - p_vis_prior) * lik_aud)

The belief is clipped to [0.001, 0.999] for numerical stability.

### Decision rule

    P(lick) = sigmoid(beta * value + bias)

where `value` depends on context belief and stimulus type:

- Visual target: value = p_vis_prior
- Auditory target: value = 1 - p_vis_prior
- Non-targets (visual or auditory): value = v_nontgt (fixed low value parameter)

### Free parameters (fit per subject, 4 total)

| Parameter | Description | Bounds |
|---|---|---|
| `beta` | Inverse temperature (decision noise) | [0.1, 20] |
| `bias` | Baseline lick tendency | [-10, 10] |
| `gamma` | Context transition/leak rate toward 0.5 | [0.001, 0.5] |
| `v_nontgt` | Value assigned to non-target stimuli | [-5, 0] |

Context belief `p_vis` is initialized to 0.5 (maximum uncertainty).

## Key differences from v1

| Aspect | v1 (`context_rl`) | v2 (`context_rl_v2`) |
|---|---|---|
| Free parameters | 6 | 4 |
| Reward likelihoods | Estimated (`p_reward_correct`, `p_reward_incorrect`) | Fixed (1.0 and 0.0, with epsilon) |
| Bayesian update | Weak (parameters collapsed to boundaries) | Sharp (deterministic likelihoods) |
| Expected `p_vis` trace | Gradual drift | Rapid context switches after informative trials |

## Features

The model uses stimulus-type indicators (`is_vis_target`, `is_vis_non_target`,
`is_aud_target`, `is_aud_non_target`) to determine stimulus category. The reward
signal for the current trial is derived from the next trial's `previous_reward`
column; the response is `target_response`.

## Usage

```bash
uv run python context_rl_v2/fit.py
```

Results are written to `context_rl_v2/REPORT.md`.
