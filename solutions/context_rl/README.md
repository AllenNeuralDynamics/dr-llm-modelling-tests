# Bayesian Context-Inference RL Model

## Model

A Bayesian context-inference model that maintains a **belief about the current
context** (visual-rewarded vs auditory-rewarded) and updates it trial-by-trial
using Bayes' rule. Unlike the standard Rescorla-Wagner model which maintains
independent Q-values per stimulus, this model explicitly represents the latent
variable that governs the task: which sensory modality is currently rewarded.

The mouse's belief `p_vis` = P(current context = visual-rewarded) is the central
state variable. It is updated after each trial based on whether the observed
reward outcome is consistent with each context hypothesis.

### Context belief update (Bayesian)

Before each trial, the prior belief is "leaked" toward uncertainty to account for
possible context switches:

    p_vis_prior = gamma * 0.5 + (1 - gamma) * p_vis

where `gamma` is the transition/leak rate. After observing the trial outcome
(response and reward), the belief is updated via Bayes' rule:

    p_vis_new = p_vis_prior * P(obs | vis_context) / P(obs)

The likelihood `P(obs | context)` depends on the stimulus responded to:
- Responded to visual target: P(reward | vis_context) = p_reward_correct,
  P(reward | aud_context) = p_reward_incorrect
- Responded to auditory target: P(reward | aud_context) = p_reward_correct,
  P(reward | vis_context) = p_reward_incorrect
- Responded to non-target or did not respond: reward is uninformative, no update

### Decision rule

    P(lick) = sigmoid(beta * value + bias)

where `value` depends on context belief and stimulus type:
- Visual target: value = p_vis
- Auditory target: value = 1 - p_vis
- Visual or auditory non-target: value = v_nontgt (fixed low value)

### Free parameters (fit per subject)

| Parameter             | Description                                        | Bounds       |
|-----------------------|----------------------------------------------------|--------------|
| `beta`                | Inverse temperature (decision noise)               | [0.1, 20]    |
| `bias`                | Baseline lick tendency                             | [-10, 10]    |
| `gamma`               | Context transition/leak rate toward 0.5            | [0.001, 0.5] |
| `p_reward_correct`    | P(reward &#124; target in correct context)         | [0.5, 1.0]   |
| `p_reward_incorrect`  | P(reward &#124; target in wrong context)           | [0.0, 0.5]   |
| `v_nontgt`            | Value assigned to non-target stimuli               | [-5, 0]      |

Context belief `p_vis` is initialized to 0.5 (maximum uncertainty).

## Rationale

The standard Rescorla-Wagner model treats each stimulus independently and has no
mechanism to represent the latent block structure. This is a fundamental limitation
because:

1. **The task is context-dependent**: The same visual target stimulus is rewarded in
   visual blocks but not in auditory blocks. A model that only tracks per-stimulus
   Q-values cannot distinguish *why* reward contingencies change -- it simply
   re-learns from scratch at each block transition.

2. **Bayesian context inference is normative**: An ideal observer in this task would
   maintain a belief distribution over possible contexts and update it using Bayes'
   rule. This model approximates that computation, providing a principled account of
   how the mouse might track block identity.

3. **The context belief trace is directly interpretable**: The model's `p_vis`
   trajectory across trials should show sharp transitions at block switches, with the
   speed of transition reflecting how quickly the mouse detects the change. This is a
   richer behavioral readout than Q-values alone.

4. **Transition rate captures individual differences**: The `gamma` parameter captures
   how much the mouse expects contexts to switch, which may vary across subjects and
   relate to behavioral flexibility.

5. **Separates reward contingency knowledge from context inference**: The
   `p_reward_correct` and `p_reward_incorrect` parameters capture the mouse's
   understanding of the reward structure, independently of its ability to track
   context.

## Features

The model uses stimulus-type indicators (`is_vis_target`, `is_vis_non_target`,
`is_aud_target`, `is_aud_non_target`) to determine stimulus category. The reward
signal for the current trial is derived from the next trial's `previous_reward`
column; the response is `target_response`.

## Usage

```bash
uv run python context_rl/fit.py
```

Results are written to `context_rl/REPORT.md`.
