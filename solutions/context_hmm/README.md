# Context-Aware HMM

## Motivation

The standard HMM (`hmm/fit.py`) applied to this dataset discovers 4 hidden states
that correspond almost perfectly to the 4 stimulus types (visual target, visual
non-target, auditory target, auditory non-target). While this captures
stimulus-driven response differences, it misses the key latent variable in the
context-switching task: **which modality is currently rewarded**.

Mice alternate between visual-rewarded blocks (~10 min, ~80-100 trials) and
auditory-rewarded blocks. Their behavior should differ across these two contexts
even for the same stimulus. For example, a mouse should respond to visual targets
more in visual-rewarded blocks and to auditory targets more in auditory-rewarded
blocks.

## Model

This model is a **2-state HMM** where the hidden states explicitly represent the
two reward contexts:

- **State 0**: Visual-rewarded context
- **State 1**: Auditory-rewarded context

### Emission structure

Unlike a generic Gaussian HMM, emissions are **Bernoulli**: each trial produces a
binary response (lick or no lick). The response probability depends on both the
hidden state and which of the 4 stimulus types was presented:

```
P(lick | state=k, stimulus=s) = resp_prob[k, s]
```

This gives 2 states x 4 stimuli = 8 response probability parameters.

### Transition structure

The transition matrix is symmetric with a single free parameter `p_switch`:

```
[[1 - p_switch, p_switch],
 [p_switch, 1 - p_switch]]
```

Since context blocks are long (~80-100 trials), `p_switch` should be small
(~0.01-0.05).

### Total parameters: 9

- 1 transition parameter (`p_switch`)
- 8 emission parameters (response probabilities for each state-stimulus pair)

## Fitting procedure

- Parameters are optimized via maximum likelihood using `scipy.optimize.minimize`
  (L-BFGS-B) on the negative log-likelihood computed with the forward algorithm.
- All parameters are logit-transformed for unconstrained optimization.
- All training subjects are pooled (shared parameters) with per-subject sequence
  separation in the forward algorithm.
- 3 random restarts to mitigate local minima.

## Prediction

For each test trial, the forward algorithm computes P(state | observations up to t),
then marginalizes over states to predict response probability:

```
P(lick_t) = sum_k P(state=k | obs_1..t) * resp_prob[k, stimulus_t]
```

## Key output

The decoded state sequence for each subject reveals when the model believes context
switches occur, providing an interpretable readout of the latent task structure that
can be compared against the true block schedule.
