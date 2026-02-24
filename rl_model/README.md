# Reinforcement Learning Model (Rescorla-Wagner)

## Model
A Rescorla-Wagner (RW) model that maintains Q-values (expected reward values) for
each of the four stimulus types: visual target, visual non-target, auditory target,
and auditory non-target. On each trial the model observes which stimulus was
presented, computes the probability of licking via a softmax (sigmoid) decision
rule, and then updates the Q-value for that stimulus based on the outcome.

### Decision rule
P(lick) = sigmoid(beta * Q[stimulus] + bias)

### Update rule
- If the mouse responded **and** received reward:
  Q[stimulus] += alpha_pos * (1 - Q[stimulus])
- If the mouse responded **and** did not receive reward:
  Q[stimulus] += alpha_neg * (0 - Q[stimulus])
- If the mouse did not respond: no update

### Free parameters (fit per subject)
| Parameter   | Description                            | Bounds       |
|-------------|----------------------------------------|--------------|
| `alpha_pos` | Learning rate for positive prediction errors | [0.001, 1]  |
| `alpha_neg` | Learning rate for negative prediction errors | [0.001, 1]  |
| `beta`      | Inverse temperature (decision noise)   | [0.1, 20]    |
| `bias`      | Baseline lick tendency                 | [-5, 5]      |

Q-values are initialized to 0.5 (no prior preference).

## Rationale
The Rescorla-Wagner model is the foundational model in computational neuroscience
for reward-driven learning. It is well suited to this context-switching task because:

1. **Cognitive interpretability**: Each parameter maps to a distinct cognitive
   process -- learning rates capture how quickly the mouse updates its expectations
   after rewarded vs unrewarded outcomes, the inverse temperature reflects decision
   stochasticity, and the bias captures baseline response tendency.
2. **Asymmetric learning**: Separate learning rates for positive and negative
   prediction errors allow the model to capture the common finding that animals
   learn faster from rewards than from omissions (or vice versa).
3. **Trial-by-trial dynamics**: Unlike static classifiers, the RW model naturally
   captures how behavior evolves across trials as the mouse learns which stimulus is
   currently rewarded, making it ideal for the alternating-block context structure.
4. **Per-subject fitting**: Individual parameter estimates allow comparison of
   learning strategies across the 113 subjects.

## Features
The model does not use the boolean features as direct regression inputs. Instead, it
uses the stimulus-type indicators (`is_vis_target`, `is_vis_non_target`,
`is_aud_target`, `is_aud_non_target`) to select which Q-value to read and update.
The reward signal for the current trial is derived from the next trial's
`previous_reward` column; the response is `target_response`.

## Usage
```bash
uv run python rl_model/fit.py
```

Results are written to `rl_model/REPORT.md`.
