# HMM Results

## Model selection

| n_components | Log-likelihood | BIC | Converged |
|---|---|---|---|
| 2 | 240,944.9 | -481,592.6 | True |
| 3 | 712,417.4 | -1,424,350.4 | True |
| 4 * | 1,086,456.4 | -2,172,219.1 | True |

\* = selected model

## Selected model: 4 hidden states

### Transition matrix

| | State 0 | State 1 | State 2 | State 3 |
|---|---|---|---|---|
| **State 0** | 0.3724 | 0.1930 | 0.2203 | 0.2143 |
| **State 1** | 0.2439 | 0.2925 | 0.2246 | 0.2389 |
| **State 2** | 0.2377 | 0.3336 | 0.1866 | 0.2420 |
| **State 3** | 0.3574 | 0.1566 | 0.2127 | 0.2734 |

### Emission means per state

| Feature | State 0 | State 1 | State 2 | State 3 |
|---|---|---|---|---|
| previous_response | 0.5313 | 0.1851 | 0.3760 | 0.3977 |
| previous_reward | 0.4427 | 0.0000 | 0.2318 | 0.2766 |
| is_vis_target | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| is_vis_non_target | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| is_aud_target | 0.7770 | 0.0000 | 0.0000 | 0.0000 |
| is_aud_non_target | 0.1566 | 0.6942 | 0.0000 | 0.0000 |

### State-conditional response rates (training data)

| State | N trials (train) | P(target_response=True) |
|---|---|---|
| 0 | 18,782 | 0.5898 |
| 1 | 14,360 | 0.0925 |
| 2 | 12,831 | 0.0372 |
| 3 | 14,565 | 0.7147 |

## Test-set evaluation

| Metric | Value |
|---|---|
| Accuracy | 0.7676 |
| Balanced accuracy | 0.7999 |
| AUC-ROC | 0.8269 |
| Log-loss | 0.4592 |

## Interpretation

The HMM discovers 4 latent states from the trial sequence data. Each state
corresponds to a distinct behavioral pattern characterized by different emission
means across the 6 features. The transition matrix captures how mice switch between
these modes over the course of a session, reflecting the block-based structure of
the context-switching task.

Prediction is performed by decoding the most likely state sequence on test data
and mapping each state to its training-set response probability. This approach
leverages the temporal structure of the task rather than treating trials as
independent observations.
