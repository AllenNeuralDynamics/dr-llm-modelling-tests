# Data

The parquet files for this experiment contain concatenated data from multiple sessions, with one
session per subject. Each subject had reached the learning criterion (Stage 3 passed). Each row corresponds to one trial, and the columns contain information about the
stimulus presented, the subject's response, and other metadata, with the following schema:
| column | type | allow in model |
|---|---|---|
| target_response | Boolean | true |
| previous_response | Boolean | true |
| previous_reward | Boolean | true |
| is_vis_target | Boolean | true |
| is_vis_non_target | Boolean | true |
| is_aud_target | Boolean | true |
| is_aud_non_target | Boolean | true |
| subject_id | Int64 | **false** |
| date | Date | **false** |
| time | Datetime(time_unit='us', time_zone=None) | **false** |
| trial_index | Int64 | **false** |
| training_set | Boolean | **false** |

- `target_response` is the actual response from the subject, which is what the model should be able to
predict. 
- `previous_response` and `previous_reward` are the response and reward on the previous trial. 
- `is_<stim_name>` columns indicate which stimulus was presented on the current trial.
- The remaining columns should not be used as features or labels for training a model ("allow in model" is false), but may be useful for other purposes such as batching,
  aggregation or filtering prior to model fitting, or for subsequent analysis of results.