# Experiment description

## Subject training curriculum
### Stage 1: visual discrimination
Mice are initially trained in a go/nogo visual discrimination task in which they are rewarded with water for licking a centrally placed spout during a response window 0.1-1 s after onset of the target stimulus (rightward-drifting vertical grating). Licks in response to the non-target stimulus (downward-drifting horizontal grating) are punished with a 3 s timeout during which the screen turns dark; the non-target stimulus is presented for up to 3 additional trials in a row until the mouse withholds licking. Licks during a 1.5 s quiescent period before the scheduled stimulus restarts the trial with a new random intertrial interval. Catch trials with no stimulus are included to assess spontaneous licking. During the first five trials of each session, the target stimulus is presented and a non-contingent reward is delivered at the end of the response window if the mouse does not respond prior to this to earn a contingent reward. Mice are trained once per day for one hour. Mice pass stage 1 after two consecutive sessions with at least 100 correct responses and d' greater than 1.5.
### Stage 2: auditory discrimination
After learning the visual discrimination task, mice are trained in an auditory discrimination task in which the target stimulus is 12 Hz amplitude-modulated (AM) noise and the non-target stimulus is 70 Hz AM noise. Other than the stimulus modality, all other aspects of this stage are the same as stage 1.
### Stage 3: context-switching task
After learning the visual and auditory discrimination tasks, mice are trained on the audio-visual context-switching task. On each trial, one of the four stimuli (visual target, visual non-target, auditory target, or auditory non-target) is presented and mice are rewarded for responding to the visual target in visual-rewarded blocks or to the auditory target in auditory-rewarded blocks. Licks in response to the non-rewarded target for a given block, or to either of the non-target stimuli, are punished with a timeout. Each session consists of six, ten-minute blocks that alternate between rewarding responses to the visual or auditory target. The initial block of each session alternates in which target is rewarded across days. In the standard version of the task, each block begins with five consecutive trials in which the rewarded-target stimulus is presented and mice are given a non-contingent reward at the end of the response window if the mouse does not respond prior to this to earn a contingent reward. For all other trials, stimuli are presented in pseudo-random order to ensure an approximately equal number of presentations of each stimulus each block.
Mice are deemed to have learned the context-switching task when their within-modality d' (comparing
responses to the rewarded target and the non-target of the same modality) and cross-modality d'
(comparing responses to the rewarded and non-rewarded targets) are both greater than 1.5 for at
least 4/6 blocks for two consecutive sessions. After learning the task, mice continue training as
described for stage 3 (except incorrect responses are no longer punished with timeouts) until they
are used for electrophysiology or behavior experiments.

## Data

The parquet files for this experiment contain data from one session per subject, after they reached the
learning criterion (Stage 3 passed). Each row corresponds to one trial, and the columns contain information about the
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