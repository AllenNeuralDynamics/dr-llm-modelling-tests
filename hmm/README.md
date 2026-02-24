# Hidden Markov Model (HMM)

## Model description

A Hidden Markov Model treats the mouse's internal cognitive state as a latent (hidden) variable that evolves over the course of a session. On each trial, the mouse is in one of *K* unobserved states, and the observable trial features (stimulus type, previous response, previous reward) are emitted according to state-specific Gaussian distributions. Transitions between states follow a first-order Markov chain.

## Rationale

The context-switching task explicitly requires mice to alternate between visual-rewarded and auditory-rewarded behavioral modes across blocks. An HMM is a natural fit because:

1. **Latent states map onto cognitive modes.** The hidden states can capture whether the mouse is currently "attending to visual" vs "attending to auditory" stimuli, or is in a disengaged/exploratory mode.
2. **The transition matrix reveals context-switching dynamics.** Off-diagonal entries quantify how readily mice shift between modes, while diagonal entries capture state persistence.
3. **Sequential structure is modeled directly.** Unlike trial-independent classifiers, the HMM uses the temporal ordering of trials, which is central to how block transitions unfold.
4. **Emission parameters are interpretable.** The mean feature vector for each state shows which stimulus-response patterns characterize each cognitive mode.

## Approach

- Each trial is encoded as a 6-dimensional feature vector (the boolean stimulus and history features, cast to float).
- A `GaussianHMM` from `hmmlearn` is fit on training data, with each subject's trial sequence treated as a separate sequence via the `lengths` parameter.
- Model selection over `n_components` in {2, 3, 4} is performed using BIC on training data.
- Prediction: hidden states are decoded on the test set, and the state-conditional probability P(target_response=True | state) (estimated from training data) is used as the predicted probability.

## Metrics

- Accuracy, balanced accuracy
- AUC-ROC
- Log-loss
