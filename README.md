# Evaluation of agentic AI tools for modelling data from the Dynamic Routing project

## Investigation 1 - Behavior
### Input
The following were given to Claude Code w/Opus 4.6 (thinking, 1M token context window), with the instruction to use `uv` for dependency management:
- [`PROMPT.md`](https://github.com/AllenNeuralDynamics/dr-llm-modelling-tests/blob/c4e2ad7f0b836352a73427edbe4b0bf66cb4c0ed/PROMPT.md)
- [`EXPERIMENT_DESCRIPTION.md`](https://github.com/AllenNeuralDynamics/dr-llm-modelling-tests/blob/c4e2ad7f0b836352a73427edbe4b0bf66cb4c0ed/EXPERIMENT_DESCRIPTION.md)
- [`data/`](https://github.com/AllenNeuralDynamics/dr-llm-modelling-tests/tree/c4e2ad7f0b836352a73427edbe4b0bf66cb4c0ed/data)

### Output
- [`PLAN.md`](https://github.com/AllenNeuralDynamics/dr-llm-modelling-tests/blob/main/PLAN.md)
- code to investigate one model per subfolder in [`solutions/`](https://github.com/AllenNeuralDynamics/dr-llm-modelling-tests/tree/c4e2ad7f0b836352a73427edbe4b0bf66cb4c0ed/solutions)
- [`SUMMARY.md`](https://github.com/AllenNeuralDynamics/dr-llm-modelling-tests/blob/c4e2ad7f0b836352a73427edbe4b0bf66cb4c0ed/SUMMARY.md)

### Notes
- initially, no guidance was given other than "do what you think is best" when prompted with choices.
- after the initial round of model investigations, the agent was prompted to "consider context", and produced the three `context_*` models 
