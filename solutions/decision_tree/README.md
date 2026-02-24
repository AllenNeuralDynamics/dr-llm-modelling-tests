# Decision Tree (CART)

## Model
Scikit-learn `DecisionTreeClassifier` trained on 6 boolean features to predict
`target_response` (lick/no-lick) in a mouse audio-visual context-switching task.

## Rationale
- Produces human-readable if/then rules that can be directly inspected by
  experimenters to understand which stimulus and history features drive
  responding.
- Naturally captures non-linear interactions (e.g., stimulus identity combined
  with previous reward) without requiring explicit interaction terms.
- The tree can be visualized as a flowchart, making it easy to communicate
  findings.
- Cross-validation over `max_depth` balances interpretability (shallow trees)
  against predictive performance.

## Features
| Feature | Description |
|---|---|
| `previous_response` | Whether the mouse licked on the previous trial |
| `previous_reward` | Whether the mouse received a reward on the previous trial |
| `is_vis_target` | Visual target stimulus presented |
| `is_vis_non_target` | Visual non-target stimulus presented |
| `is_aud_target` | Auditory target stimulus presented |
| `is_aud_non_target` | Auditory non-target stimulus presented |

## Usage
```bash
uv run python decision_tree/fit.py
```

## Outputs
- `decision_tree/REPORT.md` -- auto-generated metrics, tree rules, and feature importances
- `decision_tree/tree_plot.png` -- matplotlib visualization of the fitted tree
