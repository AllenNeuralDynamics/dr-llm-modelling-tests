# GEE (Population-Averaged GLMM) -- Results

## Fixed effects

| Feature | Coef | SE | z | p-value | OR | OR 95% CI |
|---|---|---|---|---|---|---|
| previous_response | +0.5775 | 0.0416 | +13.88 | 8.65e-44 | 1.7816 | [1.6421, 1.9331] |
| previous_reward | +0.1578 | 0.0433 | +3.65 | 2.66e-04 | 1.1710 | [1.0757, 1.2746] |
| is_vis_target | +4.9884 | 0.1279 | +39.01 | 0.00e+00 | 146.6961 | [114.1773, 188.4767] |
| is_vis_non_target | +0.7413 | 0.1260 | +5.88 | 4.06e-09 | 2.0988 | [1.6394, 2.6869] |
| is_aud_target | +4.9823 | 0.1329 | +37.50 | 1.06e-307 | 145.8105 | [112.3792, 189.1873] |
| is_aud_non_target | +2.2555 | 0.1336 | +16.88 | 6.50e-64 | 9.5400 | [7.3418, 12.3965] |
| Intercept | -4.3138 | 0.1338 | -32.25 | 3.96e-228 | 0.0134 | [0.0103, 0.0174] |

## Model summary

- **Family**: Binomial (logit link)
- **Correlation structure**: Exchangeable
- **Number of groups (subjects)**: 113
- **Scale**: 1.0000

## Test-set evaluation

| Metric | Value |
|---|---|
| Accuracy | 0.8169 |
| Balanced accuracy | 0.8335 |
| AUC-ROC | 0.8709 |
| Log-loss | 0.4113 |

## Confusion matrix

```
              Predicted 0  Predicted 1
  Actual 0        29118         8803
  Actual 1         2277        20309
```
