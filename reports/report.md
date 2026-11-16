# Evaluation report (model v0.1.0)

Training rows: 1134 (298 succeeded, 836 failed). Majority-class accuracy: 73.7%.

## Primary estimate: repeated 5-fold cross-validation (3 repeats)

| metric | mean | std |
|---|---|---|
| auc | 0.830 | 0.027 |
| balanced_acc | 0.751 | 0.038 |
| accuracy | 0.738 | 0.032 |
| f1_success | 0.610 | 0.044 |
| brier | 0.170 | 0.011 |

Accuracy is measured at a 0.5 threshold on a class-balanced score. Because only ~25% of rows are successes, balanced accuracy and AUC are the fairer headline numbers than raw accuracy.

## Hold-out (80/20) and overfitting check

| split | AUC | balanced acc | accuracy |
|---|---|---|---|
| train | 0.855 | 0.777 | 0.767 |
| test | 0.801 | 0.743 | 0.731 |

Test confusion matrix [[TN, FP], [FN, TP]]: [[120, 47], [14, 46]]

Learning curve (AUC): n=181: train 0.862 / CV 0.803; n=362: train 0.865 / CV 0.821; n=544: train 0.853 / CV 0.821; n=725: train 0.851 / CV 0.825; n=907: train 0.848 / CV 0.830

## What each input contributes (CV AUC)

| inputs | AUC | balanced acc |
|---|---|---|
| funding only | 0.692 | 0.661 |
| country only | 0.597 | 0.560 |
| industry only | 0.686 | 0.666 |
| funding + country | 0.774 | 0.730 |
| funding + industry | 0.799 | 0.731 |
| all three (final) | 0.829 | 0.752 |

## Score bands (from out-of-fold scores)

- **failure_like** (score < 0.35): 514 startups; share actually failed = 93.2%
- **mixed** (0.35 <= score <= 0.65): 330 startups; share actually succeeded = 26.1%
- **success_like** (score > 0.65): 290 startups; share actually succeeded = 61.0%

## Score vs. observed success rate in the data

| score range | n | share that succeeded |
|---|---|---|
| 0.00-0.20 | 343 | 3.8% |
| 0.20-0.35 | 171 | 12.9% |
| 0.35-0.50 | 152 | 19.1% |
| 0.50-0.65 | 178 | 32.0% |
| 0.65-0.80 | 150 | 54.0% |
| 0.80-1.00 | 140 | 68.6% |
