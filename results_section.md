## 5. Results

### 5.1 Neural network on the real test set (98 frauds, 56,863 genuine)

| | Trained on undersampled data | Trained on SMOTE data |
| :--- | :--- | :--- |
| Frauds caught | 92 | 72 |
| Frauds missed | 6 | 26 |
| Genuine transactions wrongly flagged | 1,550 | 15 |
| Recall (share of frauds caught) | about 93.9% | about 73.5% |
| Precision (share of alerts that are truly fraud) | about 5.6% | about 82.8% |
| Alerts raised per real fraud caught | about 18 | about 1.2 |
| Genuine customers wrongly blocked | about 2.7% | about 0.03% |
| Overall accuracy | about 97.3% | about 99.93% |

(The percentages are calculated from the confusion matrices printed in the notebook.)

### 5.2 Logistic Regression on the real test set

| Version | Fraud precision | Fraud recall | Fraud F1 | Average precision (PR curve) |
| :--- | :--- | :--- | :--- | :--- |
| Trained with SMOTE | 0.11 | 0.86 | 0.20 | 0.76 |
| Trained on undersampled data | not reported | not reported | not reported | 0.07 |

SMOTE-trained Logistic Regression catches about 86% of frauds, but only about 1 in 9 of its alerts is a real fraud. The much higher precision-recall score for SMOTE (0.76 versus 0.07) shows that it ranks suspicious transactions far better overall.
