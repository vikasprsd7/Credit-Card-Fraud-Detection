## Credit Card Fraud Detection Predictive Models ##

Fraud is only **0.17%** of card transactions, so a model that never flags fraud is still 99.83% "accurate". This project compares two ways of training a model on such lopsided data, **random undersampling** and **SMOTE oversampling**, and judges them with precision, recall and confusion matrices on a real, untouched test set instead of plain accuracy.

## Table of contents
 
- [Business problem](#business-problem)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Key takeaways](#key-takeaways)
- [Limitations and next steps](#limitations-and-next-steps)
- [Getting started](#getting-started)
- [Repository structure](#repository-structure)
- [Skills demonstrated](#skills-demonstrated)

## Business problem
 
A bank has a split second to approve or block a card payment, and it can be wrong in two ways:
 
- **Missed fraud (false negative):** a stolen card is accepted and money is lost.
- **False alarm (false positive):** a genuine customer is blocked, which hurts trust and increases support load.
The goal is to catch as much fraud as possible without drowning the fraud team (and customers) in false alarms.
 ## Dataset
 
- **Source:** the public *Credit Card Fraud Detection* dataset on Kaggle (ULB Machine Learning Group): <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>
- **Size:** 284,807 transactions, 31 columns, about 48 hours of data.
- **Fraud cases:** 492 (0.17%).
- **Columns:** `Time`, `Amount`, `Class` (0 = genuine, 1 = fraud) and `V1`–`V28`, which are anonymised features produced by a PCA transformation for privacy.

**Dataset Overview: European Credit Card Transactions**

This dataset archives credit card activity from September 2013 involving European users. It covers a 48-hour window and consists of 284,807 total transactions. The data is characterized by a significant class imbalance: only 492 instances are flagged as fraudulent, representing a mere 0.17% of the entire collection.


To protect **user privacy**, the original features have been obscured. Most of the input variables ($V1$ through $V28$) are numerical values derived from a **Principal Component Analysis (PCA)**. 

The only columns that remain in their original form are:
- **Time**: The number of seconds that passed between the current transaction and the very first entry in the set.
- **Amount**: The monetary value of the transaction, which is useful for cost-sensitive learning models.
- **Class**: The target variable, where **1 indicates a fraudulent transaction** and **0 indicates a legitimate one**.


**Key Considerations for Analysis**

Because of the extreme scarcity of the positive class ($0.172\%$), standard accuracy metrics will likely be misleading. When working with this data, one should focus on:
- **Area Under the Precision-Recall Curve (AUPRC)**: Since the dataset is skewed, the Confusion Matrix and AUPRC are generally more informative than the standard Accuracy score.
- **Data Resampling**: Techniques like **SMOTE (Synthetic Minority Over-sampling Technique)** or random under-sampling may be necessary to balance the training set.
- **Scaling**: While the PCA components ($V1-V28$) are already scaled, the Time and Amount features are not and will likely require normalization or standardization before being fed into most machine learning algorithms.


## Approach
 
1. **Explore:** check missing values and visualise the class imbalance and the `Amount` / `Time` distributions.
2. **Scale:** apply `RobustScaler` to `Amount` and `Time` (robust to outliers).
3. **Hold out a real test set:** stratified 5-fold split, keeping the last fold. Train: 227,846 rows. Test: 56,961 rows (98 frauds, 56,863 genuine). The test set stays unbalanced, like real life.
4. **Method A, random undersampling:** keep all 492 frauds plus 492 random genuine transactions (50/50 sample).
5. **Feature insight:** correlation heatmaps and box plots on the balanced sample. `V17, V14, V12, V10` are lower for fraud; `V11, V4, V2, V19` are higher.
6. **Outlier removal:** IQR rule (1.5 × IQR) on `V14`, `V12`, `V10`; about 950 rows remain.
7. **Visual separability:** t-SNE, PCA and Truncated SVD projections to 2D.
8. **Classic models:** Logistic Regression, K-Nearest Neighbours, SVC and Decision Tree, tuned with `GridSearchCV` and cross-validated.
9. **Method B, SMOTE:** synthetic fraud examples are created **inside** each cross-validation fold (training rows only) to avoid data leakage.
10. **Neural network comparison:** the same small Keras network (30 → 32 → 2, about 1,988 parameters, 20 epochs, Adam, learning rate 0.001) is trained on undersampled data and on SMOTE data, then both are tested on the original test set.









