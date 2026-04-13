## Credit Card Fraud Detection Predictive Models ##

**Project Objective**

- *Note:* Dealing with highly imbalanced data like this requires a careful approach to ensure that the model doesn't simply "guess" the majority class every time.

**Dataset Overview: European Credit Card Transactions**

This dataset archives credit card activity from September 2013 involving European users. It covers a 48-hour window and consists of 284,807 total transactions. The data is characterized by a significant class imbalance: only 492 instances are flagged as fraudulent, representing a mere 0.17% of the entire collection.


To protect **user privacy**, the original features have been obscured. Most of the input variables ($V1$ through $V28$) are numerical values derived from a **Principal Component Analysis (PCA)**. 

The only columns that remain in their original form are:
- **Time**: The number of seconds that passed between the current transaction and the very first entry in the set.
- **Amount**: The monetary value of the transaction, which is useful for cost-sensitive learning models.
- **Class**: The target variable, where **1 indicates a fraudulent transaction** and **0 indicates a legitimate one**.
