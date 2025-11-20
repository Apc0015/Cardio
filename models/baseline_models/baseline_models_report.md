
# CardioFusion Baseline Models Report
Generated: 2025-11-19 17:41:09

## Dataset Information
- Training samples: 454,084
- Testing samples: 113,522
- Features: 27
- Target balance: 50% No Disease, 50% Heart Disease (after SMOTE)

## Model Performance Summary
              Model  Accuracy  Precision  Recall  F1-Score  ROC-AUC  CV Mean  CV Std  Training Time (s)
Logistic Regression    0.8394     0.8481  0.8270    0.8374   0.9261   0.8377  0.0016             4.1786
      Decision Tree    0.8210     0.8291  0.8088    0.8188   0.9080   0.8197  0.0024             1.9713
      Random Forest    0.8340     0.8016  0.8876    0.8424   0.9175   0.8331  0.0016             7.2642

## Best Performing Models

- Accuracy: Logistic Regression (0.8394)
- Precision: Logistic Regression (0.8481)
- Recall: Random Forest (0.8876)
- F1-Score: Random Forest (0.8424)
- ROC-AUC: Logistic Regression (0.9261)

## Overall Best Model: Random Forest (Based on F1-Score)