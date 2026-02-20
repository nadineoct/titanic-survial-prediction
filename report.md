
## Files produced:
[classification_model_comparison.csv](classification_model_comparison.csv) 
[roc_curves_comparison.png](roc_curves_comparison.png)
[knn_accuracy_vs_k.png](knn_accuracy_vs_k.png)

# Final Analysis and Reporting

## a. Model Selection Justification

Based on the comparison table (`classification_model_comparison.csv`) and the ROC curve plot (`roc_curves_comparison.png`),  
the best model is **KNN (Optimized)**, achieving the highest ROC AUC score of **0.8478**.  

This model also performs competitively across Accuracy (**0.8212**) and F1-Score (**0.7500**),  
making it the most balanced choice for deployment.

---

## b. Hyperparameter Insights

### KNN
From the KNN tuning with the Elbow Method, we observed that performance improved up to a certain `k` before plateauing or declining.  
This demonstrates the importance of tuning hyperparameters, as smaller `k` values may lead to high variance (overfitting),  
while larger values may oversmooth the decision boundary (underfitting).

### Decision Tree
- **Unconstrained Tree**:  
  Training Accuracy = 0.9817, Testing Accuracy = 0.8156  

- **Pruned Tree**:  
  Training Accuracy = 0.8652, Testing Accuracy = 0.7598  

The unconstrained tree shows a large gap between training and testing accuracy, which is a clear sign of **overfitting**.  
After pruning, the gap reduces, indicating better generalization.

---

## c. Business Context and Metric Choice

In the Titanic survivor prediction context, the priority should be on **Recall** rather than Precision.  
- A **False Negative** (predicting someone will not survive when they actually would) could mean a passenger is denied rescue, leading to loss of life.  
- A **False Positive** (predicting survival when they would not) may waste rescue resources, but this is less critical compared to missing an actual survivor.  

Therefore, maximizing **Recall** ensures that as many true survivors as possible are identified and allocated rescue resources,  
which aligns with the life-saving goal in this real-world scenario.

---

## Quick Summary
- **Dropped columns**: ['PassengerId', 'Name', 'Ticket', 'Cabin']
- **Numeric features**: ['Age', 'SibSp', 'Parch', 'Fare']
- **Categorical features**: ['Sex', 'Embarked', 'Pclass']
- **Best k (KNN)**: 5
- **Best model (by AUC then Accuracy)**: KNN (Optimized)
- **Comparison table columns**: ['Model', 'Accuracy', 'F1-Score', 'ROC AUC']
- **DecisionTree Unconstrained (train_acc, test_acc)**: (0.9817415730337079, 0.8156424581005587)
- **DecisionTree Pruned (train_acc, test_acc)**: (0.8651685393258427, 0.7597765363128491)
