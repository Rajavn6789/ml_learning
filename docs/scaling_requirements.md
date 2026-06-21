# Which Models Need Feature Scaling?

## Scaling Recommended
- Linear Regression (especially gradient descent versions)
- Logistic Regression
- KNN
- K-Means
- SVM
- PCA
- Neural Networks / PyTorch / TensorFlow

## Scaling Usually Not Required
- Decision Trees
- Random Forest
- XGBoost
- LightGBM
- CatBoost

**Why:** Tree-based models split data based on thresholds rather than distances,
so scaling usually has little effect. Distance- and gradient-based models are
sensitive to feature magnitudes, so scaling helps them converge and compare
features fairly.
