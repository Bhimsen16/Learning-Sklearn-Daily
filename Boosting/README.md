# The Boosting Chronicles: From Stumps to Champions

This repository documents my deep-dive architectural exploration of the **Ensemble Learning Boosting Family** using Scikit-Learn and XGBoost on Python 3.14 (Fedora 43 workstation environment).

## Algorithms Explored

### 1. Adaptive Boosting (AdaBoost)
* **Concept:** Corrects training errors by dynamically adjusting sample row weights. Sequential weak learners (Decision Stumps) focus on previously misclassified data.
* **Key Hyperparameters:** `n_estimators`, `learning_rate`.

### 2. Gradient Boosting (GBM)
* **Concept:** Optimizes a cost function by training subsequent trees strictly on the **mathematical residuals (errors)** of the combined previous predictions.
* **Key Architectural Insight:** Standard deeper trees (`max_depth=3`) overfit small datasets like the Wine dataset. Limiting tree complexity to stumps (`max_depth=1`) regularized the boundaries, jumping performance from `90.74%` to `100.00%`.

### 3. Stochastic Gradient Boosting (SGB)
* **Concept:** Integrates bagging characteristics into boosting. By introducing `subsample=0.8`, each sequential tree trains on a random 80% subset of rows.
* **Outcome:** Prevents the sequential trees from memorizing outliers, yielding pristine decision boundaries and a robust `100.00%` evaluation score.

### 4. eXtreme Gradient Boosting (XGBoost)
* **Concept:** The production heavyweight champion. Implements Gradient Boosting optimized with built-in L1/L2 regularization, advanced tree pruning, and extreme CPU parallelization features.

##  Performance Benchmarks (Wine Dataset)
| Algorithm | Configuration | Test Accuracy |
| :--- | :--- | :--- |
| **AdaBoost** | 100 Trees, LR=0.5 | `92.59%` |
| **Gradient Boosting** | Default (`max_depth=3`), LR=0.1 | `90.74%` |
| **Gradient Boosting** | Optimized Stumps (`max_depth=1`), LR=0.1 | `100.00%` |
| **Stochastic Gradient Boosting** | Stumps, `subsample=0.8` | `100.00%` |
| **XGBoost** | `XGBClassifier`, `subsample=0.8` | `100.00%` |

---
*Maintained as part of a structured machine learning optimization curriculum.*