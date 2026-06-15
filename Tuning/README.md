# Hyperparameter Optimization Sprint

Welcome to the Hyperparameter Tuning module! This directory documents the transition from manual model configuration to automated, hardware-accelerated optimization pipelines using Scikit-Learn and XGBoost.

## Core Concepts Explored

* **GridSearchCV:** A systematic, brute-force search that evaluates every single coordinate intersection across a predefined grid of hyperparameters.
* **RandomizedSearchCV:** A highly efficient statistical alternative that randomly samples a fixed number of configurations (`n_iter`) from wide parameter distributions, drastically saving training time while finding top-tier estimators.
* **K-Fold Cross-Validation (`cv=5`):** Ensuring model stability and reducing variance by splitting the training dataset into 5 distinct folds, systematically rotating which fold acts as the validation set.
* **Parallel Computing (`n_jobs=-1`):** Bypassing Python's single-core limitations to distribute cross-validation chunks across all available hardware threads simultaneously.

---

## Code Architecture

### 1. Grid Search (`grid_hunt.py`)
Systematically checks a focused matrix of configurations to establish a strong benchmark performance using explicit parameters.

### 2. Randomized Search (`random_hunt.py`)
Scales the exploration space up to **750 potential combinations** across multiple parameters (`n_estimators`, `learning_rate`, `max_depth`, `subsample`). By capping execution at 10 random iterations, it finds the optimal champion model in a fraction of the time.

---

## Performance & Hardware Verification

During optimization pipelines, local hardware execution was monitored using the GNOME System Monitor under Fedora:

* **Multi-Processing Efficiency:** Setting `n_jobs=-1` successfully saturated all 12 logical threads of the CPU to near **100% utilization**, completing thousands of model iterations in seconds.
* **Optimization Results:** Both pipelines successfully isolated the optimal hyperparameter recipes, scoring an impressive **96.80% Cross-Validation Accuracy** and finalizing a robust **100.00% Accuracy on the Test Set**.

---
*Next Stop: Model Evaluation Strategy (Confusion Matrix, Precision, & Recall)*