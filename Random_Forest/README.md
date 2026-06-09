Random Forest Ecosystem & Performance Optimization
A deep-dive implementation of Ensemble Learning (Bagging) using Scikit-Learn. This module covers the architectural transition from single high-variance Decision Trees to robust parallelized tree committees, hyperparameter tuning, feature selection logic, and Out-of-Bag (OOB) validation.

The Architecture: Decision Tree vs. Random Forest
The Single Tree Vulnerability: A single Decision Tree splits data until maximum purity, making it highly prone to high variance (overfitting). It memorizes noise.

The Forest Solution: Random Forests use Bagging (Bootstrap Aggregating) and Feature Subspace Sampling. By training $N$ trees on random row subsets and random column subsets, the individual high-variance errors cancel out.

The Result: It shifts the model from individual bias to collective stability, drastically lowering variance without increasing bias.

Practical Implementations & Insights
1. Hyperparameter Grid Tuning (/Day6_First_Model)
Experimented with n_estimators and max_depth boundaries.Handeled automated scripting to find the sweet spot where the model generalizes perfectly without wasting compute cycles.

2. Feature Selection & Noise Reduction (/Day7_Feature_Importance)
Extracted MDI (Mean Decrease in Impurity) scores from the Wine dataset—identifying Flavanoids as the dominant feature ($\approx 20\%$ importance).The Constraint Test: When limiting max_depth=1 on the Iris dataset, dropping the lowest 6 features caused the Lean Forest (7 features) to outperform the Full Forest (13 features) ($98.15\%$ vs $96.30\%$). This proved that removing noisy features directly prevents overfitting under tight structural constraints.

3. Out-of-Bag (OOB) Validation (/Day8_OOB_Score)
Utilized the built-in $\approx 37\%$ left-out bootstrapped samples to grade the model during a full-fit training run.Verified that the OOB Score ($98.39\%$) closely tracks traditional test-split validation accuracy without needing to sacrifice data to a separate test partition.