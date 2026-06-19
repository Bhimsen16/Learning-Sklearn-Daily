# Model Evaluation Toolkit

This folder contains core implementations and scripts dedicated to moving beyond raw accuracy and understanding how to robustly evaluate machine learning models using Scikit-Learn.

## Key Milestones Covered

### 1. The Confusion Matrix (`Day14_Matrix/`)
* **Core Concept:** Breaks down classification results into True Positives, True Negatives, False Positives, and False Negatives.
* **Why it matters:** It exposes exactly *where* a model is misclassifying data, which raw accuracy completely hides.

### 2. Precision-Recall Trade-Off (`Day15_TradeOff/`)
* **Core Concept:** Visualizing the mathematical tug-of-war between Precision (minimizing false alarms) and Recall (minimizing missed targets).
* **Implementation:** Plotting custom decision thresholds to align model behavior with specific domain goals.

### 3. ROC-AUC Curve (`Day16_ROC/`)
* **Core Concept:** Plotting the True Positive Rate against the False Positive Rate using predicted **soft probabilities** rather than hard binary outputs.
* **Why it matters:** The Area Under the Curve (AUC) provides a single, threshold-independent benchmark score ranging from `0.5` (random guessing) to `1.0` (perfect separation).

## Environment & Tools
* **OS:** Fedora Linux
* **Language:** Python 3
* **Libraries:** `scikit-learn`, `pandas`, `matplotlib`