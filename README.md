# Titanic Survival Prediction

A machine learning classification project that predicts passenger survival aboard the Titanic using four algorithms: K-Nearest Neighbors, Gaussian Naive Bayes, Support Vector Machine, and Multilayer Perceptron.

## Project Overview

This project applies supervised learning to the Titanic dataset to classify whether a passenger survived the disaster. It covers the full ML workflow — data preprocessing, feature engineering, model training, hyperparameter tuning, and comparative evaluation — implemented across two Jupyter notebooks with slightly different feature sets and preprocessing strategies.

## Dataset

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `titanic.csv` | 1,309 | 11 | Training data with survival labels |
| `test.csv` | 19 | 10 | Holdout data for final predictions |

**Features:** `pclass`, `name`, `sex`, `age`, `sibsp`, `parch`, `ticket`, `fare`, `cabin`, `embarked`, `survived` (target)

## Repository Structure

```
titanic-survival-prediction/
├── titanic.csv          # Training dataset
├── test.csv             # Test dataset (holdout)
├── ML Project.ipynb     # KNN, Naive Bayes, SVM, MLP (with Family feature)
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Methodology

### Data Preprocessing

- Dropped non-essential columns: `name`, `ticket`, `cabin`, `fare`
- Imputed missing `age` values with the mean (Notebook 1) or median (Notebook 2)
- Imputed missing `embarked` values with the mode
- Encoded categorical variables: `sex` (male=1, female=0), `embarked` (S=1, C=2, Q=3)
- Converted `age` to integer type

### Feature Engineering

Notebook 2 introduces a `Family` feature by combining `sibsp + parch + 1`, then drops the original columns. Notebook 1 retains `sibsp` and `parch` as separate features.

### Train/Test Split

- 80/20 split with `random_state=42` (Notebook 1) or `random_state=0` (Notebook 2)
- Training set: 1,047 samples | Test set: 262 samples

### Feature Scaling

`StandardScaler` applied to all features after splitting to prevent data leakage.

## Models & Results

### Notebook 1 (`ml.ipynb`)

Features: `pclass`, `sex`, `age`, `sibsp`, `parch`, `embarked`

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| KNN (k=11) | 77.1% | 86.3% | 58.5% | 69.7% |
| Gaussian Naive Bayes | 74.8% | 76.0% | 64.4% | 69.7% |
| SVM (linear) | 75.6% | 78.1% | 63.6% | 70.1% |
| MLP (100, 50) | 77.9% | 84.9% | 61.9% | 71.6% |

**KNN Distance Metrics:**

| Metric | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
| Euclidean | 77.1% | 86.3% | 58.5% | 69.7% |
| Manhattan | 76.0% | 85.7% | 55.9% | 67.7% |
| Chebyshev | 77.9% | 86.6% | 60.2% | 71.0% |

**SVM Grid Search:** Best parameters: `C=1, gamma=0.1, kernel=rbf`

### Notebook 2 (`ML Project.ipynb`)

Features: `pclass`, `sex`, `age`, `embarked`, `Family`

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| KNN (k=17) | 82.1% | 83.5% | 66.0% | 73.7% |
| Gaussian Naive Bayes | 78.6% | 75.6% | 65.0% | 69.9% |
| SVM (tuned) | 80.9% | 86.8% | 59.0% | 70.2% |
| MLP (100, 50) | 80.2% | 81.6% | 62.0% | 70.5% |

**KNN Distance Metrics (k=17):**

| Metric | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
| Euclidean | 82.1% | 83.5% | 66.0% | 73.7% |
| Manhattan | 80.5% | 81.0% | 64.0% | 71.5% |
| Chebyshev | 82.1% | 84.4% | 65.0% | 73.4% |

**SVM Grid Search:** Best parameters: `C=100, gamma=scale, kernel=poly`

## Technologies

- **Python 3.12**
- **pandas** — data manipulation
- **scikit-learn** — ML models, preprocessing, metrics
- **matplotlib** / **seaborn** — visualization
- **Jupyter Notebook** — development environment

## Setup & Installation

```bash
# Clone the repository
git clone <repo-url>
cd titanic

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install pandas scikit-learn matplotlib seaborn jupyter

# Launch Jupyter
jupyter notebook
```

## How to Run

Open either notebook in Jupyter and execute cells top to bottom. Both notebooks are self-contained — they load the CSV files, preprocess data, train models, and display results.

- `ml.ipynb` — Original analysis with 6 features
- `ML Project.ipynb` — Refined analysis with Family feature and additional visualizations

## Key Insights

- The `Family` feature (Notebook 2) improved KNN accuracy from 77.1% to 82.1%, suggesting family size is a meaningful predictor of survival.
- KNN with Euclidean or Chebyshev distance consistently outperformed Manhattan distance across both notebooks.
- All models showed higher precision than recall, indicating they were more conservative in predicting survival (fewer false positives, more false negatives).
- The MLP model achieved competitive results without hyperparameter tuning, suggesting potential for further improvement with optimization.

## Limitations

- The test dataset (`test.csv`) contains only 19 samples, making final predictions less reliable for generalization.
- Mean/median imputation for missing ages (20% of training data) may introduce bias.
- The `fare` column was dropped despite its potential correlation with survival.
- No cross-validation was used for model selection in Notebook 1; a single train/test split was used instead.

## Future Improvements

- Apply cross-validation for more robust model evaluation
- Explore additional features (e.g., fare, cabin deck extraction, title extraction from names)
- Perform hyperparameter tuning for KNN and MLP models
- Address class imbalance with techniques like SMOTE
- Export predictions to CSV for Kaggle submission
