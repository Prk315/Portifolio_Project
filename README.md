# Student Dropout Prediction

Predicting student academic outcomes (Graduate/Dropout/Enrolled) using machine learning on real university enrollment data.

## Overview

This project analyzes student performance data to predict whether students will graduate, drop out, or remain enrolled. The goal is to identify at-risk students early so institutions can provide targeted support.

**Dataset**: [UCI ML Repository - Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- 4,424 students
- 36 features (demographics, academic performance, socioeconomic factors)
- 3 outcome classes: Graduate, Dropout, Enrolled

## Key Results

| Model | Test Accuracy | Test F1 Score | Notes |
|-------|--------------|---------------|-------|
| Logistic Regression | 76.8% | 0.758 | Best baseline model |
| Decision Tree | 72.3% | 0.724 | Baseline |
| Decision Tree + FE | 75.7% | 0.747 | +3.4% with feature engineering |
| Random Forest | 76.8% | 0.756 | Good generalization |
| Gradient Boosting | 77.0% | 0.760 | **Best overall model** |
| AdaBoost | 75.4% | 0.746 | Decent performance |

**Feature engineering improved Decision Tree accuracy by 3.4 percentage points.**

## What I Built

This is an end-to-end ML pipeline demonstrating:

### 1. Data Preprocessing & EDA
- Exploratory data analysis with 10+ visualizations
- Correlation analysis identifying key predictors
- Pipeline-based preprocessing (StandardScaler + OneHotEncoder)
- Proper train/test split with stratification

### 2. Feature Engineering
Created 9 domain-specific features:
- `approval_rate`: ratio of approved to enrolled units
- `avg_grade`: mean performance across semesters
- `grade_improvement`: academic trajectory indicator
- `total_approved`, `total_enrolled`, `total_credited`
- Age grouping and other aggregations

### 3. Model Development
- **Baseline models**: Logistic Regression, Decision Tree, KNN
- **Ensemble methods**: Random Forest, AdaBoost, Gradient Boosting
- GridSearchCV for hyperparameter tuning (5-fold CV)
- Proper evaluation with multiple metrics

### 4. Model Interpretation
- Feature importance analysis
- Confusion matrices for all models
- ROC curves (multi-class)
- Learning curves

## Top Predictive Features

From Random Forest feature importance:
1. 2nd semester grades
2. 2nd semester approved units
3. 1st semester grades
4. Previous qualification grade
5. Approval rate (engineered feature)

**Insight**: Current academic performance is the strongest predictor of outcomes, but admission qualifications and engineered engagement metrics also matter.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── data.csv                    # Original dataset
│   ├── *.joblib                    # Saved models & preprocessors
│   └── *.csv/*.npy                 # Processed data splits
├── notebook.ipynb                  # Exploratory data analysis
├── transform.ipynb                 # Data preprocessing pipeline
├── trainAndFeatureEngineer.ipynb   # Model training & feature engineering
├── hyperparameterTuning.ipynb      # Ensemble methods & tuning
└── visualizeResults.ipynb          # Results visualization
```

## Reproducibility

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/Portifolio_Project.git
cd Portifolio_Project

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Running the Pipeline
Execute notebooks in this order:
1. `notebook.ipynb` - EDA and data exploration
2. `transform.ipynb` - Preprocessing and train/test split
3. `trainAndFeatureEngineer.ipynb` - Baseline models + feature engineering
4. `hyperparameterTuning.ipynb` - Advanced ensemble models
5. `visualizeResults.ipynb` - Generate visualizations

All intermediate artifacts are saved in `data/` for reproducibility.

## Tech Stack

- **ML/Data**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Environment**: Jupyter Notebook, Python 3.8+

## Key Insights

1. **Class imbalance exists**: 40% Graduate, 26% Dropout, 14% Enrolled
   - This affects model performance on the "Enrolled" class

2. **Academic performance dominates**: Semester grades and approved units are the strongest predictors

3. **Feature engineering helps**: Adding domain knowledge improved Decision Tree performance by 3.4%

4. **Ensemble methods win**: Random Forest and Gradient Boosting outperform simple classifiers

5. **Economic factors have limited impact**: GDP, inflation, and unemployment rates show weak correlations with outcomes

## Limitations & Future Work

- **Performance ceiling**: ~77% accuracy suggests additional features or data needed
- **Class imbalance**: Could try SMOTE, class weighting, or threshold adjustment
- **Deployment**: Build REST API or Streamlit dashboard for predictions
- **Explainability**: Add SHAP values for individual prediction explanations
- **Temporal validation**: Current split is random; time-based split would be more realistic

## About

This project was developed as part of my machine learning portfolio to demonstrate practical ML engineering skills. The dataset is clean (0 missing values) which allowed me to focus on modeling techniques rather than extensive data cleaning.

## License

Dataset from UCI ML Repository. Code is MIT licensed.
