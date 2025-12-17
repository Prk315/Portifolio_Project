# Usage Guide

Quick guide to running the project components.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Running the Notebooks

Execute in this order for full pipeline:

```bash
jupyter notebook

# Then open in order:
# 1. notebook.ipynb - Exploratory data analysis
# 2. transform.ipynb - Data preprocessing
# 3. trainAndFeatureEngineer.ipynb - Baseline models
# 4. hyperparameterTuning.ipynb - Advanced models
# 5. improvedModels.ipynb - Class balancing with SMOTE
# 6. modelExplainability.ipynb - SHAP analysis
# 7. visualizeResults.ipynb - Result visualization
```

## Running the Streamlit Dashboard

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- Manual entry mode: Input student data via form
- CSV upload mode: Batch predictions from file
- Download predictions as CSV

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_preprocessing.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Model Performance Summary

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Gradient Boosting | 77.0% | 0.760 |
| Random Forest | 76.8% | 0.756 |
| Logistic Regression | 76.8% | 0.758 |

## Making Predictions

### Using Python

```python
import joblib
import numpy as np
import pandas as pd

# Load artifacts
model = joblib.load('data/best_model_final.joblib')
preprocessor = joblib.load('data/preprocessor_fe.joblib')
selector = joblib.load('data/selector.joblib')
le = joblib.load('data/label_encoder.joblib')

# Prepare your data
student_data = pd.DataFrame([{
    'Age at enrollment': 20,
    'Gender': 1,
    # ... all 36 features
}])

# Apply feature engineering (see app.py for engineer_features function)
student_fe = engineer_features(student_data)

# Transform and predict
X = preprocessor.transform(student_fe)
X_selected = selector.transform(X)
prediction = model.predict(X_selected)[0]
probabilities = model.predict_proba(X_selected)[0]

print(f"Prediction: {le.classes_[prediction]}")
print(f"Probabilities: {dict(zip(le.classes_, probabilities))}")
```

## Project Structure

```
.
├── README.md              # Main documentation
├── USAGE.md              # This file
├── requirements.txt       # Dependencies
├── app.py                # Streamlit dashboard
├── data/                 # Data and models
├── docs/                 # GitHub Pages site
├── tests/                # Unit tests
└── *.ipynb              # Jupyter notebooks
```

## Troubleshooting

### Missing Data Files
If you get errors about missing files in `data/`, run the notebooks in order to generate them.

### SHAP Installation Issues
If SHAP fails to install:
```bash
pip install shap --no-build-isolation
```

### Memory Issues
The full dataset requires ~2GB RAM. If you encounter memory errors, reduce sample sizes in notebooks.

## Further Reading

- [Dataset Source](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
