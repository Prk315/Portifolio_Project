"""
Unit tests for preprocessing pipeline
"""

import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


class TestPreprocessing:
    """Test data preprocessing functionality"""

    @classmethod
    def setup_class(cls):
        """Load artifacts before running tests"""
        cls.preprocessor = joblib.load('data/preprocessor.joblib')
        cls.preprocessor_fe = joblib.load('data/preprocessor_fe.joblib')
        cls.selector = joblib.load('data/selector.joblib')
        cls.le = joblib.load('data/label_encoder.joblib')

    def test_preprocessor_exists(self):
        """Test that preprocessor artifacts exist"""
        assert Path('data/preprocessor.joblib').exists()
        assert Path('data/preprocessor_fe.joblib').exists()
        assert Path('data/selector.joblib').exists()
        assert Path('data/label_encoder.joblib').exists()

    def test_data_files_exist(self):
        """Test that processed data files exist"""
        assert Path('data/X_train_fe.npy').exists()
        assert Path('data/X_test_fe.npy').exists()
        assert Path('data/y_train_encoded.npy').exists()
        assert Path('data/y_test_encoded.npy').exists()

    def test_data_shapes(self):
        """Test that data has correct shapes"""
        X_train = np.load('data/X_train_fe.npy')
        X_test = np.load('data/X_test_fe.npy')
        y_train = np.load('data/y_train_encoded.npy')
        y_test = np.load('data/y_test_encoded.npy')

        # Check feature dimensions match
        assert X_train.shape[1] == X_test.shape[1] == 75, \
            f"Feature count should be 75, got {X_train.shape[1]}"

        # Check sample counts
        assert len(X_train) == len(y_train), "X_train and y_train length mismatch"
        assert len(X_test) == len(y_test), "X_test and y_test length mismatch"

        # Check train/test split ratio (roughly 80/20)
        total = len(X_train) + len(X_test)
        train_ratio = len(X_train) / total
        assert 0.75 < train_ratio < 0.85, f"Train ratio {train_ratio} not around 0.80"

    def test_label_encoding(self):
        """Test that label encoder has correct classes"""
        expected_classes = ['Dropout', 'Enrolled', 'Graduate']
        assert list(self.le.classes_) == expected_classes, \
            f"Expected {expected_classes}, got {list(self.le.classes_)}"

    def test_no_missing_values(self):
        """Test that transformed data has no NaN values"""
        X_train = np.load('data/X_train_fe.npy')
        X_test = np.load('data/X_test_fe.npy')

        assert not np.isnan(X_train).any(), "X_train contains NaN values"
        assert not np.isnan(X_test).any(), "X_test contains NaN values"

    def test_feature_selection(self):
        """Test that feature selector reduces dimensions correctly"""
        # Selector should reduce from 260 to 75 features
        assert self.selector.k == 75, f"Expected k=75, got k={self.selector.k}"

        support = self.selector.get_support()
        assert support.sum() == 75, f"Expected 75 selected features, got {support.sum()}"

    def test_preprocessor_transform(self):
        """Test that preprocessor can transform new data"""
        # Load sample raw data
        X_test_raw = pd.read_csv('data/X_test_raw.csv')

        # Take first row
        sample = X_test_raw.iloc[:1]

        # Transform should work without errors
        transformed = self.preprocessor.transform(sample)

        # Check output shape
        assert transformed.shape[1] == 251, \
            f"Preprocessor should output 251 features, got {transformed.shape[1]}"

    def test_class_distribution(self):
        """Test that classes are roughly balanced in training set"""
        y_train = np.load('data/y_train_encoded.npy')

        from collections import Counter
        counts = Counter(y_train)

        # Check we have all 3 classes
        assert len(counts) == 3, f"Expected 3 classes, got {len(counts)}"

        # Each class should have at least 10% of data
        min_count = len(y_train) * 0.10
        for class_id, count in counts.items():
            assert count > min_count, \
                f"Class {class_id} has only {count} samples (<10%)"


class TestModels:
    """Test model artifacts"""

    def test_baseline_models_exist(self):
        """Test that baseline models are saved"""
        assert Path('data/best_dt.joblib').exists()
        assert Path('data/best_lr.joblib').exists()
        assert Path('data/best_knn.joblib').exists()

    def test_model_can_predict(self):
        """Test that saved model can make predictions"""
        # Load model
        try:
            model = joblib.load('data/best_model_final.joblib')
        except:
            model = joblib.load('data/best_lr.joblib')

        # Load test data
        X_test = np.load('data/X_test_fe.npy')

        # Predict
        predictions = model.predict(X_test[:10])

        # Check predictions are valid
        assert len(predictions) == 10
        assert all(p in [0, 1, 2] for p in predictions), \
            f"Invalid prediction values: {set(predictions)}"

    def test_model_probabilities(self):
        """Test that model outputs valid probabilities"""
        try:
            model = joblib.load('data/best_model_final.joblib')
        except:
            model = joblib.load('data/best_lr.joblib')

        X_test = np.load('data/X_test_fe.npy')

        # Get probabilities
        probas = model.predict_proba(X_test[:10])

        # Check shape (n_samples, n_classes)
        assert probas.shape == (10, 3), \
            f"Expected shape (10, 3), got {probas.shape}"

        # Check probabilities sum to 1
        sums = probas.sum(axis=1)
        assert np.allclose(sums, 1.0), \
            f"Probabilities don't sum to 1: {sums}"

        # Check probabilities are between 0 and 1
        assert (probas >= 0).all() and (probas <= 1).all(), \
            "Probabilities outside [0, 1] range"


class TestResults:
    """Test result files"""

    def test_results_csv_exist(self):
        """Test that result CSV files are created"""
        assert Path('data/baseline_results.csv').exists()
        assert Path('data/comparison_results.csv').exists()

    def test_results_have_content(self):
        """Test that results files are not empty"""
        baseline = pd.read_csv('data/baseline_results.csv')
        comparison = pd.read_csv('data/comparison_results.csv')

        assert len(baseline) > 0, "baseline_results.csv is empty"
        assert len(comparison) > 0, "comparison_results.csv is empty"

        # Check expected columns exist
        expected_cols = ['Model', 'Test Accuracy', 'Test F1']
        for col in expected_cols:
            assert col in baseline.columns, f"Missing column {col} in baseline_results"

    def test_accuracy_values_valid(self):
        """Test that accuracy values are in valid range"""
        comparison = pd.read_csv('data/comparison_results.csv')

        # Accuracy should be between 0 and 1
        accuracies = comparison['Test Accuracy']
        assert (accuracies >= 0).all() and (accuracies <= 1).all(), \
            f"Invalid accuracy values: {accuracies.values}"

        # For this dataset, expect accuracies > 0.6
        assert (accuracies > 0.6).all(), \
            f"Unexpectedly low accuracies: {accuracies.values}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
