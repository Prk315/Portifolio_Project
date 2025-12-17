"""Fraud detection models with imbalanced learning techniques"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from typing import Tuple, Dict, Optional
import os


class FraudDetector:
    """
    Fraud detection model with support for imbalanced learning.

    Key features:
    - Handles severe class imbalance (0.17% fraud rate)
    - Optimizes for precision-recall tradeoff
    - Supports threshold tuning for production deployment
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        use_smote: bool = True,
        use_undersampling: bool = False,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42
    ):
        """
        Args:
            model_type: One of ["logistic", "random_forest", "gradient_boosting"]
            use_smote: Whether to use SMOTE oversampling
            use_undersampling: Whether to use random undersampling
            class_weight: Class weighting strategy ("balanced" or None)
            random_state: Random seed
        """
        self.model_type = model_type
        self.use_smote = use_smote
        self.use_undersampling = use_undersampling
        self.class_weight = class_weight
        self.random_state = random_state

        self.model = None
        self.pipeline = None
        self.best_threshold = 0.5
        self.feature_names = None

        self._build_model()

    def _build_model(self):
        """Build the model pipeline"""
        steps = []

        # Sampling strategies for imbalanced data
        if self.use_smote:
            steps.append(('smote', SMOTE(random_state=self.random_state)))

        if self.use_undersampling:
            steps.append(('undersample', RandomUnderSampler(random_state=self.random_state)))

        # Model selection
        if self.model_type == "logistic":
            model = LogisticRegression(
                class_weight=self.class_weight,
                max_iter=1000,
                random_state=self.random_state,
                solver='saga',
                n_jobs=-1
            )
        elif self.model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        steps.append(('classifier', model))

        if len(steps) > 1:
            self.pipeline = ImbPipeline(steps)
            self.model = self.pipeline.named_steps['classifier']
        else:
            self.model = model
            self.pipeline = model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the model"""
        print(f"Training {self.model_type} model...")
        print(f"Training set: {len(y_train)} samples, {y_train.sum()} frauds ({y_train.mean():.4f} fraud rate)")

        self.pipeline.fit(X_train, y_train)
        print("Training complete!")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict fraud probabilities"""
        if hasattr(self.pipeline, 'predict_proba'):
            return self.pipeline.predict_proba(X)[:, 1]
        else:
            return self.pipeline.decision_function(X)

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict fraud labels.

        Args:
            X: Feature matrix
            threshold: Decision threshold (uses self.best_threshold if None)
        """
        threshold = threshold if threshold is not None else self.best_threshold
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def tune_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = "f1"
    ) -> float:
        """
        Find optimal decision threshold on validation set.

        Args:
            X_val: Validation features
            y_val: Validation labels
            metric: Optimization metric ("f1", "precision", "recall")

        Returns:
            Optimal threshold
        """
        proba = self.predict_proba(X_val)
        precisions, recalls, thresholds = precision_recall_curve(y_val, proba)

        # Compute F1 scores for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

        if metric == "f1":
            best_idx = np.argmax(f1_scores)
        elif metric == "precision":
            # Find threshold that maximizes precision while keeping recall > 0.5
            valid_indices = recalls >= 0.5
            if valid_indices.any():
                best_idx = np.argmax(precisions * valid_indices)
            else:
                best_idx = np.argmax(precisions)
        elif metric == "recall":
            # Find threshold that maximizes recall while keeping precision > 0.5
            valid_indices = precisions >= 0.5
            if valid_indices.any():
                best_idx = np.argmax(recalls * valid_indices)
            else:
                best_idx = np.argmax(recalls)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        self.best_threshold = thresholds[best_idx]

        print(f"\nOptimal threshold: {self.best_threshold:.4f}")
        print(f"At this threshold: Precision={precisions[best_idx]:.4f}, "
              f"Recall={recalls[best_idx]:.4f}, F1={f1_scores[best_idx]:.4f}")

        return self.best_threshold

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold: Optional[float] = None,
        dataset_name: str = "Test"
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Feature matrix
            y: True labels
            threshold: Decision threshold
            dataset_name: Name for logging

        Returns:
            Dictionary of metrics
        """
        threshold = threshold if threshold is not None else self.best_threshold

        y_proba = self.predict_proba(X)
        y_pred = (y_proba >= threshold).astype(int)

        metrics = {
            'roc_auc': roc_auc_score(y, y_proba),
            'avg_precision': average_precision_score(y, y_proba),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0)
        }

        print(f"\n{dataset_name} Set Evaluation (threshold={threshold:.4f}):")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Average Precision: {metrics['avg_precision']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")

        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y, y_pred)
        print(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}")
        print(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}")

        # Calculate false positive rate
        fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1])
        print(f"\nFalse Positive Rate: {fpr:.4f} ({cm[0, 1]} false alarms out of {cm[0, 0] + cm[0, 1]} legitimate transactions)")

        return metrics

    def get_feature_importance(self, feature_names: Optional[list] = None) -> pd.DataFrame:
        """Get feature importance (for tree-based models)"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_

            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importances))]

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            return importance_df
        else:
            print("Feature importance not available for this model type")
            return None

    def save(self, path: str):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'pipeline': self.pipeline,
            'best_threshold': self.best_threshold,
            'model_type': self.model_type
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk"""
        data = joblib.load(path)
        self.pipeline = data['pipeline']
        self.model = self.pipeline.named_steps.get('classifier', self.pipeline)
        self.best_threshold = data['best_threshold']
        self.model_type = data['model_type']
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    print("Fraud Detection Model Module")
    print("Supports: Logistic Regression, Random Forest, Gradient Boosting")
    print("Features: SMOTE, Undersampling, Class Weighting, Threshold Tuning")
