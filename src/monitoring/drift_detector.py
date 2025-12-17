"""Concept drift detection for monitoring model performance over time"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from collections import deque
import warnings


class DriftDetector:
    """
    Detects concept drift in streaming data.

    Concept drift occurs when the statistical properties of the target variable change,
    causing model performance to degrade. Common causes in fraud detection:
    - Fraudsters change tactics
    - Economic conditions change
    - New payment methods emerge
    - Seasonal patterns shift
    """

    def __init__(
        self,
        window_size: int = 1000,
        reference_window_size: int = 5000,
        alert_threshold: float = 0.05
    ):
        """
        Args:
            window_size: Size of current window for drift detection
            reference_window_size: Size of reference window (baseline)
            alert_threshold: P-value threshold for drift alert
        """
        self.window_size = window_size
        self.reference_window_size = reference_window_size
        self.alert_threshold = alert_threshold

        self.reference_data = deque(maxlen=reference_window_size)
        self.current_window = deque(maxlen=window_size)

        self.drift_detected = False
        self.drift_score = 0.0

    def add_sample(self, features: np.ndarray, label: Optional[int] = None):
        """
        Add a new sample to the drift detector.

        Args:
            features: Feature vector
            label: True label (if available)
        """
        sample = {
            'features': features,
            'label': label,
            'timestamp': len(self.reference_data) + len(self.current_window)
        }

        if len(self.reference_data) < self.reference_window_size:
            self.reference_data.append(sample)
        else:
            self.current_window.append(sample)

    def detect_feature_drift(self) -> Dict[str, float]:
        """
        Detect drift in feature distributions using Kolmogorov-Smirnov test.

        Returns:
            Dictionary with p-values for each feature
        """
        if len(self.current_window) < self.window_size or len(self.reference_data) < self.reference_window_size:
            return {}

        reference_features = np.array([s['features'] for s in self.reference_data])
        current_features = np.array([s['features'] for s in self.current_window])

        n_features = reference_features.shape[1]
        p_values = {}

        for i in range(n_features):
            try:
                stat, p_value = stats.ks_2samp(
                    reference_features[:, i],
                    current_features[:, i]
                )
                p_values[f'feature_{i}'] = p_value
            except Exception as e:
                p_values[f'feature_{i}'] = 1.0

        # Check if any feature shows significant drift
        min_p_value = min(p_values.values())
        self.drift_detected = min_p_value < self.alert_threshold
        self.drift_score = 1 - min_p_value

        return p_values

    def detect_label_drift(self) -> Optional[float]:
        """
        Detect drift in label distribution (fraud rate changes).

        Returns:
            P-value from chi-square test
        """
        reference_labels = [s['label'] for s in self.reference_data if s['label'] is not None]
        current_labels = [s['label'] for s in self.current_window if s['label'] is not None]

        if len(reference_labels) < 100 or len(current_labels) < 100:
            return None

        ref_fraud_rate = np.mean(reference_labels)
        curr_fraud_rate = np.mean(current_labels)

        # Chi-square test for proportion difference
        ref_fraud_count = sum(reference_labels)
        ref_total = len(reference_labels)
        curr_fraud_count = sum(current_labels)
        curr_total = len(current_labels)

        contingency_table = np.array([
            [ref_fraud_count, ref_total - ref_fraud_count],
            [curr_fraud_count, curr_total - curr_fraud_count]
        ])

        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            return p_value
        except Exception:
            return None

    def detect_performance_drift(
        self,
        predictions: List[int],
        labels: List[int],
        reference_performance: float
    ) -> Tuple[bool, float]:
        """
        Detect drift in model performance.

        Args:
            predictions: Recent predictions
            labels: True labels
            reference_performance: Baseline F1 score

        Returns:
            (drift_detected, current_performance)
        """
        if len(predictions) < 100:
            return False, 0.0

        # Calculate current F1 score
        tp = sum((p == 1 and l == 1) for p, l in zip(predictions, labels))
        fp = sum((p == 1 and l == 0) for p, l in zip(predictions, labels))
        fn = sum((p == 0 and l == 1) for p, l in zip(predictions, labels))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        current_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Alert if performance drops by more than 10%
        performance_drop = reference_performance - current_f1
        drift_detected = performance_drop > 0.1

        return drift_detected, current_f1

    def get_drift_report(self) -> Dict:
        """Get comprehensive drift report"""
        feature_p_values = self.detect_feature_drift()
        label_p_value = self.detect_label_drift()

        report = {
            'drift_detected': self.drift_detected,
            'drift_score': self.drift_score,
            'reference_window_size': len(self.reference_data),
            'current_window_size': len(self.current_window),
            'feature_drift': feature_p_values,
            'label_drift_p_value': label_p_value
        }

        if feature_p_values:
            drifted_features = [k for k, v in feature_p_values.items() if v < self.alert_threshold]
            report['drifted_features'] = drifted_features
            report['num_drifted_features'] = len(drifted_features)

        return report

    def reset_current_window(self):
        """Reset the current window (after retraining)"""
        # Move current window to reference
        for sample in self.current_window:
            self.reference_data.append(sample)
        self.current_window.clear()
        self.drift_detected = False
        self.drift_score = 0.0


class PerformanceMonitor:
    """
    Monitors model performance over time in production.

    Tracks metrics like precision, recall, F1, and latency.
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.labels = deque(maxlen=window_size)
        self.probabilities = deque(maxlen=window_size)
        self.latencies = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)

    def record(
        self,
        prediction: int,
        label: Optional[int],
        probability: float,
        latency_ms: float,
        timestamp: float
    ):
        """Record a prediction"""
        self.predictions.append(prediction)
        self.labels.append(label)
        self.probabilities.append(probability)
        self.latencies.append(latency_ms)
        self.timestamps.append(timestamp)

    def get_metrics(self) -> Dict:
        """Calculate current performance metrics"""
        # Filter out samples where label is None
        valid_indices = [i for i, label in enumerate(self.labels) if label is not None]

        if len(valid_indices) < 10:
            return {'status': 'insufficient_data'}

        preds = [self.predictions[i] for i in valid_indices]
        labels = [self.labels[i] for i in valid_indices]
        probas = [self.probabilities[i] for i in valid_indices]

        tp = sum((p == 1 and l == 1) for p, l in zip(preds, labels))
        fp = sum((p == 1 and l == 0) for p, l in zip(preds, labels))
        tn = sum((p == 0 and l == 0) for p, l in zip(preds, labels))
        fn = sum((p == 0 and l == 1) for p, l in zip(preds, labels))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(labels) if labels else 0

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'fraud_rate': np.mean(labels),
            'prediction_rate': np.mean(preds),
            'avg_probability': np.mean(probas),
            'avg_latency_ms': np.mean(self.latencies),
            'p95_latency_ms': np.percentile(self.latencies, 95),
            'window_size': len(labels)
        }

        return metrics

    def print_report(self):
        """Print performance report"""
        metrics = self.get_metrics()

        if 'status' in metrics:
            print(f"Status: {metrics['status']}")
            return

        print(f"\n{'='*60}")
        print(f"PERFORMANCE MONITORING REPORT")
        print(f"{'='*60}")
        print(f"Window size: {metrics['window_size']} samples")
        print(f"\nClassification Metrics:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}")
        print(f"  FN: {metrics['false_negatives']}, TN: {metrics['true_negatives']}")
        print(f"\nRates:")
        print(f"  Actual fraud rate:    {metrics['fraud_rate']:.4f}")
        print(f"  Predicted fraud rate: {metrics['prediction_rate']:.4f}")
        print(f"\nLatency:")
        print(f"  Average: {metrics['avg_latency_ms']:.2f}ms")
        print(f"  P95:     {metrics['p95_latency_ms']:.2f}ms")


if __name__ == "__main__":
    print("Drift Detection Module")
    print("\nDetects concept drift in streaming fraud detection:")
    print("  - Feature distribution drift (KS test)")
    print("  - Label distribution drift (chi-square test)")
    print("  - Model performance drift")
    print("\nTriggers retraining when drift is detected")
