"""Simulated streaming inference for fraud detection"""

import numpy as np
import pandas as pd
import time
from typing import Callable, Optional, Dict, List
from collections import deque
import json


class StreamSimulator:
    """
    Simulates real-time transaction processing.

    In production, this would be replaced by:
    - Kafka consumer reading from transaction stream
    - Real-time feature store (Redis/Memcached)
    - Model serving infrastructure (TensorFlow Serving, AWS SageMaker, etc.)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        model,
        feature_extractor=None,
        batch_size: int = 1,
        delay_ms: int = 0
    ):
        """
        Args:
            data: DataFrame with transactions sorted by time
            model: Trained fraud detection model
            feature_extractor: Optional real-time feature extractor
            batch_size: Number of transactions to process at once
            delay_ms: Delay between batches (milliseconds) to simulate real-time
        """
        self.data = data.sort_values('Time').reset_index(drop=True)
        self.model = model
        self.feature_extractor = feature_extractor
        self.batch_size = batch_size
        self.delay_ms = delay_ms

        self.position = 0
        self.predictions = []
        self.latencies = []
        self.alerts = []

    def stream(
        self,
        callback: Optional[Callable] = None,
        alert_threshold: float = 0.8,
        max_transactions: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Simulate streaming inference.

        Args:
            callback: Optional function called for each prediction
            alert_threshold: Probability threshold for generating alerts
            max_transactions: Maximum number of transactions to process

        Returns:
            DataFrame with predictions and metadata
        """
        total = min(len(self.data), max_transactions or len(self.data))
        print(f"Starting stream simulation: {total} transactions")

        while self.position < total:
            batch_end = min(self.position + self.batch_size, total)
            batch = self.data.iloc[self.position:batch_end]

            # Simulate processing time
            start_time = time.time()

            # Extract features
            feature_cols = [col for col in batch.columns if col not in ['Time', 'Class']]
            X_batch = batch[feature_cols].values
            y_batch = batch['Class'].values

            # Make predictions
            probas = self.model.predict_proba(X_batch)
            preds = self.model.predict(X_batch)

            # Calculate latency
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            # Process each transaction in the batch
            for i, (idx, row) in enumerate(batch.iterrows()):
                result = {
                    'transaction_id': idx,
                    'time': row['Time'],
                    'amount': row['Amount'],
                    'true_label': y_batch[i],
                    'predicted_label': preds[i],
                    'fraud_probability': probas[i],
                    'latency_ms': latency_ms / len(batch),
                    'is_alert': probas[i] >= alert_threshold
                }

                self.predictions.append(result)
                self.latencies.append(result['latency_ms'])

                # Generate alert for high-risk transactions
                if result['is_alert']:
                    self.alerts.append(result)

                # Call callback if provided
                if callback:
                    callback(result)

            self.position = batch_end

            # Simulate real-time delay
            if self.delay_ms > 0:
                time.sleep(self.delay_ms / 1000)

            # Progress update
            if self.position % 1000 == 0:
                print(f"Processed {self.position}/{total} transactions...")

        print(f"\nStream simulation complete!")
        print(f"Total transactions: {len(self.predictions)}")
        print(f"Alerts generated: {len(self.alerts)}")
        print(f"Average latency: {np.mean(self.latencies):.2f}ms")
        print(f"Max latency: {np.max(self.latencies):.2f}ms")

        return pd.DataFrame(self.predictions)

    def get_performance_metrics(self) -> Dict:
        """Calculate streaming performance metrics"""
        if not self.predictions:
            return {}

        df = pd.DataFrame(self.predictions)

        true_positives = ((df['true_label'] == 1) & (df['predicted_label'] == 1)).sum()
        false_positives = ((df['true_label'] == 0) & (df['predicted_label'] == 1)).sum()
        true_negatives = ((df['true_label'] == 0) & (df['predicted_label'] == 0)).sum()
        false_negatives = ((df['true_label'] == 1) & (df['predicted_label'] == 0)).sum()

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            'total_transactions': len(df),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_latency_ms': np.mean(self.latencies),
            'p95_latency_ms': np.percentile(self.latencies, 95),
            'p99_latency_ms': np.percentile(self.latencies, 99),
            'max_latency_ms': np.max(self.latencies),
            'throughput_per_sec': len(df) / (np.sum(self.latencies) / 1000) if self.latencies else 0
        }

        return metrics

    def reset(self):
        """Reset the simulator"""
        self.position = 0
        self.predictions = []
        self.latencies = []
        self.alerts = []


class TransactionBuffer:
    """
    Maintains a sliding window buffer of recent transactions.

    In production, this would be backed by Redis or a similar in-memory store.
    """

    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, transaction: Dict):
        """Add a transaction to the buffer"""
        self.buffer.append(transaction)

    def get_recent(self, n: int = 100) -> List[Dict]:
        """Get the n most recent transactions"""
        return list(self.buffer)[-n:]

    def get_stats(self) -> Dict:
        """Get statistics from recent transactions"""
        if not self.buffer:
            return {}

        amounts = [t['amount'] for t in self.buffer]
        times = [t['time'] for t in self.buffer]

        return {
            'count': len(self.buffer),
            'avg_amount': np.mean(amounts),
            'max_amount': np.max(amounts),
            'min_amount': np.min(amounts),
            'std_amount': np.std(amounts),
            'time_span_hours': (times[-1] - times[0]) / 3600 if len(times) > 1 else 0
        }

    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()


def alert_callback(transaction: Dict):
    """Example callback function for high-risk transactions"""
    if transaction['is_alert']:
        print(f"⚠️  ALERT: High fraud risk detected!")
        print(f"   Transaction ID: {transaction['transaction_id']}")
        print(f"   Amount: ${transaction['amount']:.2f}")
        print(f"   Fraud Probability: {transaction['fraud_probability']:.2%}")
        print(f"   True Label: {'FRAUD' if transaction['true_label'] == 1 else 'LEGITIMATE'}")
        print()


if __name__ == "__main__":
    print("Stream Simulator Module")
    print("Simulates real-time transaction processing and fraud detection")
