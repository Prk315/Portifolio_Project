"""Time-aware feature engineering for fraud detection"""

import pandas as pd
import numpy as np
from typing import List, Optional


class TimeWindowFeatures:
    """
    Creates time-window features that simulate real-time fraud detection.

    In production, these features would be computed from a sliding window
    of recent transactions stored in a fast database (e.g., Redis).
    """

    def __init__(self, window_sizes: List[int] = [10, 50, 100, 500]):
        """
        Args:
            window_sizes: List of window sizes (number of transactions) to compute features over
        """
        self.window_sizes = window_sizes

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic time-based features"""
        df = df.copy()

        # Time in hours (original Time is in seconds)
        df['Time_hours'] = df['Time'] / 3600

        # Time of day features (cyclical encoding)
        # Assuming Time represents seconds from start, we can create hour of day features
        hours = (df['Time'] / 3600) % 24
        df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hours / 24)

        return df

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling window features that capture recent transaction patterns.

        These simulate what we'd see in a real-time system where we track:
        - Recent transaction velocity
        - Recent fraud rate
        - Recent amount statistics
        """
        df = df.copy()
        df = df.sort_values('Time').reset_index(drop=True)

        for window in self.window_sizes:
            # Rolling statistics for Amount
            df[f'amount_mean_{window}'] = df['Amount'].rolling(
                window=window, min_periods=1
            ).mean()

            df[f'amount_std_{window}'] = df['Amount'].rolling(
                window=window, min_periods=1
            ).std().fillna(0)

            df[f'amount_max_{window}'] = df['Amount'].rolling(
                window=window, min_periods=1
            ).max()

            # Transaction velocity (transactions per hour in the window)
            df[f'velocity_{window}'] = window / (
                df['Time'].rolling(window=window, min_periods=1).apply(
                    lambda x: (x.max() - x.min()) / 3600 + 0.01  # +0.01 to avoid div by zero
                )
            )

            # Deviation from rolling mean
            df[f'amount_deviation_{window}'] = (
                df['Amount'] - df[f'amount_mean_{window}']
            ) / (df[f'amount_std_{window}'] + 1e-6)

        return df

    def add_transaction_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features based on transaction sequences.

        In a streaming system, these would track patterns like:
        - Time since last transaction
        - Number of transactions in last N seconds
        """
        df = df.copy()
        df = df.sort_values('Time').reset_index(drop=True)

        # Time since last transaction
        df['time_delta'] = df['Time'].diff().fillna(0)

        # Cumulative transaction count (position in sequence)
        df['transaction_number'] = np.arange(len(df))

        # Amount relative to recent history
        df['amount_vs_mean_100'] = df['Amount'] / (df['Amount'].rolling(100, min_periods=1).mean() + 1e-6)

        return df

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all time-based features"""
        df = self.add_time_features(df)
        df = self.add_rolling_features(df)
        df = self.add_transaction_sequence_features(df)

        print(f"Created {len([c for c in df.columns if c not in ['Time', 'Class', 'Amount'] + [f'V{i}' for i in range(1, 29)]])} new features")

        return df

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get names of all engineered features"""
        base_features = ['Time', 'Class', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        return [col for col in df.columns if col not in base_features]


class RealTimeFeatureExtractor:
    """
    Simulates real-time feature extraction from a stream of transactions.

    In production, this would interface with:
    - Redis/Memcached for fast access to recent transactions
    - Feature store (e.g., Feast) for consistent online/offline features
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.transaction_buffer = []

    def update(self, transaction: dict) -> dict:
        """
        Update buffer with new transaction and compute features.

        Args:
            transaction: Dictionary containing transaction data

        Returns:
            Dictionary with original transaction + computed features
        """
        # Add to buffer
        self.transaction_buffer.append(transaction)

        # Keep only last window_size transactions
        if len(self.transaction_buffer) > self.window_size:
            self.transaction_buffer = self.transaction_buffer[-self.window_size:]

        # Compute features from buffer
        recent_amounts = [t['Amount'] for t in self.transaction_buffer]
        recent_times = [t['Time'] for t in self.transaction_buffer]

        features = transaction.copy()

        # Rolling statistics
        features['amount_mean'] = np.mean(recent_amounts)
        features['amount_std'] = np.std(recent_amounts)
        features['amount_max'] = np.max(recent_amounts)

        # Time-based features
        if len(recent_times) > 1:
            time_deltas = np.diff(recent_times)
            features['avg_time_delta'] = np.mean(time_deltas)
            features['velocity'] = len(recent_times) / ((recent_times[-1] - recent_times[0]) / 3600 + 0.01)
        else:
            features['avg_time_delta'] = 0
            features['velocity'] = 0

        # Deviation from recent mean
        features['amount_deviation'] = (
            transaction['Amount'] - features['amount_mean']
        ) / (features['amount_std'] + 1e-6)

        return features

    def reset(self):
        """Reset the buffer"""
        self.transaction_buffer = []


if __name__ == "__main__":
    print("Time-based feature engineering module for real-time fraud detection")
    print(f"Default window sizes: {TimeWindowFeatures().window_sizes}")
