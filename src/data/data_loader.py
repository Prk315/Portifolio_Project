"""Data loading and preprocessing for Credit Card Fraud Detection"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import joblib
import os


class FraudDataLoader:
    """Handles loading and preprocessing of credit card fraud data with temporal awareness"""

    def __init__(self, data_path: str = "data/creditcard.csv"):
        self.data_path = data_path
        self.scaler = StandardScaler()

    def load_data(self) -> pd.DataFrame:
        """Load credit card fraud dataset"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                "Please download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
            )

        df = pd.read_csv(self.data_path)
        print(f"Loaded {len(df)} transactions")
        print(f"Fraud rate: {df['Class'].mean():.4f} ({df['Class'].sum()} fraudulent transactions)")

        return df

    def temporal_split(
        self,
        df: pd.DataFrame,
        train_size: float = 0.6,
        val_size: float = 0.2,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally (not randomly) to simulate real-world deployment

        In production, we train on past data and predict future transactions.
        This is crucial for time-series/streaming scenarios.
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1.0"

        # Sort by time
        df_sorted = df.sort_values('Time').reset_index(drop=True)

        n = len(df_sorted)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))

        train_df = df_sorted.iloc[:train_end]
        val_df = df_sorted.iloc[train_end:val_end]
        test_df = df_sorted.iloc[val_end:]

        print(f"\nTemporal Split:")
        print(f"Train: {len(train_df)} transactions, fraud rate: {train_df['Class'].mean():.4f}")
        print(f"Val:   {len(val_df)} transactions, fraud rate: {val_df['Class'].mean():.4f}")
        print(f"Test:  {len(test_df)} transactions, fraud rate: {test_df['Class'].mean():.4f}")

        return train_df, val_df, test_df

    def prepare_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        scale_amount: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features for modeling

        Note: V1-V28 are already PCA-transformed features from the original data.
        We only need to scale the Amount feature.
        """
        feature_cols = [col for col in train_df.columns if col not in ['Time', 'Class']]

        X_train = train_df[feature_cols].values
        y_train = train_df['Class'].values

        X_val = val_df[feature_cols].values
        y_val = val_df['Class'].values

        X_test = test_df[feature_cols].values
        y_test = test_df['Class'].values

        if scale_amount:
            # Only scale the Amount column (last column)
            X_train_scaled = X_train.copy()
            X_val_scaled = X_val.copy()
            X_test_scaled = X_test.copy()

            X_train_scaled[:, -1] = self.scaler.fit_transform(X_train[:, -1].reshape(-1, 1)).ravel()
            X_val_scaled[:, -1] = self.scaler.transform(X_val[:, -1].reshape(-1, 1)).ravel()
            X_test_scaled[:, -1] = self.scaler.transform(X_test[:, -1].reshape(-1, 1)).ravel()

            return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_scaler(self, path: str = "data/scaler.joblib"):
        """Save the fitted scaler"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        print(f"Scaler saved to {path}")

    def load_scaler(self, path: str = "data/scaler.joblib"):
        """Load a fitted scaler"""
        self.scaler = joblib.load(path)
        print(f"Scaler loaded from {path}")


def download_instructions():
    """Print instructions for downloading the dataset"""
    print("""
    To download the Credit Card Fraud Detection dataset:

    1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    2. Download creditcard.csv
    3. Place it in the 'data/' directory

    The dataset contains:
    - 284,807 transactions
    - 492 frauds (0.172% fraud rate - highly imbalanced!)
    - Features V1-V28: PCA-transformed features
    - Time: seconds elapsed between transactions
    - Amount: transaction amount
    - Class: 1 for fraud, 0 for legitimate
    """)


if __name__ == "__main__":
    download_instructions()
