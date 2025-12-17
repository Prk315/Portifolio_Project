"""Main entry point for real-time fraud detection system"""

import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.data_loader import FraudDataLoader
from src.features.time_features import TimeWindowFeatures
from src.models.fraud_detector import FraudDetector
from src.streaming.stream_simulator import StreamSimulator
from src.streaming.kafka_mock import FraudDetectionPipeline
from src.monitoring.drift_detector import DriftDetector, PerformanceMonitor
from src.monitoring.alerting import AlertingSystem, console_alert_handler


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_model(config: dict):
    """Train fraud detection model"""
    print("="*60)
    print("TRAINING FRAUD DETECTION MODEL")
    print("="*60)

    # Load data
    data_loader = FraudDataLoader(data_path=config['data']['path'])
    df = data_loader.load_data()

    # Temporal split
    train_df, val_df, test_df = data_loader.temporal_split(
        df,
        train_size=config['data']['temporal_split']['train_size'],
        val_size=config['data']['temporal_split']['val_size'],
        test_size=config['data']['temporal_split']['test_size']
    )

    # Feature engineering
    print("\nEngineering time-based features...")
    feature_engineer = TimeWindowFeatures(window_sizes=config['features']['time_windows'])

    train_df_fe = feature_engineer.create_all_features(train_df)
    val_df_fe = feature_engineer.create_all_features(val_df)
    test_df_fe = feature_engineer.create_all_features(test_df)

    # Prepare features
    feature_cols = [col for col in train_df_fe.columns if col not in ['Time', 'Class']]
    X_train = train_df_fe[feature_cols].values
    y_train = train_df_fe['Class'].values
    X_val = val_df_fe[feature_cols].values
    y_val = val_df_fe['Class'].values
    X_test = test_df_fe[feature_cols].values
    y_test = test_df_fe['Class'].values

    # Train model
    print("\nTraining model...")
    detector = FraudDetector(
        model_type=config['model']['type'],
        use_smote=config['model']['use_smote'],
        use_undersampling=config['model']['use_undersampling'],
        class_weight=config['model']['class_weight'],
        random_state=config['model']['random_state']
    )

    detector.fit(X_train, y_train)

    # Tune threshold
    print("\nTuning decision threshold on validation set...")
    detector.tune_threshold(X_val, y_val, metric=config['threshold']['metric'])

    # Evaluate
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)

    val_metrics = detector.evaluate(X_val, y_val, dataset_name="Validation")
    test_metrics = detector.evaluate(X_test, y_test, dataset_name="Test")

    # Feature importance
    if config['model']['type'] in ['random_forest', 'gradient_boosting']:
        print("\nTop 10 Most Important Features:")
        importance_df = detector.get_feature_importance(feature_cols)
        if importance_df is not None:
            print(importance_df.head(10).to_string(index=False))

    # Save model
    model_path = Path(config['output']['model_path'])
    model_path.mkdir(parents=True, exist_ok=True)
    detector.save(str(model_path / "fraud_detector.joblib"))

    return detector, test_df_fe, test_metrics


def run_streaming_simulation(detector, test_df, config: dict):
    """Run streaming inference simulation"""
    print("\n" + "="*60)
    print("STREAMING INFERENCE SIMULATION")
    print("="*60)

    # Setup alerting
    alerting = AlertingSystem(
        fraud_probability_threshold=config['alerting']['fraud_probability_threshold'],
        high_risk_threshold=config['alerting']['high_risk_threshold']
    )
    alerting.register_callback(console_alert_handler)

    # Create alert callback for stream
    def stream_callback(result):
        alerting.check_transaction(
            transaction_id=f"txn_{result['transaction_id']}",
            fraud_probability=result['fraud_probability'],
            amount=result['amount'],
            timestamp=result['time']
        )

    # Run stream simulation
    simulator = StreamSimulator(
        data=test_df,
        model=detector,
        batch_size=config['streaming']['batch_size'],
        delay_ms=config['streaming']['delay_ms']
    )

    results_df = simulator.stream(
        callback=stream_callback,
        alert_threshold=config['streaming']['alert_threshold'],
        max_transactions=10000  # Limit for demo
    )

    # Print metrics
    metrics = simulator.get_performance_metrics()
    print("\n" + "="*60)
    print("STREAMING PERFORMANCE METRICS")
    print("="*60)
    print(f"Total transactions: {metrics['total_transactions']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"\nLatency Statistics:")
    print(f"  Average: {metrics['avg_latency_ms']:.2f}ms")
    print(f"  P95: {metrics['p95_latency_ms']:.2f}ms")
    print(f"  P99: {metrics['p99_latency_ms']:.2f}ms")
    print(f"  Max: {metrics['max_latency_ms']:.2f}ms")
    print(f"  Throughput: {metrics['throughput_per_sec']:.0f} transactions/sec")

    # Alert summary
    alerting.print_alert_summary()

    return results_df, alerting


def run_kafka_pipeline(detector, test_df, config: dict):
    """Run Kafka mock pipeline"""
    print("\n" + "="*60)
    print("KAFKA STREAMING PIPELINE")
    print("="*60)

    pipeline = FraudDetectionPipeline(
        model=detector,
        input_topic=config['kafka']['input_topic'],
        output_topic=config['kafka']['output_topic'],
        alert_topic=config['kafka']['alert_topic']
    )

    pipeline.run(
        transactions_df=test_df,
        max_transactions=5000,
        alert_threshold=config['streaming']['alert_threshold']
    )

    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Real-Time Fraud Detection System")
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'stream', 'kafka', 'full'],
        default='full',
        help='Execution mode'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    if args.mode in ['train', 'full']:
        detector, test_df, test_metrics = train_model(config)

    if args.mode in ['stream', 'full']:
        if args.mode == 'stream':
            # Load trained model
            from joblib import load
            model_path = Path(config['output']['model_path']) / "fraud_detector.joblib"
            data = load(model_path)
            detector = FraudDetector()
            detector.pipeline = data['pipeline']
            detector.best_threshold = data['best_threshold']

            # Load test data
            data_loader = FraudDataLoader(data_path=config['data']['path'])
            df = data_loader.load_data()
            _, _, test_df = data_loader.temporal_split(df)
            feature_engineer = TimeWindowFeatures(window_sizes=config['features']['time_windows'])
            test_df = feature_engineer.create_all_features(test_df)

        results_df, alerting = run_streaming_simulation(detector, test_df, config)

    if args.mode in ['kafka', 'full']:
        if args.mode == 'kafka':
            # Load trained model and test data (similar to stream mode)
            pass

        pipeline = run_kafka_pipeline(detector, test_df, config)

    print("\n" + "="*60)
    print("EXECUTION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
