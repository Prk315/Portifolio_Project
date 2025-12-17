"""Mock Kafka pipeline for fraud detection streaming"""

import json
import time
from typing import Dict, List, Callable, Optional
from queue import Queue, Empty
from threading import Thread
import pandas as pd


class MockKafkaProducer:
    """
    Simulates a Kafka producer that publishes transactions to a topic.

    In production:
    - Replace with kafka-python or confluent-kafka
    - Connect to actual Kafka brokers
    - Handle partitioning, acknowledgments, retries
    """

    def __init__(self, topic: str = "transactions"):
        self.topic = topic
        self.sent_count = 0

    def send(self, message: Dict):
        """
        Send a message to the topic.

        Args:
            message: Transaction data as dictionary
        """
        # In real Kafka, this would serialize and send to brokers
        self.sent_count += 1
        return {
            'topic': self.topic,
            'partition': 0,
            'offset': self.sent_count,
            'timestamp': time.time()
        }

    def flush(self):
        """Flush any pending messages"""
        pass

    def close(self):
        """Close the producer"""
        pass


class MockKafkaConsumer:
    """
    Simulates a Kafka consumer that reads transactions from a topic.

    In production:
    - Replace with kafka-python or confluent-kafka
    - Handle consumer groups, offset management, rebalancing
    - Implement error handling and retry logic
    """

    def __init__(self, topic: str = "transactions", group_id: str = "fraud-detector"):
        self.topic = topic
        self.group_id = group_id
        self.message_queue = Queue()
        self.running = False
        self.consumed_count = 0

    def subscribe(self, topics: List[str]):
        """Subscribe to topics"""
        pass

    def poll(self, timeout_ms: int = 1000) -> Optional[Dict]:
        """
        Poll for messages.

        Args:
            timeout_ms: Timeout in milliseconds

        Returns:
            Message dictionary or None
        """
        try:
            message = self.message_queue.get(timeout=timeout_ms / 1000)
            self.consumed_count += 1
            return message
        except Empty:
            return None

    def commit(self):
        """Commit current offsets"""
        pass

    def close(self):
        """Close the consumer"""
        self.running = False


class FraudDetectionPipeline:
    """
    End-to-end fraud detection streaming pipeline.

    Simulates:
    1. Transaction producer → Kafka topic
    2. Consumer reads transactions
    3. Feature extraction
    4. Model inference
    5. Alert generation
    6. Results published to output topic
    """

    def __init__(
        self,
        model,
        input_topic: str = "transactions",
        output_topic: str = "fraud-predictions",
        alert_topic: str = "fraud-alerts"
    ):
        self.model = model
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.alert_topic = alert_topic

        self.producer = MockKafkaProducer(topic=output_topic)
        self.alert_producer = MockKafkaProducer(topic=alert_topic)
        self.consumer = MockKafkaConsumer(topic=input_topic)

        self.processing_times = []
        self.predictions = []
        self.alerts = []

    def process_transaction(self, transaction: Dict) -> Dict:
        """
        Process a single transaction through the fraud detection pipeline.

        Args:
            transaction: Raw transaction data

        Returns:
            Prediction result with metadata
        """
        start_time = time.time()

        # Extract features (in production, this would query feature store)
        feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
        features = [transaction.get(col, 0) for col in feature_cols]
        X = [features]

        # Make prediction
        fraud_proba = self.model.predict_proba(X)[0]
        fraud_pred = self.model.predict(X)[0]

        # Calculate processing time
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000

        result = {
            'transaction_id': transaction.get('transaction_id', 'unknown'),
            'timestamp': transaction.get('Time', time.time()),
            'amount': transaction.get('Amount', 0),
            'fraud_probability': float(fraud_proba),
            'is_fraud': bool(fraud_pred),
            'processing_time_ms': processing_time_ms,
            'model_version': '1.0'
        }

        self.processing_times.append(processing_time_ms)
        self.predictions.append(result)

        return result

    def run(
        self,
        transactions_df: pd.DataFrame,
        max_transactions: Optional[int] = None,
        alert_threshold: float = 0.8,
        delay_ms: int = 0
    ):
        """
        Run the streaming pipeline.

        Args:
            transactions_df: DataFrame with transactions
            max_transactions: Maximum number of transactions to process
            alert_threshold: Threshold for generating alerts
            delay_ms: Delay between transactions (for simulation)
        """
        print(f"Starting Kafka-based fraud detection pipeline...")
        print(f"Input topic: {self.input_topic}")
        print(f"Output topic: {self.output_topic}")
        print(f"Alert topic: {self.alert_topic}")
        print(f"Alert threshold: {alert_threshold}")
        print()

        transactions_df = transactions_df.sort_values('Time')
        n = min(len(transactions_df), max_transactions or len(transactions_df))

        for i, (idx, row) in enumerate(transactions_df.head(n).iterrows()):
            # Convert row to transaction dict
            transaction = row.to_dict()
            transaction['transaction_id'] = f"txn_{idx}"

            # Process transaction
            result = self.process_transaction(transaction)

            # Publish prediction to output topic
            self.producer.send(result)

            # Generate alert if high risk
            if result['fraud_probability'] >= alert_threshold:
                alert = {
                    **result,
                    'alert_type': 'HIGH_FRAUD_RISK',
                    'severity': 'HIGH' if result['fraud_probability'] >= 0.95 else 'MEDIUM'
                }
                self.alert_producer.send(alert)
                self.alerts.append(alert)

                print(f"🚨 ALERT: Transaction {result['transaction_id']} - "
                      f"Fraud probability: {result['fraud_probability']:.2%}")

            # Progress update
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{n} transactions... "
                      f"Avg latency: {sum(self.processing_times[-1000:]) / 1000:.2f}ms")

            # Simulate real-time delay
            if delay_ms > 0:
                time.sleep(delay_ms / 1000)

        print(f"\nPipeline processing complete!")
        self.print_stats()

    def print_stats(self):
        """Print pipeline statistics"""
        print(f"\n{'='*60}")
        print(f"PIPELINE STATISTICS")
        print(f"{'='*60}")
        print(f"Total transactions processed: {len(self.predictions)}")
        print(f"Total alerts generated: {len(self.alerts)}")
        print(f"Alert rate: {len(self.alerts) / len(self.predictions) * 100:.2f}%")
        print(f"\nLatency Statistics:")
        print(f"  Mean: {sum(self.processing_times) / len(self.processing_times):.2f}ms")
        print(f"  P50:  {sorted(self.processing_times)[len(self.processing_times)//2]:.2f}ms")
        print(f"  P95:  {sorted(self.processing_times)[int(len(self.processing_times)*0.95)]:.2f}ms")
        print(f"  P99:  {sorted(self.processing_times)[int(len(self.processing_times)*0.99)]:.2f}ms")
        print(f"  Max:  {max(self.processing_times):.2f}ms")

        if self.predictions:
            fraud_predictions = sum(1 for p in self.predictions if p['is_fraud'])
            print(f"\nPrediction Distribution:")
            print(f"  Predicted Fraud: {fraud_predictions} ({fraud_predictions/len(self.predictions)*100:.2f}%)")
            print(f"  Predicted Legitimate: {len(self.predictions) - fraud_predictions}")

    def get_results_df(self) -> pd.DataFrame:
        """Get predictions as DataFrame"""
        return pd.DataFrame(self.predictions)

    def get_alerts_df(self) -> pd.DataFrame:
        """Get alerts as DataFrame"""
        return pd.DataFrame(self.alerts)

    def close(self):
        """Close all connections"""
        self.producer.close()
        self.alert_producer.close()
        self.consumer.close()


class KafkaConfig:
    """
    Configuration for Kafka connection.

    In production, this would contain:
    - Bootstrap servers
    - Security settings (SSL, SASL)
    - Consumer group settings
    - Serialization config
    """

    def __init__(self):
        self.bootstrap_servers = ['localhost:9092']
        self.security_protocol = 'PLAINTEXT'
        self.sasl_mechanism = None
        self.sasl_username = None
        self.sasl_password = None

    def to_dict(self) -> Dict:
        """Convert to configuration dictionary"""
        return {
            'bootstrap.servers': ','.join(self.bootstrap_servers),
            'security.protocol': self.security_protocol
        }


if __name__ == "__main__":
    print("Mock Kafka Pipeline Module")
    print("\nThis module simulates a Kafka-based streaming pipeline:")
    print("  1. Transactions → Kafka topic")
    print("  2. Fraud detection consumer")
    print("  3. Real-time inference")
    print("  4. Predictions → Output topic")
    print("  5. High-risk alerts → Alert topic")
    print("\nIn production, replace with real Kafka (kafka-python or confluent-kafka)")
