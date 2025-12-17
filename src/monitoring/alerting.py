"""Alerting system for fraud detection"""

import time
from typing import Dict, List, Optional, Callable
from collections import deque
from enum import Enum
import json


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    """Types of alerts"""
    HIGH_FRAUD_PROBABILITY = "HIGH_FRAUD_PROBABILITY"
    UNUSUAL_AMOUNT = "UNUSUAL_AMOUNT"
    RAPID_TRANSACTIONS = "RAPID_TRANSACTIONS"
    MODEL_DRIFT = "MODEL_DRIFT"
    PERFORMANCE_DEGRADATION = "PERFORMANCE_DEGRADATION"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class Alert:
    """Represents a fraud detection alert"""

    def __init__(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        data: Optional[Dict] = None,
        timestamp: Optional[float] = None
    ):
        self.alert_type = alert_type
        self.severity = severity
        self.message = message
        self.data = data or {}
        self.timestamp = timestamp or time.time()
        self.alert_id = f"{alert_type.value}_{int(self.timestamp * 1000)}"

    def to_dict(self) -> Dict:
        """Convert alert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'data': self.data,
            'timestamp': self.timestamp
        }

    def __repr__(self):
        return f"Alert({self.severity.value}, {self.alert_type.value}): {self.message}"


class AlertingSystem:
    """
    Comprehensive alerting system for fraud detection.

    In production, this would integrate with:
    - PagerDuty / Opsgenie for on-call alerts
    - Slack / Email for notifications
    - Dashboard / Monitoring tools (Grafana, Datadog)
    - SIEM systems for security operations
    """

    def __init__(
        self,
        fraud_probability_threshold: float = 0.8,
        high_risk_threshold: float = 0.95,
        unusual_amount_multiplier: float = 3.0,
        rapid_transaction_window_seconds: int = 300,
        rapid_transaction_threshold: int = 5
    ):
        """
        Args:
            fraud_probability_threshold: Threshold for fraud alerts
            high_risk_threshold: Threshold for high-severity alerts
            unusual_amount_multiplier: Multiplier for unusual amount detection
            rapid_transaction_window_seconds: Time window for rapid transactions
            rapid_transaction_threshold: Number of transactions to trigger alert
        """
        self.fraud_probability_threshold = fraud_probability_threshold
        self.high_risk_threshold = high_risk_threshold
        self.unusual_amount_multiplier = unusual_amount_multiplier
        self.rapid_transaction_window = rapid_transaction_window_seconds
        self.rapid_transaction_threshold = rapid_transaction_threshold

        self.alerts = []
        self.alert_callbacks = []
        self.recent_transactions = deque(maxlen=1000)
        self.alert_counts = {severity: 0 for severity in AlertSeverity}

    def register_callback(self, callback: Callable[[Alert], None]):
        """Register a callback to be called when alerts are triggered"""
        self.alert_callbacks.append(callback)

    def check_transaction(
        self,
        transaction_id: str,
        fraud_probability: float,
        amount: float,
        timestamp: float,
        additional_data: Optional[Dict] = None
    ) -> List[Alert]:
        """
        Check a transaction and generate alerts if needed.

        Args:
            transaction_id: Unique transaction identifier
            fraud_probability: Model prediction probability
            amount: Transaction amount
            timestamp: Transaction timestamp
            additional_data: Additional transaction data

        Returns:
            List of generated alerts
        """
        alerts = []
        additional_data = additional_data or {}

        # Store transaction
        self.recent_transactions.append({
            'id': transaction_id,
            'probability': fraud_probability,
            'amount': amount,
            'timestamp': timestamp
        })

        # Check 1: High fraud probability
        if fraud_probability >= self.high_risk_threshold:
            alert = Alert(
                alert_type=AlertType.HIGH_FRAUD_PROBABILITY,
                severity=AlertSeverity.CRITICAL,
                message=f"Critical fraud risk detected: {fraud_probability:.2%} probability",
                data={
                    'transaction_id': transaction_id,
                    'fraud_probability': fraud_probability,
                    'amount': amount,
                    **additional_data
                },
                timestamp=timestamp
            )
            alerts.append(alert)

        elif fraud_probability >= self.fraud_probability_threshold:
            alert = Alert(
                alert_type=AlertType.HIGH_FRAUD_PROBABILITY,
                severity=AlertSeverity.HIGH,
                message=f"High fraud risk detected: {fraud_probability:.2%} probability",
                data={
                    'transaction_id': transaction_id,
                    'fraud_probability': fraud_probability,
                    'amount': amount,
                    **additional_data
                },
                timestamp=timestamp
            )
            alerts.append(alert)

        # Check 2: Unusual amount
        if len(self.recent_transactions) >= 100:
            recent_amounts = [t['amount'] for t in list(self.recent_transactions)[-100:]]
            avg_amount = sum(recent_amounts) / len(recent_amounts)

            if amount > avg_amount * self.unusual_amount_multiplier:
                alert = Alert(
                    alert_type=AlertType.UNUSUAL_AMOUNT,
                    severity=AlertSeverity.MEDIUM,
                    message=f"Unusual transaction amount: ${amount:.2f} (avg: ${avg_amount:.2f})",
                    data={
                        'transaction_id': transaction_id,
                        'amount': amount,
                        'average_amount': avg_amount,
                        'multiplier': amount / avg_amount
                    },
                    timestamp=timestamp
                )
                alerts.append(alert)

        # Check 3: Rapid transactions
        recent_time_window = [
            t for t in self.recent_transactions
            if timestamp - t['timestamp'] <= self.rapid_transaction_window
        ]
        if len(recent_time_window) >= self.rapid_transaction_threshold:
            alert = Alert(
                alert_type=AlertType.RAPID_TRANSACTIONS,
                severity=AlertSeverity.MEDIUM,
                message=f"Rapid transaction pattern: {len(recent_time_window)} transactions in {self.rapid_transaction_window}s",
                data={
                    'transaction_id': transaction_id,
                    'transaction_count': len(recent_time_window),
                    'time_window_seconds': self.rapid_transaction_window
                },
                timestamp=timestamp
            )
            alerts.append(alert)

        # Store and trigger callbacks for all alerts
        for alert in alerts:
            self._trigger_alert(alert)

        return alerts

    def alert_drift(self, drift_report: Dict):
        """Generate alert for concept drift"""
        severity = AlertSeverity.HIGH if drift_report.get('drift_score', 0) > 0.9 else AlertSeverity.MEDIUM

        alert = Alert(
            alert_type=AlertType.MODEL_DRIFT,
            severity=severity,
            message=f"Concept drift detected: {len(drift_report.get('drifted_features', []))} features drifted",
            data=drift_report,
            timestamp=time.time()
        )

        self._trigger_alert(alert)

    def alert_performance_degradation(
        self,
        current_f1: float,
        reference_f1: float,
        metrics: Dict
    ):
        """Generate alert for model performance degradation"""
        performance_drop = reference_f1 - current_f1

        severity = AlertSeverity.CRITICAL if performance_drop > 0.2 else AlertSeverity.HIGH

        alert = Alert(
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=severity,
            message=f"Model performance degraded: F1 dropped from {reference_f1:.4f} to {current_f1:.4f}",
            data={
                'current_f1': current_f1,
                'reference_f1': reference_f1,
                'performance_drop': performance_drop,
                **metrics
            },
            timestamp=time.time()
        )

        self._trigger_alert(alert)

    def alert_system_error(self, error_message: str, error_data: Optional[Dict] = None):
        """Generate alert for system errors"""
        alert = Alert(
            alert_type=AlertType.SYSTEM_ERROR,
            severity=AlertSeverity.HIGH,
            message=f"System error: {error_message}",
            data=error_data or {},
            timestamp=time.time()
        )

        self._trigger_alert(alert)

    def _trigger_alert(self, alert: Alert):
        """Store alert and trigger callbacks"""
        self.alerts.append(alert)
        self.alert_counts[alert.severity] += 1

        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Error in alert callback: {e}")

    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        alert_type: Optional[AlertType] = None,
        last_n: Optional[int] = None
    ) -> List[Alert]:
        """
        Retrieve alerts with optional filtering.

        Args:
            severity: Filter by severity
            alert_type: Filter by alert type
            last_n: Return only last N alerts

        Returns:
            List of alerts
        """
        alerts = self.alerts

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]

        if last_n:
            alerts = alerts[-last_n:]

        return alerts

    def get_alert_summary(self) -> Dict:
        """Get summary of alerts"""
        return {
            'total_alerts': len(self.alerts),
            'by_severity': {
                severity.value: count
                for severity, count in self.alert_counts.items()
            },
            'by_type': self._count_by_type(),
            'recent_alerts': [a.to_dict() for a in self.alerts[-10:]]
        }

    def _count_by_type(self) -> Dict:
        """Count alerts by type"""
        counts = {}
        for alert in self.alerts:
            alert_type = alert.alert_type.value
            counts[alert_type] = counts.get(alert_type, 0) + 1
        return counts

    def print_alert_summary(self):
        """Print alert summary"""
        summary = self.get_alert_summary()

        print(f"\n{'='*60}")
        print(f"ALERT SUMMARY")
        print(f"{'='*60}")
        print(f"Total alerts: {summary['total_alerts']}")
        print(f"\nBy severity:")
        for severity, count in summary['by_severity'].items():
            print(f"  {severity}: {count}")
        print(f"\nBy type:")
        for alert_type, count in summary['by_type'].items():
            print(f"  {alert_type}: {count}")

    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts = []
        self.alert_counts = {severity: 0 for severity in AlertSeverity}


# Example alert callbacks

def console_alert_handler(alert: Alert):
    """Print alerts to console"""
    icon = {
        AlertSeverity.LOW: "ℹ️",
        AlertSeverity.MEDIUM: "⚠️",
        AlertSeverity.HIGH: "🚨",
        AlertSeverity.CRITICAL: "🔥"
    }.get(alert.severity, "•")

    print(f"{icon} [{alert.severity.value}] {alert.message}")


def slack_alert_handler(alert: Alert):
    """Mock Slack alert handler (would use Slack API in production)"""
    if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
        # In production: send to Slack webhook
        print(f"[SLACK] Sending alert to #fraud-alerts: {alert.message}")


def email_alert_handler(alert: Alert):
    """Mock email alert handler (would use SMTP in production)"""
    if alert.severity == AlertSeverity.CRITICAL:
        # In production: send email to on-call team
        print(f"[EMAIL] Sending critical alert to fraud-team@company.com: {alert.message}")


if __name__ == "__main__":
    print("Alerting System Module")
    print("\nProvides comprehensive alerting for:")
    print("  - High fraud probability transactions")
    print("  - Unusual transaction patterns")
    print("  - Model drift")
    print("  - Performance degradation")
    print("  - System errors")
    print("\nIntegrates with notification systems (Slack, email, PagerDuty)")
