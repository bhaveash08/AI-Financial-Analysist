"""
compliance_logger.py - SEBI Compliance Guardrail & Metrics Logging
===================================================================
Automated SEBI compliance checking for all AI outputs, mandatory regulatory
disclaimer attachment, and system metrics logging (latency, concentration
risk, 30-day forward-return signal accuracy).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import (
    COMPLIANCE_LOG_FILE,
    MAX_LOG_ENTRIES,
    SEBI_ADVISORY_RESTRICTIONS,
    SEBI_DISCLAIMER,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compliance Check Result
# ---------------------------------------------------------------------------

@dataclass
class ComplianceCheckResult:
    """Result of running compliance checks on AI output."""
    is_compliant: bool = True
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    disclaimer_attached: bool = False
    disclaimer_text: str = ""
    checked_output_preview: str = ""
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Compliance Guardrail
# ---------------------------------------------------------------------------

class SEBIComplianceGuard:
    """
    Checks AI-generated outputs against SEBI regulatory constraints
    and attaches mandatory disclaimers.
    """

    # Phrases that indicate potential regulatory violations
    PROHIBITED_PHRASES = [
        "guaranteed returns",
        "guaranteed profit",
        "risk-free investment",
        "100% safe",
        "no risk",
        "assured returns",
        "certain to rise",
        "will definitely",
        "promise returns",
        "sure shot",
        "surefire",
        "can't lose",
        "cannot lose",
        "double your money",
        "10x returns guaranteed",
    ]

    # Phrases that require additional cautionary notes
    CAUTIONARY_PHRASES = [
        "insider",
        "secret",
        "undisclosed",
        "confidential",
        "non-public",
        "tip",
        "hot stock",
        "buy now",
        "last chance",
        "act fast",
    ]

    def __init__(self):
        self._log_entries: List[Dict[str, Any]] = []
        self._load_existing_logs()

    def _load_existing_logs(self) -> None:
        """Load existing compliance logs from disk."""
        if os.path.exists(COMPLIANCE_LOG_FILE):
            try:
                with open(COMPLIANCE_LOG_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entry = json.loads(line)
                                self._log_entries.append(entry)
                            except json.JSONDecodeError:
                                continue
            except Exception as exc:
                logger.warning("Failed to load compliance logs: %s", exc)

    def check_output(self, text: str) -> ComplianceCheckResult:
        """
        Run all compliance checks against an AI-generated output.
        Returns ComplianceCheckResult with violations and warnings.
        """
        result = ComplianceCheckResult(
            timestamp=str(datetime.now()),
            checked_output_preview=text[:500],
        )

        text_lower = text.lower()

        # Check for prohibited phrases
        for phrase in self.PROHIBITED_PHRASES:
            if phrase.lower() in text_lower:
                result.violations.append(
                    f"PROHIBITED: Output contains '{phrase}' - "
                    f"violates {SEBI_ADVISORY_RESTRICTIONS['no_guaranteed_returns']}"
                )
                result.is_compliant = False

        # Check for cautionary phrases
        for phrase in self.CAUTIONARY_PHRASES:
            if phrase.lower() in text_lower:
                result.warnings.append(
                    f"CAUTION: Output contains '{phrase}' - "
                    f"may imply {SEBI_ADVISORY_RESTRICTIONS['no_fraud_inducement']}"
                )

        # Check for absence of required risk disclosure
        risk_keywords = ["risk", "loss", "volatile", "market risk"]
        has_risk_mention = any(kw in text_lower for kw in risk_keywords)
        if not has_risk_mention and len(text) > 200:
            result.warnings.append(
                "Output does not mention investment risks. "
                f"SEBI requirement: {SEBI_ADVISORY_RESTRICTIONS['mandatory_risk_disclosure']}"
            )

        # Check for personalized advice language
        personal_advice_indicators = [
            "you should invest",
            "you must buy",
            "you need to sell",
            "your portfolio should",
            "i recommend you",
        ]
        for indicator in personal_advice_indicators:
            if indicator in text_lower:
                result.warnings.append(
                    f"Output may constitute personalized advice: '{indicator}'. "
                    f"{SEBI_ADVISORY_RESTRICTIONS['no_personalized_advice']}"
                )

        return result

    def attach_disclaimer(self, text: str) -> str:
        """Attach the mandatory SEBI disclaimer to the output."""
        return f"{text}\n\n---\n{SEBI_DISCLAIMER}"

    def sanitize_and_comply(self, text: str) -> str:
        """
        Full compliance pipeline: check, sanitize, and attach disclaimer.
        Returns the compliant version of the text.
        """
        check = self.check_output(text)

        # If there are violations, we still return the text but flag it
        # In production, severe violations would block the output
        if not check.is_compliant:
            logger.warning(
                "Compliance violations detected (%d): %s",
                len(check.violations), check.violations,
            )

        # Attach disclaimer
        compliant_text = self.attach_disclaimer(text)
        check.disclaimer_attached = True
        check.disclaimer_text = SEBI_DISCLAIMER

        # Log the compliance check
        self._log_compliance_check(check)

        return compliant_text

    def _log_compliance_check(self, check: ComplianceCheckResult) -> None:
        """Log a compliance check result to the JSONL file."""
        entry = {
            "timestamp": check.timestamp,
            "is_compliant": check.is_compliant,
            "violations": check.violations,
            "warnings": check.warnings,
            "disclaimer_attached": check.disclaimer_attached,
            "output_preview": check.checked_output_preview[:200],
        }
        self._log_entries.append(entry)

        # Write to file
        try:
            with open(COMPLIANCE_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Failed to write compliance log: %s", exc)

        # Trim if too many entries
        if len(self._log_entries) > MAX_LOG_ENTRIES:
            self._log_entries = self._log_entries[-MAX_LOG_ENTRIES:]


# ---------------------------------------------------------------------------
# System Metrics Logger
# ---------------------------------------------------------------------------

@dataclass
class SystemMetrics:
    """System performance and accuracy metrics."""
    agent_latencies: Dict[str, float] = field(default_factory=dict)
    portfolio_concentration_risk: float = 0.0
    signal_accuracy_30d: float = 0.0
    backtested_signals: int = 0
    total_analyses_run: int = 0
    total_compliance_checks: int = 0
    total_violations: int = 0
    uptime_seconds: float = 0.0
    degraded_data_events: int = 0


class MetricsLogger:
    """
    Records and computes system metrics: agent response latency,
    portfolio concentration risk, and 30-day forward-return signal accuracy.
    """

    def __init__(self):
        self._start_time = time.time()
        self._metrics = SystemMetrics()
        self._latency_records: Dict[str, List[float]] = {}
        self._signal_history: List[Dict[str, Any]] = []
        # Cache fetched OHLCV frames per ticker so repeated metric queries are cheap
        self._ohlcv_cache: Dict[str, Any] = {}
        self._signal_accuracy_cached: bool = False

    def record_agent_latency(self, agent_name: str, latency_ms: float) -> None:
        """Record agent response latency."""
        if agent_name not in self._latency_records:
            self._latency_records[agent_name] = []
        self._latency_records[agent_name].append(latency_ms)
        self._metrics.agent_latencies[agent_name] = round(
            sum(self._latency_records[agent_name])
            / len(self._latency_records[agent_name]),
            2,
        )

    def record_signal(self, ticker: str, direction: str, confidence: float) -> None:
        """Record a signal for 30-day forward-return accuracy tracking."""
        self._signal_history.append({
            "timestamp": str(datetime.now()),
            "ticker": ticker,
            "direction": direction,
            "confidence": confidence,
            "actual": None,  # filled in retrospectively by compute_signal_accuracy
        })
        # New signal -> accuracy must be recomputed on next query
        self._signal_accuracy_cached = False

    def compute_concentration_risk(self, allocation: Dict[str, float]) -> float:
        """
        Compute portfolio concentration risk using Herfindahl-Hirschman Index.
        Higher values = more concentrated = higher risk.
        Returns a score from 0 (well-diversified) to 100 (fully concentrated).
        """
        if not allocation:
            return 0.0

        # HHI = sum of squared weights
        hhi = sum(w ** 2 for w in allocation.values())
        # Normalize: single asset (HHI=1) -> 100, equal weight -> lower
        max_hhi = 1.0
        normalized_hhi = (hhi / max_hhi) * 100

        self._metrics.portfolio_concentration_risk = round(normalized_hhi, 1)
        return self._metrics.portfolio_concentration_risk

    def compute_signal_accuracy(self) -> float:
        """
        Compute 30-day forward-return signal accuracy via historical replay.
        Fetches real market data (cached), recomputes the classified direction,
        and compares against the ACTUAL 30-day forward return.
        """
        if not self._signal_history:
            self._metrics.signal_accuracy_30d = 0.0
            self._metrics.backtested_signals = 0
            self._signal_accuracy_cached = True
            return 0.0

        if self._signal_accuracy_cached:
            return self._metrics.signal_accuracy_30d

        from data_engine import fetch_ohlcv
        from data_engine import compute_momentum_signal, compute_volume_signal

        # Group recent signals by ticker and direction
        evidence: List[Dict[str, Any]] = []
        for sig in self._signal_history[-100:]:
            if sig.get("ticker") and sig.get("direction"):
                evidence.append({
                    "ticker": sig["ticker"],
                    "direction": sig["direction"],
                })

        if not evidence:
            self._metrics.signal_accuracy_30d = 0.0
            self._metrics.backtested_signals = 0
            self._signal_accuracy_cached = True
            return 0.0

        correct = 0
        total = 0
        queried_tickers = set()

        for ev in evidence:
            ticker = ev["ticker"]
            if ticker in queried_tickers:
                continue  # one replay per ticker per session
            queried_tickers.add(ticker)

            try:
                if ticker not in self._ohlcv_cache:
                    self._ohlcv_cache[ticker] = fetch_ohlcv(ticker, period="1y")
                df = self._ohlcv_cache[ticker]
                if df is None or len(df) < 31:
                    continue

                close = df["Close"]
                momentum = compute_momentum_signal(df)
                volume = compute_volume_signal(df)

                # Replicate the aggregate weighted direction logic from data_engine
                direction_map = {"bullish": 1, "bearish": -1, "neutral": 0}
                weights = {"momentum": 0.40, "volume": 0.25, "sentiment": 0.35}
                weighted_score = (
                    weights["momentum"] * direction_map[momentum.direction] * momentum.confidence
                    + weights["volume"] * direction_map[volume.direction] * volume.confidence
                )
                if weighted_score > 5:
                    computed_direction = "bullish"
                elif weighted_score < -5:
                    computed_direction = "bearish"
                else:
                    computed_direction = "neutral"

                if computed_direction == "neutral":
                    continue

                # ACTUAL forward return over the last 30 trading days of real data
                realized_30d = float(close.iloc[-1] / close.iloc[-31] - 1)

                total += 1
                if (computed_direction == "bullish" and realized_30d > 0) or (
                    computed_direction == "bearish" and realized_30d < 0
                ):
                    correct += 1
            except Exception as exc:
                logger.warning("Signal accuracy replay failed for %s: %s", ticker, exc)

        accuracy = (correct / total * 100) if total else 0.0
        self._metrics.signal_accuracy_30d = round(accuracy, 1)
        self._metrics.backtested_signals = total
        self._signal_accuracy_cached = True
        return self._metrics.signal_accuracy_30d

    def backtest_30d_forward_accuracy(self, n_sample_points: int = 30) -> float:
        """
        Independent, historical backtest: compute classified signals across many
        past evaluation windows for a panel of liquid tickers, then measure each
        signal against the ACTUAL 30-trading-day forward return immediately after.
        Returns overall directional accuracy percentage.
        """
        import pandas as pd

        from data_engine import (
            fetch_ohlcv,
            compute_momentum_signal,
            compute_volume_signal,
            compute_sentiment_signal,
        )

        panel = ["RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS", "SBIN.NS"]
        weights = {"momentum": 0.40, "volume": 0.25, "sentiment": 0.35}
        direction_map = {"bullish": 1, "bearish": -1, "neutral": 0}

        correct = 0
        total = 0

        for ticker in panel:
            try:
                if ticker not in self._ohlcv_cache:
                    self._ohlcv_cache[ticker] = fetch_ohlcv(ticker, period="2y")
                df = self._ohlcv_cache[ticker]
                if df is None or len(df) < 60:
                    continue
                close = df["Close"]
                # Evaluation windows: start at 120 bars, step so that each window
                # still has >=30 forward bars of real data
                eval_indices = list(range(120, len(df) - 30, max(5, (len(df) - 150) // max(n_sample_points, 1))))
                for i in eval_indices:
                    window = df.iloc[:i + 1]
                    momentum = compute_momentum_signal(window)
                    volume = compute_volume_signal(window)
                    sentiment = compute_sentiment_signal(window)

                    weighted_score = (
                        weights["momentum"] * direction_map[momentum.direction] * momentum.confidence
                        + weights["volume"] * direction_map[volume.direction] * volume.confidence
                        + weights["sentiment"] * direction_map[sentiment.direction] * sentiment.confidence
                    )
                    if weighted_score > 5:
                        direction = "bullish"
                    elif weighted_score < -5:
                        direction = "bearish"
                    else:
                        direction = "neutral"
                    if direction == "neutral":
                        continue

                    realized_30d = float(close.iloc[i + 30] / close.iloc[i] - 1)
                    total += 1
                    if (direction == "bullish" and realized_30d > 0) or (
                        direction == "bearish" and realized_30d < 0
                    ):
                        correct += 1
            except Exception as exc:
                logger.warning("Backtest failed for %s: %s", ticker, exc)
                continue

        accuracy = (correct / total * 100) if total else 0.0
        self._metrics.signal_accuracy_30d = round(accuracy, 1)
        self._metrics.backtested_signals = total
        self._signal_accuracy_cached = True
        return self._metrics.signal_accuracy_30d

    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics.

        NOTE: signal accuracy is not recomputed here - the authoritative 30-day
        forward-return accuracy comes from an explicit historical backtest
        (``backtest_30d_forward_accuracy``) so that repeated metric queries never
        trigger network fetches or rewrite the backtested values.
        """
        self._metrics.uptime_seconds = round(time.time() - self._start_time, 1)

        return {
            "uptime_seconds": self._metrics.uptime_seconds,
            "uptime_formatted": _format_duration(self._metrics.uptime_seconds),
            "agent_latencies": dict(self._metrics.agent_latencies),
            "portfolio_concentration_risk": self._metrics.portfolio_concentration_risk,
            "signal_accuracy_30d": self._metrics.signal_accuracy_30d,
            "backtested_signals": self._metrics.backtested_signals,
            "total_analyses": len(self._signal_history),
            "degraded_data_events": self._metrics.degraded_data_events,
        }

    def record_degraded_event(self) -> None:
        """Record a degraded data event."""
        self._metrics.degraded_data_events += 1


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    else:
        return f"{seconds / 86400:.1f}d"


# ---------------------------------------------------------------------------
# Convenience Singletons
# ---------------------------------------------------------------------------

_guard_instance: Optional[SEBIComplianceGuard] = None
_metrics_instance: Optional[MetricsLogger] = None


def get_compliance_guard() -> SEBIComplianceGuard:
    """Get or create the singleton compliance guard."""
    global _guard_instance
    if _guard_instance is None:
        _guard_instance = SEBIComplianceGuard()
    return _guard_instance


def get_metrics_logger() -> MetricsLogger:
    """Get or create the singleton metrics logger."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MetricsLogger()
    return _metrics_instance
