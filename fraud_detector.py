"""
fraud_detector.py - Telegram & FinInfluencer Tip Scanner / Pump-and-Dump Detector
===================================================================================
Accepts unverified text tips and rumors, cross-evaluates them against order book
anomalies, illiquid volumes, and SEBI disclosures to flag potential pump-and-dump schemes.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    ISIN_MAP,
    PUMP_AND_DUMP_SIGNAL_THRESHOLDS,
    PUMP_KEYWORDS,
)
from data_engine import fetch_ohlcv, compute_volume_signal, compute_rsi

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fraud Analysis Output
# ---------------------------------------------------------------------------

@dataclass
class FraudAnalysisResult:
    """Result of analyzing a stock tip for potential fraud."""
    input_text: str
    detected_ticker: Optional[str] = None
    detected_isin: Optional[str] = None
    detected_company: Optional[str] = None
    red_flag_score: float = 0.0
    risk_level: str = "LOW"  # LOW / MEDIUM / HIGH / CRITICAL
    flags: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    recommendation: str = "No immediate fraud indicators detected."
    pump_keywords_found: List[str] = field(default_factory=list)
    volume_anomaly: bool = False
    price_surge_detected: bool = False
    illiquid_stock: bool = False
    sebi_disclosures_match: bool = False
    detailed_report: str = ""


# ---------------------------------------------------------------------------
# Ticker / Company Name Extraction from Free Text
# ---------------------------------------------------------------------------

def extract_ticker_from_text(text: str) -> Optional[str]:
    """
    Attempt to extract a known ticker or company name from free text.
    Searches against the ISIN_MAP in config.py.
    """
    text_upper = text.upper()
    text_lower = text.lower()

    # Direct ticker mention (e.g., RELIANCE, INFY)
    for isin, info in ISIN_MAP.items():
        ticker_base = info["ticker"].replace(".NS", "")
        if ticker_base in text_upper:
            return info["ticker"]
        if info["name"].lower() in text_lower:
            return info["ticker"]

    # Check for ISIN in text
    isin_pattern = r"INE\d{5}[A-Z]\d{6}"
    match = re.search(isin_pattern, text_upper)
    if match:
        found_isin = match.group()
        info = ISIN_MAP.get(found_isin)
        if info:
            return info["ticker"]

    return None


# ---------------------------------------------------------------------------
# Pump Keyword Scanner
# ---------------------------------------------------------------------------

def scan_pump_keywords(text: str) -> List[str]:
    """Scan text for pump-and-dump indicator keywords."""
    text_lower = text.lower()
    found = []
    for keyword in PUMP_KEYWORDS:
        if keyword.lower() in text_lower:
            found.append(keyword)
    return found


# ---------------------------------------------------------------------------
# Market Data Cross-Evaluation
# ---------------------------------------------------------------------------

def evaluate_market_anomalies(ticker: str) -> Dict[str, Any]:
    """
    Cross-evaluate a ticker's market data for fraud indicators:
    - Volume spikes
    - Price surges
    - Illiquid stock characteristics
    """
    result: Dict[str, Any] = {
        "volume_anomaly": False,
        "price_surge": False,
        "illiquid_stock": False,
        "volume_ratio": 1.0,
        "recent_price_change_pct": 0.0,
        "avg_volume": 0.0,
    }

    try:
        df = fetch_ohlcv(ticker, period="3mo")
        if df is None or df.empty or len(df) < 20:
            return result

        close = df["Close"]
        volume = df["Volume"]

        # Volume analysis
        vol_signal = compute_volume_signal(df)
        result["volume_ratio"] = vol_signal.volume_ratio
        result["volume_anomaly"] = vol_signal.is_anomaly
        result["avg_volume"] = vol_signal.avg_volume_20d

        # Check illiquid stock
        avg_vol = vol_signal.avg_volume_20d
        if avg_vol < PUMP_AND_DUMP_SIGNAL_THRESHOLDS["illiquid_stock_avg_volume"]:
            result["illiquid_stock"] = True

        # Price surge detection (last 5 days)
        if len(close) >= 5:
            five_day_return = (close.iloc[-1] / close.iloc[-5] - 1) * 100
            result["recent_price_change_pct"] = round(five_day_return, 2)
            if five_day_return > PUMP_AND_DUMP_SIGNAL_THRESHOLDS["price_surge_pct"] * 100:
                result["price_surge"] = True

        # Single day extreme move
        if len(close) >= 2:
            daily_return = (close.iloc[-1] / close.iloc[-2] - 1) * 100
            if abs(daily_return) > 10:
                result["price_surge"] = True
                result["recent_price_change_pct"] = round(daily_return, 2)

    except Exception as exc:
        logger.warning("Market anomaly evaluation failed for %s: %s", ticker, exc)

    return result


# ---------------------------------------------------------------------------
# SEBI Disclosure Cross-Check
# ---------------------------------------------------------------------------

def check_sebi_disclosures(ticker: str) -> Dict[str, Any]:
    """
    Check if a ticker has any SEBI regulatory alerts or
    if the company is flagged in SEBI circulars.
    """
    from rag_engine import get_rag_engine

    result: Dict[str, Any] = {
        "has_regulatory_mentions": False,
        "has_fraud_mentions": False,
        "regulatory_sources": [],
    }

    try:
        rag = get_rag_engine()
        reg_results = rag.retrieve(
            query="SEBI regulatory fraud penalty warning",
            ticker_filter=ticker,
            n_results=3,
        )
        fraud_results = rag.retrieve(
            query="fraudulent pump dump manipulation",
            ticker_filter=ticker,
            n_results=3,
        )

        if reg_results:
            result["has_regulatory_mentions"] = True
            result["regulatory_sources"] = [r.source for r in reg_results]

        if fraud_results:
            for r in fraud_results:
                text_lower = r.chunk_text.lower()
                if any(w in text_lower for w in ["fraud", "manipulation", "penalty", "pump"]):
                    result["has_fraud_mentions"] = True
                    break

    except Exception as exc:
        logger.warning("SEBI disclosure check failed for %s: %s", ticker, exc)

    return result


# ---------------------------------------------------------------------------
# Main Fraud Analysis Pipeline
# ---------------------------------------------------------------------------

def analyze_tip(text: str) -> FraudAnalysisResult:
    """
    Complete fraud analysis pipeline for an unverified stock tip.
    Steps:
    1. Scan for pump keywords
    2. Extract ticker from text
    3. Cross-evaluate market data anomalies
    4. Check SEBI disclosures
    5. Compute composite red-flag score
    """
    result = FraudAnalysisResult(input_text=text)

    # Step 1: Pump keyword scan
    pump_kw = scan_pump_keywords(text)
    result.pump_keywords_found = pump_kw
    keyword_score = min(len(pump_kw) * 12, 50)  # Max 50 points from keywords

    # Step 2: Extract ticker
    ticker = extract_ticker_from_text(text)
    result.detected_ticker = ticker

    if ticker:
        # Find ISIN and company name
        for isin, info in ISIN_MAP.items():
            if info["ticker"] == ticker:
                result.detected_isin = isin
                result.detected_company = info["name"]
                break

    # Step 3: Market data cross-evaluation
    market_score = 0.0
    if ticker:
        market = evaluate_market_anomalies(ticker)
        result.volume_anomaly = market["volume_anomaly"]
        result.price_surge_detected = market["price_surge"]
        result.illiquid_stock = market["illiquid_stock"]

        if market["volume_anomaly"]:
            market_score += 15
            result.flags.append(
                f"Volume anomaly: {market['volume_ratio']:.1f}x average"
            )
            result.evidence.append(
                f"Current volume is {market['volume_ratio']:.1f}x the 20-day average "
                f"({market['avg_volume']:,.0f} shares)"
            )

        if market["price_surge"]:
            market_score += 20
            result.flags.append(
                f"Abnormal price move: {market['recent_price_change_pct']:+.1f}% recently"
            )

        if market["illiquid_stock"]:
            market_score += 10
            result.flags.append(
                f"Illiquid stock: avg volume {market['avg_volume']:,.0f} "
                f"(below {PUMP_AND_DUMP_SIGNAL_THRESHOLDS['illiquid_stock_avg_volume']:,})"
            )

    # Step 4: SEBI disclosure check
    sebi_score = 0.0
    if ticker:
        sebi = check_sebi_disclosures(ticker)
        result.sebi_disclosures_match = sebi["has_fraud_mentions"] or sebi["has_regulatory_mentions"]

        if sebi["has_fraud_mentions"]:
            sebi_score += 15
            result.flags.append("SEBI fraud-related disclosure found")
        if sebi["has_regulatory_mentions"]:
            sebi_score += 5
            result.flags.append(
                f"SEBI regulatory mentions: {', '.join(sebi['regulatory_sources'][:2])}"
            )

    # Step 5: Composite red-flag score
    result.red_flag_score = round(
        min(keyword_score + market_score + sebi_score, 100), 1
    )

    # Risk level classification
    if result.red_flag_score >= 75:
        result.risk_level = "CRITICAL"
        result.recommendation = (
            "CRITICAL: Strong indicators of potential pump-and-dump scheme. "
            "DO NOT act on this tip. Report to SEBI if possible."
        )
    elif result.red_flag_score >= 50:
        result.risk_level = "HIGH"
        result.recommendation = (
            "HIGH RISK: Multiple red flags detected. Exercise extreme caution. "
            "This tip exhibits characteristics of market manipulation."
        )
    elif result.red_flag_score >= 25:
        result.risk_level = "MEDIUM"
        result.recommendation = (
            "MEDIUM RISK: Some concerning patterns detected. "
            "Verify independently before taking any action."
        )
    else:
        result.risk_level = "LOW"
        result.recommendation = (
            "No immediate fraud indicators detected. "
            "However, always verify tips independently and be wary of unsolicited advice."
        )

    # Build detailed report
    report_lines = [
        "=" * 60,
        "FRAUD DETECTION ANALYSIS REPORT",
        "=" * 60,
        f"Input Text: \"{text[:200]}{'...' if len(text) > 200 else ''}\"",
        f"Detected Ticker: {ticker or 'Not identified'}",
        f"Company: {result.detected_company or 'N/A'}",
        f"ISIN: {result.detected_isin or 'N/A'}",
        "",
        "--- Scoring Breakdown ---",
        f"Pump Keywords Found: {len(pump_kw)} (score: {keyword_score})",
        f"  Keywords: {', '.join(pump_kw) if pump_kw else 'None'}",
        f"Market Anomalies (score: {market_score})",
        f"  Volume Anomaly: {result.volume_anomaly}",
        f"  Price Surge: {result.price_surge_detected}",
        f"  Illiquid Stock: {result.illiquid_stock}",
        f"SEBI Disclosures (score: {sebi_score})",
        f"  Fraud Mentions: {result.sebi_disclosures_match}",
        "",
        f"RED FLAG SCORE: {result.red_flag_score}/100",
        f"RISK LEVEL: {result.risk_level}",
        f"RECOMMENDATION: {result.recommendation}",
        "",
        "--- Flags ---",
    ]
    for flag in result.flags:
        report_lines.append(f"  * {flag}")
    if not result.flags:
        report_lines.append("  No flags raised.")

    report_lines.append("")
    report_lines.append("--- Supporting Evidence ---")
    for ev in result.evidence:
        report_lines.append(f"  - {ev}")
    if not result.evidence:
        report_lines.append("  No market data evidence available.")

    result.detailed_report = "\n".join(report_lines)
    return result
