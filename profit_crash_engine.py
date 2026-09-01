"""
profit_crash_engine.py - Profit Projection & Macro Crash Early Warning
========================================================================
Simulates Min/Median/Max expected returns based on investment duration and asset CAGR.
Analyzes Nifty 50 valuation metrics and volatility trends (India VIX) to issue
dynamic market crash risk warnings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import ASSET_CLASS_CAGR, CRASH_WARNING_CONFIG
from data_engine import fetch_nifty_data, fetch_vix, compute_sma, compute_rsi

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profit Projection
# ---------------------------------------------------------------------------

@dataclass
class ProfitProjection:
    """Simulated return projections for a portfolio over a given duration."""
    capital: float
    duration_months: int
    scenarios: Dict[str, Dict[str, float]] = field(default_factory=dict)
    projection_curve_min: List[float] = field(default_factory=list)
    projection_curve_median: List[float] = field(default_factory=list)
    projection_curve_max: List[float] = field(default_factory=list)
    time_labels: List[int] = field(default_factory=list)
    summary: str = ""


def simulate_profit_projection(
    capital: float,
    allocation: Dict[str, float],
    duration_months: int,
    steps: int = 12,
) -> ProfitProjection:
    """
    Simulate Min/Median/Max expected returns over the investment duration.
    allocation: dict of asset_class -> weight (0.0 to 1.0)
    Returns projection curves for plotting.
    """
    result = ProfitProjection(capital=capital, duration_months=duration_months)

    if capital <= 0 or duration_months <= 0:
        result.summary = "Invalid inputs for projection."
        return result

    years = duration_months / 12.0
    step_months = max(1, duration_months // steps)
    time_points = list(range(0, duration_months + 1, step_months))
    if time_points[-1] != duration_months:
        time_points.append(duration_months)

    result.time_labels = time_points

    # Weighted CAGR across all asset classes
    weighted_min = 0.0
    weighted_median = 0.0
    weighted_max = 0.0

    for asset_class, weight in allocation.items():
        cagr = ASSET_CLASS_CAGR.get(asset_class, {"min": 0.05, "median": 0.07, "max": 0.09})
        weighted_min += weight * cagr["min"]
        weighted_median += weight * cagr["median"]
        weighted_max += weight * cagr["max"]

    # Generate curves
    for scenario_name, weighted_cagr in [
        ("min", weighted_min), ("median", weighted_median), ("max", weighted_max)
    ]:
        curve = []
        for t in time_points:
            y = t / 12.0
            value = capital * (1 + weighted_cagr) ** y
            curve.append(round(value, 2))

        if scenario_name == "min":
            result.projection_curve_min = curve
        elif scenario_name == "median":
            result.projection_curve_median = curve
        else:
            result.projection_curve_max = curve

    # Per-asset-class scenarios
    for asset_class, weight in allocation.items():
        cagr = ASSET_CLASS_CAGR.get(asset_class, {"min": 0.05, "median": 0.07, "max": 0.09})
        amount = capital * weight
        result.scenarios[asset_class] = {
            "invested": round(amount, 2),
            "min_value": round(amount * (1 + cagr["min"]) ** years, 2),
            "median_value": round(amount * (1 + cagr["median"]) ** years, 2),
            "max_value": round(amount * (1 + cagr["max"]) ** years, 2),
            "min_return_pct": round(cagr["min"] * 100, 1),
            "median_return_pct": round(cagr["median"] * 100, 1),
            "max_return_pct": round(cagr["max"] * 100, 1),
        }

    result.summary = (
        f"Projection for Rs {capital:,.0f} over {duration_months} months:\n"
        f"  Conservative: Rs {result.projection_curve_min[-1]:,.0f} "
        f"({weighted_min:.1%} CAGR)\n"
        f"  Expected: Rs {result.projection_curve_median[-1]:,.0f} "
        f"({weighted_median:.1%} CAGR)\n"
        f"  Optimistic: Rs {result.projection_curve_max[-1]:,.0f} "
        f"({weighted_max:.1%} CAGR)"
    )
    return result


# ---------------------------------------------------------------------------
# Crash Early Warning System
# ---------------------------------------------------------------------------

@dataclass
class CrashWarningOutput:
    """Output from the macro crash early warning analysis."""
    overall_risk_level: str = "LOW"  # LOW / MODERATE / ELEVATED / HIGH / EXTREME
    overall_risk_score: float = 0.0
    vix_analysis: Dict[str, Any] = field(default_factory=dict)
    nifty_valuation: Dict[str, Any] = field(default_factory=dict)
    trend_analysis: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    detailed_report: str = ""


def analyze_crash_risk() -> CrashWarningOutput:
    """
    Analyze macro indicators for early crash warning signals:
    1. India VIX level and trend
    2. Nifty 50 price trend and momentum
    3. Moving average crossovers
    4. Historical volatility regime
    """
    result = CrashWarningOutput()
    risk_score = 0.0

    # --- VIX Analysis ---
    try:
        vix_df = fetch_vix()
        if vix_df is not None and not vix_df.empty and len(vix_df) >= 5:
            current_vix = float(vix_df["Close"].iloc[-1])
            avg_vix_20d = float(vix_df["Close"].tail(20).mean())
            vix_trend = "rising" if current_vix > avg_vix_20d else "falling"
            vix_change_5d = current_vix - float(vix_df["Close"].iloc[-min(5, len(vix_df))])

            result.vix_analysis = {
                "current_vix": round(current_vix, 2),
                "avg_vix_20d": round(avg_vix_20d, 2),
                "trend": vix_trend,
                "5d_change": round(vix_change_5d, 2),
            }

            config = CRASH_WARNING_CONFIG
            if current_vix >= config["vix_extreme_threshold"]:
                risk_score += 30
                result.warnings.append(
                    f"VIX EXTREME: {current_vix:.1f} (threshold: {config['vix_extreme_threshold']})"
                )
            elif current_vix >= config["vix_elevated_threshold"]:
                risk_score += 15
                result.warnings.append(
                    f"VIX ELEVATED: {current_vix:.1f} (threshold: {config['vix_elevated_threshold']})"
                )

            if vix_trend == "rising" and vix_change_5d > 3:
                risk_score += 10
                result.warnings.append(
                    f"VIX rising sharply: +{vix_change_5d:.1f} in 5 days"
                )
    except Exception as exc:
        logger.warning("VIX crash analysis failed: %s", exc)
        result.warnings.append("VIX data unavailable - crash analysis partially degraded")

    # --- Nifty 50 Trend Analysis ---
    try:
        nifty_df = fetch_nifty_data()
        if nifty_df is not None and not nifty_df.empty and len(nifty_df) >= 50:
            close = nifty_df["Close"]
            current_price = float(close.iloc[-1])

            sma_20 = compute_sma(close, 20)
            sma_50 = compute_sma(close, 50)
            rsi = compute_rsi(close, 14)

            sma20_val = float(sma_20.iloc[-1])
            sma50_val = float(sma_50.iloc[-1])
            current_rsi = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

            # Price vs SMA position
            below_sma20 = current_price < sma20_val
            below_sma50 = current_price < sma50_val

            # 1-month return
            if len(close) >= 22:
                monthly_return = (current_price / float(close.iloc[-22]) - 1) * 100
            else:
                monthly_return = 0.0

            # 3-month return
            if len(close) >= 66:
                quarterly_return = (current_price / float(close.iloc[-66]) - 1) * 100
            else:
                quarterly_return = 0.0

            result.nifty_valuation = {
                "current_price": round(current_price, 2),
                "sma_20": round(sma20_val, 2),
                "sma_50": round(sma50_val, 2),
                "rsi_14": round(current_rsi, 2),
                "monthly_return_pct": round(monthly_return, 2),
                "quarterly_return_pct": round(quarterly_return, 2),
                "below_sma_20": below_sma20,
                "below_sma_50": below_sma50,
            }

            if below_sma20 and below_sma50:
                risk_score += 15
                result.warnings.append(
                    f"Nifty {current_price:.0f} below both SMA20 ({sma20_val:.0f}) "
                    f"and SMA50 ({sma50_val:.0f})"
                )
            elif below_sma50:
                risk_score += 8
                result.warnings.append(f"Nifty below SMA50 ({sma50_val:.0f})")

            if monthly_return < -5:
                risk_score += 12
                result.warnings.append(f"Nifty down {monthly_return:.1f}% in 1 month")
            elif monthly_return < -10:
                risk_score += 20
                result.warnings.append(f"Nifty down {monthly_return:.1f}% in 1 month - significant decline")

            if current_rsi < 30:
                risk_score += 5
                result.warnings.append(f"Nifty RSI oversold at {current_rsi:.1f}")
            elif current_rsi > 80:
                risk_score -= 5  # Very overbought can mean strong momentum

            # Trend analysis
            daily_returns = close.pct_change().dropna().tail(20)
            up_pct = (daily_returns > 0).sum() / len(daily_returns) if len(daily_returns) > 0 else 0.5
            result.trend_analysis = {
                "up_days_20d_pct": round(up_pct * 100, 1),
                "trend": "bearish" if up_pct < 0.35 else ("bullish" if up_pct > 0.65 else "neutral"),
            }

            if up_pct < 0.35:
                risk_score += 10
                result.warnings.append(
                    f"Bearish trend: only {up_pct:.0%} up days in last 20"
                )
        else:
            result.warnings.append("Insufficient Nifty data for crash analysis")

    except Exception as exc:
        logger.warning("Nifty crash analysis failed: %s", exc)
        result.warnings.append("Nifty data unavailable - crash analysis partially degraded")

    # Normalize risk score to 0-100
    result.overall_risk_score = round(min(max(risk_score, 0), 100), 1)

    # Classify risk level
    if result.overall_risk_score >= 70:
        result.overall_risk_level = "EXTREME"
    elif result.overall_risk_score >= 50:
        result.overall_risk_level = "HIGH"
    elif result.overall_risk_score >= 30:
        result.overall_risk_level = "ELEVATED"
    elif result.overall_risk_score >= 15:
        result.overall_risk_level = "MODERATE"
    else:
        result.overall_risk_level = "LOW"

    # Recommendations
    if result.overall_risk_level in ("HIGH", "EXTREME"):
        result.recommendations = [
            "Consider reducing equity exposure and increasing allocation to debt/gold",
            "Implement stop-loss orders on high-beta positions",
            "Review portfolio concentration risk",
            "Avoid fresh lumpsum equity investments until VIX normalizes",
            "Consider hedging through index put options",
        ]
    elif result.overall_risk_level == "ELEVATED":
        result.recommendations = [
            "Monitor portfolio more closely in coming sessions",
            "Avoid increasing small-cap/mid-cap positions",
            "Maintain adequate emergency liquidity",
        ]
    elif result.overall_risk_level == "MODERATE":
        result.recommendations = [
            "Market conditions are generally stable",
            "Continue with systematic investment plan (SIP)",
            "Review allocation quarterly",
        ]
    else:
        result.recommendations = [
            "Market conditions appear favorable",
            "Continue with existing investment strategy",
            "Opportunistic accumulation during any brief dips",
        ]

    # Build detailed report
    report_lines = [
        "=" * 60,
        "MACRO CRASH EARLY WARNING REPORT",
        "=" * 60,
        f"Overall Risk Level: {result.overall_risk_level}",
        f"Risk Score: {result.overall_risk_score}/100",
        "",
    ]
    if result.vix_analysis:
        va = result.vix_analysis
        report_lines.extend([
            "--- VIX Analysis ---",
            f"  Current VIX: {va.get('current_vix', 'N/A')}",
            f"  20D Average: {va.get('avg_vix_20d', 'N/A')}",
            f"  Trend: {va.get('trend', 'N/A')}",
            f"  5D Change: {va.get('5d_change', 'N/A')}",
        ])
    if result.nifty_valuation:
        nv = result.nifty_valuation
        report_lines.extend([
            "",
            "--- Nifty 50 Valuation ---",
            f"  Current: {nv.get('current_price', 'N/A')}",
            f"  SMA20: {nv.get('sma_20', 'N/A')}",
            f"  SMA50: {nv.get('sma_50', 'N/A')}",
            f"  RSI(14): {nv.get('rsi_14', 'N/A')}",
            f"  Monthly Return: {nv.get('monthly_return_pct', 'N/A')}%",
        ])

    if result.warnings:
        report_lines.extend(["", "--- Active Warnings ---"])
        for w in result.warnings:
            report_lines.append(f"  WARNING: {w}")

    report_lines.extend(["", "--- Recommendations ---"])
    for r in result.recommendations:
        report_lines.append(f"  -> {r}")

    result.detailed_report = "\n".join(report_lines)
    return result
