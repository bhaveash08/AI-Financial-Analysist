"""
riskometer.py - Interactive SEBI Risk-o-Meter Plotly Gauge
==========================================================
Renders an interactive Plotly gauge visualization mapping stock volatility and beta
to official SEBI Risk-o-Meter levels (Low, Moderately Low, Moderate, Moderately High,
High, Very High).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import SEBI_RISK_CATEGORIES
from data_engine import fetch_ohlcv
from isin_analyzer import compute_volatility, compute_beta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SEBI Risk-o-Meter Thresholds
# ---------------------------------------------------------------------------

# Volatility thresholds (annualized) mapped to SEBI risk categories
VOLATILITY_THRESHOLDS = {
    "Low": (0.0, 0.10),
    "Moderately Low": (0.10, 0.18),
    "Moderate": (0.18, 0.28),
    "Moderately High": (0.28, 0.40),
    "High": (0.40, 0.55),
    "Very High": (0.55, 1.0),
}

# Beta thresholds mapped to SEBI risk categories
BETA_THRESHOLDS = {
    "Low": (0.0, 0.5),
    "Moderately Low": (0.5, 0.8),
    "Moderate": (0.8, 1.0),
    "Moderately High": (1.0, 1.3),
    "High": (1.3, 1.6),
    "Very High": (1.6, 3.0),
}

# Gauge colors for each risk level
RISK_COLORS = {
    "Low": "#00C853",
    "Moderately Low": "#64DD17",
    "Moderate": "#FFD600",
    "Moderately High": "#FF9100",
    "High": "#FF3D00",
    "Very High": "#D50000",
}


# ---------------------------------------------------------------------------
# Risk Assessment
# ---------------------------------------------------------------------------

@dataclass
class RiskAssessment:
    """Risk assessment result for a stock."""
    ticker: str
    volatility: float = 0.0
    beta: float = 1.0
    volatility_risk: str = "Moderate"
    beta_risk: str = "Moderate"
    composite_risk: str = "Moderate"
    risk_score: float = 50.0  # 0 (lowest) to 100 (highest risk)
    risk_index: int = 3  # 0-5 mapping to SEBI categories
    details: Dict[str, Any] = field(default_factory=dict)
    sources: list = field(default_factory=list)


def assess_risk(
    ticker: str,
    market_returns: Optional[pd.Series] = None,
) -> RiskAssessment:
    """
    Compute comprehensive risk assessment mapping to SEBI Risk-o-Meter.
    Uses volatility, beta, and price patterns to determine risk level.
    """
    result = RiskAssessment(ticker=ticker)

    try:
        df = fetch_ohlcv(ticker, period="2y")
        if df is None or df.empty or len(df) < 60:
            result.composite_risk = "Moderate"
            result.risk_score = 50.0
            result.details["warning"] = "Insufficient data for accurate risk assessment"
            return result

        close = df["Close"]

        # Volatility
        result.volatility = compute_volatility(close)
        result.volatility_risk = _classify_by_volatility(result.volatility)

        # Beta
        stock_returns = close.pct_change().dropna()
        if market_returns is not None:
            result.beta = compute_beta(stock_returns, market_returns)
        else:
            result.beta = 1.0  # Default if market returns unavailable
        result.beta_risk = _classify_by_beta(result.beta)

        # Composite risk score (weighted: 60% volatility, 40% beta)
        vol_score = _risk_to_score(result.volatility_risk)
        beta_score = _risk_to_score(result.beta_risk)
        result.risk_score = round(vol_score * 0.6 + beta_score * 0.4, 1)
        result.composite_risk = SEBI_RISK_CATEGORIES[min(int(result.risk_score / 100 * 5), 5)]
        result.risk_index = SEBI_RISK_CATEGORIES.index(result.composite_risk)

        result.details = {
            "volatility_annualized": round(result.volatility, 4),
            "volatility_pct": f"{result.volatility:.1%}",
            "beta": round(result.beta, 2),
            "volatility_risk": result.volatility_risk,
            "beta_risk": result.beta_risk,
            "risk_score": result.risk_score,
        }
        result.sources = [
            "data_engine.py: Annualized volatility calculation",
            "isin_analyzer.py: Beta vs Nifty 50",
            "config.py: SEBI Risk-o-Meter thresholds",
        ]

    except Exception as exc:
        logger.error("Risk assessment failed for %s: %s", ticker, exc)
        result.composite_risk = "Moderate"
        result.risk_score = 50.0
        result.details["error"] = str(exc)

    return result


def _classify_by_volatility(vol: float) -> str:
    """Classify volatility into SEBI risk category."""
    for category, (low, high) in VOLATILITY_THRESHOLDS.items():
        if low <= vol < high:
            return category
    return "Very High"


def _classify_by_beta(beta: float) -> str:
    """Classify beta into SEBI risk category."""
    for category, (low, high) in BETA_THRESHOLDS.items():
        if low <= beta < high:
            return category
    return "Very High"


def _risk_to_score(risk_level: str) -> float:
    """Convert risk level to numeric score (0-100)."""
    scores = {
        "Low": 8,
        "Moderately Low": 25,
        "Moderate": 42,
        "Moderately High": 58,
        "High": 75,
        "Very High": 92,
    }
    return scores.get(risk_level, 50)


# ---------------------------------------------------------------------------
# Plotly Gauge Visualization
# ---------------------------------------------------------------------------

def create_risk_gauge(assessment: RiskAssessment) -> go.Figure:
    """
    Create an interactive Plotly gauge chart matching SEBI's Risk-o-Meter format.
    6 colored segments: Low -> Very High.
    """
    fig = go.Figure()

    # Build the gauge with 6 steps
    steps = []
    for i, category in enumerate(SEBI_RISK_CATEGORIES):
        color = RISK_COLORS[category]
        steps.append({
            "range": [i * (100 / 6), (i + 1) * (100 / 6)],
            "color": color,
            "thickness": 0.75,
        })

    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=assessment.risk_score,
        number={"suffix": "%", "font": {"size": 32, "color": "#ffffff"}},
        title={
            "text": f"<b>{assessment.ticker}</b><br>"
                    f"<span style='font-size:16px;color:{RISK_COLORS.get(assessment.composite_risk, '#ffffff')}'>"
                    f"Risk Level: {assessment.composite_risk}</span>",
            "font": {"size": 18, "color": "#ffffff"},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 2,
                "tickcolor": "#ffffff",
                "tickfont": {"color": "#ffffff"},
                "dtick": 100 / 6,
                "tickvals": [
                    (i * 100 / 6 + (i + 1) * 100 / 6) / 2
                    for i in range(6)
                ],
                "ticktext": [cat.replace(" ", "<br>") for cat in SEBI_RISK_CATEGORIES],
            },
            "bar": {"color": "#ffffff", "thickness": 0.3},
            "steps": steps,
            "threshold": {
                "line": {"color": "#ffffff", "width": 4},
                "thickness": 0.8,
                "value": assessment.risk_score,
            },
        },
    ))

    fig.update_layout(
        height=400,
        margin=dict(l=30, r=30, t=80, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ffffff"},
    )

    return fig


# ---------------------------------------------------------------------------
# Portfolio-Level Risk Assessment
# ---------------------------------------------------------------------------

def create_portfolio_risk_summary(
    allocations: Dict[str, float],
    individual_risks: Dict[str, RiskAssessment],
) -> Dict[str, Any]:
    """
    Compute portfolio-level risk based on individual stock risks and weights.
    """
    weighted_risk_score = 0.0
    risk_breakdown: Dict[str, Dict[str, Any]] = {}

    for asset_class, weight in allocations.items():
        if asset_class in individual_risks:
            ra = individual_risks[asset_class]
            weighted_risk_score += weight * ra.risk_score
            risk_breakdown[asset_class] = {
                "weight": weight,
                "risk_score": ra.risk_score,
                "risk_level": ra.composite_risk,
                "volatility": ra.volatility,
                "beta": ra.beta,
            }

    portfolio_risk_score = round(weighted_risk_score, 1)
    portfolio_risk_index = min(int(portfolio_risk_score / 100 * 5), 5)
    portfolio_risk_level = SEBI_RISK_CATEGORIES[portfolio_risk_index]

    return {
        "portfolio_risk_score": portfolio_risk_score,
        "portfolio_risk_level": portfolio_risk_level,
        "portfolio_risk_index": portfolio_risk_index,
        "risk_breakdown": risk_breakdown,
    }
