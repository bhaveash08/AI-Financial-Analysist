"""
profiler_allocator.py - Risk Profile-Based Portfolio Allocator
=============================================================
Inputs: User Risk Profile (Conservative, Moderate, Aggressive, Very High),
Investment Horizon, and Capital Amount (INR).
Generates customized asset allocation tables with projected returns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import ASSET_CLASS_CAGR, RISK_PROFILES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class AllocationItem:
    """Single asset class allocation entry."""
    asset_class: str
    allocation_pct: float
    amount_inr: float
    expected_cagr_min: float = 0.0
    expected_cagr_median: float = 0.0
    expected_cagr_max: float = 0.0
    risk_notes: str = ""


@dataclass
class PortfolioAllocation:
    """Complete portfolio allocation output."""
    risk_profile: str
    investment_horizon_months: int
    capital_amount_inr: float
    allocations: List[AllocationItem] = field(default_factory=list)
    projected_1y_return_min: float = 0.0
    projected_1y_return_median: float = 0.0
    projected_1y_return_max: float = 0.0
    projected_value_1y_min: float = 0.0
    projected_value_1y_median: float = 0.0
    projected_value_1y_max: float = 0.0
    total_allocation_pct: float = 0.0
    warnings: List[str] = field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# Core Allocator
# ---------------------------------------------------------------------------

def generate_allocation(
    risk_profile: str,
    horizon_months: int,
    capital_inr: float,
) -> PortfolioAllocation:
    """
    Generate a customized portfolio allocation based on risk profile,
    investment horizon, and available capital.
    """
    result = PortfolioAllocation(
        risk_profile=risk_profile,
        investment_horizon_months=horizon_months,
        capital_amount_inr=capital_inr,
    )

    # Validate inputs
    profile = risk_profile.strip().title()
    if profile not in RISK_PROFILES:
        result.warnings.append(
            f"Unknown risk profile '{risk_profile}'. Defaulting to Moderate."
        )
        profile = "Moderate"

    if capital_inr <= 0:
        result.warnings.append("Capital must be positive. Using minimum Rs 10,000.")
        capital_inr = 10_000

    if horizon_months <= 0:
        result.warnings.append("Investment horizon must be positive. Defaulting to 12 months.")
        horizon_months = 12

    allocations_map = RISK_PROFILES[profile]

    # Horizon-based adjustments
    horizon_factor = _get_horizon_adjustment(horizon_months, profile)

    total_pct = 0.0
    for asset_class, base_pct in allocations_map.items():
        adjusted_pct = base_pct * horizon_factor.get(asset_class, 1.0)
        adjusted_pct = max(0, min(adjusted_pct, 0.60))  # Cap any single asset at 60%
        total_pct += adjusted_pct

        cagr = ASSET_CLASS_CAGR.get(asset_class, {"min": 0.05, "median": 0.07, "max": 0.09})

        amount = capital_inr * adjusted_pct

        # Horizon-specific risk notes
        risk_notes = _get_asset_risk_notes(asset_class, horizon_months)

        result.allocations.append(AllocationItem(
            asset_class=asset_class,
            allocation_pct=round(adjusted_pct * 100, 1),
            amount_inr=round(amount, 2),
            expected_cagr_min=cagr["min"],
            expected_cagr_median=cagr["median"],
            expected_cagr_max=cagr["max"],
            risk_notes=risk_notes,
        ))

    # Normalize allocations if total exceeds 100%
    if total_pct > 1.0:
        factor = 1.0 / total_pct
        for item in result.allocations:
            item.allocation_pct = round(item.allocation_pct * factor, 1)
            item.amount_inr = round(capital_inr * item.allocation_pct / 100, 2)
        result.warnings.append(
            f"Allocations normalized from {total_pct * 100:.1f}% to 100%"
        )

    result.total_allocation_pct = round(
        sum(a.allocation_pct for a in result.allocations), 1
    )

    # Project returns
    _compute_projected_returns(result, horizon_months)

    # Generate summary
    result.summary = _build_summary(result)

    return result


def _get_horizon_adjustment(months: int, profile: str) -> Dict[str, float]:
    """
    Adjust allocation weights based on investment horizon.
    Shorter horizons shift toward debt/liquid; longer horizons shift toward equity.
    """
    if months <= 12:
        return {
            "Large Cap": 0.9,
            "Mid Cap": 0.6,
            "Small Cap": 0.3,
            "Debt / Liquid Funds": 1.5,
            "Sovereign Gold Bonds": 1.2,
            "Fixed Deposit / T-Bills": 1.8,
        }
    elif months <= 36:
        return {
            "Large Cap": 1.0,
            "Mid Cap": 0.9,
            "Small Cap": 0.6,
            "Debt / Liquid Funds": 1.2,
            "Sovereign Gold Bonds": 1.1,
            "Fixed Deposit / T-Bills": 1.2,
        }
    elif months <= 60:
        return {
            "Large Cap": 1.1,
            "Mid Cap": 1.1,
            "Small Cap": 1.0,
            "Debt / Liquid Funds": 0.9,
            "Sovereign Gold Bonds": 0.9,
            "Fixed Deposit / T-Bills": 0.8,
        }
    else:
        return {
            "Large Cap": 1.2,
            "Mid Cap": 1.2,
            "Small Cap": 1.3,
            "Debt / Liquid Funds": 0.7,
            "Sovereign Gold Bonds": 0.8,
            "Fixed Deposit / T-Bills": 0.6,
        }


def _get_asset_risk_notes(asset_class: str, horizon_months: int) -> str:
    """Generate risk notes for a specific asset class based on horizon."""
    notes = {
        "Large Cap": "Suitable for all horizons. Lower volatility compared to mid/small cap.",
        "Mid Cap": "Moderate to high volatility. Ideal for 3-5 year horizons.",
        "Small Cap": "High volatility. Best suited for 5+ year horizons.",
        "Debt / Liquid Funds": "Low risk, stable returns. Good for short-term parking.",
        "Sovereign Gold Bonds": "Government-backed. Gold price volatility but 2.5% annual interest.",
        "Fixed Deposit / T-Bills": "Capital protected. Returns may lag inflation over long periods.",
    }
    base = notes.get(asset_class, "")

    if horizon_months <= 12 and asset_class in ("Small Cap", "Mid Cap"):
        base += " WARNING: Short horizon increases risk of capital loss."
    if horizon_months > 60 and asset_class in ("Fixed Deposit / T-Bills",):
        base += " NOTE: May significantly underperform over long horizons."

    return base


def _compute_projected_returns(result: PortfolioAllocation, horizon_months: int) -> None:
    """Compute projected portfolio returns across min/median/max scenarios."""
    years = horizon_months / 12.0

    weighted_min = 0.0
    weighted_median = 0.0
    weighted_max = 0.0

    for item in result.allocations:
        weight = item.allocation_pct / 100.0
        weighted_min += weight * item.expected_cagr_min
        weighted_median += weight * item.expected_cagr_median
        weighted_max += weight * item.expected_cagr_max

    result.projected_1y_return_min = round(weighted_min, 4)
    result.projected_1y_return_median = round(weighted_median, 4)
    result.projected_1y_return_max = round(weighted_max, 4)

    capital = result.capital_amount_inr
    result.projected_value_1y_min = round(capital * (1 + weighted_min) ** years, 2)
    result.projected_value_1y_median = round(capital * (1 + weighted_median) ** years, 2)
    result.projected_value_1y_max = round(capital * (1 + weighted_max) ** years, 2)


def _build_summary(result: PortfolioAllocation) -> str:
    """Build a human-readable summary of the allocation."""
    lines = [
        f"Portfolio Allocation Summary",
        f"Profile: {result.risk_profile} | Horizon: {result.investment_horizon_months} months | "
        f"Capital: Rs {result.capital_amount_inr:,.0f}",
        "",
        "Asset Allocation:",
    ]
    for item in result.allocations:
        lines.append(
            f"  {item.asset_class}: {item.allocation_pct}% "
            f"(Rs {item.amount_inr:,.0f})"
        )
    lines.append("")
    lines.append(
        f"Expected CAGR Range: "
        f"{result.projected_1y_return_min:.1%} - {result.projected_1y_return_max:.1%} "
        f"(median: {result.projected_1y_return_median:.1%})"
    )
    lines.append(
        f"Projected Value ({result.investment_horizon_months}mo): "
        f"Rs {result.projected_value_1y_min:,.0f} - Rs {result.projected_value_1y_max:,.0f} "
        f"(median: Rs {result.projected_value_1y_median:,.0f})"
    )
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in result.warnings:
            lines.append(f"  ! {w}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Allocation Table for Streamlit
# ---------------------------------------------------------------------------

def allocation_to_dataframe(result: PortfolioAllocation) -> pd.DataFrame:
    """Convert PortfolioAllocation to a pandas DataFrame for Streamlit display."""
    rows = []
    for item in result.allocations:
        rows.append({
            "Asset Class": item.asset_class,
            "Allocation %": item.allocation_pct,
            "Amount (INR)": f"Rs {item.amount_inr:,.0f}",
            "Min CAGR": f"{item.expected_cagr_min:.1%}",
            "Median CAGR": f"{item.expected_cagr_median:.1%}",
            "Max CAGR": f"{item.expected_cagr_max:.1%}",
            "Risk Notes": item.risk_notes,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Agent-Output Personalization (Risk-Profile Aware)
# ---------------------------------------------------------------------------

@dataclass
class PersonalizedRecommendation:
    """
    The SAME market signal, adjusted for a specific user's risk profile.
    Demonstrates that identical market inputs yield different agent outputs
    for different users.
    """
    base_recommendation: str
    base_confidence: float
    risk_profile: str
    horizon_months: int
    adjusted_recommendation: str
    adjusted_confidence: float
    adjustment_notes: List[str] = field(default_factory=list)


def personalize_recommendation(
    base_recommendation: str,
    base_confidence: float,
    risk_profile: str,
    horizon_months: int = 36,
) -> PersonalizedRecommendation:
    """
    Modify a raw synthesis recommendation (BUY/HOLD/SELL) based on the user's
    stored risk profile and investment horizon.

    The raw signal is identical for every user; the profile transforms it:
      - Conservative: downgrades BUY unless extremely high confidence, caps confidence
      - Moderate:     neutral, slight confidence dampening at extremes
      - Aggressive:   keeps BUY, relaxes the confidence bar, may convert strong HOLD
      - Very High:    activates even at moderate confidence, keeps conviction high
    """
    reco = base_recommendation.strip().upper()
    confidence = max(0.0, min(100.0, float(base_confidence)))
    profile = risk_profile.strip().title()
    if profile not in RISK_PROFILES:
        profile = "Moderate"

    horizon_bars = {
        "Conservative": (80.0, 85.0, "down_up"),
        "Moderate": (70.0, 75.0, "moderate"),
        "Aggressive": (60.0, 65.0, "up_up"),
        "Very High": (50.0, 55.0, "up_up"),
    }
    buy_floor, sell_floor, mode = horizon_bars[profile]

    adjusted_reco = reco
    adjusted_conf = confidence
    notes: List[str] = [
        f"Raw signal (identical for all users): {reco} @ {confidence:.1f}% confidence"
    ]

    # Translate based on profile stance
    if reco == "BUY":
        if confidence >= buy_floor:
            adjusted_reco = "BUY"
            adjusted_conf = confidence
            notes.append(f"Conviction above {buy_floor:.0f}% threshold -> maintain BUY")
        else:
            adjusted_reco = "HOLD"
            adjusted_conf = min(adjusted_conf, buy_floor - 5)
            notes.append(
                f"BUY signal lacks conviction for a {profile} investor "
                f"(threshold {buy_floor:.0f}%) -> downgraded to HOLD"
            )
    elif reco == "SELL":
        if confidence >= sell_floor:
            adjusted_reco = "SELL"
            notes.append(f"Escape signal confirmed above {sell_floor:.0f}% threshold")
        else:
            adjusted_reco = "HOLD"
            notes.append(
                f"SELL signal trimmed to HOLD for {profile} profile (below "
                f"{sell_floor:.0f}% conviction)"
            )
    elif reco == "HOLD":
        # Aggressive / Very High may convert a bullish-leaning HOLD into BUY
        if mode == "up_up" and confidence >= buy_floor:
            adjusted_reco = "BUY"
            adjusted_conf = min(adjusted_conf, 90)
            notes.append(
                f"HOLD converted to BUY for the {profile} profile due to "
                f"sufficient bullish conviction ({confidence:.1f}% >= {buy_floor:.0f}%)"
            )
        else:
            adjusted_reco = "HOLD"
            notes.append(f"Aggression level ({profile}) too low to convert HOLD into an entry")

    # Horizon override: very short horizons dampen equity conviction
    if horizon_months <= 12 and adjusted_reco == "BUY":
        adjusted_reco = "HOLD"
        adjusted_conf = min(adjusted_conf, 65.0)
        notes.append(
            f"Horizon override: {horizon_months} months is short for equity entry "
            f"-> BUY softened to HOLD"
        )

    adjusted_conf = round(max(0.0, min(100.0, adjusted_conf)), 1)

    return PersonalizedRecommendation(
        base_recommendation=reco,
        base_confidence=round(confidence, 1),
        risk_profile=profile,
        horizon_months=horizon_months,
        adjusted_recommendation=adjusted_reco,
        adjusted_confidence=adjusted_conf,
        adjustment_notes=notes,
    )


def personalize_across_profiles(
    base_recommendation: str,
    base_confidence: float,
    horizon_months: int = 36,
) -> Dict[str, PersonalizedRecommendation]:
    """
    Run personalization across all four risk profiles on the SAME market signal,
    providing an explicit side-by-side demonstration that identical market inputs
    produce different agent outputs per user profile.
    """
    return {
        profile: personalize_recommendation(
            base_recommendation, base_confidence, profile, horizon_months
        )
        for profile in ["Conservative", "Moderate", "Aggressive", "Very High"]
    }
