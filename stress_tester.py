"""
stress_tester.py - Macro "What-If" Stress Tester
=================================================
Simulates portfolio drawdowns under macroeconomic shock scenarios
such as RBI rate hikes, crude oil spikes, FII outflows, etc.
Uses predefined scenarios from config.py and computes impact on
each asset class in the portfolio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import STRESS_SCENARIOS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class StressScenarioResult:
    """Result of applying a single stress scenario to a portfolio."""
    scenario_name: str
    duration_months: int = 0
    asset_impacts: Dict[str, float] = field(default_factory=dict)
    portfolio_impact_pct: float = 0.0
    portfolio_drawdown_inr: float = 0.0
    portfolio_stressed_value: float = 0.0
    narrative: str = ""


@dataclass
class StressTestOutput:
    """Complete stress test output with all scenarios."""
    capital: float
    allocation: Dict[str, float]
    scenarios: List[StressScenarioResult] = field(default_factory=list)
    worst_case_scenario: str = ""
    worst_case_drawdown_pct: float = 0.0
    worst_case_drawdown_inr: float = 0.0
    avg_scenario_impact_pct: float = 0.0
    summary: str = ""


# ---------------------------------------------------------------------------
# Scenario Narratives
# ---------------------------------------------------------------------------

SCENARIO_NARRATIVES: Dict[str, str] = {
    "RBI Hikes Rates by 50bps": (
        "When the RBI raises the repo rate by 50 basis points, borrowing costs increase "
        "across the economy. Equity valuations compress as discount rates rise, while "
        "debt portfolios face mark-to-market losses. Large caps tend to be more resilient "
        "due to stronger balance sheets."
    ),
    "Crude Oil Spikes 20%": (
        "India imports ~85% of its crude oil requirements. A 20% spike raises the current "
        "account deficit, increases input costs for manufacturing, and fuels inflation. "
        "Oil-sensitive sectors (airlines, chemicals, paints) face margin pressure."
    ),
    "Massive FII Capital Outflow ($10B+)": (
        "Foreign Institutional Investors pulling out $10B+ creates massive selling pressure "
        "on Indian equities, particularly in large caps where FII holdings are concentrated. "
        "The rupee depreciates, further amplifying the impact."
    ),
    "Global Recession Trigger": (
        "A global recession triggers risk-off sentiment, causing capital flight from emerging "
        "markets. Export-oriented sectors face demand destruction. Defensive sectors (pharma, "
        "FMCG) may relatively outperform but cannot avoid the broad downturn."
    ),
    "INR Depreciates 10% vs USD": (
        "A 10% rupee depreciation increases import costs, widens the trade deficit, and "
        "can trigger FII selling. IT and pharma exporters benefit, while import-dependent "
        "sectors (oil & gas, electronics) face margin erosion."
    ),
    "Domestic Political Instability": (
        "Political uncertainty triggers policy paralysis and investor uncertainty. "
        "Infrastructure spending may be delayed, reform momentum stalls, and foreign "
        "investors adopt a wait-and-watch approach. Market may correct 10-20%."
    ),
}


# ---------------------------------------------------------------------------
# Core Stress Testing Engine
# ---------------------------------------------------------------------------

def run_stress_test(
    capital: float,
    allocation: Dict[str, float],
    selected_scenarios: Optional[List[str]] = None,
) -> StressTestOutput:
    """
    Run macro stress tests on a portfolio allocation.
    capital: total portfolio value in INR
    allocation: dict of asset_class -> weight (0.0 to 1.0)
    selected_scenarios: list of scenario names to test (None = all)
    """
    result = StressTestOutput(capital=capital, allocation=allocation)

    if capital <= 0:
        result.summary = "Capital must be positive for stress testing."
        return result

    scenarios_to_run = selected_scenarios or list(STRESS_SCENARIOS.keys())

    for scenario_name in scenarios_to_run:
        scenario_config = STRESS_SCENARIOS.get(scenario_name)
        if scenario_config is None:
            logger.warning("Unknown scenario: %s", scenario_name)
            continue

        stress_result = _apply_scenario(scenario_name, capital, allocation, scenario_config)
        result.scenarios.append(stress_result)

    if result.scenarios:
        # Find worst case
        worst = min(result.scenarios, key=lambda x: x.portfolio_impact_pct)
        result.worst_case_scenario = worst.scenario_name
        result.worst_case_drawdown_pct = worst.portfolio_impact_pct
        result.worst_case_drawdown_inr = worst.portfolio_drawdown_inr

        # Average impact
        result.avg_scenario_impact_pct = round(
            sum(s.portfolio_impact_pct for s in result.scenarios) / len(result.scenarios), 2
        )

        result.summary = (
            f"Stress Test Summary ({len(result.scenarios)} scenarios):\n"
            f"  Worst Case: {result.worst_case_scenario} "
            f"({result.worst_case_drawdown_pct:.1f}% / Rs {result.worst_case_drawdown_inr:,.0f})\n"
            f"  Average Impact: {result.avg_scenario_impact_pct:.1f}%\n"
            f"  Capital at Risk: Rs {capital:,.0f}"
        )

    return result


def _apply_scenario(
    scenario_name: str,
    capital: float,
    allocation: Dict[str, float],
    config: Dict[str, float],
) -> StressScenarioResult:
    """Apply a single stress scenario to the portfolio."""
    result = StressScenarioResult(
        scenario_name=scenario_name,
        duration_months=int(config.get("duration_months", 6)),
    )

    # Map asset classes to their impact fields in the scenario config
    impact_mapping = {
        "Large Cap": "large_cap_impact",
        "Mid Cap": "mid_cap_impact",
        "Small Cap": "small_cap_impact",
        "Debt / Liquid Funds": "debt_impact",
        "Sovereign Gold Bonds": "sgb_impact",
        "Fixed Deposit / T-Bills": "fd_impact",
    }

    weighted_impact = 0.0

    for asset_class, weight in allocation.items():
        impact_key = impact_mapping.get(asset_class)
        if impact_key and impact_key in config:
            impact_pct = config[impact_key]
        else:
            # Default: small negative impact if asset class not in scenario
            impact_pct = -0.02

        result.asset_impacts[asset_class] = round(impact_pct * 100, 2)
        weighted_impact += weight * impact_pct

    result.portfolio_impact_pct = round(weighted_impact * 100, 2)
    result.portfolio_drawdown_inr = round(capital * abs(weighted_impact), 2)
    result.portfolio_stressed_value = round(capital * (1 + weighted_impact), 2)

    # Generate narrative
    base_narrative = SCENARIO_NARRATIVES.get(
        scenario_name,
        f"Under {scenario_name}, the portfolio would face an estimated "
        f"{abs(result.portfolio_impact_pct):.1f}% impact."
    )
    result.narrative = (
        f"{base_narrative}\n\n"
        f"Impact on your Rs {capital:,.0f} portfolio:\n"
        f"  Stressed Value: Rs {result.portfolio_stressed_value:,.0f}\n"
        f"  Drawdown: Rs {result.portfolio_drawdown_inr:,.0f} ({result.portfolio_impact_pct:.1f}%)\n"
        f"  Recovery Period: ~{result.duration_months} months\n\n"
        f"Asset-level impacts:\n" +
        "\n".join(
            f"  {ac}: {imp:+.1f}%"
            for ac, imp in result.asset_impacts.items()
        )
    )

    return result


# ---------------------------------------------------------------------------
# Sensitivity Analysis
# ---------------------------------------------------------------------------

def run_sensitivity_analysis(
    capital: float,
    allocation: Dict[str, float],
    variable: str = "large_cap_impact",
    range_pct: Tuple[float, float] = (-0.30, 0.10),
    steps: int = 20,
) -> pd.DataFrame:
    """
    Run sensitivity analysis varying a single scenario parameter.
    Returns a DataFrame of impact vs parameter value.
    """
    low, high = range_pct
    values = np.linspace(low, high, steps)
    rows = []

    for val in values:
        # Create a modified allocation with the stress parameter
        modified_config = {
            variable: val,
            "duration_months": 6,
        }
        # Default other impacts
        for key in ["large_cap_impact", "mid_cap_impact", "small_cap_impact",
                     "debt_impact", "sgb_impact", "fd_impact"]:
            if key not in modified_config:
                modified_config[key] = 0.0

        weighted_impact = 0.0
        impact_mapping = {
            "Large Cap": "large_cap_impact",
            "Mid Cap": "mid_cap_impact",
            "Small Cap": "small_cap_impact",
            "Debt / Liquid Funds": "debt_impact",
            "Sovereign Gold Bonds": "sgb_impact",
            "Fixed Deposit / T-Bills": "fd_impact",
        }

        for asset_class, weight in allocation.items():
            key = impact_mapping.get(asset_class, "")
            weighted_impact += weight * modified_config.get(key, 0.0)

        rows.append({
            "Parameter Value": f"{val:.1%}",
            "Portfolio Impact %": round(weighted_impact * 100, 2),
            "Stressed Value INR": round(capital * (1 + weighted_impact), 2),
            "Drawdown INR": round(capital * abs(weighted_impact), 2),
        })

    return pd.DataFrame(rows)
