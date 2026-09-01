"""
config.py - System Settings, API Configurations, SEBI Risk Boundaries & Mock Fallback Toggles
===============================================================================================
Central configuration hub for the PS-01 Autonomous Financial Intelligence Platform.
All modules pull their settings, thresholds, and regulatory boundaries from here.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Global Toggles
# ---------------------------------------------------------------------------
USE_MOCK_DATA: bool = os.getenv("PS01_USE_MOCK", "false").lower() == "true"
DEGRADED_DATA_MODE: bool = False  # flipped at runtime when feeds fail

# ---------------------------------------------------------------------------
# API & Data-Feed Configuration
# ---------------------------------------------------------------------------
YAHOO_FINANCE_TLD: str = "in"  # Indian market tickers (.NS / .BO)
MARKET_INDEX_TICKER: str = "^NSEI"  # Nifty 50
INDIA_VIX_TICKER: str = "^INDIAVIX"
CRUDE_TICKER: str = "CL=F"
US10Y_TICKER: str = "^TNX"

DATA_FETCH_TIMEOUT: int = 30  # seconds
MAX_RETRY_ATTEMPTS: int = 3
RETRY_BACKOFF_FACTOR: float = 1.5

# ---------------------------------------------------------------------------
# ISIN-to-Ticker Mapping (Indian Equities)
# ---------------------------------------------------------------------------
ISIN_MAP: Dict[str, Dict[str, str]] = {
    "INE002A01018": {"ticker": "RELIANCE.NS", "name": "Reliance Industries Ltd", "exchange": "NSE"},
    "INE040A01012": {"ticker": "HDFCBANK.NS", "name": "HDFC Bank Ltd", "exchange": "NSE"},
    "INE009A01021": {"ticker": "INFY.NS", "name": "Infosys Ltd", "exchange": "NSE"},
    "INE467B01014": {"ticker": "TCS.NS", "name": "Tata Consultancy Services Ltd", "exchange": "NSE"},
    "INE154A01026": {"ticker": "LT.NS", "name": "Larsen & Toubro Ltd", "exchange": "NSE"},
    "INE023A01015": {"ticker": "ITC.NS", "name": "ITC Ltd", "exchange": "NSE"},
    "INE142A01011": {"ticker": "ONGC.NS", "name": "Oil & Natural Gas Corp Ltd", "exchange": "NSE"},
    "INE481G01011": {"ticker": "HCLTECH.NS", "name": "HCL Technologies Ltd", "exchange": "NSE"},
    "INE098A01019": {"ticker": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank Ltd", "exchange": "NSE"},
    "INE528G01035": {"ticker": "SBIN.NS", "name": "State Bank of India", "exchange": "NSE"},
    "INE528F01037": {"ticker": "BHARTIARTL.NS", "name": "Bharti Airtel Ltd", "exchange": "NSE"},
    "INE792A01010": {"ticker": "ICICIBANK.NS", "name": "ICICI Bank Ltd", "exchange": "NSE"},
    "INE357D01014": {"ticker": "AXISBANK.NS", "name": "Axis Bank Ltd", "exchange": "NSE"},
    "INE010A01010": {"ticker": "TATAMOTORS.NS", "name": "Tata Motors Ltd", "exchange": "NSE"},
    "INE075A01022": {"ticker": "WIPRO.NS", "name": "Wipro Ltd", "exchange": "NSE"},
    "INE120G01017": {"ticker": "ASIANPAINT.NS", "name": "Asian Paints Ltd", "exchange": "NSE"},
    "INE029A01011": {"ticker": "POWERGRID.NS", "name": "Power Grid Corp of India Ltd", "exchange": "NSE"},
    "INE062A01015": {"ticker": "TATACONSUM.NS", "name": "Tata Consumer Products Ltd", "exchange": "NSE"},
    "INE192A01025": {"ticker": "MARUTI.NS", "name": "Maruti Suzuki India Ltd", "exchange": "NSE"},
    "INE237A01028": {"ticker": "ULTRACEMCO.NS", "name": "UltraTech Cement Ltd", "exchange": "NSE"},
}

# ---------------------------------------------------------------------------
# Technical Indicator Parameters
# ---------------------------------------------------------------------------
@dataclass
class IndicatorParams:
    RSI_PERIOD: int = 14
    RSI_OVERBOUGHT: float = 70.0
    RSI_OVERSOLD: float = 30.0
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    SMA_SHORT: int = 20
    SMA_LONG: int = 50
    EMA_SHORT: int = 12
    EMA_LONG: int = 26
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD: float = 2.0
    ATR_PERIOD: int = 14
    VOLUME_SMA_PERIOD: int = 20
    VOLUME_SPIKE_THRESHOLD: float = 2.0  # 2x average volume = anomaly

INDICATOR_PARAMS = IndicatorParams()

# ---------------------------------------------------------------------------
# SEBI Compliance & Risk Boundaries
# ---------------------------------------------------------------------------
SEBI_DISCLAIMER: str = (
    "DISCLAIMER: This platform is for educational and informational purposes only. "
    "It does not constitute investment advice, a recommendation, or an offer to buy or "
    "sell any securities. Past performance is not indicative of future results. Investments "
    "in securities market are subject to market risks. Read all related documents carefully "
    "before investing. SEBI Registration pending. The platform is an AI research tool, "
    "not a registered investment advisor under SEBI (Investment Advisers) Regulations, 2013."
)

SEBI_ADVISORY_RESTRICTIONS: Dict[str, str] = {
    "no_guaranteed_returns": "This platform cannot guarantee or promise any investment returns.",
    "no_personalized_advice": "AI-generated outputs are generic analysis, not personalized financial advice.",
    "no_fraud_inducement": "The platform must not be used to induce fraudulent market activity.",
    "mandatory_risk_disclosure": "All equity investments carry market risk. Users must assess their own risk tolerance.",
    "no_insdider_trading": "The platform does not access or promote insider information. All data is publicly available.",
    "fee_disclosure": "This platform charges no advisory fees. Any commercial arrangements will be disclosed separately.",
}

SEBI_RISK_CATEGORIES: List[str] = [
    "Low",
    "Moderately Low",
    "Moderate",
    "Moderately High",
    "High",
    "Very High",
]

# ---------------------------------------------------------------------------
# Risk Profile Allocation Templates (% of portfolio)
# ---------------------------------------------------------------------------
RISK_PROFILES: Dict[str, Dict[str, float]] = {
    "Conservative": {
        "Large Cap": 0.35,
        "Mid Cap": 0.05,
        "Small Cap": 0.00,
        "Debt / Liquid Funds": 0.40,
        "Sovereign Gold Bonds": 0.10,
        "Fixed Deposit / T-Bills": 0.10,
    },
    "Moderate": {
        "Large Cap": 0.35,
        "Mid Cap": 0.15,
        "Small Cap": 0.05,
        "Debt / Liquid Funds": 0.25,
        "Sovereign Gold Bonds": 0.10,
        "Fixed Deposit / T-Bills": 0.10,
    },
    "Aggressive": {
        "Large Cap": 0.30,
        "Mid Cap": 0.25,
        "Small Cap": 0.15,
        "Debt / Liquid Funds": 0.15,
        "Sovereign Gold Bonds": 0.10,
        "Fixed Deposit / T-Bills": 0.05,
    },
    "Very High": {
        "Large Cap": 0.25,
        "Mid Cap": 0.25,
        "Small Cap": 0.25,
        "Debt / Liquid Funds": 0.10,
        "Sovereign Gold Bonds": 0.10,
        "Fixed Deposit / T-Bills": 0.05,
    },
}

# Historical CAGR benchmarks per asset class (approx long-term Indian market)
ASSET_CLASS_CAGR: Dict[str, Dict[str, float]] = {
    "Large Cap": {"min": 0.08, "median": 0.13, "max": 0.18},
    "Mid Cap": {"min": 0.06, "median": 0.16, "max": 0.24},
    "Small Cap": {"min": 0.02, "median": 0.19, "max": 0.32},
    "Debt / Liquid Funds": {"min": 0.05, "median": 0.07, "max": 0.09},
    "Sovereign Gold Bonds": {"min": 0.06, "median": 0.10, "max": 0.14},
    "Fixed Deposit / T-Bills": {"min": 0.05, "median": 0.065, "max": 0.075},
}

# ---------------------------------------------------------------------------
# Macro Stress Scenarios
# ---------------------------------------------------------------------------
STRESS_SCENARIOS: Dict[str, Dict[str, float]] = {
    "RBI Hikes Rates by 50bps": {
        "large_cap_impact": -0.08,
        "mid_cap_impact": -0.12,
        "small_cap_impact": -0.16,
        "debt_impact": -0.04,
        "sgb_impact": -0.02,
        "fd_impact": 0.01,
        "duration_months": 6,
    },
    "Crude Oil Spikes 20%": {
        "large_cap_impact": -0.06,
        "mid_cap_impact": -0.09,
        "small_cap_impact": -0.14,
        "debt_impact": -0.02,
        "sgb_impact": 0.03,
        "fd_impact": 0.00,
        "duration_months": 3,
    },
    "Massive FII Capital Outflow ($10B+)": {
        "large_cap_impact": -0.12,
        "mid_cap_impact": -0.18,
        "small_cap_impact": -0.25,
        "debt_impact": -0.05,
        "sgb_impact": 0.02,
        "fd_impact": 0.01,
        "duration_months": 6,
    },
    "Global Recession Trigger": {
        "large_cap_impact": -0.20,
        "mid_cap_impact": -0.28,
        "small_cap_impact": -0.35,
        "debt_impact": -0.08,
        "sgb_impact": 0.05,
        "fd_impact": 0.02,
        "duration_months": 12,
    },
    "INR Depreciates 10% vs USD": {
        "large_cap_impact": -0.05,
        "mid_cap_impact": -0.08,
        "small_cap_impact": -0.11,
        "debt_impact": -0.03,
        "sgb_impact": 0.04,
        "fd_impact": 0.00,
        "duration_months": 4,
    },
    "Domestic Political Instability": {
        "large_cap_impact": -0.10,
        "mid_cap_impact": -0.15,
        "small_cap_impact": -0.22,
        "debt_impact": -0.06,
        "sgb_impact": 0.03,
        "fd_impact": 0.01,
        "duration_months": 3,
    },
}

# ---------------------------------------------------------------------------
# Crash Warning Thresholds
# ---------------------------------------------------------------------------
CRASH_WARNING_CONFIG: Dict[str, float] = {
    "vix_extreme_threshold": 30.0,
    "vix_elevated_threshold": 22.0,
    "nifty_pe_upper_bound": 28.0,
    "nifty_pe_lower_bound": 15.0,
    "fii_net_sell_threshold_billion": -5.0,
    "advance_decline_ratio_panic": 0.4,
}

# ---------------------------------------------------------------------------
# Fraud Detection Keywords & Thresholds
# ---------------------------------------------------------------------------
PUMP_KEYWORDS: List[str] = [
    "guaranteed", "double your money", "secret tip", "insider info",
    "must buy", "buy now", "last chance", "100% returns", "risk free",
    "multibagger alert", "get rich quick", "hot stock", "sure shot",
    "tip", "buy before", "moon", "rocket", "jackpot", "lottery",
]

PUMP_AND_DUMP_SIGNAL_THRESHOLDS: Dict[str, float] = {
    "volume_spike_ratio": 3.0,
    "price_surge_pct": 0.15,
    "illiquid_stock_avg_volume": 100000,
    "red_flag_score_threshold": 60.0,
}

# ---------------------------------------------------------------------------
# Localization / Vernacular Settings
# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
}

# ---------------------------------------------------------------------------
# System Prompt for Synthesis Agent
# ---------------------------------------------------------------------------
MASTER_SYNTHESIS_SYSTEM_PROMPT: str = """You are the Master Synthesis Agent of PS-01 Financial Intelligence Platform.
Your role is to unify outputs from the Technical, Fundamental, and Sentiment agents into
a single coherent investment thesis. You must:
1. Weigh conflicting signals and explain the reasoning.
2. Produce a final recommendation: BUY, HOLD, or SELL.
3. Assign a confidence score (0-100%).
4. Cite specific indicators and document references.
5. Include the mandatory SEBI disclaimer.
6. Never promise guaranteed returns. Always highlight risks.
7. Output in structured JSON format with fields: recommendation, confidence, rationale, risks, sources."""

# ---------------------------------------------------------------------------
# Compliance Logger Settings
# ---------------------------------------------------------------------------
COMPLIANCE_LOG_FILE: str = "compliance_metrics.jsonl"
MAX_LOG_ENTRIES: int = 10000

# ---------------------------------------------------------------------------
# ChromaDB Settings
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = "chroma_db"
CHROMA_COLLECTION_NAME: str = "sebi_filings"
CHROMA_DISTANCE_FN: str = "cosine"
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 50

# ---------------------------------------------------------------------------
# Application Layout
# ---------------------------------------------------------------------------
APP_TITLE: str = "FinTradai"
APP_ICON: str = "📊"
APP_LAYOUT: str = "wide"

# Chart timeframes available for ISIN analysis
CHART_TIMEFRAMES: Dict[str, int] = {
    "1M": 30,
    "3M": 90,
    "6M": 180,
    "1Y": 365,
    "2Y": 730,
    "5Y": 1825,
}
