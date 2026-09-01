"""
isin_analyzer.py - ISIN Mapper, Interactive Plotly Candlestick Charts & Rating Engine
=====================================================================================
Resolves ISIN codes to NSE/BSE tickers, renders dynamic Plotly candlestick charts
with custom timeframes, and generates BUY/HOLD/SELL recommendations with CAGR returns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import ASSET_CLASS_CAGR, CHART_TIMEFRAMES, ISIN_MAP
from data_engine import fetch_ohlcv, compute_rsi, compute_macd, compute_sma, INDICATOR_PARAMS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ISIN Mapper
# ---------------------------------------------------------------------------

def resolve_isin(isin: str) -> Optional[Dict[str, str]]:
    """
    Resolve an ISIN code to ticker and company name.
    Returns dict with keys: ticker, name, exchange.
    Returns None if ISIN is not found in the local mapping.
    """
    normalized = isin.strip().upper()
    mapping = ISIN_MAP.get(normalized)
    if mapping:
        return dict(mapping)
    logger.warning("ISIN %s not found in local mapping", isin)
    return None


def get_available_isins() -> List[Dict[str, str]]:
    """Return all available ISINs with their details."""
    results = []
    for isin, info in ISIN_MAP.items():
        results.append({"isin": isin, **info})
    return results


# ---------------------------------------------------------------------------
# Candlestick Chart Generator
# ---------------------------------------------------------------------------

def create_candlestick_chart(
    ticker: str,
    period_days: int = 365,
    show_volume: bool = True,
    show_sma: bool = True,
    show_bollinger: bool = False,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Generate an interactive Plotly candlestick chart with optional overlays.
    Supports custom timeframes via period_days.
    """
    df = fetch_ohlcv(ticker, period="max")
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"No data available for {ticker}",
            annotations=[dict(text="Data unavailable", showarrow=False, font=dict(size=20))],
        )
        return fig

    # Filter to requested period
    if period_days and len(df) > period_days:
        df = df.tail(period_days).copy()

    # Create subplots: candlestick + volume
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
            subplot_titles=[title or f"{ticker} Price Chart", "Volume"],
        )
    else:
        fig = make_subplots(rows=1, cols=1)

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # SMA overlays
    if show_sma:
        sma_20 = compute_sma(df["Close"], INDICATOR_PARAMS.SMA_SHORT)
        sma_50 = compute_sma(df["Close"], INDICATOR_PARAMS.SMA_LONG)
        fig.add_trace(
            go.Scatter(x=df.index, y=sma_20, name=f"SMA {INDICATOR_PARAMS.SMA_SHORT}",
                       line=dict(color="#ff9800", width=1)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=sma_50, name=f"SMA {INDICATOR_PARAMS.SMA_LONG}",
                       line=dict(color="#2196f3", width=1)),
            row=1, col=1,
        )

    # Bollinger Bands
    if show_bollinger:
        period = INDICATOR_PARAMS.BOLLINGER_PERIOD
        sma = compute_sma(df["Close"], period)
        std = df["Close"].rolling(window=period).std()
        upper = sma + INDICATOR_PARAMS.BOLLINGER_STD * std
        lower = sma - INDICATOR_PARAMS.BOLLINGER_STD * std
        fig.add_trace(
            go.Scatter(x=df.index, y=upper, name="BB Upper",
                       line=dict(color="#9c27b0", width=1, dash="dash")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=lower, name="BB Lower",
                       line=dict(color="#9c27b0", width=1, dash="dash"),
                       fill="tonexty", fillcolor="rgba(156,39,176,0.1)"),
            row=1, col=1,
        )

    # Volume bars
    if show_volume:
        colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="Volume",
                   marker_color=colors, opacity=0.6),
            row=2, col=1,
        )

    # Layout
    fig.update_layout(
        title=title or f"{ticker} - Interactive Candlestick Chart",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=600 if show_volume else 450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=60, b=30),
    )
    fig.update_xaxes(title_text="Date", row=2 if show_volume else 1, col=1)
    fig.update_yaxes(title_text="Price (INR)", row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


# ---------------------------------------------------------------------------
# Rating Engine
# ---------------------------------------------------------------------------

@dataclass
class StockRating:
    """Comprehensive stock rating output."""
    ticker: str
    isin: str
    company_name: str
    current_price: float = 0.0
    cagr_1y: Optional[float] = None
    cagr_3y: Optional[float] = None
    cagr_5y: Optional[float] = None
    recommendation: str = "HOLD"
    confidence: float = 50.0
    rationale: str = ""
    risks: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    technical_score: float = 50.0
    fundamental_score: float = 50.0
    momentum_score: float = 50.0


def compute_cagr(current: float, initial: float, years: float) -> Optional[float]:
    """Compute Compound Annual Growth Rate."""
    if initial <= 0 or current <= 0 or years <= 0:
        return None
    return round((current / initial) ** (1 / years) - 1, 4)


def compute_volatility(price_series: pd.Series, annualize: bool = True) -> float:
    """Compute annualized price volatility."""
    returns = price_series.pct_change().dropna()
    vol = float(returns.std())
    if annualize:
        vol *= np.sqrt(252)
    return round(vol, 4)


def compute_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """Compute beta against the market (Nifty 50)."""
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    if len(aligned) < 30:
        return 1.0
    cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    var = aligned.iloc[:, 1].var()
    if var == 0:
        return 1.0
    return round(cov / var, 4)


def rate_stock(
    isin: str,
    market_returns: Optional[pd.Series] = None,
) -> StockRating:
    """
    Generate a comprehensive BUY / HOLD / SELL rating for a stock.
    Combines technical signals, CAGR analysis, volatility, and beta.
    """
    mapping = resolve_isin(isin)
    if mapping is None:
        return StockRating(
            ticker="UNKNOWN", isin=isin, company_name="Unknown",
            recommendation="HOLD", rationale=f"ISIN {isin} not found in database.",
            risks=["Unresolved ISIN - data unavailable"],
        )

    ticker = mapping["ticker"]
    name = mapping["name"]
    rating = StockRating(ticker=ticker, isin=isin, company_name=name)

    # Fetch 5-year data for CAGR
    df = fetch_ohlcv(ticker, period="5y")
    if df is None or df.empty or len(df) < 30:
        rating.rationale = "Insufficient historical data for comprehensive rating."
        rating.risks.append("Degraded analysis due to limited price history")
        return rating

    close = df["Close"]
    rating.current_price = round(float(close.iloc[-1]), 2)

    # CAGR calculations
    if len(close) >= 252:
        rating.cagr_1y = compute_cagr(
            float(close.iloc[-1]), float(close.iloc[-252]), 1.0
        )
    if len(close) >= 756:
        rating.cagr_3y = compute_cagr(
            float(close.iloc[-1]), float(close.iloc[-756]), 3.0
        )
    if len(close) >= 1260:
        rating.cagr_5y = compute_cagr(
            float(close.iloc[-1]), float(close.iloc[-1260]), 5.0
        )

    # Technical scoring
    rsi_series = compute_rsi(close)
    current_rsi = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0
    macd_line, signal_line, histogram = compute_macd(close)
    current_macd_hist = float(histogram.iloc[-1]) if len(histogram) > 0 else 0.0

    tech_score = 50.0
    if current_rsi < INDICATOR_PARAMS.RSI_OVERSOLD:
        tech_score += 20  # Oversold = bullish opportunity
    elif current_rsi > INDICATOR_PARAMS.RSI_OVERBOUGHT:
        tech_score -= 20
    if current_macd_hist > 0:
        tech_score += 15
    else:
        tech_score -= 15

    # Volatility penalty
    volatility = compute_volatility(close)
    rating.sources.append(f"Annualized Volatility = {volatility:.2%}")
    if volatility > 0.50:
        tech_score -= 10
        rating.risks.append(f"High volatility ({volatility:.1%}) indicates elevated risk")
    elif volatility < 0.15:
        tech_score += 5

    # Beta analysis
    if market_returns is not None:
        stock_returns = close.pct_change().dropna()
        beta = compute_beta(stock_returns, market_returns)
        rating.sources.append(f"Beta vs Nifty = {beta:.2f}")
        if beta > 1.5:
            tech_score -= 5
            rating.risks.append(f"High beta ({beta:.2f}) - amplified downside risk")
        elif beta < 0.7:
            tech_score += 5

    rating.technical_score = max(0, min(100, tech_score))

    # Fundamental scoring (from RAG - simplified heuristic here)
    fundamental_score = 50.0
    if rating.cagr_1y is not None:
        if rating.cagr_1y > 0.15:
            fundamental_score += 15
        elif rating.cagr_1y < 0:
            fundamental_score -= 10
    if rating.cagr_3y is not None:
        if rating.cagr_3y > 0.12:
            fundamental_score += 10
        elif rating.cagr_3y < 0:
            fundamental_score -= 15
    rating.fundamental_score = max(0, min(100, fundamental_score))

    # Momentum score
    momentum_score = 50.0
    if rating.cagr_1y is not None:
        momentum_score += min(rating.cagr_1y * 100, 30)
    if current_rsi < 40:
        momentum_score += 10
    elif current_rsi > 70:
        momentum_score -= 10
    rating.momentum_score = max(0, min(100, momentum_score))

    # Final recommendation
    composite = (
        0.40 * rating.technical_score
        + 0.30 * rating.fundamental_score
        + 0.30 * rating.momentum_score
    )
    rating.confidence = round(composite, 1)

    if composite >= 65:
        rating.recommendation = "BUY"
    elif composite <= 35:
        rating.recommendation = "SELL"
    else:
        rating.recommendation = "HOLD"

    # Build rationale
    rationale_parts = [
        f"Technical Score: {rating.technical_score:.0f}/100 (RSI={current_rsi:.1f})",
        f"Fundamental Score: {rating.fundamental_score:.0f}/100",
        f"Momentum Score: {rating.momentum_score:.0f}/100",
        f"Composite: {composite:.1f}/100 => {rating.recommendation}",
    ]
    if rating.cagr_1y is not None:
        rationale_parts.append(f"1Y CAGR: {rating.cagr_1y:.1%}")
    if rating.cagr_3y is not None:
        rationale_parts.append(f"3Y CAGR: {rating.cagr_3y:.1%}")
    if rating.cagr_5y is not None:
        rationale_parts.append(f"5Y CAGR: {rating.cagr_5y:.1%}")
    rating.rationale = " | ".join(rationale_parts)
    rating.sources.append("data_engine.py technical indicators")
    rating.sources.append("ISIN mapping: config.py ISIN_MAP")

    return rating
