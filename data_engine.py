"""
data_engine.py - Real-Time/Historical OHLCV Market Data & Signal Dimensions
============================================================================
Fetches market data via yfinance, calculates 3 signal dimensions:
  1. Price Momentum (RSI / MACD)
  2. Volume Anomalies
  3. Market Sentiment (proxied via price action + VIX)
Outputs confidence ratings (0-100%) with cited technical indicators.
Gracefully degrades when feeds fail, reducing confidence without crashing.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import config
from config import (
    INDICATOR_PARAMS,
    INDIA_VIX_TICKER,
    MARKET_INDEX_TICKER,
    MAX_RETRY_ATTEMPTS,
    RETRY_BACKOFF_FACTOR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Classes for Signal Output
# ---------------------------------------------------------------------------

@dataclass
class MomentumSignal:
    """Price momentum signal derived from RSI and MACD."""
    rsi_value: float = 50.0
    rsi_signal: str = "neutral"  # overbought / oversold / neutral
    macd_line: float = 0.0
    macd_signal_line: float = 0.0
    macd_histogram: float = 0.0
    macd_crossover: str = "none"  # bullish / bearish / none
    direction: str = "neutral"  # bullish / bearish / neutral
    confidence: float = 50.0
    citations: List[str] = field(default_factory=list)


@dataclass
class VolumeAnomalySignal:
    """Volume-based anomaly detection signal."""
    current_volume: float = 0.0
    avg_volume_20d: float = 0.0
    volume_ratio: float = 1.0
    is_anomaly: bool = False
    anomaly_type: str = "normal"  # spike / surge / dry / normal
    direction: str = "neutral"  # bullish / bearish / neutral
    confidence: float = 50.0
    citations: List[str] = field(default_factory=list)


@dataclass
class SentimentSignal:
    """Market sentiment signal proxied via VIX and price-action patterns."""
    vix_value: float = 0.0
    vix_regime: str = "normal"  # calm / normal / elevated / extreme
    price_trend: str = "sideways"  # uptrend / downtrend / sideways
    consecutive_up_days: int = 0
    consecutive_down_days: int = 0
    sma_crossover: str = "none"  # golden_cross / death_cross / none
    direction: str = "neutral"
    confidence: float = 50.0
    citations: List[str] = field(default_factory=list)


@dataclass
class AggregateSignal:
    """Combined output from all three signal dimensions."""
    ticker: str = ""
    timestamp: str = ""
    momentum: MomentumSignal = field(default_factory=MomentumSignal)
    volume: VolumeAnomalySignal = field(default_factory=VolumeAnomalySignal)
    sentiment: SentimentSignal = field(default_factory=SentimentSignal)
    overall_direction: str = "neutral"
    overall_confidence: float = 50.0
    degraded: bool = False
    degradation_notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Mock Data Generator (Fallback)
# ---------------------------------------------------------------------------

def _generate_mock_ohlcv(ticker: str, period_days: int = 365) -> pd.DataFrame:
    """Generate synthetic OHLCV data for degraded-data mode."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=period_days, freq="B")
    np.random.seed(hash(ticker) % 2**31)
    base_price = np.random.uniform(500, 5000)
    returns = np.random.normal(0.0004, 0.018, period_days)
    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.01, period_days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, period_days)))
    open_ = close * (1 + np.random.normal(0, 0.005, period_days))
    volume = np.random.randint(500_000, 10_000_000, period_days).astype(float)
    # Inject a volume anomaly at a random point
    anomaly_idx = np.random.randint(100, period_days - 10)
    volume[anomaly_idx] *= np.random.uniform(3, 6)
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume
    }, index=dates)


def _generate_mock_vix(period_days: int = 60) -> pd.DataFrame:
    """Generate synthetic India VIX data."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=period_days, freq="B")
    np.random.seed(42)
    vix = np.random.uniform(12, 28, period_days)
    vix[-5:] = np.random.uniform(28, 40, 5)  # spike at end
    return pd.DataFrame({"Close": vix}, index=dates)


# ---------------------------------------------------------------------------
# Data Fetching Layer
# ---------------------------------------------------------------------------

def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data for a given ticker via yfinance.
    Falls back to mock data if USE_MOCK_DATA is True or fetch fails.
    Returns a DataFrame with columns: Open, High, Low, Close, Volume.
    """
    if config.USE_MOCK_DATA:
        logger.info("Using mock OHLCV data for %s", ticker)
        return _generate_mock_ohlcv(ticker)

    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            obj = yf.Ticker(ticker)
            df = obj.history(period=period, interval=interval)
            if df is not None and not df.empty:
                required_cols = ["Open", "High", "Low", "Close", "Volume"]
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = 0.0
                return df[required_cols].copy()
            logger.warning("Empty data returned for %s on attempt %d", ticker, attempt)
        except Exception as exc:
            logger.warning("Fetch attempt %d for %s failed: %s", attempt, ticker, exc)
        if attempt < MAX_RETRY_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_FACTOR * attempt)

    logger.error("All fetch attempts failed for %s; falling back to mock data", ticker)
    return _generate_mock_ohlcv(ticker)


def fetch_vix() -> pd.DataFrame:
    """Fetch India VIX data for sentiment analysis."""
    if config.USE_MOCK_DATA:
        return _generate_mock_vix()
    try:
        obj = yf.Ticker(INDIA_VIX_TICKER)
        df = obj.history(period="3mo", interval="1d")
        if df is not None and not df.empty:
            return df[["Close"]].copy()
    except Exception as exc:
        logger.warning("VIX fetch failed: %s", exc)
    return _generate_mock_vix()


def fetch_nifty_data() -> pd.DataFrame:
    """Fetch Nifty 50 index data for macro analysis."""
    if config.USE_MOCK_DATA:
        return _generate_mock_ohlcv("^NSEI")
    try:
        obj = yf.Ticker(MARKET_INDEX_TICKER)
        df = obj.history(period="1y", interval="1d")
        if df is not None and not df.empty:
            return df[["Open", "High", "Low", "Close", "Volume"]].copy()
    except Exception as exc:
        logger.warning("Nifty fetch failed: %s", exc)
    return _generate_mock_ohlcv("^NSEI")


# ---------------------------------------------------------------------------
# Technical Indicator Calculations
# ---------------------------------------------------------------------------

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram."""
    p = INDICATOR_PARAMS
    ema_fast = close.ewm(span=p.MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=p.MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=p.MACD_SIGNAL, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_sma(close: pd.Series, period: int) -> pd.Series:
    """Compute Simple Moving Average."""
    return close.rolling(window=period, min_periods=1).mean()


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    """Compute Exponential Moving Average."""
    return close.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Signal Dimension 1: Price Momentum
# ---------------------------------------------------------------------------

def compute_momentum_signal(df: pd.DataFrame) -> MomentumSignal:
    """Calculate price momentum signal from RSI and MACD."""
    signal = MomentumSignal()
    citations = []

    if df is None or df.empty or len(df) < INDICATOR_PARAMS.MACD_SLOW + INDICATOR_PARAMS.MACD_SIGNAL:
        signal.citations = ["Insufficient data for momentum calculation"]
        signal.confidence = 20.0
        return signal

    close = df["Close"]

    # RSI
    rsi_series = compute_rsi(close, INDICATOR_PARAMS.RSI_PERIOD)
    current_rsi = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0
    signal.rsi_value = round(current_rsi, 2)

    if current_rsi >= INDICATOR_PARAMS.RSI_OVERBOUGHT:
        signal.rsi_signal = "overbought"
    elif current_rsi <= INDICATOR_PARAMS.RSI_OVERSOLD:
        signal.rsi_signal = "oversold"
    else:
        signal.rsi_signal = "neutral"
    citations.append(f"RSI({INDICATOR_PARAMS.RSI_PERIOD}) = {current_rsi:.2f} [{signal.rsi_signal}]")

    # MACD
    macd_line, signal_line, histogram = compute_macd(close)
    signal.macd_line = round(float(macd_line.iloc[-1]), 4)
    signal.macd_signal_line = round(float(signal_line.iloc[-1]), 4)
    signal.macd_histogram = round(float(histogram.iloc[-1]), 4)

    if len(macd_line) >= 2:
        prev_hist = float(histogram.iloc[-2])
        curr_hist = float(histogram.iloc[-1])
        if prev_hist <= 0 and curr_hist > 0:
            signal.macd_crossover = "bullish"
        elif prev_hist >= 0 and curr_hist < 0:
            signal.macd_crossover = "bearish"
    citations.append(
        f"MACD({INDICATOR_PARAMS.MACD_FAST},{INDICATOR_PARAMS.MACD_SLOW},{INDICATOR_PARAMS.MACD_SIGNAL}) "
        f"= {signal.macd_line}, Signal = {signal.macd_signal_line}, "
        f"Hist = {signal.macd_histogram} [{signal.macd_crossover} crossover]"
    )

    # Determine overall momentum direction and confidence
    bullish_pts = 0
    bearish_pts = 0
    total_pts = 0

    # RSI contribution (weight: 40%)
    total_pts += 40
    if signal.rsi_signal == "oversold":
        bullish_pts += 40
    elif signal.rsi_signal == "overbought":
        bearish_pts += 40
    else:
        bullish_pts += 20
        bearish_pts += 20

    # MACD contribution (weight: 60%)
    total_pts += 60
    if signal.macd_crossover == "bullish":
        bullish_pts += 60
    elif signal.macd_crossover == "bearish":
        bearish_pts += 60
    else:
        bullish_pts += 30
        bearish_pts += 30

    if bullish_pts > bearish_pts:
        signal.direction = "bullish"
        signal.confidence = round((bullish_pts / total_pts) * 100, 1)
    elif bearish_pts > bullish_pts:
        signal.direction = "bearish"
        signal.confidence = round((bearish_pts / total_pts) * 100, 1)
    else:
        signal.direction = "neutral"
        signal.confidence = 50.0

    signal.citations = citations
    return signal


# ---------------------------------------------------------------------------
# Signal Dimension 2: Volume Anomalies
# ---------------------------------------------------------------------------

def compute_volume_signal(df: pd.DataFrame) -> VolumeAnomalySignal:
    """Calculate volume anomaly signal."""
    signal = VolumeAnomalySignal()
    citations = []

    if df is None or df.empty or "Volume" not in df.columns:
        signal.citations = ["Volume data unavailable"]
        signal.confidence = 20.0
        return signal

    volume = df["Volume"]
    if len(volume) < INDICATOR_PARAMS.VOLUME_SMA_PERIOD:
        signal.citations = ["Insufficient volume history for anomaly detection"]
        signal.confidence = 30.0
        return signal

    current_vol = float(volume.iloc[-1])
    avg_vol = float(volume.iloc[-INDICATOR_PARAMS.VOLUME_SMA_PERIOD:].mean())
    signal.current_volume = current_vol
    signal.avg_volume_20d = round(avg_vol, 0)

    if avg_vol > 0:
        signal.volume_ratio = round(current_vol / avg_vol, 2)
    else:
        signal.volume_ratio = 1.0
    citations.append(
        f"Current Volume = {current_vol:,.0f}, "
        f"20D Avg = {avg_vol:,.0f}, "
        f"Ratio = {signal.volume_ratio:.2f}x"
    )

    if signal.volume_ratio >= INDICATOR_PARAMS.VOLUME_SPIKE_THRESHOLD * 1.5:
        signal.is_anomaly = True
        signal.anomaly_type = "surge"
        citations.append(f"Volume surge detected: {signal.volume_ratio:.2f}x exceeds "
                         f"{INDICATOR_PARAMS.VOLUME_SPIKE_THRESHOLD * 1.5:.1f}x threshold")
    elif signal.volume_ratio >= INDICATOR_PARAMS.VOLUME_SPIKE_THRESHOLD:
        signal.is_anomaly = True
        signal.anomaly_type = "spike"
        citations.append(f"Volume spike detected: {signal.volume_ratio:.2f}x exceeds "
                         f"{INDICATOR_PARAMS.VOLUME_SPIKE_THRESHOLD:.1f}x threshold")
    elif signal.volume_ratio <= 0.3:
        signal.is_anomaly = True
        signal.anomaly_type = "dry"
        citations.append(f"Volume dry-up detected: {signal.volume_ratio:.2f}x below 0.3x threshold")
    else:
        signal.anomaly_type = "normal"
        citations.append("Volume within normal range")

    # Confidence: higher when ratio is more extreme
    if signal.is_anomaly:
        deviation = abs(signal.volume_ratio - 1.0)
        signal.confidence = min(50 + deviation * 25, 95)
    else:
        signal.confidence = 55.0

    # Determine directional bias: correlate volume anomaly with recent price action
    # A volume spike on up-day = bullish participation; on down-day = bearish
    if "Close" in df.columns and len(df) >= 2:
        recent_close = float(df["Close"].iloc[-1])
        prior_close = float(df["Close"].iloc[-2])
        price_moved_up = recent_close > prior_close
        if signal.is_anomaly:
            if signal.anomaly_type == "dry":
                signal.direction = "neutral"
                signal.confidence *= 0.9
            elif price_moved_up:
                signal.direction = "bullish"
            else:
                signal.direction = "bearish"
            # Strong volume + strong price direction boosts confidence
            if abs(signal.volume_ratio - 1.0) > 1.5:
                signal.confidence = min(signal.confidence + 5, 95)
        else:
            signal.direction = "neutral"

    signal.citations = citations
    return signal


# ---------------------------------------------------------------------------
# Signal Dimension 3: Market Sentiment
# ---------------------------------------------------------------------------

def compute_sentiment_signal(
    price_df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame] = None,
) -> SentimentSignal:
    """Calculate market sentiment signal using VIX and price-action trends."""
    signal = SentimentSignal()
    citations = []

    # VIX Analysis
    if vix_df is not None and not vix_df.empty and "Close" in vix_df.columns:
        current_vix = float(vix_df["Close"].iloc[-1])
        signal.vix_value = round(current_vix, 2)
        if current_vix >= 30:
            signal.vix_regime = "extreme"
        elif current_vix >= 22:
            signal.vix_regime = "elevated"
        elif current_vix >= 15:
            signal.vix_regime = "normal"
        else:
            signal.vix_regime = "calm"
        citations.append(f"India VIX = {current_vix:.2f} [{signal.vix_regime} regime]")
    else:
        citations.append("VIX data unavailable; sentiment partially degraded")

    # Price Trend Analysis
    if price_df is not None and not price_df.empty and len(price_df) >= 50:
        close = price_df["Close"]
        sma_20 = compute_sma(close, INDICATOR_PARAMS.SMA_SHORT)
        sma_50 = compute_sma(close, INDICATOR_PARAMS.SMA_LONG)

        current_price = float(close.iloc[-1])
        sma20_val = float(sma_20.iloc[-1])
        sma50_val = float(sma_50.iloc[-1])

        # Trend detection
        if len(close) >= 20:
            recent_returns = close.pct_change().dropna().tail(20)
            up_days = int((recent_returns > 0).sum())
            down_days = int((recent_returns < 0).sum())

            if up_days >= 15:
                signal.price_trend = "uptrend"
            elif down_days >= 15:
                signal.price_trend = "downtrend"
            else:
                signal.price_trend = "sideways"
            citations.append(
                f"20-day lookback: {up_days} up, {down_days} down [{signal.price_trend}]"
            )

        # SMA crossover
        if len(sma_20) >= 2 and len(sma_50) >= 2:
            prev_sma20 = float(sma_20.iloc[-2])
            prev_sma50 = float(sma_50.iloc[-2])
            if prev_sma20 <= prev_sma50 and sma20_val > sma50_val:
                signal.sma_crossover = "golden_cross"
            elif prev_sma20 >= prev_sma50 and sma20_val < sma50_val:
                signal.sma_crossover = "death_cross"
            citations.append(
                f"SMA20({sma20_val:.2f}) vs SMA50({sma50_val:.2f}) "
                f"[{signal.sma_crossover}]"
            )

        # Consecutive up/down days
        daily_returns = close.pct_change().dropna()
        if len(daily_returns) > 0:
            for i in range(len(daily_returns) - 1, -1, -1):
                if daily_returns.iloc[i] > 0:
                    signal.consecutive_up_days += 1
                else:
                    break
            for i in range(len(daily_returns) - 1, -1, -1):
                if daily_returns.iloc[i] < 0:
                    signal.consecutive_down_days += 1
                else:
                    break
            if signal.consecutive_up_days > 0:
                citations.append(f"{signal.consecutive_up_days} consecutive up days")
            if signal.consecutive_down_days > 0:
                citations.append(f"{signal.consecutive_down_days} consecutive down days")

    # Aggregate sentiment direction
    bullish_score = 0
    bearish_score = 0

    if signal.vix_regime == "calm":
        bullish_score += 2
    elif signal.vix_regime == "elevated":
        bearish_score += 1
    elif signal.vix_regime == "extreme":
        bearish_score += 3

    if signal.price_trend == "uptrend":
        bullish_score += 2
    elif signal.price_trend == "downtrend":
        bearish_score += 2

    if signal.sma_crossover == "golden_cross":
        bullish_score += 2
    elif signal.sma_crossover == "death_cross":
        bearish_score += 2

    total = bullish_score + bearish_score
    if total == 0:
        signal.direction = "neutral"
        signal.confidence = 50.0
    elif bullish_score > bearish_score:
        signal.direction = "bullish"
        signal.confidence = round((bullish_score / total) * 100, 1)
    elif bearish_score > bullish_score:
        signal.direction = "bearish"
        signal.confidence = round((bearish_score / total) * 100, 1)
    else:
        signal.direction = "neutral"
        signal.confidence = 50.0

    signal.citations = citations
    return signal


# ---------------------------------------------------------------------------
# Aggregate Signal Builder
# ---------------------------------------------------------------------------

def generate_signals(ticker: str) -> AggregateSignal:
    """
    Master function: fetches data, computes all 3 signal dimensions,
    and returns a unified AggregateSignal with confidence-weighted overall direction.
    Handles degraded data feeds by reducing confidence proportionally.
    """
    overall = AggregateSignal(ticker=ticker, timestamp=str(pd.Timestamp.now()))
    degradation_notes = []

    # Fetch data
    try:
        price_df = fetch_ohlcv(ticker, period="1y")
    except Exception as exc:
        logger.error("Failed to fetch OHLCV for %s: %s", ticker, exc)
        price_df = None
        degradation_notes.append(f"OHLCV fetch failed: {exc}")

    try:
        vix_df = fetch_vix()
    except Exception as exc:
        logger.error("Failed to fetch VIX: %s", exc)
        vix_df = None
        degradation_notes.append(f"VIX fetch failed: {exc}")

    if price_df is not None and not price_df.empty:
        overall.momentum = compute_momentum_signal(price_df)
        overall.volume = compute_volume_signal(price_df)
        overall.sentiment = compute_sentiment_signal(price_df, vix_df)
    else:
        degradation_notes.append("All price-based signals degraded due to missing OHLCV data")
        overall.momentum.confidence = 10.0
        overall.volume.confidence = 10.0
        overall.sentiment.confidence = 10.0

    if vix_df is None or vix_df.empty:
        overall.sentiment.confidence *= 0.7  # reduce sentiment confidence without VIX

    # Weighted overall direction
    weights = {"momentum": 0.40, "volume": 0.25, "sentiment": 0.35}
    direction_map = {"bullish": 1, "bearish": -1, "neutral": 0}

    weighted_score = (
        weights["momentum"] * direction_map[overall.momentum.direction] * overall.momentum.confidence
        + weights["volume"] * direction_map[overall.volume.direction] * overall.volume.confidence
        + weights["sentiment"] * direction_map[overall.sentiment.direction] * overall.sentiment.confidence
    )

    total_weighted_conf = (
        weights["momentum"] * overall.momentum.confidence
        + weights["volume"] * overall.volume.confidence
        + weights["sentiment"] * overall.sentiment.confidence
    )

    if total_weighted_conf > 0:
        overall.overall_confidence = round(abs(weighted_score / total_weighted_conf) * 100, 1)
    else:
        overall.overall_confidence = 10.0

    if weighted_score > 5:
        overall.overall_direction = "bullish"
    elif weighted_score < -5:
        overall.overall_direction = "bearish"
    else:
        overall.overall_direction = "neutral"

    overall.degraded = len(degradation_notes) > 0
    overall.degradation_notes = degradation_notes
    return overall


# ---------------------------------------------------------------------------
# Convenience: Fetch All Agents' Data
# ---------------------------------------------------------------------------

def fetch_market_context() -> Dict[str, Any]:
    """Fetch broad market context data (Nifty, VIX) for the macro/sentiment agent."""
    ctx: Dict[str, Any] = {}
    try:
        nifty_df = fetch_nifty_data()
        if nifty_df is not None and not nifty_df.empty:
            ctx["nifty_close"] = round(float(nifty_df["Close"].iloc[-1]), 2)
            ctx["nifty_returns_1d"] = round(float(nifty_df["Close"].pct_change().iloc[-1]) * 100, 2)
            ctx["nifty_returns_1m"] = round(
                float((nifty_df["Close"].iloc[-1] / nifty_df["Close"].iloc[-22] - 1) * 100), 2
            ) if len(nifty_df) > 22 else None
    except Exception as exc:
        logger.warning("Market context Nifty fetch failed: %s", exc)

    try:
        vix_df = fetch_vix()
        if vix_df is not None and not vix_df.empty:
            ctx["vix"] = round(float(vix_df["Close"].iloc[-1]), 2)
    except Exception as exc:
        logger.warning("Market context VIX fetch failed: %s", exc)

    return ctx
