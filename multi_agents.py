"""
multi_agents.py - Parallel Agent System, Master Synthesis & Multi-Agent Debate Engine
=======================================================================================
Implements parallel Technical, Fundamental, and Sentiment/Macro agents using
concurrent.futures.ThreadPoolExecutor. Includes a Master Synthesis Agent that
unifies outputs and a Bull vs. Bear Debate Engine for conflicting signals.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import MASTER_SYNTHESIS_SYSTEM_PROMPT, SEBI_DISCLAIMER
from data_engine import (
    AggregateSignal,
    fetch_market_context,
    fetch_ohlcv,
    generate_signals,
)
from isin_analyzer import StockRating, rate_stock, compute_volatility, compute_beta
from rag_engine import (
    CorporateBreakthroughExtractor,
    RetrievalResult,
    get_breakthrough_extractor,
    get_rag_engine,
)
from data_engine import fetch_nifty_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent Output Contracts
# ---------------------------------------------------------------------------

@dataclass
class AgentOutput:
    """Standard contract for all agent outputs."""
    agent_name: str
    ticker: str
    timestamp: str = ""
    direction: str = "neutral"  # bullish / bearish / neutral
    confidence: float = 50.0
    recommendation: str = "HOLD"
    key_signals: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class SynthesisOutput:
    """Master synthesis output combining all agent perspectives."""
    ticker: str
    final_recommendation: str = "HOLD"
    confidence: float = 50.0
    rationale: str = ""
    risks: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    agent_outputs: List[AgentOutput] = field(default_factory=list)
    has_conflict: bool = False
    debate_triggered: bool = False
    synthesis_latency_ms: float = 0.0


@dataclass
class DebateTurn:
    """Single turn in the Bull vs. Bear debate."""
    speaker: str  # "Bull" or "Bear"
    argument: str
    confidence: float
    supporting_evidence: List[str] = field(default_factory=list)


@dataclass
class DebateOutput:
    """Complete multi-agent debate result."""
    ticker: str
    bullArguments: List[DebateTurn] = field(default_factory=list)
    bearArguments: List[DebateTurn] = field(default_factory=list)
    verdict: str = "HOLD"
    verdict_confidence: float = 50.0
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Technical Agent
# ---------------------------------------------------------------------------

def run_technical_agent(ticker: str) -> AgentOutput:
    """
    Analyzes momentum and chart patterns using technical indicators.
    Sources: Price OHLCV, RSI, MACD, moving averages, volume.
    """
    start = time.time()
    output = AgentOutput(agent_name="Technical Agent", ticker=ticker)

    try:
        signals: AggregateSignal = generate_signals(ticker)
        output.raw_data = {
            "rsi": signals.momentum.rsi_value,
            "macd_histogram": signals.momentum.macd_histogram,
            "volume_ratio": signals.volume.volume_ratio,
            "sma_crossover": signals.sentiment.sma_crossover,
            "overall_direction": signals.overall_direction,
            "overall_confidence": signals.overall_confidence,
        }

        output.direction = signals.overall_direction
        output.confidence = signals.overall_confidence
        output.timestamp = signals.timestamp

        # Build key signals
        output.key_signals = [
            f"RSI({signals.momentum.rsi_period}) = {signals.momentum.rsi_value}"
            if hasattr(signals.momentum, 'rsi_period') else
            f"RSI = {signals.momentum.rsi_value} [{signals.momentum.rsi_signal}]",
            f"MACD Crossover: {signals.momentum.macd_crossover}",
            f"Volume Ratio: {signals.volume.volume_ratio}x [{signals.volume.anomaly_type}]",
            f"Price Trend: {signals.sentiment.price_trend}",
            f"VIX Regime: {signals.sentiment.vix_regime}",
            f"SMA Crossover: {signals.sentiment.sma_crossover}",
        ]

        if signals.volume.is_anomaly:
            output.key_signals.append(
                f"ALERT: Volume anomaly detected ({signals.volume.anomaly_type})"
            )

        output.risks = []
        if signals.momentum.rsi_signal == "overbought":
            output.risks.append("RSI overbought - potential pullback risk")
        if signals.sentiment.vix_regime in ("elevated", "extreme"):
            output.risks.append(f"VIX {signals.sentiment.vix_regime} - elevated market volatility")
        if signals.volume.is_anomaly and signals.volume.anomaly_type == "dry":
            output.risks.append("Low volume indicates weak conviction")
        if signals.degraded:
            output.risks.extend([f"Degraded: {n}" for n in signals.degradation_notes])

        output.sources = [
            "data_engine.py: RSI, MACD, SMA, Volume analysis",
            "data_engine.py: VIX sentiment regime",
        ]
        # Surface explicit indicator citations so the user sees cited reasoning
        output.sources.extend(signals.momentum.citations)
        output.sources.extend(signals.volume.citations)
        output.sources.extend(signals.sentiment.citations)

    except Exception as exc:
        logger.error("Technical Agent failed for %s: %s", ticker, exc)
        output.direction = "neutral"
        output.confidence = 10.0
        output.risks.append(f"Technical analysis failed: {exc}")

    output.latency_ms = round((time.time() - start) * 1000, 2)
    return output


# ---------------------------------------------------------------------------
# Fundamental Agent
# ---------------------------------------------------------------------------

def run_fundamental_agent(ticker: str, isin: Optional[str] = None) -> AgentOutput:
    """
    Analyzes fundamentals using RAG filings, balance sheets, and breakthrough events.
    Sources: SEBI filings, earnings transcripts, corporate disclosures.
    """
    start = time.time()
    output = AgentOutput(agent_name="Fundamental Agent", ticker=ticker)

    try:
        # Get RAG engine and breakthrough extractor
        rag = get_rag_engine()
        extractor = get_breakthrough_extractor()

        # Retrieve relevant filings
        filings = rag.retrieve(
            query="financial performance revenue profit growth",
            ticker_filter=ticker,
            n_results=5,
        )

        # Retrieve corporate breakthroughs
        breakthroughs = []
        if isin:
            breakthroughs = extractor.extract(ticker)

        # Score based on retrieved information
        fundamental_score = 50.0
        key_signals: List[str] = []

        if filings:
            output.sources.extend([f.source for f in filings[:3]])
            for f in filings:
                text = f.chunk_text.lower()
                if any(w in text for w in ["growth", "increase", "expanded", "surge"]):
                    fundamental_score += 8
                    key_signals.append(f"Positive filing: {f.source}")
                if any(w in text for w in ["decline", "loss", "npa", "deterioration"]):
                    fundamental_score -= 8
                    key_signals.append(f"Concern in filing: {f.source}")

        if breakthroughs:
            for bt in breakthroughs[:3]:
                fundamental_score += 5 * bt.relevance_score
                key_signals.append(
                    f"Breakthrough [{bt.event_type}]: {bt.description[:80]}..."
                )
                output.sources.append(bt.source)
        else:
            key_signals.append("No significant breakthrough events found in disclosures")

        fundamental_score = max(0, min(100, fundamental_score))

        # RAG chunk attribution
        rag_stats = rag.get_collection_stats()
        output.raw_data = {
            "rag_chunks_total": rag_stats["total_chunks"],
            "filings_found": len(filings),
            "breakthroughs_found": len(breakthroughs),
            "fundamental_score": fundamental_score,
        }

        output.confidence = fundamental_score
        if fundamental_score >= 65:
            output.direction = "bullish"
            output.recommendation = "BUY"
        elif fundamental_score <= 35:
            output.direction = "bearish"
            output.recommendation = "SELL"
        else:
            output.direction = "neutral"
            output.recommendation = "HOLD"

        output.key_signals = key_signals

        # Risks
        output.risks = []
        if len(filings) == 0:
            output.risks.append("No filings found for this ticker in knowledge base")
            output.confidence *= 0.7
        if len(breakthroughs) == 0:
            output.risks.append("No recent corporate breakthrough disclosures found")
        if rag_stats["total_chunks"] < 10:
            output.risks.append("Limited knowledge base coverage - analysis may be incomplete")

        output.sources.insert(0, "rag_engine.py: SEBI filings, earnings transcripts")

    except Exception as exc:
        logger.error("Fundamental Agent failed for %s: %s", ticker, exc)
        output.direction = "neutral"
        output.confidence = 10.0
        output.risks.append(f"Fundamental analysis failed: {exc}")

    output.latency_ms = round((time.time() - start) * 1000, 2)
    return output


# ---------------------------------------------------------------------------
# Sentiment & Macro Agent
# ---------------------------------------------------------------------------

def run_sentiment_macro_agent(ticker: str) -> AgentOutput:
    """
    Analyzes market sentiment, macro conditions, and news vibes.
    Sources: India VIX, Nifty 50 trends, global indicators.
    """
    start = time.time()
    output = AgentOutput(agent_name="Sentiment & Macro Agent", ticker=ticker)

    try:
        # Market context
        ctx = fetch_market_context()
        output.raw_data = ctx

        # Sentiment signals
        signals = generate_signals(ticker)
        sentiment = signals.sentiment

        sentiment_score = 50.0
        key_signals: List[str] = []

        # VIX analysis
        if ctx.get("vix"):
            vix = ctx["vix"]
            key_signals.append(f"India VIX: {vix:.2f} [{sentiment.vix_regime}]")
            if sentiment.vix_regime == "calm":
                sentiment_score += 10
            elif sentiment.vix_regime == "elevated":
                sentiment_score -= 10
                output.risks.append("VIX elevated - market uncertainty")
            elif sentiment.vix_regime == "extreme":
                sentiment_score -= 25
                output.risks.append("VIX extreme - high fear in market")

        # Nifty trend
        if ctx.get("nifty_close"):
            key_signals.append(f"Nifty 50: {ctx['nifty_close']:.2f}")
        if ctx.get("nifty_returns_1d") is not None:
            ret_1d = ctx["nifty_returns_1d"]
            key_signals.append(f"Nifty 1D Return: {ret_1d:.2f}%")
            if ret_1d < -1:
                sentiment_score -= 10
            elif ret_1d > 1:
                sentiment_score += 5
        if ctx.get("nifty_returns_1m") is not None:
            ret_1m = ctx["nifty_returns_1m"]
            key_signals.append(f"Nifty 1M Return: {ret_1m:.2f}%")
            if ret_1m < -5:
                sentiment_score -= 15
            elif ret_1m > 5:
                sentiment_score += 10

        # Price trend sentiment
        key_signals.append(f"Price Trend: {sentiment.price_trend}")
        if sentiment.price_trend == "uptrend":
            sentiment_score += 10
        elif sentiment.price_trend == "downtrend":
            sentiment_score -= 10

        if sentiment.sma_crossover == "golden_cross":
            sentiment_score += 8
            key_signals.append("Golden Cross detected - bullish signal")
        elif sentiment.sma_crossover == "death_cross":
            sentiment_score -= 8
            key_signals.append("Death Cross detected - bearish signal")
            output.risks.append("Death cross pattern - potential downtrend")

        sentiment_score = max(0, min(100, sentiment_score))
        output.confidence = sentiment_score

        if sentiment_score >= 60:
            output.direction = "bullish"
            output.recommendation = "BUY"
        elif sentiment_score <= 40:
            output.direction = "bearish"
            output.recommendation = "SELL"
        else:
            output.direction = "neutral"
            output.recommendation = "HOLD"

        output.key_signals = key_signals
        output.sources = [
            "data_engine.py: VIX, Nifty index analysis",
            "data_engine.py: Price trend & SMA crossover",
        ]

    except Exception as exc:
        logger.error("Sentiment & Macro Agent failed for %s: %s", ticker, exc)
        output.direction = "neutral"
        output.confidence = 10.0
        output.risks.append(f"Sentiment analysis failed: {exc}")

    output.latency_ms = round((time.time() - start) * 1000, 2)
    return output


# ---------------------------------------------------------------------------
# Parallel Execution Engine
# ---------------------------------------------------------------------------

def run_parallel_agents(
    ticker: str,
    isin: Optional[str] = None,
) -> List[AgentOutput]:
    """
    Execute all three agents in parallel using ThreadPoolExecutor.
    Returns list of AgentOutput from Technical, Fundamental, and Sentiment agents.
    """
    results: List[AgentOutput] = []

    agent_configs = [
        ("technical", lambda: run_technical_agent(ticker)),
        ("fundamental", lambda: run_fundamental_agent(ticker, isin)),
        ("sentiment", lambda: run_sentiment_macro_agent(ticker)),
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(fn): name
            for name, fn in agent_configs
        }

        for future in as_completed(futures):
            agent_name = futures[future]
            try:
                result = future.result(timeout=60)
                results.append(result)
                logger.info(
                    "Agent '%s' completed for %s in %.1fms (direction=%s, conf=%.1f%%)",
                    agent_name, ticker, result.latency_ms,
                    result.direction, result.confidence,
                )
            except Exception as exc:
                logger.error("Agent '%s' failed for %s: %s", agent_name, ticker, exc)
                results.append(AgentOutput(
                    agent_name=agent_name,
                    ticker=ticker,
                    direction="neutral",
                    confidence=0.0,
                    risks=[f"Agent execution failed: {exc}"],
                ))

    return results


# ---------------------------------------------------------------------------
# Master Synthesis Agent
# ---------------------------------------------------------------------------

def synthesize_agent_outputs(
    ticker: str,
    agent_outputs: List[AgentOutput],
) -> SynthesisOutput:
    """
    Unify parallel agent outputs into a single structured investment thesis.
    Weighs conflicting signals and produces a final recommendation.
    """
    start = time.time()
    synthesis = SynthesisOutput(ticker=ticker)
    synthesis.agent_outputs = agent_outputs

    if not agent_outputs:
        synthesis.final_recommendation = "HOLD"
        synthesis.confidence = 10.0
        synthesis.rationale = "No agent outputs available for synthesis."
        synthesis.risks.append("Complete agent failure - no analysis available")
        return synthesis

    # Compute weighted direction
    direction_scores = {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0}
    agent_weights = {
        "Technical Agent": 0.40,
        "Fundamental Agent": 0.35,
        "Sentiment & Macro Agent": 0.25,
    }

    all_sources: List[str] = []
    all_risks: List[str] = []

    for ao in agent_outputs:
        weight = agent_weights.get(ao.agent_name, 0.33)
        direction_scores[ao.direction] += weight * ao.confidence
        all_sources.extend(ao.sources)
        all_risks.extend(ao.risks)

    # Determine final direction
    best_dir = max(direction_scores, key=direction_scores.get)
    total_weight = sum(agent_weights.get(ao.agent_name, 0.33) for ao in agent_outputs)
    synthesis.confidence = round(
        direction_scores[best_dir] / total_weight if total_weight > 0 else 50.0, 1
    )

    # Check for conflicts between Technical and Fundamental
    tech_output = next((a for a in agent_outputs if a.agent_name == "Technical Agent"), None)
    fund_output = next((a for a in agent_outputs if a.agent_name == "Fundamental Agent"), None)

    if tech_output and fund_output:
        if tech_output.direction != fund_output.direction:
            if tech_output.direction != "neutral" and fund_output.direction != "neutral":
                synthesis.has_conflict = True
                synthesis.debate_triggered = True
                logger.info("Conflict detected between Technical and Fundamental agents for %s", ticker)

    # Final recommendation
    if synthesis.confidence >= 60 and best_dir == "bullish":
        synthesis.final_recommendation = "BUY"
    elif synthesis.confidence >= 60 and best_dir == "bearish":
        synthesis.final_recommendation = "SELL"
    else:
        synthesis.final_recommendation = "HOLD"

    # Build rationale
    rationale_parts = []
    for ao in agent_outputs:
        rationale_parts.append(
            f"[{ao.agent_name}] Direction={ao.direction}, "
            f"Confidence={ao.confidence:.1f}%, "
            f"Signals={len(ao.key_signals)}"
        )
    rationale_parts.append(f"Weighted Direction: {best_dir} ({synthesis.confidence:.1f}%)")
    rationale_parts.append(f"Final: {synthesis.final_recommendation}")
    synthesis.rationale = " | ".join(rationale_parts)

    synthesis.sources = list(set(all_sources))
    synthesis.risks = list(set(all_risks))
    synthesis.risks.insert(0, SEBI_DISCLAIMER)
    synthesis.synthesis_latency_ms = round((time.time() - start) * 1000, 2)

    return synthesis


# ---------------------------------------------------------------------------
# Multi-Agent Debate Engine
# ---------------------------------------------------------------------------

def run_debate(
    ticker: str,
    tech_output: AgentOutput,
    fund_output: AgentOutput,
) -> DebateOutput:
    """
    Trigger a dynamic Bull vs. Bear debate when Technical and Fundamental agents
    produce conflicting signals. Each side builds arguments from their evidence.
    """
    debate = DebateOutput(ticker=ticker)

    # Determine which agent is bullish and which is bearish
    if tech_output.direction == "bullish" and fund_output.direction == "bearish":
        bull_agent, bear_agent = tech_output, fund_output
    elif tech_output.direction == "bearish" and fund_output.direction == "bullish":
        bull_agent, bear_agent = fund_output, tech_output
    else:
        # No real conflict; use both to build balanced arguments
        bull_agent = tech_output if tech_output.confidence >= fund_output.confidence else fund_output
        bear_agent = fund_output if fund_output.confidence >= tech_output.confidence else tech_output

    # Bull arguments
    bull_evidence = bull_agent.key_signals + [
        s for s in bull_agent.sources[:2]
    ]
    debate.bullArguments.append(DebateTurn(
        speaker="Bull",
        argument=(
            f"The {bull_agent.agent_name} indicates a {bull_agent.direction} outlook "
            f"with {bull_agent.confidence:.1f}% confidence. "
            f"Key evidence: {'; '.join(bull_agent.key_signals[:3])}"
        ),
        confidence=bull_agent.confidence,
        supporting_evidence=bull_evidence,
    ))

    # Add a second bull argument if available
    if bull_agent.risks:
        debate.bullArguments.append(DebateTurn(
            speaker="Bull",
            argument=(
                f"While risks exist ({'; '.join(bull_agent.risks[:2])}), "
                f"the weight of evidence favors the upside."
            ),
            confidence=bull_agent.confidence * 0.8,
            supporting_evidence=[],
        ))

    # Bear arguments
    bear_evidence = bear_agent.key_signals + [
        s for s in bear_agent.sources[:2]
    ]
    debate.bearArguments.append(DebateTurn(
        speaker="Bear",
        argument=(
            f"The {bear_agent.agent_name} signals a {bear_agent.direction} outlook "
            f"with {bear_agent.confidence:.1f}% confidence. "
            f"Key concerns: {'; '.join(bear_agent.key_signals[:3])}"
        ),
        confidence=bear_agent.confidence,
        supporting_evidence=bear_evidence,
    ))

    if bear_agent.risks:
        debate.bearArguments.append(DebateTurn(
            speaker="Bear",
            argument=(
                f"Key risks: {'; '.join(bear_agent.risks[:3])}. "
                f"These cannot be ignored."
            ),
            confidence=bear_agent.confidence * 0.85,
            supporting_evidence=bear_agent.risks[:3],
        ))

    # Verdict based on confidence-weighted scoring
    bull_total_conf = sum(t.confidence for t in debate.bullArguments)
    bear_total_conf = sum(t.confidence for t in debate.bearArguments)

    if bull_total_conf > bear_total_conf * 1.1:
        debate.verdict = "BUY"
        debate.verdict_confidence = round(
            bull_total_conf / (bull_total_conf + bear_total_conf) * 100, 1
        )
        debate.reasoning = (
            "Bull case is stronger based on confidence-weighted evidence. "
            f"Overall BUY with {debate.verdict_confidence:.1f}% conviction."
        )
    elif bear_total_conf > bull_total_conf * 1.1:
        debate.verdict = "SELL"
        debate.verdict_confidence = round(
            bear_total_conf / (bull_total_conf + bear_total_conf) * 100, 1
        )
        debate.reasoning = (
            "Bear case is stronger based on confidence-weighted evidence. "
            f"Overall SELL with {debate.verdict_confidence:.1f}% conviction."
        )
    else:
        debate.verdict = "HOLD"
        debate.verdict_confidence = 50.0
        debate.reasoning = (
            "Bull and Bear cases are evenly matched. "
            "Recommendation: HOLD pending further information."
        )

    return debate


# ---------------------------------------------------------------------------
# Convenience: Full Analysis Pipeline
# ---------------------------------------------------------------------------

def run_full_analysis(ticker: str, isin: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute the complete multi-agent analysis pipeline:
    1. Run parallel agents
    2. Synthesize outputs
    3. If conflict detected, trigger debate
    Returns a dictionary with all results.
    """
    # Step 1: Parallel agents
    agent_outputs = run_parallel_agents(ticker, isin)

    # Step 2: Synthesis
    synthesis = synthesize_agent_outputs(ticker, agent_outputs)

    # Step 3: Debate (if conflict)
    debate_result = None
    if synthesis.debate_triggered:
        tech_out = next(
            (a for a in agent_outputs if a.agent_name == "Technical Agent"), None
        )
        fund_out = next(
            (a for a in agent_outputs if a.agent_name == "Fundamental Agent"), None
        )
        if tech_out and fund_out:
            debate_result = run_debate(ticker, tech_out, fund_out)
            # Override synthesis verdict with debate verdict
            synthesis.final_recommendation = debate_result.verdict
            synthesis.confidence = debate_result.verdict_confidence

    return {
        "ticker": ticker,
        "isin": isin,
        "agent_outputs": agent_outputs,
        "synthesis": synthesis,
        "debate": debate_result,
    }
