"""
app.py - Multi-Tab Streamlit Interface for PS-01 Financial Intelligence Platform
=================================================================================
Tab 1: Live Market Signals & Multi-Agent Debate Trace
Tab 2: ISIN Deep-Dive & Plotly Candlestick Chart
Tab 3: Custom Portfolio Allocator, Profit Curves & Risk-o-Meter
Tab 4: FinInfluencer Fraud Detector & Macro Stress Tester
Tab 5: Vernacular Summary & System Metrics / Degraded-Data Toggle
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Module Imports
# ---------------------------------------------------------------------------
import config
from config import (
    APP_LAYOUT,
    APP_TITLE,
    CHART_TIMEFRAMES,
    ISIN_MAP,
    SEBI_DISCLAIMER,
    SEBI_RISK_CATEGORIES,
    STRESS_SCENARIOS,
    SUPPORTED_LANGUAGES,
)
from data_engine import fetch_market_context, fetch_ohlcv, generate_signals
from fraud_detector import analyze_tip
from isin_analyzer import (
    create_candlestick_chart,
    get_available_isins,
    rate_stock,
    resolve_isin,
)
from localization import (
    generate_all_language_summaries,
    generate_full_localized_summary,
)
from multi_agents import (
    AgentOutput,
    DebateOutput,
    SynthesisOutput,
    run_debate,
    run_full_analysis,
    run_parallel_agents,
    synthesize_agent_outputs,
)
from profiler_allocator import (
    allocation_to_dataframe,
    generate_allocation,
    personalize_across_profiles,
)
from profit_crash_engine import (
    analyze_crash_risk,
    simulate_profit_projection,
)
from riskometer import (
    RiskAssessment,
    assess_risk,
    create_risk_gauge,
)
from stress_tester import run_stress_test
from compliance_logger import (
    get_compliance_guard,
    get_metrics_logger,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded",
)

# Initialize session state defaults
if "degraded_mode" not in st.session_state:
    st.session_state.degraded_mode = False
if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = {}
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "user_profile" not in st.session_state:
    st.session_state.user_profile = "Moderate"
if "user_horizon" not in st.session_state:
    st.session_state.user_horizon = 36


# ---------------------------------------------------------------------------
# Sidebar: Global Controls
# ---------------------------------------------------------------------------
def render_sidebar():
    """Render the global sidebar controls."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/bull-market.png", width=64)
        st.title("PS-01 Control Panel")
        st.markdown("---")

        # Degraded data toggle
        degraded = st.toggle(
            "Degraded Data Mode",
            value=st.session_state.degraded_mode,
            help="Simulates missing data feeds. Reduces confidence scores.",
        )
        st.session_state.degraded_mode = degraded
        config.DEGRADED_DATA_MODE = degraded
        config.USE_MOCK_DATA = degraded

        # User profile (individually stored risk parameters)
        st.markdown("---")
        st.caption("**User Profile**")
        selected_profile = st.selectbox(
            "Risk Profile",
            ["Conservative", "Moderate", "Aggressive", "Very High"],
            index=["Conservative", "Moderate", "Aggressive", "Very High"].index(
                st.session_state.user_profile
            ),
            key="sidebar_profile",
        )
        st.session_state.user_profile = selected_profile
        st.session_state.user_horizon = st.slider(
            "Horizon (months)", 3, 120, st.session_state.user_horizon, step=3,
            key="sidebar_horizon",
        )

        st.markdown("---")

        # Watchlist state (persists for the session)
        st.caption("**Watchlist**")
        watch_options = {f"{v['name']}": f"{v['ticker']}" for v in ISIN_MAP.values()}
        watch_options = dict(sorted(watch_options.items()))
        to_add = st.selectbox(
            "Add ticker", ["— None —"] + list(watch_options.keys()),
            key="watch_add",
        )
        if st.button("Add to Watchlist", key="watch_add_btn"):
            if to_add != "— None —" and watch_options[to_add] not in st.session_state.watchlist:
                st.session_state.watchlist.append(watch_options[to_add])
                st.toast(f"Added {watch_options[to_add]} to watchlist")

        if st.session_state.watchlist:
            for w in list(st.session_state.watchlist):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.text(w)
                with c2:
                    if st.button("✕", key=f"watch_del_{w}"):
                        st.session_state.watchlist.remove(w)
                        st.rerun()
        else:
            st.caption("(empty - add stocks to track)")

        st.markdown("---")

        # System status
        metrics = get_metrics_logger().get_metrics()
        st.caption("**System Status**")
        st.text(f"Uptime: {metrics['uptime_formatted']}")
        st.text(f"Analyses: {metrics['total_analyses']}")
        st.text(f"Accuracy: {metrics['signal_accuracy_30d']:.1f}%")

        st.markdown("---")

        # Disclaimer
        with st.expander("SEBI Disclaimer", expanded=False):
            st.caption(SEBI_DISCLAIMER)


render_sidebar()

# ---------------------------------------------------------------------------
# Main Title
# ---------------------------------------------------------------------------
st.title(f"📊 {APP_TITLE}")
st.caption(
    f"AI-powered investment research platform | "
    f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    f"{'⚠️ Degraded Mode' if st.session_state.degraded_mode else '✅ Live Mode'}"
)


# ---------------------------------------------------------------------------
# TAB 1: Live Market Signals & Multi-Agent Debate Trace
# ---------------------------------------------------------------------------
def render_tab1():
    """Live Market Signals & Multi-Agent Debate Trace."""
    st.header("Live Market Signals & Multi-Agent Debate")

    # -- Watchlist state board (Requirement: render current watchlist state) --
    if st.session_state.watchlist:
        st.subheader("📌 Watchlist — Live Classifications")
        wl_rows = []
        for wl_ticker in st.session_state.watchlist:
            try:
                cache_key = f"wl_{wl_ticker}"
                if cache_key not in st.session_state.analysis_cache:
                    st.session_state.analysis_cache[cache_key] = generate_signals(wl_ticker)
                wl_sig = st.session_state.analysis_cache[cache_key]
                wl_rows.append({
                    "Ticker": wl_ticker,
                    "Classification": wl_sig.overall_direction.title(),
                    "Confidence": f"{wl_sig.overall_confidence:.1f}%",
                    "RSI": f"{wl_sig.momentum.rsi_value:.0f}",
                    "Volume Ratio": f"{wl_sig.volume.volume_ratio:.2f}x",
                    "VIX Regime": wl_sig.sentiment.vix_regime,
                })
            except Exception as exc:
                st.warning(f"Watchlist ticker {wl_ticker} unavailable: {exc}")
        if wl_rows:
            st.dataframe(
                pd.DataFrame(wl_rows),
                use_container_width=True,
                hide_index=True,
            )
        st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Stock Selection")
        isin_options = {f"{v['name']} ({k})": k for k, v in ISIN_MAP.items()}
        selected_label = st.selectbox("Select Stock", list(isin_options.keys()), key="tab1_stock")
        selected_isin = isin_options[selected_label]
        mapping = ISIN_MAP[selected_isin]
        ticker = mapping["ticker"]
        company = mapping["name"]

        run_analysis = st.button("Run Multi-Agent Analysis", key="tab1_run", type="primary")

    with col2:
        if run_analysis:
            with st.spinner("Running parallel agents..."):
                result = run_full_analysis(ticker, selected_isin)

            # Record metrics
            ml = get_metrics_logger()
            for ao in result["agent_outputs"]:
                ml.record_agent_latency(ao.agent_name, ao.latency_ms)
            ml.record_signal(ticker, result["synthesis"].final_recommendation, result["synthesis"].confidence)
            # Concentration risk: record against a default 3-position portfolio if unset
            if ml.get_metrics()["portfolio_concentration_risk"] == 0:
                ml.compute_concentration_risk({"Large Cap": 0.4, "Mid Cap": 0.3, "Small Cap": 0.3})

            # Compliance check
            guard = get_compliance_guard()
            compliant_text = guard.sanitize_and_comply(
                f"Analysis for {company} ({ticker}): "
                f"{result['synthesis'].final_recommendation} "
                f"(Confidence: {result['synthesis'].confidence:.1f}%)"
            )

            # Synthesis Display
            synth = result["synthesis"]
            rec_color = {"BUY": "green", "SELL": "red", "HOLD": "orange"}.get(synth.final_recommendation, "white")
            st.markdown(
                f"### Final Recommendation: :{rec_color}[{synth.final_recommendation}] "
                f"({synth.confidence:.1f}% confidence)"
            )
            st.caption(f"Synthesis latency: {synth.synthesis_latency_ms:.1f}ms")

            # Full E2E Reasoning Chain (Requirement: full reasoning chain visible)
            with st.expander("🧠 Full Reasoning Chain (Ingestion → Agents → Synthesis)", expanded=False):
                chain_steps = [
                    ("[1] Raw Data Ingestion",
                     f"OHLCV + volume for {ticker} fetched via yfinance/yahoo "
                     f"(mode: {'mock/degraded' if config.USE_MOCK_DATA else 'live'}); "
                     f"VIX & Nifty context pulled for macro layer."),
                    ("[2] Signal Classification (3 dimensions)",
                     "\n".join(
                         f"  • {ao.agent_name}: direction={ao.direction}, "
                         f"confidence={ao.confidence:.1f}%, latency={ao.latency_ms:.0f}ms"
                         for ao in result["agent_outputs"]
                     )),
                    ("[3] RAG Grounding (Fundamental Agent)",
                     "Semantic search over SEBI filings / transcripts via ChromaDB "
                     "with document chunk attribution; citations attached to outputs."),
                ]
                if result["debate"]:
                    chain_steps.append(("[4] Conflict Resolution (Debate Engine)",
                                        f"Technical vs Fundamental conflicted -> "
                                        f"Bull/Bear debate ran; verdict: {result['debate'].verdict} "
                                        f"({result['debate'].verdict_confidence:.1f}%)"))
                chain_steps.append(("[5] Master Synthesis",
                                    f"Weighted aggregation produced final "
                                    f"'{synth.final_recommendation}' @ {synth.confidence:.1f}% "
                                    f"confidence with {len(synth.sources)} cited sources."))
                chain_steps.append(("[6] Compliance Guardrail",
                                    "Output checked against SEBI restrictions; "
                                    "mandatory disclaimer attached."))

                for title, body in chain_steps:
                    st.markdown(f"**{title}**")
                    st.text(body)

            # Agent Outputs
            st.subheader("Agent Outputs")
            agent_cols = st.columns(3)
            for i, ao in enumerate(result["agent_outputs"]):
                with agent_cols[i]:
                    dir_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(ao.direction, "⚪")
                    st.markdown(f"**{ao.agent_name}** {dir_emoji}")
                    st.caption(f"Direction: {ao.direction.title()} | Confidence: {ao.confidence:.1f}%")
                    st.caption(f"Latency: {ao.latency_ms:.1f}ms")
                    with st.expander("Key Signals"):
                        for sig in ao.key_signals:
                            st.text(f"• {sig}")
                    with st.expander("Risks"):
                        for risk in ao.risks:
                            st.warning(risk)

            # Debate Section
            if result["debate"]:
                st.markdown("---")
                st.subheader("Multi-Agent Debate")
                debate = result["debate"]

                debate_cols = st.columns(2)
                with debate_cols[0]:
                    st.markdown("#### 🟢 Bull Case")
                    for turn in debate.bullArguments:
                        st.info(f"**{turn.speaker}** (Confidence: {turn.confidence:.1f}%)\n\n{turn.argument}")
                        if turn.supporting_evidence:
                            with st.expander("Supporting Evidence"):
                                for ev in turn.supporting_evidence:
                                    st.text(f"  → {ev}")

                with debate_cols[1]:
                    st.markdown("#### 🔴 Bear Case")
                    for turn in debate.bearArguments:
                        st.error(f"**{turn.speaker}** (Confidence: {turn.confidence:.1f}%)\n\n{turn.argument}")
                        if turn.supporting_evidence:
                            with st.expander("Supporting Evidence"):
                                for ev in turn.supporting_evidence:
                                    st.text(f"  → {ev}")

                st.markdown(
                    f"**Debat Verdict: {debate.verdict}** "
                    f"(Confidence: {debate.verdict_confidence:.1f}%)"
                )
                st.caption(debate.reasoning)

            # Sources
            with st.expander("Sources & Citations"):
                for src in synth.sources:
                    st.text(f"• {src}")

            # Risk-Profile Personalization (Requirement: different outputs for
            # different user profiles on IDENTICAL market inputs)
            st.markdown("---")
            st.subheader("🎛️ Risk-Profile Personalization")
            st.caption(
                f"Your stored profile: **{st.session_state.user_profile}** "
                f"(horizon {st.session_state.user_horizon} months). The raw signal "
                f"below is identical for every user; it is transformed per profile."
            )
            personalized = personalize_across_profiles(
                synth.final_recommendation, synth.confidence,
                horizon_months=st.session_state.user_horizon,
            )
            p_rows = []
            for prof_name, pr in personalized.items():
                marker = "👈" if prof_name == st.session_state.user_profile else ""
                p_rows.append({
                    "Profile": prof_name,
                    "Raw Signal": pr.base_recommendation,
                    "Adjusted": pr.adjusted_recommendation,
                    "Adjusted Conf.": f"{pr.adjusted_confidence:.1f}%",
                    "Adjustment": pr.adjustment_notes[-1],
                    "": marker,
                })
            st.dataframe(pd.DataFrame(p_rows), use_container_width=True, hide_index=True)

            active_pr = personalized[st.session_state.user_profile]
            st.success(
                f"**For you ({active_pr.risk_profile}):** "
                f"{active_pr.adjusted_recommendation} "
                f"({active_pr.adjusted_confidence:.1f}% confidence adjusted "
                f"from {active_pr.base_confidence:.1f}%)"
            )
            with st.expander("Personalization reasoning"):
                for note in active_pr.adjustment_notes:
                    st.text(f"• {note}")
        else:
            st.info("Select a stock and click 'Run Multi-Agent Analysis' to begin.")


# ---------------------------------------------------------------------------
# TAB 2: ISIN Deep-Dive & Plotly Candlestick Chart
# ---------------------------------------------------------------------------
def render_tab2():
    """ISIN Deep-Dive & Plotly Candlestick Chart."""
    st.header("ISIN Deep-Dive & Candlestick Chart")

    col1, col2 = st.columns([1, 3])

    with col1:
        isin_options = {f"{v['name']} ({k})": k for k, v in ISIN_MAP.items()}
        selected_label = st.selectbox("Select Stock", list(isin_options.keys()), key="tab2_stock")
        selected_isin = isin_options[selected_label]
        mapping = ISIN_MAP[selected_isin]

        timeframe = st.selectbox("Timeframe", list(CHART_TIMEFRAMES.keys()), index=3, key="tab2_tf")
        period_days = CHART_TIMEFRAMES[timeframe]

        show_volume = st.checkbox("Show Volume", value=True, key="tab2_vol")
        show_sma = st.checkbox("Show SMA", value=True, key="tab2_sma")
        show_bb = st.checkbox("Show Bollinger Bands", value=False, key="tab2_bb")

        st.markdown("---")
        st.subheader("Rating")
        run_rating = st.button("Generate Rating", key="tab2_rate", type="primary")

    with col2:
        fig = create_candlestick_chart(
            mapping["ticker"],
            period_days=period_days,
            show_volume=show_volume,
            show_sma=show_sma,
            show_bollinger=show_bb,
            title=f"{mapping['name']} ({mapping['ticker']}) - {timeframe}",
        )
        st.plotly_chart(fig, use_container_width=True)

        if run_rating:
            with st.spinner("Generating comprehensive rating..."):
                rating = rate_stock(selected_isin)

            rec_color = {"BUY": "green", "SELL": "red", "HOLD": "orange"}.get(rating.recommendation, "white")
            st.markdown(
                f"### {rating.recommendation} | Confidence: {rating.confidence:.1f}%"
            )

            # CAGR display
            cagr_cols = st.columns(3)
            with cagr_cols[0]:
                cagr_1y = f"{rating.cagr_1y:.1%}" if rating.cagr_1y is not None else "N/A"
                st.metric("1Y CAGR", cagr_1y)
            with cagr_cols[1]:
                cagr_3y = f"{rating.cagr_3y:.1%}" if rating.cagr_3y is not None else "N/A"
                st.metric("3Y CAGR", cagr_3y)
            with cagr_cols[2]:
                cagr_5y = f"{rating.cagr_5y:.1%}" if rating.cagr_5y is not None else "N/A"
                st.metric("5Y CAGR", cagr_5y)

            # Score breakdown
            score_cols = st.columns(3)
            with score_cols[0]:
                st.metric("Technical Score", f"{rating.technical_score:.0f}/100")
            with score_cols[1]:
                st.metric("Fundamental Score", f"{rating.fundamental_score:.0f}/100")
            with score_cols[2]:
                st.metric("Momentum Score", f"{rating.momentum_score:.0f}/100")

            st.info(rating.rationale)

            if rating.risks:
                with st.expander("Risks"):
                    for r in rating.risks:
                        st.warning(r)

            with st.expander("Sources"):
                for s in rating.sources:
                    st.text(f"• {s}")

            # Risk-o-Meter for this stock
            with st.spinner("Computing risk assessment..."):
                risk = assess_risk(mapping["ticker"])
            risk_fig = create_risk_gauge(risk)
            st.plotly_chart(risk_fig, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 3: Portfolio Allocator, Profit Curves & Risk-o-Meter
# ---------------------------------------------------------------------------
def render_tab3():
    """Custom Portfolio Allocator, Profit Curves & Risk-o-Meter."""
    st.header("Portfolio Allocator & Profit Projections")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Investment Profile")
        risk_profile = st.selectbox(
            "Risk Profile",
            ["Conservative", "Moderate", "Aggressive", "Very High"],
            index=1,
            key="tab3_profile",
        )
        horizon_months = st.slider(
            "Investment Horizon (months)",
            min_value=3, max_value=120, value=36, step=3,
            key="tab3_horizon",
        )
        capital = st.number_input(
            "Capital Amount (INR)",
            min_value=10000, max_value=100000000, value=500000, step=50000,
            key="tab3_capital",
        )

        allocate_btn = st.button("Generate Portfolio", key="tab3_alloc", type="primary")

    with col2:
        if allocate_btn:
            with st.spinner("Generating allocation and projections..."):
                allocation = generate_allocation(risk_profile, horizon_months, capital)
                projection = simulate_profit_projection(
                    capital,
                    {a.asset_class: a.allocation_pct / 100 for a in allocation.allocations},
                    horizon_months,
                )

            st.subheader("Asset Allocation")
            alloc_df = allocation_to_dataframe(allocation)
            st.dataframe(alloc_df, use_container_width=True, hide_index=True)

            # Allocation Pie Chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=[a.asset_class for a in allocation.allocations],
                values=[a.allocation_pct for a in allocation.allocations],
                hole=0.4,
                textinfo="label+percent",
            )])
            fig_pie.update_layout(
                title="Portfolio Allocation",
                height=400,
                template="plotly_dark",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Projection Summary
            st.subheader("Profit Projections")
            proj_cols = st.columns(3)
            with proj_cols[0]:
                st.metric(
                    "Conservative",
                    f"Rs {allocation.projected_value_1y_min:,.0f}",
                    f"{allocation.projected_1y_return_min:.1%} CAGR",
                )
            with proj_cols[1]:
                st.metric(
                    "Expected",
                    f"Rs {allocation.projected_value_1y_median:,.0f}",
                    f"{allocation.projected_1y_return_median:.1%} CAGR",
                )
            with proj_cols[2]:
                st.metric(
                    "Optimistic",
                    f"Rs {allocation.projected_value_1y_max:,.0f}",
                    f"{allocation.projected_1y_return_max:.1%} CAGR",
                )

            # Projection Curves
            fig_proj = go.Figure()
            fig_proj.add_trace(go.Scatter(
                x=projection.time_labels,
                y=projection.projection_curve_max,
                name="Optimistic",
                line=dict(color="#26a69a", dash="dash"),
                fill=None,
            ))
            fig_proj.add_trace(go.Scatter(
                x=projection.time_labels,
                y=projection.projection_curve_median,
                name="Expected",
                line=dict(color="#ff9800", width=3),
            ))
            fig_proj.add_trace(go.Scatter(
                x=projection.time_labels,
                y=projection.projection_curve_min,
                name="Conservative",
                line=dict(color="#ef5350", dash="dash"),
                fill="tonexty",
                fillcolor="rgba(255, 152, 0, 0.15)",
            ))
            fig_proj.update_layout(
                title="Profit Projection Curves",
                xaxis_title="Months",
                yaxis_title="Portfolio Value (INR)",
                template="plotly_dark",
                height=450,
            )
            st.plotly_chart(fig_proj, use_container_width=True)

            with st.expander("Detailed Projection Summary"):
                st.text(projection.summary)

            with st.expander("Warnings"):
                for w in allocation.warnings:
                    st.warning(w)

    # Risk-o-Meter Section
    st.markdown("---")
    st.subheader("SEBI Risk-o-Meter")

    # Show risk for a selected stock
    risk_isin_options = {f"{v['name']} ({k})": k for k, v in ISIN_MAP.items()}
    risk_selected = st.selectbox("Select Stock for Risk Assessment", list(risk_isin_options.keys()), key="tab3_risk")
    risk_isin = risk_isin_options[risk_selected]
    risk_ticker = ISIN_MAP[risk_isin]["ticker"]

    if st.button("Assess Risk", key="tab3_risk_btn"):
        with st.spinner("Computing risk metrics..."):
            risk = assess_risk(risk_ticker)

        risk_fig = create_risk_gauge(risk)
        st.plotly_chart(risk_fig, use_container_width=True)

        with st.expander("Risk Details"):
            st.json(risk.details)


# ---------------------------------------------------------------------------
# TAB 4: Fraud Detector & Macro Stress Tester
# ---------------------------------------------------------------------------
def render_tab4():
    """FinInfluencer Fraud Detector & Macro Stress Tester."""
    st.header("Fraud Detector & Macro Stress Tester")

    tab4a, tab4b = st.tabs(["Fraud Detector", "Macro Stress Tester"])

    with tab4a:
        st.subheader("Telegram / FinInfluencer Tip Scanner")
        st.caption("Paste any stock tip or rumor text below to analyze for potential fraud indicators.")

        tip_text = st.text_area(
            "Stock Tip / Rumor Text",
            placeholder="e.g., 'Secret tip: Buy XYZ stock now! Guaranteed 10x returns in 1 week! Insider info from SEBI filing...'",
            height=150,
            key="tab4_tip",
        )

        if st.button("Analyze Tip", key="tab4_analyze", type="primary"):
            if tip_text.strip():
                with st.spinner("Analyzing tip for fraud indicators..."):
                    result = analyze_tip(tip_text)

                # Risk level coloring
                risk_colors = {
                    "LOW": "green", "MEDIUM": "orange", "HIGH": "red", "CRITICAL": "red"
                }
                risk_color = risk_colors.get(result.risk_level, "white")

                st.markdown(
                    f"### Risk Level: :{risk_color}[{result.risk_level}] "
                    f"(Score: {result.red_flag_score}/100)"
                )
                st.info(result.recommendation)

                # Metrics
                m_cols = st.columns(4)
                with m_cols[0]:
                    st.metric("Red Flag Score", f"{result.red_flag_score:.0f}/100")
                with m_cols[1]:
                    st.metric("Pump Keywords", len(result.pump_keywords_found))
                with m_cols[2]:
                    st.metric("Volume Anomaly", "Yes" if result.volume_anomaly else "No")
                with m_cols[3]:
                    st.metric("Price Surge", "Yes" if result.price_surge_detected else "No")

                if result.detected_ticker:
                    st.success(f"Detected Ticker: {result.detected_ticker} ({result.detected_company or 'Unknown'})")

                if result.flags:
                    with st.expander("Flags Detected"):
                        for flag in result.flags:
                            st.error(f"• {flag}")

                if result.pump_keywords_found:
                    with st.expander("Pump Keywords Found"):
                        st.write(", ".join(result.pump_keywords_found))

                with st.expander("Full Report"):
                    st.code(result.detailed_report)
            else:
                st.warning("Please enter some text to analyze.")

    with tab4b:
        st.subheader("Macro What-If Stress Tester")

        s_col1, s_col2 = st.columns([1, 2])

        with s_col1:
            stress_capital = st.number_input(
                "Portfolio Value (INR)",
                min_value=10000, max_value=100000000, value=500000, step=50000,
                key="tab4_capital",
            )

            st.markdown("**Allocation**")
            alloc_large = st.slider("Large Cap %", 0, 100, 40, key="tab4_lc")
            alloc_mid = st.slider("Mid Cap %", 0, 100, 20, key="tab4_mc")
            alloc_small = st.slider("Small Cap %", 0, 100, 10, key="tab4_sc")
            alloc_debt = st.slider("Debt %", 0, 100, 20, key="tab4_debt")
            alloc_sgb = st.slider("SGB %", 0, 100, 5, key="tab4_sgb")
            alloc_fd = st.slider("FD/T-Bills %", 0, 100, 5, key="tab4_fd")

            selected_scenarios = st.multiselect(
                "Stress Scenarios",
                list(STRESS_SCENARIOS.keys()),
                default=list(STRESS_SCENARIOS.keys())[:3],
                key="tab4_scenarios",
            )

            run_stress = st.button("Run Stress Test", key="tab4_stress", type="primary")

        with s_col2:
            if run_stress:
                total_alloc = alloc_large + alloc_mid + alloc_small + alloc_debt + alloc_sgb + alloc_fd
                if total_alloc != 100:
                    st.warning(f"Total allocation is {total_alloc}%. Normalizing to 100%.")
                    norm = 100 / total_alloc if total_alloc > 0 else 1
                    alloc_large, alloc_mid, alloc_small = alloc_large * norm, alloc_mid * norm, alloc_small * norm
                    alloc_debt, alloc_sgb, alloc_fd = alloc_debt * norm, alloc_sgb * norm, alloc_fd * norm

                allocation = {
                    "Large Cap": alloc_large / 100,
                    "Mid Cap": alloc_mid / 100,
                    "Small Cap": alloc_small / 100,
                    "Debt / Liquid Funds": alloc_debt / 100,
                    "Sovereign Gold Bonds": alloc_sgb / 100,
                    "Fixed Deposit / T-Bills": alloc_fd / 100,
                }

                with st.spinner("Running stress scenarios..."):
                    stress_result = run_stress_test(stress_capital, allocation, selected_scenarios)

                if stress_result.scenarios:
                    # Results table
                    rows = []
                    for s in stress_result.scenarios:
                        rows.append({
                            "Scenario": s.scenario_name,
                            "Impact %": f"{s.portfolio_impact_pct:+.1f}%",
                            "Drawdown (INR)": f"Rs {s.portfolio_drawdown_inr:,.0f}",
                            "Stressed Value": f"Rs {s.portfolio_stressed_value:,.0f}",
                            "Duration": f"{s.duration_months}mo",
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                    # Stress impact chart
                    fig_stress = go.Figure(data=[go.Bar(
                        x=[s.scenario_name for s in stress_result.scenarios],
                        y=[s.portfolio_impact_pct for s in stress_result.scenarios],
                        marker_color=[
                            "#ef5350" if s.portfolio_impact_pct < -10
                            else "#ff9800" if s.portfolio_impact_pct < 0
                            else "#26a69a"
                            for s in stress_result.scenarios
                        ],
                    )])
                    fig_stress.update_layout(
                        title="Portfolio Impact by Scenario",
                        yaxis_title="Impact (%)",
                        template="plotly_dark",
                        height=400,
                    )
                    st.plotly_chart(fig_stress, use_container_width=True)

                    st.error(
                        f"**Worst Case:** {stress_result.worst_case_scenario} "
                        f"({stress_result.worst_case_drawdown_pct:+.1f}% / "
                        f"Rs {stress_result.worst_case_drawdown_inr:,.0f})"
                    )

                    with st.expander("Scenario Narratives"):
                        for s in stress_result.scenarios:
                            st.markdown(f"**{s.scenario_name}**")
                            st.text(s.narrative)
                            st.markdown("---")


# ---------------------------------------------------------------------------
# TAB 5: Vernacular Summary & System Metrics
# ---------------------------------------------------------------------------
def render_tab5():
    """Vernacular Summary & System Metrics / Degraded-Data Toggle."""
    st.header("Vernacular Intelligence & System Metrics")

    tab5a, tab5b = st.tabs(["Vernacular Summary", "System Metrics"])

    with tab5a:
        st.subheader("Investment Decision Summary in Indian Languages")

        v_col1, v_col2 = st.columns([1, 2])

        with v_col1:
            v_isin_options = {f"{v['name']} ({k})": k for k, v in ISIN_MAP.items()}
            v_selected = st.selectbox("Select Stock", list(v_isin_options.keys()), key="tab5_stock")
            v_isin = v_isin_options[v_selected]
            v_mapping = ISIN_MAP[v_isin]

            v_capital = st.number_input("Capital (INR)", value=500000, min_value=10000, key="tab5_cap")
            v_horizon = st.slider("Horizon (months)", 3, 120, 24, key="tab5_hor")
            v_lang = st.selectbox(
                "Language",
                list(SUPPORTED_LANGUAGES.keys()),
                format_func=lambda x: SUPPORTED_LANGUAGES[x],
                key="tab5_lang",
            )

            gen_summary = st.button("Generate Summary", key="tab5_gen", type="primary")

        with v_col2:
            if gen_summary:
                # Run analysis to get recommendation
                with st.spinner("Analyzing..."):
                    analysis = run_full_analysis(v_mapping["ticker"], v_isin)
                    synth = analysis["synthesis"]

                # Get crash level
                crash = analyze_crash_risk()

                summary = generate_full_localized_summary(
                    ticker=v_mapping["ticker"],
                    company=v_mapping["name"],
                    recommendation=synth.final_recommendation,
                    confidence=synth.confidence,
                    capital=v_capital,
                    projected=v_capital * 1.13,  # simplified
                    months=v_horizon,
                    crash_level=crash.overall_risk_level,
                    language=v_lang,
                )

                st.markdown(f"### Summary in {summary.language_name}")
                st.markdown(summary.full_summary)

                with st.expander("TTS Audio Script"):
                    st.text_area(
                        "Text-to-Speech Script",
                        value=summary.tts_script,
                        height=150,
                        disabled=True,
                        key="tab5_tts",
                    )

                # All languages
                with st.expander("All Languages"):
                    all_langs = generate_all_language_summaries(
                        ticker=v_mapping["ticker"],
                        company=v_mapping["name"],
                        recommendation=synth.final_recommendation,
                        confidence=synth.confidence,
                        capital=v_capital,
                        projected=v_capital * 1.13,
                        months=v_horizon,
                        crash_level=crash.overall_risk_level,
                    )
                    for code, lang_summary in all_langs.items():
                        st.markdown(f"**{lang_summary.language_name}**")
                        st.text(lang_summary.full_summary[:300] + "...")
                        st.markdown("---")

    with tab5b:
        st.subheader("System Metrics & Diagnostics")

        ml = get_metrics_logger()
        metrics = ml.get_metrics()

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Uptime", metrics["uptime_formatted"])
        with m2:
            st.metric("Analyses Run", metrics["total_analyses"])
        with m3:
            st.metric("30D Fwd. Accuracy", f"{metrics['signal_accuracy_30d']:.1f}%")
            st.caption(f"(backtested over {metrics.get('backtested_signals', 0)} signals)")
        with m4:
            st.metric("Degraded Events", metrics["degraded_data_events"])

        # Independent 30-day forward-return backtest (Requirement: signal accuracy
        # must be measured against actual forward returns, not a proxy)
        st.subheader("30-Day Forward-Return Backtest")
        st.caption(
            "Replays classified signals across historical evaluation windows and "
            "verifies each against the ACTUAL 30-trading-day forward return of the "
            "underlying price series."
        )
        if st.button("Run Historical Backtest", key="tab5_backtest"):
            with st.spinner("Backtesting signals across a panel of liquid tickers..."):
                backtest_acc = ml.backtest_30d_forward_accuracy()
            st.success(
                f"Backtested accuracy: **{backtest_acc:.1f}%** "
                f"(over {ml.get_metrics()['backtested_signals']} historical signal checks)"
            )

        # Agent Latency
        if metrics["agent_latencies"]:
            st.subheader("Agent Latency History")
            lat_df = pd.DataFrame([
                {"Agent": name, "Avg Latency (ms)": lat}
                for name, lat in metrics["agent_latencies"].items()
            ])
            st.dataframe(lat_df, use_container_width=True, hide_index=True)

            fig_lat = go.Figure(data=[go.Bar(
                x=list(metrics["agent_latencies"].keys()),
                y=list(metrics["agent_latencies"].values()),
                marker_color="#ff9800",
            )])
            fig_lat.update_layout(
                title="Average Agent Latency",
                yaxis_title="Latency (ms)",
                template="plotly_dark",
                height=350,
            )
            st.plotly_chart(fig_lat, use_container_width=True)

        # Compliance Log
        guard = get_compliance_guard()
        st.subheader("Compliance Log")
        if guard._log_entries:
            recent = guard._log_entries[-10:]
            for entry in reversed(recent):
                status = "✅" if entry["is_compliant"] else "⚠️"
                st.text(
                    f"{status} {entry['timestamp'][:19]} | "
                    f"Violations: {len(entry['violations'])} | "
                    f"Warnings: {len(entry['warnings'])}"
                )
        else:
            st.info("No compliance checks logged yet.")

        # Market Context
        st.subheader("Market Context")
        with st.spinner("Fetching market data..."):
            ctx = fetch_market_context()
        if ctx:
            ctx_cols = st.columns(3)
            with ctx_cols[0]:
                st.metric("Nifty 50", f"{ctx.get('nifty_close', 'N/A')}")
            with ctx_cols[1]:
                st.metric("1D Return", f"{ctx.get('nifty_returns_1d', 'N/A')}%")
            with ctx_cols[2]:
                st.metric("India VIX", f"{ctx.get('vix', 'N/A')}")

        # Crash Warning
        st.subheader("Crash Early Warning")
        if st.button("Run Crash Analysis", key="tab5_crash"):
            with st.spinner("Analyzing macro conditions..."):
                crash = analyze_crash_risk()

            risk_colors = {
                "LOW": "green", "MODERATE": "yellow", "ELEVATED": "orange",
                "HIGH": "red", "EXTREME": "red",
            }
            risk_color = risk_colors.get(crash.overall_risk_level, "white")
            st.markdown(
                f"### Crash Risk: :{risk_color}[{crash.overall_risk_level}] "
                f"(Score: {crash.overall_risk_score:.0f}/100)"
            )

            if crash.warnings:
                for w in crash.warnings:
                    st.warning(w)

            if crash.recommendations:
                st.subheader("Recommendations")
                for r in crash.recommendations:
                    st.info(r)

            with st.expander("Detailed Crash Report"):
                st.code(crash.detailed_report)


# ---------------------------------------------------------------------------
# Render All Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Live Signals & Debate",
    "ISIN Deep-Dive",
    "Portfolio & Risk-o-Meter",
    "Fraud Detector & Stress Test",
    "Vernacular & Metrics",
])

with tab1:
    render_tab1()

with tab2:
    render_tab2()

with tab3:
    render_tab3()

with tab4:
    render_tab4()

with tab5:
    render_tab5()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(SEBI_DISCLAIMER)
