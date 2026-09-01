# AI-Financial-Analysist
AI based Financial analysist

INSTRUCTION TO RUN THE PROGRAM:

1. Download the all the provided files on to your computer.
2. Install all the dependencies and libraires in the requirement file.
3. Open terminal in the folder where you have the files and run the below command:
   streamlit run app.py
4.Then open the local URL printed in the terminal (default http://localhost:8501).


Explaination of the code:


AI Financial Analyst

An advanced, multi-agent AI framework engineered to automate institutional-grade financial analysis, assess market risks, evaluate portfolio stress, and ensure strict regulatory compliance. This solution leverages modern LLM orchestration, structured data ingestion, and specialized micro-services to provide real-time, audit-ready financial insights through a sleek, interactive dashboard. 

### 🎯 The Problem & The Solution

### The Challenge

Modern financial analysis requires processing vast amounts of disparate data—including raw stock tickers, macroeconomic indicators, complex SEC filings, and global news. Human analysts spend up to 80% of their time aggregating data and checking compliance rather than executing strategic decisions. 

### Our Solution: Multi-Agent Synergy

This project replaces traditional, fragmented tools with a unified, *collaborative multi-agent framework*. Instead of relying on a single generic AI prompt, your files construct a specialized network of autonomous agents that mimic a real-world Wall Street research team: 

                  ┌────────────────────────┐
                  │   Streamlit Frontend   │
                  │        (app.py)        │
                  └───────────┬────────────┘
                              │
                  ┌───────────▼────────────┐
                  │    Orchestration       │
                  │   (multi_agents.py)    │
                  └─────┬────────────┬─────┘
                        │            │
         ┌──────────────▼─┐        ┌─▼──────────────┐
         │  Data Engine   │        │  RAG Engine    │
         │(data_engine.py)│        │ (rag_engine.py)│
         └──────┬─────────┘        └─┬──────────────┘
                │                    │
 ┌──────────────┼────────────────────┼──────────────┐
 │ 📊 ANALYTICS │ 🛡️ RISK & COMPLIANCE │ 🔎 SECURITY  │
 │ • profiler_  │ • riskometer.py    │ • fraud_     │
 │   allocator  │ • stress_tester.py │   detector.py│
 │ • profit_    │ • compliance_      │ • isin_      │
 │   crash      │   logger.py        │   analyzer.py│
 └──────────────┴────────────────────┴──────────────┘

1. *The Ingestion Phase:* data_engine.py pulls raw numerical metrics while rag_engine.py parses text-heavy financial disclosures.
2. *The Intelligence Phase:* multi_agents.py assigns tasks to domain-specific personas (e.g., Fundamental Analyst, Technical Expert) who cross-reference the ingested data.
3. *The Guardrail Phase:* Before any recommendation is pushed to the UI, risk (riskometer.py, stress_tester.py) and compliance layers (compliance_logger.py) stress-test and log the decisions for complete audit transparency.

### ✨ Key Features

### 🤖 1. Autonomous Multi-Agent Orchestration

* *Domain Specialization:* Leverages task-specific AI agents that run parallel chains of thought to analyze market fundamentals, trend charts, and asset valuations.
* *Intelligent Consensus:* Synthesizes conflicting market data points into a single, cohesive, and actionable investment report.

### 📊 2. Deep Financial Risk Modeling & Stress Testing

* *The "Riskometer":* Dynamically calculates real-time volatility ratings and risk profiles for individual tickers or blended portfolios.
* *Macro Stress Simulation:* Simulates historical black-swan events and market crashes to project worst-case drawdowns and portfolio vulnerabilities.
* *Profit-Crash Protection:* Back-tests extreme business revenue drops to assess long-term corporate survival.

### 🛡️ 3. Regulatory Compliance & Enterprise Auditing

* *Immutable Compliance Trail:* Logs agent decisions, parameters, and prompts into a persistent compliance_metrics.jsonl audit file to protect operations against AI hallucinations.
* *Asset Verification:* Decodes global ISIN (International Securities Identification Number) keys automatically to track and map assets across international borders.
* *Fraud Safeguards:* Audits real-time data flows for anomalous, high-risk financial markers or fraudulent patterns.

### 🌍 4. Global Market Readyness

* *Smart Portfolio Allocator:* Recommends asset weighting variations based on custom risk profiles, from conservative income to aggressive growth.
* *Localization Support:* Integrates structural regional formatting and multi-lingual dictionary engines to ensure localized dashboard deployment.

### 📂 File Directory Blueprint

* app.py / config.py — Application entry point, layout engine, and central global configurations.
* multi_agents.py — Orchestrates agent communication workflows, prompt templates, and reasoning chains.
* data_engine.py / rag_engine.py — Data pipelines combining API market feeds and vector-style text retrieval.
* riskometer.py / stress_tester.py / profit_crash_engine.py — Mathematical evaluation structures for portfolio threat modeling.
* profiler_allocator.py — Rebalances portfolios using user-defined risk parameters.
* fraud_detector.py / isin_analyzer.py — Anomaly tracking tools and security identification helpers.
* compliance_logger.py / compliance_metrics.jsonl — Local governance framework recording runtime agent operations.
* localization.py — Dynamic translation infrastructure.

### 🛠️ Instructions to Run the Program

Follow these quick steps to deploy and run the platform locally on your machine: 

### 1. Download the Project Files

Clone this repository to your local computer using Git, or download the workspace ZIP file: 

bash

git clone https://github.com/bhaveash08/AI-Financial-Analysist.git
cd AI-Financial-Analysist

Use code with caution.

### 2. Set Up a Virtual Environment (Recommended)

Create and isolate dependencies to avoid environmental conflicts: 

bash

# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

Use code with caution.

### 3. Install Dependencies

Install all required packages and deep learning/financial libraries listed in the setup manifest: 

bash

pip install -r requirements.txt

Use code with caution.

### 4. Launch the Server

Initialize the Streamlit web framework directly from your terminal: 

bash

streamlit run app.py

Use code with caution.

### 5. Access the Platform

Open your standard web browser and navigate to the local interface address displayed in your terminal channel: 

http://localhost:8501
