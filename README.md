# 🔬 DataLens AI

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**DataLens AI** is a production-grade, grounded exploratory data analysis (EDA), automated data quality, machine learning, and AI analyst platform. Built on the core philosophy that **deterministic Python and statistics calculate the truth, while AI reasons and interprets it**, DataLens AI completely eliminates LLM hallucinations in tabular data analysis.

It comes equipped with a **bespoke custom frontend** (HTML5, Vanilla CSS design system, responsive Vanilla JS, and Chart.js) powered by a high-performance **FastAPI backend REST server**.

---

## 🏛️ System Architecture

```
                       CUSTOM MODERN WEB FRONTEND
            (HTML5 Semantic Views • Vanilla CSS Design System • Chart.js)
                                     │
                                     ▼  RESTful JSON APIs
                      FASTAPI BACKEND SERVER (server.py)
            (File Ingestion • Delimiter Inferences • Async Endpoints)
                                     │
                                     ▼
                   🛡️ AI PRIVACY SCANNER (PrivacyScanner)
             (Masks Emails, Phones, PAN, Aadhaar, SSN, Passwords)
                                     │
                                     ▼
                    DATA PROFILER (Semantic Classifier)
              (Identifier, Numerical, Categorical, Date, Bool)
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
DATA QUALITY ENGINE         STATISTICAL ENGINE           VISUALIZER ENGINE
(Missing, Dups, IQR Outliers) (Summary, Skew, Corr Matrix) (Interactive Chart.js)
         │                           │                           │
         └───────────────────────────┼───────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
 MACHINE LEARNING          📚 RAG KNOWLEDGE BASE       🤖 AGENT ORCHESTRATOR
(Classification, Regr,     (Statistical & ML Metric     (Insight, Anomaly,
  Clustering, Importance)     Concept Retrieval)         Chat & Report Agents)
         │                           │                           │
         └───────────────────────────┼───────────────────────────┘
                                     │
                                     ▼
              LOCAL-FIRST GROUNDED AI REASONING
             (Optional Gemini only when configured)
                  + MATHEMATICAL EVIDENCE VERIFICATION
                                     │
                                     ▼
                    EXECUTIVE REPORTS & REPRODUCIBILITY
                (Download Cleaned CSV, PDF, JSON SHA-256)
```

---

## 🌟 Core Features

1. **🎨 Bespoke Custom Frontend**:
   - Modern, responsive single-page application built with semantic HTML5, fluid Vanilla CSS, and reactive Vanilla JS.
   - Client-side rendering powered by **Chart.js** with zero UI lag.
   - Live drag-and-drop CSV upload, interactive data cleaning studio, paginated data explorer, and real-time AI analyst chat.
2. **🛡️ AI Privacy & Safety Scanner**:
   - Scans datasets for Personally Identifiable Information (PII) before any data touches an external LLM.
   - Detects emails, phone numbers, credit card sequences, government IDs (Aadhaar, PAN, SSN), and passwords.
   - Generates a **Privacy Safety Score (0–100)** and automatically masks sensitive fields into redacted tokens.
3. **📊 Deterministic Semantic Profiler**:
   - Classifies columns into `Identifier`, `Numerical`, `Categorical`, `Date`, and `Boolean` semantic types.
   - Computes sparsity rates, memory footprint, dimensions, and unique cardinality counts.
4. **🧼 Data Quality & Cleaning Engine**:
   - Identifies missing data gaps, column-by-column missing percentages, and duplicate rows.
   - Computes statistical anomalies using the **Interquartile Range (IQR)** method ($Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}$) and Z-scores.
   - Provides a one-click interactive data cleaning pipeline (deduplication, mean/median/mode imputation, IQR capping/trimming).
5. **🔢 Statistical & Correlation Engine**:
   - Univariate summary: Mean, Median, Mode, Standard Deviation, Variance, Min, Max, Quartiles, Skewness, and Kurtosis.
   - Bivariate dynamics: Computes Pearson (linear) and Spearman (rank-order) correlation matrices.
6. **🧠 Machine Learning Studio**:
   - Automatic problem type detection (Supervised Regression vs Classification vs Unsupervised Clustering).
   - Model training (Random Forest, Linear Regression, Logistic Regression, K-Means) with $R^2$, RMSE, Accuracy, Precision, Recall, F1, interactive Confusion Matrix, and Feature Importance rankings.
7. **📚 RAG Knowledge Base**:
   - Indexed domain knowledge repository of statistical formulas, ML interpretations, and financial metrics (e.g. Debt-to-Equity, Kurtosis, Silhouette Score).
8. **🤖 Agentic Orchestration & Grounded Chatbot**:
   - Multi-agent coordination: `InsightAgent`, `AnomalyAgent`, `ChatAgent`, and `ReportAgent`.
   - Natural language query answering with deterministic Python tool routing.
   - **Evidence Verification**: Attaches verifiable mathematical tags like `[Evidence: Pearson r=0.82]` to all AI claims.
9. **📑 Multi-Format Reports & Reproducibility**:
   - One-click export of **Executive PDF Audit Reports**, **Cleaned CSV Datasets**, and **Cryptographic JSON Audit Logs** (with SHA-256 checksums).

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/DataLensAI.git
cd DataLensAI

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Custom Web Studio

```bash
python server.py
```
Open your browser at **[http://localhost:8000](http://localhost:8000)**!

### 3. Run the Command-Line Interface (CLI)

```bash
python main.py data/auto_data_analyzer_test.csv
```

### 4. Run Automated Test Suite

```bash
python -m unittest discover -s tests
```

### Optional AI Configuration

DataLens AI works without any external API key. It uses deterministic local analytics first, can call a local Ollama-compatible model when available, and only uses Gemini as an optional enhancement when `GEMINI_API_KEY` or `GOOGLE_API_KEY` is configured.

```bash
# Optional local model settings
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama3.1

# Optional Gemini enhancement
export GEMINI_API_KEY=your_key_here
```

---

## 📂 Project Structure

```
DataLensAI/
│
├── server.py                   # FastAPI Production Web Server
├── main.py                     # Command-Line Interface Terminal Runner
├── requirements.txt            # Project Dependencies
├── README.md                   # Full Portfolio Documentation
├── .env.example                # Secrets & Config Template
├── .gitignore                  # Security & Build Exclusions
│
├── static/                     # Bespoke Custom Frontend Assets
│   ├── index.html              # Modern SPA Markup
│   ├── css/
│   │   └── style.css           # Custom Vanilla CSS Design System
│   └── js/
│       └── app.js              # Reactive Vanilla JS App & Chart.js Controllers
│
├── data/
│   └── auto_data_analyzer_test.csv # Sample HR & Attrition Dataset
│
├── tests/
│   ├── test_pipeline.py        # Core Engine Unit Tests
│   └── test_server.py          # FastAPI Endpoint Integration Tests
│
└── datalens/                   # Core Analytical Engines
    ├── __init__.py             # Package Exports
    ├── loader.py               # Ingestion with Auto-Encoding & Sampling
    ├── privacy.py              # AI Safety & PII Scanner
    ├── profiler.py             # Semantic Type Inference & Schema
    ├── quality.py              # Missing, Dups, IQR Outliers & Cleaning
    ├── statistics.py           # Descriptive Stats, Skewness, Correlations
    ├── visualizer.py           # Chart Recommendations & Figures
    ├── ml_engine.py            # Supervised & Unsupervised ML Studio
    ├── rag_engine.py           # Statistical Knowledge Base
    ├── agents.py               # Autonomous Multi-Agent Orchestrator
    ├── ai_engine.py            # Grounded AI Analyst & Tool Router
    ├── reports.py              # PDF, MD & Reproducibility JSON Generator
    └── logger.py               # Structured Production Logging
```

---

## 📜 License
MIT License. Built for data scientists, engineers, and analysts who demand grounded truth in AI analysis.

DataLensAI/
│
├── server.py                   # High-Performance FastAPI Backend REST Server
├── main.py                     # Terminal Command-Line Interface
├── requirements.txt            # All Dependencies (fastapi, pandas, scikit-learn, fpdf2, etc.)
├── README.md                   # Full Portfolio Documentation & Diagrams
│
├── static/                     # Bespoke Custom Frontend (HTML5 / Vanilla CSS / Vanilla JS)
│   ├── index.html              # Semantic, Responsive Single-Page Dashboard
│   ├── css/
│   │   └── style.css           # Custom Design System (HSL Tokens, Glassmorphism, Micro-Animations)
│   └── js/
│       └── app.js              # Reactive State Controller, Chart.js Integrations & Chat UI
│
├── tests/
│   ├── test_pipeline.py        # Core Engine Unit Tests
│   └── test_server.py          # FastAPI Endpoint Integration Tests
│
└── datalens/                   # Core Analytical Engines
    ├── loader.py               # Resilient Ingestion & Encoding Detection
    ├── privacy.py              # AI Safety & PII Scanner (Emails, Phones, PAN, Aadhaar, SSN)
    ├── profiler.py             # Semantic Type Inference (ID, Num, Cat, Date, Bool)
    ├── quality.py              # Missing Values, Duplicates, IQR Outliers & Cleaning
    ├── statistics.py           # Descriptive Stats, Skewness & Correlations
    ├── ml_engine.py            # Supervised (Reg/Clf) & Unsupervised Clustering ML Studio
    ├── rag_engine.py           # Statistical Knowledge Base
    ├── agents.py               # Autonomous Multi-Agent Orchestrator
    ├── ai_engine.py            # Grounded AI Analyst + Evidence Verification
    └── reports.py              # Executive PDF, Cleaned CSV & Reproducibility JSON Audits
