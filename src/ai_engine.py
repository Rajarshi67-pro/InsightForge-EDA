"""
Grounded AI Engine for InsightForge EDA
Integrates LLM reasoning with deterministic calculation tools, Privacy Scanner,
Evidence Verification, and RAG Knowledge Base.
"""

import os
import re
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from .logger import app_logger
from .privacy import PrivacyScanner
from .rag_engine import RAGEngine
from .agents import AgentOrchestrator
from .llm_providers import BaseLLMProvider, LocalOllamaProvider, GeminiProvider


class AIEngine:
    """
    Grounded AI Analyst that combines deterministic calculations, privacy protection,
    RAG domain knowledge, and evidence-backed LLM interpretation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        profiler_dict: Dict[str, Any],
        quality_dict: Dict[str, Any],
        stats_dict: Dict[str, Any],
        api_key: Optional[str] = None,
    ):
        self.raw_df = df.copy()
        # 1. Privacy Safety Scan & Masking
        self.privacy_scanner = PrivacyScanner(df)
        self.df = self.privacy_scanner.mask_dataframe()
        self.privacy_report = self.privacy_scanner.to_dict()

        self.profiler = profiler_dict
        self.quality = quality_dict
        self.stats = stats_dict
        self.api_key = api_key
        
        # 2. Agent Orchestrator (Deterministic Actionable Insights Engine)
        self.orchestrator = AgentOrchestrator(self.df, self.profiler, self.quality, self.stats)
        
        # 3. Initialize LLM Providers
        self._init_llm_providers()

    def _init_llm_providers(self):
        """Initializes available LLM providers. Tries local first, then external APIs."""
        self.providers: List[BaseLLMProvider] = []
        
        local_provider = LocalOllamaProvider()
        if local_provider.is_available():
            self.providers.append(local_provider)
            
        gemini_provider = GeminiProvider(api_key=self.api_key)
        if gemini_provider.is_available():
            self.providers.append(gemini_provider)

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Calls the first available LLM provider to generate text."""
        for provider in self.providers:
            result = provider.generate(prompt)
            if result:
                app_logger.info(f"Successfully generated response using {provider.get_provider_name()}")
                return result
        return None

    def get_engine_status(self) -> Dict[str, Any]:
        """Exposes AI availability for API/UI status badges."""
        local_ai = next((p for p in self.providers if p.get_provider_name() == "Local Ollama"), None)
        gemini_ai = next((p for p in self.providers if p.get_provider_name() == "Gemini"), None)
        
        return {
            "local_ai_enabled": local_ai is not None,
            "local_model": getattr(local_ai, "model", None) if local_ai else None,
            "local_base_url": getattr(local_ai, "base_url", None) if local_ai else None,
            "external_llm_available": gemini_ai is not None,
            "external_model": getattr(gemini_ai, "model_name", None) if gemini_ai else None,
        }

    def _build_grounded_context(self) -> str:
        """Constructs a factual, deterministic, privacy-sanitized summary context."""
        lines = []
        lines.append("=== PRIVACY & DATASET HEALTH ===")
        lines.append(f"- Privacy Safety Score: {self.privacy_report.get('privacy_safety_score')}/100")
        lines.append(f"- Health Score: {self.quality.get('health_score')}/100")
        lines.append(f"- Total Rows: {self.profiler.get('total_rows')}")
        lines.append(f"- Total Columns: {self.profiler.get('total_cols')}")
        lines.append(f"- Duplicate Rows: {self.quality.get('duplicate_rows')} ({self.quality.get('duplicate_pct')}%)")
        lines.append(f"- Missing Cells: {self.quality.get('total_missing_cells')} ({self.profiler.get('missing_cells_pct')}%)")

        lines.append("\n=== STATISTICAL GROUND TRUTH ===")
        num_stats = self.stats.get("numerical", {})
        for col, s in list(num_stats.items())[:6]:
            lines.append(f"- `{col}`: Mean={s.get('mean')}, Median={s.get('median_50')}, Std={s.get('std')}, Min={s.get('min')}, Max={s.get('max')}, Skewness={s.get('skewness')} ({s.get('skewness_label')})")

        return "\n".join(lines)

    def generate_executive_insights(self) -> str:
        """
        Generates an executive briefing by strictly summarizing the output of the 
        deterministic Actionable Insights Engine.
        """
        # Always run the deterministic engine first (Source of Truth)
        audit = self.orchestrator.run_autonomous_audit()
        deterministic_report = audit["executive_briefing"]

        # If we have an AI provider, ask it to summarize the deterministic findings.
        # We explicitly enforce that it does NOT invent statistics.
        if self.providers:
            prompt = f"""You are InsightForge EDA, an elite Principal Data Scientist.
Your task is to summarize the following STRICTLY VERIFIED mathematical findings generated by our deterministic Actionable Insights Engine.

VERIFIED DETERMINISTIC FINDINGS:
{deterministic_report}

Generate a concise, authoritative executive summary in GitHub-flavored Markdown.
RULES:
1. DO NOT invent, hallucinate, or calculate any new numbers.
2. Only summarize the findings, evidence, and actions provided above.
3. Group the insights logically into a cohesive business narrative.
4. Keep the tone professional, impactful, and direct."""
            
            ai_summary = self._call_llm(prompt)
            if ai_summary:
                return f"> 🤖 **AI-Generated Executive Summary** *(Synthesized from deterministic rule-based findings)*\n\n{ai_summary}\n\n---\n\n### 📐 Raw Deterministic Actionable Insights\n\n{deterministic_report}"

        # If no AI is available (or generation fails), just return the raw deterministic report
        return deterministic_report

    def answer_query(self, user_query: str) -> Dict[str, Any]:
        """
        Answers user questions using deterministic calculation tools + LLM reasoning + RAG knowledge retrieval.
        """
        app_logger.info(f"Processing query: '{user_query}'")
        q = user_query.strip().lower()

        # 0. Conversational Greetings Handler
        if q in ["hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening", "who are you", "what can you do"]:
            rows = self.profiler.get("total_rows", len(self.raw_df))
            cols = self.profiler.get("total_cols", len(self.raw_df.columns))
            return {
                "answer": f"Hello! 👋 I am your **InsightForge EDA Executive Analyst**.\n\nI am actively analyzing your dataset with **{rows:,} rows** and **{cols} features**.\n\nHere are practical questions you can ask me right now:\n- 📈 *'What are the strongest correlations in this data?'*\n- 💰 *'What is the highest and lowest salary by department?'*\n- 🚨 *'Show outliers in Experience or Performance'* \n- 🧼 *'Summarize data hygiene and missingness risks'*\n- 🧠 *'Train an ML model on Salary or Attrition'*",
                "data": None,
            }

        # 1. Deterministic Correlation Queries
        if "correlation" in q or "relationship" in q or "correlated" in q:
            strong_corrs = self.stats.get("correlations", {}).get("strong_correlations", [])
            if strong_corrs:
                corr_rows = []
                for c in strong_corrs:
                    corr_rows.append({
                        "Feature 1": c["col1"],
                        "Feature 2": c["col2"],
                        "Pearson (r)": c["pearson"],
                        "Strength": f"{c['strength'].capitalize()} {c['direction']}"
                    })
                top_c = strong_corrs[0]
                return {
                    "answer": f"The strongest linear correlation is between **`{top_c['col1']}`** and **`{top_c['col2']}`** with a Pearson **$r = {top_c['pearson']}$** ({top_c['strength']} {top_c['direction']}).\n\n`[Evidence: Pearson Correlation Matrix calculation]`",
                    "data": corr_rows,
                }
            else:
                return {
                    "answer": f"{self.stats.get('correlations', {}).get('interpretation_guide', '')}\n\nNo strong linear correlations ($|r| \\ge 0.5$) were detected between numerical columns in this dataset.\n\n`[Evidence: Full pairwise correlation matrix evaluated]`",
                    "data": None,
                }

        # 2. Deterministic calculations before any generative model.
        agg_match = re.search(r"(highest|maximum|max|lowest|minimum|min|average|avg|mean)\s+([\w_]+)(?:\s+(?:by|in|for|per)\s+([\w_]+))?", q)
        if agg_match:
            op, target_term, group_term = agg_match.groups()
            target_col = self._match_column_name(target_term)
            group_col = self._match_column_name(group_term) if group_term else None

            if target_col and pd.api.types.is_numeric_dtype(self.raw_df[target_col]):
                if group_col and group_col in self.raw_df.columns:
                    if op in ["highest", "maximum", "max"]:
                        res = self.raw_df.groupby(group_col)[target_col].max().reset_index().sort_values(by=target_col, ascending=False)
                        top_grp = res.iloc[0][group_col]
                        top_val = res.iloc[0][target_col]
                        return {
                            "answer": f"The `{group_col}` with the highest `{target_col}` is **{top_grp}** with a value of **{top_val:,.2f}**.\n\n`[Evidence: Group Maximum calculation]`",
                            "data": res.to_dict(orient="records"),
                        }
                    if op in ["lowest", "minimum", "min"]:
                        res = self.raw_df.groupby(group_col)[target_col].min().reset_index().sort_values(by=target_col, ascending=True)
                        low_grp = res.iloc[0][group_col]
                        low_val = res.iloc[0][target_col]
                        return {
                            "answer": f"The `{group_col}` with the lowest `{target_col}` is **{low_grp}** with a value of **{low_val:,.2f}**.\n\n`[Evidence: Group Minimum calculation]`",
                            "data": res.to_dict(orient="records"),
                        }

                    res = self.raw_df.groupby(group_col)[target_col].mean().round(2).reset_index().sort_values(by=target_col, ascending=False)
                    top_grp = res.iloc[0][group_col]
                    top_val = res.iloc[0][target_col]
                    return {
                        "answer": f"The average `{target_col}` across `{group_col}` is highest in **{top_grp}** at **{top_val:,.2f}**.\n\n`[Evidence: Group Mean calculation]`",
                        "data": res.to_dict(orient="records"),
                    }

                if op in ["highest", "maximum", "max"]:
                    val = self.raw_df[target_col].max()
                    return {"answer": f"The maximum value of `{target_col}` is **{val:,.2f}**.\n\n`[Evidence: Max={val}]`", "data": None}
                if op in ["lowest", "minimum", "min"]:
                    val = self.raw_df[target_col].min()
                    return {"answer": f"The minimum value of `{target_col}` is **{val:,.2f}**.\n\n`[Evidence: Min={val}]`", "data": None}

                val = self.raw_df[target_col].mean()
                return {"answer": f"The overall average (mean) of `{target_col}` is **{val:,.2f}**.\n\n`[Evidence: Mean={val:,.2f}]`", "data": None}

        if "missing" in q or "null" in q or "na" in q:
            missing_data = self.raw_df.isna().sum().reset_index()
            missing_data.columns = ["Column", "Missing Count"]
            missing_data = missing_data[missing_data["Missing Count"] > 0]
            if missing_data.empty:
                return {"answer": "There are **0 missing values** across all columns in this dataset.\n\n`[Evidence: 100% Completeness]`", "data": None}
            return {
                "answer": f"Detected missing values in **{len(missing_data)} column(s)**:\n\n`[Evidence: Total Missing Cells = {self.quality.get('total_missing_cells')}]`",
                "data": missing_data.to_dict(orient="records"),
            }

        if "outlier" in q or "anomaly" in q or "extreme" in q:
            outliers = self.quality.get("outliers", {})
            outlier_rows = []
            method = "Z-Score" if "z-score" in q.lower() or "z score" in q.lower() else "IQR"
            method_key = "z_score" if method == "Z-Score" else "iqr"

            for col, o in outliers.items():
                info = o.get(method_key, {})
                if info.get("outlier_count", 0) > 0:
                    outlier_rows.append({
                        "Column": col,
                        "Method": method,
                        "Outliers": info["outlier_count"],
                        "Percentage": f"{info['outlier_pct']}%",
                        "Severity": info.get("potential_severity", "Unknown"),
                        "Lower Bound": info.get("lower_bound"),
                        "Upper Bound": info.get("upper_bound"),
                        "Outlier Values": str(info.get("outlier_values", [])),
                    })
            if outlier_rows:
                return {
                    "answer": f"Detected {method} outliers in **{len(outlier_rows)} column(s)**:\n\n`[Evidence: {method} method]`",
                    "data": outlier_rows,
                }
            return {"answer": f"No statistical outliers detected via {method} method.\n\n`[Evidence: Zero values outside {method} bounds]`", "data": None}

        # 3. InsightForge EDA Grounded Reasoning with Full Dataset Context
        grounded_ctx = self._build_grounded_context()
        rag_docs = RAGEngine.search(user_query)
        rag_context = "\n".join([f"• **{d['title']}**: {d['content']}" for d in rag_docs]) if rag_docs else ""

        prompt = f"""You are InsightForge EDA, an expert, friendly, and practical data scientist.
Dataset Context:
{grounded_ctx}

Relevant Knowledge Base Articles:
{rag_context}

User question: {user_query}

Provide a direct, crystal-clear, and helpful answer in Markdown. Keep explanations simple, practical, and grounded in the data. Cite exact numbers and column names where helpful. Never mention third-party AI provider names, refer exclusively to InsightForge EDA."""

        res_text = self._call_llm(prompt)
        if res_text:
            return {"answer": f"> 🤖 **AI-Generated Answer**\n\n{res_text}", "data": None}

        # 4. Conceptual definitions (only if explicitly asking "what is", "explain") fallback if no LLM
        if any(term in q for term in ["what is", "explain", "definition of", "how does"]):
            if rag_docs:
                top_d = rag_docs[0]
                return {
                    "answer": f"### 📚 Knowledge Base: {top_d['title']}\n\n{top_d['content']}\n\n*Retrieved via InsightForge EDA RAG Engine.*",
                    "data": None,
                }

        return {
            "answer": f"I parsed your question: **'{user_query}'**.\n\nHere are specific questions you can ask me:\n- *'What is the average Salary by Department?'*\n- *'Show outliers in Experience'* \n- *'What are the strongest correlations?'*\n- *'Check for duplicate rows'*\n- *'Highest Performance_Score by Department'*",
            "data": None,
        }

    def _match_column_name(self, term: Optional[str]) -> Optional[str]:
        """Fuzzy matches user terms against DataFrame column names."""
        if not term:
            return None
        term_clean = re.sub(r"[^\w]", "", term.lower())
        for col in self.raw_df.columns:
            col_clean = re.sub(r"[^\w]", "", str(col).lower())
            if term_clean == col_clean or term_clean in col_clean or col_clean in term_clean:
                return str(col)
        return None
