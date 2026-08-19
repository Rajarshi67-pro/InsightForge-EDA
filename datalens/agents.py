"""
Agentic Multi-Agent Orchestration Engine for DataLens AI
Coordinates specialized autonomous agents (InsightAgent, AnomalyAgent, ReportAgent)
to generate actionable insights.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from .logger import app_logger
from .rag_engine import RAGEngine


class InsightAgent:
    """Specialized agent for macro pattern recognition, feature correlations, and distributions."""

    def __init__(self, stats_dict: Dict[str, Any], profiler_dict: Dict[str, Any]):
        self.stats = stats_dict
        self.profiler = profiler_dict

    def run(self) -> List[Dict[str, Any]]:
        insights = []
        
        # 1. Strong correlations
        corrs = self.stats.get("correlations", {}).get("strong_correlations", [])
        for c in corrs[:3]:
            # Pearson or Spearman
            is_nl = c.get("is_nonlinear", False)
            method = "Spearman" if is_nl else "Pearson"
            val = c.get("spearman") if is_nl else c.get("pearson")
            
            insights.append({
                "category": "Correlation Dynamics",
                "finding": f"Strong {c.get('strength', '').lower()} relationship between `{c['col1']}` and `{c['col2']}`.",
                "evidence": f"{method} r = {val}",
                "importance": "Metrics moving together suggest underlying trends. Note: Correlation does not imply causation.",
                "action": "Investigate shared confounding factors before concluding direct causality.",
            })

        num_stats = self.stats.get("numerical", {})
        for col, s in num_stats.items():
            # 2. Severe skewness
            if abs(s.get("skewness", 0)) > 1.0:
                insights.append({
                    "category": "Distribution Asymmetry",
                    "finding": f"`{col}` exhibits severe {s['skewness_label'].lower()}.",
                    "evidence": f"Skewness = {s['skewness']} (Mean: {s['mean']}, Median: {s['median_50']})",
                    "importance": "Highly skewed features can degrade the performance of linear machine learning models.",
                    "action": "Apply a log or Box-Cox transformation to normalize the distribution.",
                })
            
            # 3. Low-variance columns
            variance = s.get("variance", 0)
            if variance == 0:
                insights.append({
                    "category": "Low-Variance Feature",
                    "finding": f"`{col}` has zero variance (constant value).",
                    "evidence": f"Variance = {variance}",
                    "importance": "Constant features provide zero predictive power to machine learning models.",
                    "action": f"Safely drop `{col}` from the dataset.",
                })
            elif variance > 0 and s.get("std", 1) < 0.01:
                insights.append({
                    "category": "Low-Variance Feature",
                    "finding": f"`{col}` has extremely low variance.",
                    "evidence": f"Standard Deviation = {s['std']}",
                    "importance": "Features with near-zero variance often contribute noise rather than signal.",
                    "action": "Consider scaling or evaluating if the feature can be safely removed.",
                })

        cat_stats = self.stats.get("categorical", {})
        for col, s in cat_stats.items():
            # 4. High-cardinality categorical columns
            if s.get("unique", 0) > 50:
                insights.append({
                    "category": "High Cardinality",
                    "finding": f"`{col}` contains an unusually high number of unique categories.",
                    "evidence": f"{s['unique']} unique values.",
                    "importance": "One-hot encoding this feature will cause severe dimensionality expansion (Curse of Dimensionality).",
                    "action": "Use target encoding, frequency encoding, or group rare categories into an 'Other' bucket.",
                })
            
            # 5. Class imbalance
            if s.get("top_percentage", 0) > 80.0 and s.get("unique", 0) > 1:
                insights.append({
                    "category": "Class Imbalance",
                    "finding": f"`{col}` is heavily dominated by a single class.",
                    "evidence": f"'{s['top']}' accounts for {s['top_percentage']}% of all observations.",
                    "importance": "Models trained on imbalanced targets will become heavily biased toward the majority class.",
                    "action": "Apply SMOTE, downsample the majority class, or use class weights during training.",
                })

        # 6. Potential identifier columns
        columns_info = self.profiler.get("columns", [])
        for col_info in columns_info:
            if col_info.get("semantic_type") == "Identifier":
                insights.append({
                    "category": "Identifier Column Detected",
                    "finding": f"`{col_info['column_name']}` appears to be a primary key or identifier.",
                    "evidence": f"Semantic type inferred as Identifier with {col_info.get('unique_count')} unique values.",
                    "importance": "Identifiers act as noise and can cause data leakage or severe overfitting if passed to a model.",
                    "action": f"Drop `{col_info['column_name']}` before training machine learning models.",
                })

        return insights


class AnomalyAgent:
    """Specialized agent for detecting data hygiene hazards, duplicate records, and outliers."""

    def __init__(self, quality_dict: Dict[str, Any]):
        self.quality = quality_dict

    def run(self) -> List[Dict[str, Any]]:
        anomalies = []
        
        # 1. High duplicate rate
        dups = self.quality.get("duplicate_rows", 0)
        dup_pct = self.quality.get("duplicate_pct", 0)
        if dup_pct > 1.0:
            anomalies.append({
                "category": "Redundancy Risk",
                "finding": f"High duplicate row frequency detected.",
                "evidence": f"{dups} duplicate row(s) ({dup_pct}% of dataset).",
                "importance": "Duplicate records artificially inflate confidence and bias model training.",
                "action": "Deduplicate dataset before performing statistical analysis or ML training.",
            })

        # 2. High missingness
        missing = self.quality.get("missing_details", [])
        for m in missing:
            if m.get("missing_pct", 0) > 5.0:
                anomalies.append({
                    "category": "Missing Data Gap",
                    "finding": f"`{m['column']}` is suffering from significant missingness.",
                    "evidence": f"Missing {m['missing_count']} values ({m['missing_pct']}% of column).",
                    "importance": "High missingness causes algorithms to drop rows or hallucinate patterns during imputation.",
                    "action": f"Apply robust imputation (Iterative/KNN) or consider dropping `{m['column']}` if >40% missing.",
                })

        # 3. Extreme outliers (Medium or High severity)
        outliers = self.quality.get("outliers", {})
        for col, info in outliers.items():
            iqr = info.get("iqr", {})
            if iqr.get("outlier_count", 0) > 0 and iqr.get("potential_severity") in ["Medium", "High"]:
                anomalies.append({
                    "category": "Statistical Outlier (IQR)",
                    "finding": f"`{col}` contains a concerning density of extreme values.",
                    "evidence": f"{iqr['outlier_count']} extreme value(s) ({iqr.get('outlier_pct', 0)}%).",
                    "importance": "Extreme outliers disproportionately skew loss functions in gradient descent models.",
                    "action": "Use IQR Capping (Winsorization) or robust scaling before modeling.",
                })
            
            z_score = info.get("z_score", {})
            if z_score.get("outlier_count", 0) > 0 and z_score.get("potential_severity") in ["Medium", "High"]:
                anomalies.append({
                    "category": "Statistical Outlier (Z-Score)",
                    "finding": f"`{col}` violates standard normal distribution bounds.",
                    "evidence": f"{z_score['outlier_count']} extreme value(s) (|z| > {z_score.get('threshold', 3.0)}).",
                    "importance": "Models assuming gaussian distributions will fail to predict accurately.",
                    "action": "Investigate the underlying data generation process for errors.",
                })

        return anomalies


class ReportAgent:
    """Specialized agent for compiling multi-agent outputs into a coherent executive report."""

    def __init__(self, insight_agent: InsightAgent, anomaly_agent: AnomalyAgent):
        self.insights = insight_agent.run()
        self.anomalies = anomaly_agent.run()

    def compile(self) -> str:
        lines = []
        lines.append("## 🤖 Autonomous Actionable Insights Engine\n")

        def _format_insight(item: Dict[str, str]) -> str:
            return (
                f"- **{item.get('finding', 'Unknown finding')}**\n"
                f"  - *Evidence*: {item.get('evidence', 'No evidence provided.')}\n"
                f"  - *Why it matters*: {item.get('importance', 'No impact specified.')}\n"
                f"  - *Actionable Recommendation*: **{item.get('action', 'No action recommended.')}**\n"
            )

        lines.append("### 🔍 1. Macro Insights & Patterns")
        if self.insights:
            for item in self.insights:
                lines.append(_format_insight(item))
        else:
            lines.append("- No significant macro patterns or actionable insights detected in this dataset.\n")

        lines.append("### ⚠️ 2. Anomaly & Hygiene Hazards")
        if self.anomalies:
            for item in self.anomalies:
                lines.append(_format_insight(item))
        else:
            lines.append("- Dataset is exceptionally clean. No actionable hazards or anomalies detected.\n")

        return "\n".join(lines)


class AgentOrchestrator:
    """Main orchestrator coordinating all specialized analysis sub-agents."""

    def __init__(self, df: pd.DataFrame, profiler_dict: Dict[str, Any], quality_dict: Dict[str, Any], stats_dict: Dict[str, Any]):
        self.df = df
        self.insight_agent = InsightAgent(stats_dict, profiler_dict)
        self.anomaly_agent = AnomalyAgent(quality_dict)
        self.report_agent = ReportAgent(self.insight_agent, self.anomaly_agent)
        app_logger.info("AgentOrchestrator initialized with 4 autonomous sub-agents.")

    def run_autonomous_audit(self) -> Dict[str, Any]:
        """Runs all sub-agents and aggregates findings."""
        return {
            "insights": self.insight_agent.run(),
            "anomalies": self.anomaly_agent.run(),
            "executive_briefing": self.report_agent.compile(),
        }
