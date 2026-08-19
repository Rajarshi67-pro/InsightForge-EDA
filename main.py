"""
DataLens AI - Terminal Command-Line Interface & Engine Runner
Phases 1-6 CLI Profiler, Quality Auditor, Statistical Engine, and Grounded AI Analyst.
"""

import sys
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from datalens.profiler import DataProfiler
from datalens.quality import DataQualityEngine
from datalens.statistics import StatisticalEngine
from datalens.ai_engine import AIEngine


def format_header(title: str, width: int = 65) -> str:
    line = "═" * width
    return f"\n{line}\n  {title.center(width - 4)}\n{line}"


def format_subhead(title: str, width: int = 65) -> str:
    return f"\n{title}\n" + "─" * width


def run_cli_pipeline(file_path: str):
    if not os.path.exists(file_path):
        print(f"\n[!] Error: File '{file_path}' does not exist.")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"\n[!] Error reading CSV file: {e}")
        return

    print(format_header("DATALENS AI v1.0 - DATA PROFILER & ANALYST"))
    print(f"File Path    : {file_path}")
    print(f"Total Rows   : {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")

    # 1. Profiler
    profiler = DataProfiler(df)
    prof_dict = profiler.to_dict()

    print(format_subhead("1. COLUMN SEMANTIC CLASSIFICATION"))
    print(f"{'Column Name':<22} | {'Semantic Type':<14} | {'Dtype':<10} | {'Missing':<10} | {'Unique':<8}")
    print("─" * 65)
    for col in prof_dict["columns"]:
        m_str = f"{col['missing_count']} ({col['missing_pct']}%)"
        print(f"{col['column_name']:<22} | {col['semantic_type']:<14} | {col['pandas_dtype']:<10} | {m_str:<10} | {col['unique_count']:<8}")

    # 2. Data Quality
    quality_engine = DataQualityEngine(df, profiler_summary=prof_dict)
    qual_dict = quality_engine.to_dict()

    print(format_subhead("2. DATA QUALITY & HEALTH AUDIT"))
    print(f"Overall Health Score : {qual_dict['health_score']}/100")
    print(f"Duplicate Rows       : {qual_dict['duplicate_rows']} ({qual_dict['duplicate_pct']}%)")
    print(f"Total Missing Cells  : {qual_dict['total_missing_cells']} ({prof_dict['missing_cells_pct']}%)")

    outliers = qual_dict.get("outliers", {})
    if outliers:
        print("\n  [IQR Outlier Anomalies]")
        for c, info in outliers.items():
            iqr = info["iqr"]
            if iqr["outlier_count"] > 0:
                print(f"  • {c:<18}: {iqr['outlier_count']} outlier(s) (Bounds: [{iqr['lower_bound']}, {iqr['upper_bound']}]) -> Values: {iqr['outlier_values']}")
    else:
        print("  • No statistical IQR outliers detected.")

    # 3. Statistical Highlights & Correlations
    stats_engine = StatisticalEngine(df)
    stats_dict = stats_engine.to_dict()

    print(format_subhead("3. STATISTICAL INSIGHTS & CORRELATIONS"))
    num_stats = stats_dict.get("numerical", {})
    for col, s in list(num_stats.items())[:4]:
        print(f"  • {col:<18}: Mean={s['mean']:<8} Median={s['median_50']:<8} Std={s['std']:<8} Skew={s['skewness']:<6} ({s['skewness_label']})")

    strong_corrs = stats_dict.get("correlations", {}).get("strong_correlations", [])
    if strong_corrs:
        print("\n  [Key Pairwise Correlations]")
        for c in strong_corrs[:3]:
            print(f"  • {c['col1']} <-> {c['col2']}: Pearson r = {c['pearson']} ({c['strength']})")

    # 4. Grounded AI Insights
    ai_engine = AIEngine(df, prof_dict, qual_dict, stats_dict)
    insights = ai_engine.generate_executive_insights()

    print(format_subhead("4. GROUNDED AI EXECUTIVE SUMMARY"))
    print(insights)

    print(format_header("ANALYSIS COMPLETE"))
    print("💡 To explore interactive charts, ML Studio, and the web UI, run:\n   python server.py\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        default_test = os.path.join(os.path.dirname(__file__), "data", "auto_data_analyzer_test.csv")
        prompt_msg = f"Enter CSV file path (press Enter for '{default_test}'): "
        user_input = input(prompt_msg).strip()
        csv_file = user_input if user_input else default_test

    run_cli_pipeline(csv_file)