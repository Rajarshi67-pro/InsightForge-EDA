"""
Visualization Engine for InsightForge AI
Generates intelligent, interactive Plotly charts tailored to data types and relationships.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


class VisualizerEngine:
    """
    Constructs responsive, theme-consistent Plotly visualizations and automated chart recommendations.
    """

    COLOR_PALETTE = ["#4361EE", "#3A0CA3", "#7209B7", "#F72585", "#4CC9F0", "#4895EF", "#560BAD"]
    THEME_TEMPLATE = "plotly_white"

    @staticmethod
    def create_histogram(df: pd.DataFrame, col: str, nbins: int = 25) -> go.Figure:
        """Builds a distribution histogram with marginal boxplot."""
        clean_df = df.dropna(subset=[col])
        fig = px.histogram(
            clean_df,
            x=col,
            nbins=nbins,
            marginal="box",
            title=f"Distribution of {col}",
            color_discrete_sequence=["#4361EE"],
            template=VisualizerEngine.THEME_TEMPLATE,
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title=col,
            yaxis_title="Frequency",
            bargap=0.05,
        )
        return fig

    @staticmethod
    def create_box_plot(df: pd.DataFrame, y_col: str, x_col: Optional[str] = None) -> go.Figure:
        """Builds an interactive box & whisker plot, optionally grouped by a category."""
        clean_df = df.dropna(subset=[y_col] + ([x_col] if x_col else []))
        fig = px.box(
            clean_df,
            y=y_col,
            x=x_col,
            color=x_col if x_col else None,
            points="outliers",
            title=f"Box Plot: {y_col}" + (f" grouped by {x_col}" if x_col else ""),
            color_discrete_sequence=VisualizerEngine.COLOR_PALETTE,
            template=VisualizerEngine.THEME_TEMPLATE,
        )
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        return fig

    @staticmethod
    def create_bar_chart(df: pd.DataFrame, col: str, top_n: int = 15) -> go.Figure:
        """Builds a frequency bar chart for categorical variables."""
        counts = df[col].value_counts().head(top_n).reset_index()
        counts.columns = [col, "Count"]
        
        fig = px.bar(
            counts,
            x=col,
            y="Count",
            text="Count",
            title=f"Top Categories in {col}",
            color=col,
            color_discrete_sequence=VisualizerEngine.COLOR_PALETTE,
            template=VisualizerEngine.THEME_TEMPLATE,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False,
            xaxis_tickangle=-30 if len(counts) > 5 else 0,
        )
        return fig

    @staticmethod
    def create_scatter_plot(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: Optional[str] = None,
        add_trendline: bool = False,
    ) -> go.Figure:
        """Builds a scatter plot for relationship analysis."""
        subset_cols = [x_col, y_col] + ([color_col] if color_col else [])
        clean_df = df.dropna(subset=subset_cols)

        fig = px.scatter(
            clean_df,
            x=x_col,
            y=y_col,
            color=color_col if color_col else None,
            trendline="ols" if add_trendline else None,
            title=f"Relationship: {x_col} vs {y_col}",
            color_discrete_sequence=VisualizerEngine.COLOR_PALETTE,
            template=VisualizerEngine.THEME_TEMPLATE,
        )
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        return fig

    @staticmethod
    def create_line_chart(df: pd.DataFrame, date_col: str, val_col: str) -> go.Figure:
        """Builds a time-series line chart."""
        temp_df = df.dropna(subset=[date_col, val_col]).copy()
        temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce")
        temp_df = temp_df.dropna(subset=[date_col]).sort_values(by=date_col)

        fig = px.line(
            temp_df,
            x=date_col,
            y=val_col,
            markers=True,
            title=f"Trend Analysis: {val_col} over {date_col}",
            color_discrete_sequence=["#4361EE"],
            template=VisualizerEngine.THEME_TEMPLATE,
        )
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        return fig

    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame, method: str = "pearson") -> Optional[go.Figure]:
        """Builds an annotated correlation heatmap."""
        num_df = df.select_dtypes(include=[np.number])
        # Filter out id columns
        filtered_cols = [c for c in num_df.columns if not (str(c).lower().endswith("id") or "_id" in str(c).lower())]
        if len(filtered_cols) < 2:
            return None

        corr_matrix = num_df[filtered_cols].corr(method=method).round(2)
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title=f"{method.capitalize()} Correlation Heatmap",
            template=VisualizerEngine.THEME_TEMPLATE,
        )
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        return fig

    @staticmethod
    def create_missing_values_bar(df: pd.DataFrame) -> Optional[go.Figure]:
        """Builds a bar chart showing missing values per column."""
        missing = df.isna().sum()
        missing = missing[missing > 0].reset_index()
        if missing.empty:
            return None

        missing.columns = ["Column", "MissingCount"]
        missing["Percentage"] = ((missing["MissingCount"] / len(df)) * 100).round(1)

        fig = px.bar(
            missing,
            x="Column",
            y="MissingCount",
            text="Percentage",
            title="Missing Values by Column (%)",
            color="MissingCount",
            color_continuous_scale="Reds",
            template=VisualizerEngine.THEME_TEMPLATE,
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
        return fig

    @classmethod
    def recommend_visualizations(cls, df: pd.DataFrame, column_profiles: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Determines the most insightful automated visualizations based on strict column type rules.
        """
        recommendations = []
        
        # Sort columns to pick 'top' ones based on simple heuristics (e.g., fewest missing values, then uniqueness)
        def score_col(info):
            return (info.get('missing_pct', 100), -info.get('unique_count', 0))
            
        num_cols = sorted([c for c, info in column_profiles.items() if info.get("semantic_type") == "Numerical"],
                          key=lambda c: score_col(column_profiles[c]))
        cat_cols = sorted([c for c, info in column_profiles.items() if info.get("semantic_type") == "Categorical"],
                          key=lambda c: score_col(column_profiles[c]))
        date_cols = sorted([c for c, info in column_profiles.items() if info.get("semantic_type") == "Date"],
                          key=lambda c: score_col(column_profiles[c]))

        # Rule 1: Numerical -> Histogram & Box Plot (Standalone)
        if num_cols:
            primary_num = num_cols[0]
            recommendations.append({
                "type": "histogram",
                "title": f"Distribution of {primary_num}",
                "description": f"Displays frequency spread, central tendency, and skewness for {primary_num}.",
                "fig": cls.create_histogram(df, primary_num),
            })
            recommendations.append({
                "type": "box",
                "title": f"Box Plot of {primary_num}",
                "description": f"Highlights statistical outliers and quartiles for {primary_num}.",
                "fig": cls.create_box_plot(df, y_col=primary_num),
            })

        # Rule 2: Categorical -> Bar Chart
        if cat_cols:
            primary_cat = cat_cols[0]
            recommendations.append({
                "type": "bar",
                "title": f"Category Breakdown: {primary_cat}",
                "description": f"Shows distribution and most frequent classes in {primary_cat}.",
                "fig": cls.create_bar_chart(df, primary_cat),
            })

        # Rule 3: Multiple Numerical -> Correlation Heatmap
        if len(num_cols) >= 2:
            heat_fig = cls.create_correlation_heatmap(df)
            if heat_fig:
                recommendations.append({
                    "type": "heatmap",
                    "title": "Correlation Matrix",
                    "description": "Exposes linear relationships and collinearity among numerical features.",
                    "fig": heat_fig,
                })

        # Rule 4: Numerical + Numerical -> Scatter Plot
        if len(num_cols) >= 2:
            x_num = num_cols[0]
            y_num = num_cols[1]
            recommendations.append({
                "type": "scatter",
                "title": f"Relationship: {x_num} vs {y_num}",
                "description": f"Analyzes the correlation and clustering between {x_num} and {y_num}.",
                "fig": cls.create_scatter_plot(df, x_col=x_num, y_col=y_num, add_trendline=False),
            })

        # Rule 5: Datetime + Numerical -> Line Chart
        if date_cols and num_cols:
            primary_date = date_cols[0]
            primary_num = num_cols[0]
            recommendations.append({
                "type": "line",
                "title": f"{primary_num} Trend Over Time",
                "description": f"Tracks progression of {primary_num} across {primary_date}.",
                "fig": cls.create_line_chart(df, date_col=primary_date, val_col=primary_num),
            })

        # Rule 6: Categorical + Numerical -> Grouped Box Plot / Aggregated Bar
        if cat_cols and num_cols:
            primary_cat = cat_cols[0]
            primary_num = num_cols[0]
            recommendations.append({
                "type": "box",
                "title": f"{primary_num} by {primary_cat}",
                "description": f"Analyzes variance and median of {primary_num} across {primary_cat} groups.",
                "fig": cls.create_box_plot(df, y_col=primary_num, x_col=primary_cat),
            })

        return recommendations
