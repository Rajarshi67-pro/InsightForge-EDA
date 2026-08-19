"""
RAG (Retrieval-Augmented Generation) Knowledge Base for DataLens AI
Maintains an indexed domain knowledge repository of statistical concepts, ML metrics,
and data engineering best practices to enrich AI reasoning.
"""

from typing import List, Dict, Any, Optional
import re
from .logger import app_logger


class RAGEngine:
    """
    Knowledge retriever over statistical definitions, ML interpretations, and data science concepts.
    """

    KNOWLEDGE_DOCUMENTS = [
        {
            "id": "stat_pearson",
            "title": "Pearson Correlation Coefficient (r)",
            "keywords": ["pearson", "correlation", "linear", "relationship", "r-value", "collinearity"],
            "content": (
                "The Pearson correlation coefficient (r) measures linear correlation between two continuous variables. "
                "Values range from -1.0 to +1.0. |r| >= 0.7 indicates strong correlation, 0.4 to 0.7 indicates moderate, "
                "and < 0.4 indicates weak. A positive r means as X increases, Y increases; a negative r means as X increases, Y decreases."
            ),
        },
        {
            "id": "stat_spearman",
            "title": "Spearman Rank Correlation (rho)",
            "keywords": ["spearman", "rank", "monotonic", "non-linear", "outlier-robust"],
            "content": (
                "Spearman's rank correlation coefficient assesses monotonic relationships (whether linear or non-linear). "
                "It ranks values before computing correlation, making it robust against non-normal distributions and extreme outliers."
            ),
        },
        {
            "id": "quality_iqr",
            "title": "Interquartile Range (IQR) Outlier Detection",
            "keywords": ["iqr", "outlier", "anomaly", "quartile", "bounds", "tukey", "box plot"],
            "content": (
                "The IQR method defines outliers as values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR], where IQR = Q3 - Q1. "
                "Unlike Z-scores, IQR is non-parametric and resistant to extreme values. "
                "Handling strategies include trimming (dropping) or capping/Winsorization (clipping to bounds)."
            ),
        },
        {
            "id": "stat_skewness",
            "title": "Skewness and Distribution Asymmetry",
            "keywords": ["skewness", "skew", "asymmetry", "tail", "normal distribution", "log transform"],
            "content": (
                "Skewness quantifies distribution asymmetry. Skewness between -0.5 and +0.5 is fairly symmetrical. "
                "Skewness > +1.0 indicates strong positive (right-tailed) skew, where mean > median. "
                "Highly skewed features in regression often benefit from log, square root, or Box-Cox power transformations."
            ),
        },
        {
            "id": "ml_r2_score",
            "title": "Coefficient of Determination (R² Score)",
            "keywords": ["r2", "r-squared", "regression", "fit", "variance explained", "model score"],
            "content": (
                "R² represents the proportion of variance in the dependent variable explained by independent features. "
                "1.0 indicates perfect prediction, 0.0 indicates predicting the mean value, and negative values indicate "
                "worse performance than the simple mean."
            ),
        },
        {
            "id": "ml_classification_metrics",
            "title": "Precision, Recall, and F1-Score",
            "keywords": ["precision", "recall", "f1", "f1-score", "classification", "confusion matrix", "imbalance"],
            "content": (
                "Precision is TP / (TP + FP) (accuracy of positive predictions). "
                "Recall is TP / (TP + FN) (proportion of actual positives identified). "
                "F1-score is the harmonic mean: 2 * (Precision * Recall) / (Precision + Recall), ideal for imbalanced class distributions."
            ),
        },
        {
            "id": "ml_clustering_silhouette",
            "title": "Silhouette Score in Clustering",
            "keywords": ["silhouette", "kmeans", "clustering", "k-means", "cluster quality", "unsupervised"],
            "content": (
                "Silhouette score measures how similar an object is to its own cluster compared to other clusters. "
                "Values range from -1 to +1. Scores > 0.5 suggest solid cluster separation; scores near 0 indicate overlapping clusters."
            ),
        },
        {
            "id": "finance_debt_equity",
            "title": "Debt-to-Equity Ratio",
            "keywords": ["debt", "equity", "financial", "leverage", "ratio", "solvency"],
            "content": (
                "Debt-to-Equity (D/E) ratio = Total Liabilities / Shareholder Equity. "
                "It measures financial leverage and risk. A high D/E indicates aggressive growth financed by debt, "
                "increasing vulnerability during economic downturns."
            ),
        },
    ]

    @classmethod
    def search(cls, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Searches indexed documents using token overlap and keyword relevance."""
        q_tokens = set(re.findall(r"\w+", query.lower()))
        scored_docs = []

        for doc in cls.KNOWLEDGE_DOCUMENTS:
            score = 0
            # Check keywords match
            for kw in doc["keywords"]:
                if kw in query.lower():
                    score += 5
                elif any(tok in kw for tok in q_tokens):
                    score += 2
            
            # Check content token overlap
            content_tokens = set(re.findall(r"\w+", doc["content"].lower()))
            overlap = len(q_tokens.intersection(content_tokens))
            score += overlap

            if score > 0:
                scored_docs.append({"doc": doc, "score": score})

        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        results = [item["doc"] for item in scored_docs[:top_k]]
        app_logger.info(f"RAG search for '{query}' returned {len(results)} relevant documents.")
        return results
