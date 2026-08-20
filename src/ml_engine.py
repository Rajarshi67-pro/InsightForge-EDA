"""
Machine Learning Engine for InsightForge AI
Supports automated and user-guided Classification, Regression, and Clustering workflows.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    silhouette_score,
)
import plotly.express as px
import plotly.graph_objects as go


class MLEngine:
    """
    Handles preprocessing, automated model selection, training, evaluation, and feature importance.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def detect_problem_type(self, target_col: str) -> str:
        """Determines whether the target is suitable for Classification or Regression."""
        series = self.df[target_col].dropna()
        if pd.api.types.is_numeric_dtype(series):
            unique_count = series.nunique()
            if unique_count <= 10 and series.dtype in [np.int64, np.int32, int]:
                return "Classification"
            return "Regression"
        else:
            return "Classification"

    def _preprocess(
        self,
        target_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str], Optional[LabelEncoder]]:
        """Cleans, encodes, and scales features and target."""
        df_clean = self.df.copy()

        # Drop identifiers
        id_cols = [c for c in df_clean.columns if str(c).lower().endswith("id") or "_id" in str(c).lower()]
        df_clean = df_clean.drop(columns=[c for c in id_cols if c != target_col], errors="ignore")

        if feature_cols:
            selected_features = [f for f in feature_cols if f in df_clean.columns and f != target_col]
        else:
            selected_features = [c for c in df_clean.columns if c != target_col]

        # Handle missing values in features
        for col in selected_features:
            if df_clean[col].isna().sum() > 0:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    mode_val = df_clean[col].mode()
                    fill = mode_val[0] if len(mode_val) > 0 else "Missing"
                    df_clean[col] = df_clean[col].fillna(fill)

        # One-hot encode categorical features
        X_df = pd.get_dummies(df_clean[selected_features], drop_first=True)
        feature_names = list(X_df.columns)

        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X_df)

        y = None
        target_encoder = None
        if target_col is not None:
            y_series = df_clean[target_col].copy()
            # If missing target, drop those rows from X and y
            valid_mask = y_series.notna()
            X = X[valid_mask]
            y_series = y_series[valid_mask]

            if not pd.api.types.is_numeric_dtype(y_series):
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y_series.astype(str))
            else:
                y = y_series.values

        return X, y, feature_names, target_encoder

    def train_regression(
        self,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        model_type: str = "Random Forest",
    ) -> Dict[str, Any]:
        """Trains a regression model and computes performance metrics."""
        X, y, feature_names, _ = self._preprocess(target_col, feature_cols)
        if len(X) < 5:
            return {"error": "Insufficient data points for training regression model."}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_type == "Linear Regression":
            model = LinearRegression()
            model.fit(X_train, y_train)
            importances = np.abs(model.coef_)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            importances = model.feature_importances_

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        # Feature importance ranking
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values(by="Importance", ascending=False).head(10)

        # Actual vs Predicted figure
        fig_pred = px.scatter(
            x=y_test,
            y=y_pred,
            labels={"x": "Actual", "y": "Predicted"},
            title=f"Actual vs Predicted: {target_col}",
            template="plotly_white",
        )
        fig_pred.add_trace(go.Scatter(
            x=[min(y_test), max(y_test)],
            y=[min(y_test), max(y_test)],
            mode="lines",
            name="Ideal Fit",
            line=dict(color="red", dash="dash"),
        ))

        fig_imp = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Feature Importances",
            color="Importance",
            color_continuous_scale="Blues",
            template="plotly_white",
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed"))

        return {
            "task": "Regression",
            "model_name": model_type,
            "target": target_col,
            "metrics": {
                "R2_Score": round(float(r2), 3),
                "RMSE": round(float(rmse), 3),
                "MAE": round(float(mae), 3),
                "MSE": round(float(mse), 3),
            },
            "feature_importance": importance_df.to_dict(orient="records"),
            "fig_prediction": fig_pred,
            "fig_importance": fig_imp,
        }

    def train_classification(
        self,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        model_type: str = "Random Forest",
    ) -> Dict[str, Any]:
        """Trains a classification model and computes metrics and confusion matrix."""
        X, y, feature_names, target_encoder = self._preprocess(target_col, feature_cols)
        if len(X) < 5:
            return {"error": "Insufficient data points for training classification model."}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            importances = np.mean(np.abs(model.coef_), axis=0) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            importances = model.feature_importances_

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        class_labels = [str(c) for c in (target_encoder.classes_ if target_encoder else np.unique(y))]
        num_classes = len(class_labels)
        cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))

        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=class_labels,
            y=class_labels,
            labels=dict(x="Predicted Class", y="Actual Class"),
            title=f"Confusion Matrix ({model_type})",
            color_continuous_scale="Blues",
            template="plotly_white",
        )

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values(by="Importance", ascending=False).head(10)

        fig_imp = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Feature Importances",
            color="Importance",
            color_continuous_scale="Purples",
            template="plotly_white",
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed"))

        return {
            "task": "Classification",
            "model_name": model_type,
            "target": target_col,
            "classes": class_labels,
            "metrics": {
                "Accuracy": round(float(acc) * 100, 2),
                "Precision": round(float(prec) * 100, 2),
                "Recall": round(float(rec) * 100, 2),
                "F1_Score": round(float(f1) * 100, 2),
            },
            "confusion_matrix": cm.tolist(),
            "feature_importance": importance_df.to_dict(orient="records"),
            "fig_confusion_matrix": fig_cm,
            "fig_importance": fig_imp,
        }

    def train_clustering(self, n_clusters: int = 3, feature_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Performs K-Means clustering and analyzes cluster profiles."""
        X, _, feature_names, _ = self._preprocess(target_col=None, feature_cols=feature_cols)
        if len(X) < n_clusters:
            return {"error": "Not enough samples to cluster with specified cluster count."}

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        sil_score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0.0

        # Cluster breakdown
        df_clusters = self.df.copy()
        df_clusters["Cluster"] = [f"Cluster {l}" for l in labels]

        fig_scatter = None
        num_cols = [c for c in df_clusters.select_dtypes(include=[np.number]).columns if not str(c).lower().endswith("id")]
        if len(num_cols) >= 2:
            fig_scatter = px.scatter(
                df_clusters,
                x=num_cols[0],
                y=num_cols[1],
                color="Cluster",
                title=f"K-Means Clusters ({num_cols[0]} vs {num_cols[1]})",
                template="plotly_white",
            )

        return {
            "task": "Clustering",
            "k": n_clusters,
            "silhouette_score": round(float(sil_score), 3),
            "cluster_counts": df_clusters["Cluster"].value_counts().to_dict(),
            "fig_clusters": fig_scatter,
        }
