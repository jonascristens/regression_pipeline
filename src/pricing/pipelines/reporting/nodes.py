"""
This is a boilerplate pipeline 'reporting'
generated using Kedro 0.19.11
"""

import logging
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from dython.model_utils import ks_abc, metric_graph
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from pricing.utils import expand_args

logger = logging.getLogger(__name__)

matplotlib.use("Agg")


@expand_args
def predict(X_test: pd.DataFrame, fitted_pipeline: Pipeline) -> pd.DataFrame:
    """
    Generate predictions and prediction probabilities using a fitted pipeline.

    Args:
        X_test (pd.DataFrame): The test data to predict on.
        fitted_pipeline (Pipeline): The fitted machine learning pipeline.

    Returns:
        pd.DataFrame: A DataFrame containing the predictions and the prediction
            probabilities.
    """

    return fitted_pipeline.predict(X_test), fitted_pipeline.predict_proba(X_test)[:, 1]


@expand_args
def evaluate_model_metrics(
    y_test: pd.Series, y_pred: pd.Series, y_pred_proba: pd.Series
) -> dict[str, Any]:
    """
    Evaluate and return various performance metrics for a classification model.
    Args:
        y_test (pd.Series): True labels of the test dataset.
        y_pred (pd.Series): Predicted labels by the model.
        y_pred_proba (pd.Series): Predicted probabilities by the model.
    Returns:
        dict[str, Any]: A dictionary containing the following metrics:
            - "AUC": Area Under the ROC Curve.
            - "Precision": Precision score.
            - "Recall": Recall score.
            - "F1-Score": F1 score.
            - "Log-Loss": Logarithmic loss.
            - "Accuracy": Accuracy score.
            - "KS-abc": Kolmogorov-Smirnov area between curves.
    """

    report = {
        "AUC": roc_auc_score(y_test, y_pred_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "Log-Loss": log_loss(y_test, y_pred_proba),
        "Accuracy": accuracy_score(y_test, y_pred),
        "KS-abc": ks_abc(y_test, y_pred_proba)["abc"],
    }
    return report


@expand_args
def plot_roc_auc(y_test: pd.Series, y_pred_proba: pd.Series):
    """
    Plot the ROC AUC curve using the true labels and predicted probabilities.

    Args:
        y_test (pd.Series): True labels of the test dataset.
        y_pred_proba (pd.Series): Predicted probabilities for the positive class.

    Returns:
        matplotlib.figure.Figure: The figure object containing the ROC AUC plot.
    """
    return metric_graph(y_test, y_pred_proba, "roc")["ax"].get_figure()


@expand_args
def plot_precision_recall(y_test: pd.Series, y_pred_proba: pd.Series):
    """Plot the Precision-Recall curve."""
    return metric_graph(y_test, y_pred_proba, "pr")["ax"].get_figure()


@expand_args
def plot_confusion_matrix(y_test: pd.Series, y_pred: pd.Series):
    """
    Plots the confusion matrix for the given true and predicted labels.

    Args:
        y_test (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.

    Returns:
        matplotlib.figure.Figure: The figure object containing the confusion
            matrix plot.
    """

    return ConfusionMatrixDisplay.from_predictions(y_test, y_pred).figure_


@expand_args
def plot_calibration_curve(y_test: pd.Series, y_pred_proba: pd.Series):
    """
    Plots the calibration curve for the given true labels and predicted probabilities.

    Args:
        y_test (pd.Series): True labels.
        y_pred_proba (pd.Series): Predicted probabilities for the positive class.
    Returns:
        matplotlib.figure.Figure: The figure object containing the calibration
            curve plot.
    """
    return CalibrationDisplay.from_predictions(y_test, y_pred_proba).figure_


@expand_args
def plot_ks_abc(y_test: pd.Series, y_pred_proba: pd.Series) -> plt.Figure:
    """
    Plots the Kolmogorov-Smirnov statistic for binary classification.

    Args:
        y_test (pd.Series): True labels.
        y_pred (pd.Series): Predicted probabilities for the positive class.

    Returns:
        matplotlib.figure.Figure: The figure object containing the KS plot.
    """
    return ks_abc(y_test, y_pred_proba)["ax"].get_figure()


@expand_args
def plot_actual_vs_expected(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    benchmark: pd.Series | None = None,
    exposure: pd.Series | None = None,
    q=10,
):
    """
    Plots the actual vs expected values for each feature in the test dataset.

    Args:
        X_test (pd.DataFrame): The test dataset features.
        y_test (pd.Series): The actual target values.
        y_pred (pd.Series): The predicted target values.
        benchmark (pd.Series, optional): The benchmark target values for comparison.
            Defaults to None.
        exposure (pd.Series, optional): The exposure values for weighting.
            Defaults to None.
        q (int, optional): The number of quantiles for binning numerical features.
            Defaults to 10.

    Returns:
        dict: A dictionary where keys are filenames and values are matplotlib figures.
    """
    if exposure is None:
        exposure = np.ones(len(X_test))

    plots_dict = {}
    # Iterate over all feature columns
    for feature in X_test.columns:
        temp_df = X_test[[feature]].copy()
        temp_df["Observed"] = y_test
        temp_df["Predicted"] = y_pred
        temp_df["Exposure"] = exposure

        if benchmark is not None:
            temp_df["Benchmark"] = benchmark

        # Handle numerical vs categorical features
        if pd.api.types.is_numeric_dtype(temp_df[feature]):
            temp_df["Feature_Binned"] = pd.qcut(
                temp_df[feature], q=q, duplicates="drop", labels=False
            )
        else:
            temp_df["Feature_Binned"] = temp_df[feature]  # Keep categorical as-is

        # Aggregate with weighted means
        agg_dict = {
            "Observed": lambda x: np.average(
                x,
                weights=temp_df.loc[x.index, "Exposure"],  # noqa: B023
            ),
            "Predicted": lambda x: np.average(
                x,
                weights=temp_df.loc[x.index, "Exposure"],  # noqa: B023
            ),
            "Exposure": "sum",
        }

        if benchmark is not None:
            agg_dict["Benchmark"] = lambda x: np.average(
                x,
                weights=temp_df.loc[x.index, "Exposure"],  # noqa: B023
            )

        df_grouped = (
            temp_df.groupby("Feature_Binned", observed=True).agg(agg_dict).reset_index()
        )

        # Compute weighted difference
        df_grouped["Weighted_Diff"] = df_grouped["Predicted"] - df_grouped["Observed"]

        # --- Seaborn Plotting ---
        fig, axes = plt.subplots(
            2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
        )

        # Top Panel: Line Plot
        sns.lineplot(
            x="Feature_Binned",
            y="Observed",
            data=df_grouped,
            marker="s",
            color="red",
            label="Observed avg vs exposure",
            ax=axes[0],
        )
        sns.lineplot(
            x="Feature_Binned",
            y="Predicted",
            data=df_grouped,
            marker="s",
            color="teal",
            label="Predicted avg vs exposure",
            ax=axes[0],
        )

        if benchmark is not None:
            sns.lineplot(
                x="Feature_Binned",
                y="Benchmark",
                data=df_grouped,
                marker="s",
                color="purple",
                label="Benchmark avg vs exposure",
                ax=axes[0],
            )

        # Bar Plot for Weighted Difference
        sns.barplot(
            x="Feature_Binned",
            y="Weighted_Diff",
            data=df_grouped,
            color="orange",
            alpha=0.5,
            ax=axes[0],
        )

        # Labels & Grid
        axes[0].set_ylabel("Mean weighted by exposure")
        axes[0].legend()
        axes[0].grid(axis="y", linestyle="--", alpha=0.7)

        # Bottom Panel: Bar Plot for Exposure
        sns.barplot(
            x="Feature_Binned",
            y="Exposure",
            data=df_grouped,
            color="orange",
            edgecolor="gray",
            ax=axes[1],
        )

        # Labels & Grid
        axes[1].set_ylabel("Sum of Exposure")
        axes[1].set_xlabel(f"Feature: {feature} (Numerical Quantiles + Categorical)")
        axes[1].grid(axis="y", linestyle="--", alpha=0.7)
        # Ensure the number of ticks matches the labels
        axes[1].set_xticks(
            range(len(df_grouped["Feature_Binned"]))
        )  # Set fixed tick positions
        axes[1].set_xticklabels(
            df_grouped["Feature_Binned"], rotation=45
        )  # Apply labels

        # Layout
        plt.tight_layout()
        plots_dict[f"{feature}_avse_plot.png"] = fig

    return plots_dict


@expand_args
def generate_shap_values(fitted_pipeline, X_test) -> np.ndarray:
    """
    Generate SHAP values for a given fitted pipeline and test data.

    Args:
        fitted_pipeline (Pipeline): A fitted scikit-learn pipeline object.
        X_test (pd.DataFrame or np.ndarray): The test data to explain.

    Returns:
        np.ndarray: SHAP values for the test data.
    """

    # Create SHAP explainer
    if len(fitted_pipeline) > 1:
        X_test = fitted_pipeline[:-1].transform(X_test)
    explainer = shap.Explainer(fitted_pipeline[-1])
    shap_values = explainer(X_test)
    return shap_values


@expand_args
def plot_shap_feature_importance(shap_values, X_test, feature_names=None) -> plt.Figure:
    """
    Plots the SHAP feature importance.

    Args:
        shap_values (shap.Explanation or numpy.ndarray): SHAP values for the features.
        X_test (pandas.DataFrame or numpy.ndarray): Test dataset used to extract
            feature names if not provided.
        feature_names (list of str, optional): list of feature names. If None,
            feature names are extracted from X_test if it is a DataFrame.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object containing the SHAP
            feature importance plot.
    """

    # Ensure feature names are extracted if X_test is a DataFrame
    if feature_names is None and hasattr(X_test, "columns"):
        feature_names = X_test.columns.tolist()

    # Create feature importance plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax = shap.plots.bar(shap_values, ax=ax, show=False)
    ax.set_title("SHAP Feature Importance Plot")
    plt.tight_layout()
    return fig


@expand_args
def plot_shap_beeswarm(shap_values, X_test, feature_names=None) -> plt.Figure:
    """
    Generates a SHAP beeswarm plot for the given SHAP values and test dataset.

    Args:
        shap_values (shap.Explanation or numpy.ndarray): SHAP values for the test
            dataset.
        X_test (pandas.DataFrame or numpy.ndarray): Test dataset used to compute SHAP
            values.
        feature_names (list of str, optional): list of feature names. If None,
            feature names will be extracted from X_test if it is a DataFrame.

    Returns:
        matplotlib.figure.Figure: The generated SHAP beeswarm plot as a Matplotlib
            Figure object.
    """

    # Ensure feature names are extracted if X_test is a DataFrame
    if feature_names is None and hasattr(X_test, "columns"):
        feature_names = X_test.columns.tolist()

    # Create beeswarm plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax = shap.plots.beeswarm(shap_values, show=False)
    ax.set_title("SHAP Beeswarm Plot")
    plt.tight_layout()
    return fig


@expand_args
def plot_sklearn_pdp(
    fitted_pipeline: Pipeline, X_test: pd.DataFrame, features: list | None = None
):
    """
    Generates partial dependence plots for specified features using a fitted
        scikit-learn pipeline.
    Args:
        fitted_pipeline (Pipeline): A fitted scikit-learn pipeline.
        X_test (pd.DataFrame): The test dataset containing features for which
            partial dependence plots are to be generated.
        features (list | None, optional): list of feature names for which to generate
            partial dependence plots. If None, numeric features from X_test
            will be used. Defaults to None.
    Returns:
        dict: A dictionary where keys are filenames of the plots and values
            are the corresponding matplotlib figure objects.
    """

    # Ensure feature names are extracted if X_test is a DataFrame
    if features is None and hasattr(X_test, "columns"):
        features = X_test.select_dtypes("number").columns.tolist()

    plots_dict = {}
    for feature in features:  # type: ignore
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        PartialDependenceDisplay.from_estimator(
            fitted_pipeline,
            X_test,
            features=[feature],
            ax=ax,
        )

        ax.set_title(f"Partial Dependence: {feature}")
        plt.tight_layout()
        plots_dict[f"{feature}_pdp_plot.png"] = fig

    return plots_dict
