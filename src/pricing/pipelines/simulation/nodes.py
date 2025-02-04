"""
This is a boilerplate pipeline 'simulation'
generated using Kedro 0.19.11
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from pricing.utils import expand_args


def get_simulation_cols(min_change, max_change, price_steps, col_prefix):
    return [
        f"{col_prefix}{int(round(((i) * 100), 0)):03d}"
        for i in np.arange(min_change, max_change, price_steps).round(2)
    ]


def compute_price_dependent_cols(
    df: pd.DataFrame,
    discount_new: float,
    discount_col: str,
) -> pd.DataFrame:
    """
    Updates the specified discount column in the DataFrame with a new discount value.

    Args:
        df (pd.DataFrame): The input DataFrame containing pricing data.
        discount_new (float): The new discount value to be applied.
        discount_col (str): The name of the column in the DataFrame to update with
            the new discount value.

    Returns:
        pd.DataFrame: The updated DataFrame with the new discount value
            applied to the specified column.
    """

    df[discount_col] = discount_new

    # Add other dependent columns here when needed

    return df


def update_financial_metrics(
    df: pd.DataFrame,
    discount_new: float,
    pred_prob_col: str,
    revenue_col: str,
) -> pd.DataFrame:
    """
    Update financial metrics with new discount and calculate converted revenue.
    Args:
        df (pd.DataFrame): DataFrame with financial data.
        discount_new (float): New discount rate.
        pred_prob_col (str): Column with predicted probabilities.
        revenue_col (str): Column with revenue values.
    Returns:
        pd.DataFrame: Updated DataFrame with new discount and converted revenue columns.
    """

    # Generate a suffix for the columns based on the price change percentage
    discount_label = f"{int(round((discount_new) * 100, 0)):03d}"

    # Store the updated revenue in a new column with the price change suffix
    discount_col = f"discount_{discount_label}"
    df[discount_col] = discount_new

    # Calculate and store the converted revenue
    converted_revenue_col = f"converted_revenue_{discount_label}"
    df[converted_revenue_col] = (
        (1 - df[discount_col]) * df[pred_prob_col] * df[revenue_col]
    )

    return df


def predict_conversion_probability(
    df: pd.DataFrame,
    fitted_pipeline: Pipeline,
    discount_col: str,
    discount_new: float = 0.00,
    pred_prob_col: str = "pred_prob_000",
    revenue_col: str = "revenue",
) -> pd.DataFrame:
    """
    Predicts the conversion probability for each row in the DataFrame
    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        fitted_pipeline (Pipeline): The fitted machine learning pipeline used for
            prediction.
        discount_col (str): The name of the column containing discount information.
        discount_new (float, optional): The new discount value to be applied. Defaults
            to 0.00.
        pred_prob_col (str, optional): The name of the column to store the predicted
            probabilities. Defaults to "pred_prob_000".
        revenue_col (str, optional): The name of the column containing revenue
            information. Defaults to "revenue".
    Returns:
        pd.DataFrame: The updated DataFrame with predicted probabilities and
            recalculated financial metrics.
    """

    # Create a copy of the DataFrame to avoid modifying the original
    df_ = df.copy()

    # Update any columns that are dependent on the premium
    df_ = compute_price_dependent_cols(df_, discount_new, discount_col=discount_col)

    # Predict churn probability using the fitted pipeline and store
    # in the specified column
    df[pred_prob_col] = fitted_pipeline.predict_proba(df_)[:, 1]

    # Update the DataFrame with financial metrics based on conversion prediction
    df = update_financial_metrics(
        df=df,
        discount_new=discount_new,
        pred_prob_col=pred_prob_col,
        revenue_col=revenue_col,
    )

    return df


@expand_args
def simulate_discount_effect(
    df: pd.DataFrame,
    fitted_pipeline: Pipeline,
    revenue_col: str,
    discount_col: str,
    min_discount: float = 0.0,
    max_discount: float = 1.0,
    price_steps: float = 0.01,
) -> pd.DataFrame:
    """
    Simulates the effect of various discount levels on KPIs.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        fitted_pipeline (Pipeline): The fitted machine learning pipeline
            used for prediction.
        revenue_col (str): The name of the column containing revenue data.
        discount_col (str): The name of the column containing discount data.
        min_discount (float, optional): The minimum discount level to simulate.
            Defaults to 0.0.
        max_discount (float, optional): The maximum discount level to simulate.
            Defaults to 1.0.
        price_steps (float, optional): The step size for discount levels.
            Defaults to 0.01.

    Returns:
        pd.DataFrame: The DataFrame with added columns for predicted conversion
            probabilities at each discount level.
    """

    # Loop over each price change step, computing predictions and sensitivity
    for discount in np.arange(min_discount, max_discount, price_steps):
        discount = round(discount, 2)  # Round to two decimal places
        pred_prob_col = f"pred_prob_{int(round((discount * 100), 0)):03d}"
        df = predict_conversion_probability(
            df,
            fitted_pipeline,
            discount_new=discount,
            discount_col=discount_col,
            pred_prob_col=pred_prob_col,
            revenue_col=revenue_col,
        )

    return df


@expand_args
def find_optimal_discount(
    df: pd.DataFrame,
    min_discount: float,
    max_discount: float,
    price_steps: float,
    col_name_to_optimize: str = "converted_revenue_",
) -> float:
    """
    Finds the optimal discount by calculating the maximum revenue
    within the specified discount range and price steps.
    Args:
        df (pd.DataFrame): The DataFrame containing revenue data.
        min_discount (float): The minimum discount value to consider.
        max_discount (float): The maximum discount value to consider.
        price_steps (float): The step size between discount values.
        col_name_to_optimize (str, optional): The base name of the columns to optimize.
            Defaults to "converted_revenue_".
    Returns:
        pd.DataFrame: The DataFrame with additional columns for the optimized revenue
            and the corresponding column name.
    """

    revenue_cols = get_simulation_cols(
        min_discount, max_discount, price_steps, col_name_to_optimize
    )

    df[f"{col_name_to_optimize}optimized"] = df[revenue_cols].max(axis=1)
    df[f"{col_name_to_optimize}optimized_col"] = df[revenue_cols].idxmax(axis=1)

    return df


@expand_args
def plot_simulation_results(
    df: pd.DataFrame,
    min_discount: float,
    max_discount: float,
    price_steps: float,
    conversion_col_prefix: str = "pred_prob_",
    revenue_col_prefix: str = "converted_revenue_",
) -> pd.DataFrame:
    """
    Plots the mean conversion probability over a range of price changes.

    Args:
        df (pd.DataFrame): DataFrame containing the conversion data.
        min_change (float): Minimum price change percentage.
        max_change (float): Maximum price change percentage.
        price_steps (float): Step size for price changes.
        col_prefix (str, optional): Prefix for the column names.

    Returns:
        pd.DataFrame: A plot of the mean conversion probabilities.
    """

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    conversion_cols = get_simulation_cols(
        min_discount, max_discount, price_steps, conversion_col_prefix
    )

    revenue_cols = get_simulation_cols(
        min_discount, max_discount, price_steps, revenue_col_prefix
    )

    (
        df[conversion_cols]
        .mean()
        .plot(
            ax=axes[0],
            title="Predicted conversion probability relative to the discount change",
        )
    )

    (
        df[revenue_cols]
        .mean()
        .plot(ax=axes[1], title="Predicted revenue relative to the discount change")
    )

    plt.tight_layout()
    return fig
