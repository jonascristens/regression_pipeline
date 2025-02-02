"""This is a boilerplate pipeline 'inference'
generated using Kedro 0.19.8
"""

# from sklearn.tree import DecisionTreeClassifier

import logging

import pandas as pd
from sklearn.pipeline import Pipeline

from pricing.utils import expand_args

logger = logging.getLogger(__name__)


@expand_args
def predict(
    df: pd.DataFrame, fitted_pipeline: Pipeline, pred_col: str, pred_proba_col: str
) -> pd.DataFrame:
    """Predicts the target label and cluster for each row in the given DataFrame using the provided pipeline.

    Args:
        pipeline (Pipeline): A scikit-learn pipeline object that includes the necessary
        preprocessing steps and the model.
        df (pd.DataFrame): The input DataFrame containing the data to be predicted.
        inference_params (Dict[str, Any]): A dictionary containing the following keys:
            - 'cluster_col': The name of the column to store the node identifier for
            each row.
            - 'predicted_target_col': The name of the column to store the predicted
            target label for each row.

    Returns:
        pd.DataFrame: The input DataFrame with additional columns for the node
        identifier and the predicted target label.
    """
    # determine the node identifier for each row in the dataset

    # predict the target label for each row in the dataset
    df[pred_col] = fitted_pipeline.predict(df)
    df[pred_proba_col] = fitted_pipeline.predict_proba(df)[:, 1]

    return df
