"""This is a boilerplate pipeline 'modelling'
generated using Kedro 0.19.8
"""

import logging
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, plot_importance, plot_metric
from matplotlib.figure import Figure
from optuna.samplers import TPESampler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import (TimeSeriesSplit, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

from pricing.utils import expand_args

logger = logging.getLogger(__name__)

matplotlib.use("Agg")


@expand_args
def fix_target(
    df: pd.DataFrame, original_target_col: str, target_col: str
) -> pd.DataFrame:
    """
    Transforms the target column in the DataFrame based on the original target column values.

    Args:
        df (pd.DataFrame): The input DataFrame containing the target columns.
        original_target_col (str): The name of the original target column in the DataFrame.
        target_col (str): The name of the new target column to be created in the DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the new target column added.

    Raises:
        ValueError: If an unknown final outcome is encountered in the original target column.
    """

    def _determine_target(x):
        if x == "Accepted":
            return 1
        elif x == "Rejected":
            return 0
        elif x in ["In Progress", np.nan, None]:
            return np.nan
        else:
            raise ValueError(f"Unknown final outcome: {x}")

    df[target_col] = df[original_target_col].apply(_determine_target)
    return df


@expand_args
def filter_dataset(
    df: pd.DataFrame,
    target_col: str,
    organization_id: str | None = None,
    clip_min: int = 0,
    clip_max: int = 99,
    discount_col: str = "discount_shp",
) -> pd.DataFrame:
    """
    Filters the dataset based on specified conditions and removes duplicates.
    Args:
        df (pd.DataFrame): The input DataFrame to be filtered.
        target_col (str): The name of the target column to check for non-null values.
        organization_id (str | None, optional): The organization ID to filter by. Defaults to None.
        clip_min (int, optional): The minimum value for the discount column filter. Defaults to 0.
        clip_max (int, optional): The maximum value for the discount column filter. Defaults to 99.
        discount_col (str, optional): The name of the discount column to apply the filter on. Defaults to "discount_shp".
    Returns:
        pd.DataFrame: The filtered DataFrame with duplicates removed.
    """
    conditions = (
        (df[discount_col] <= clip_max)
        & (df[discount_col] >= clip_min)
        & (df[target_col].notna())
    )

    if organization_id is not None:
        conditions &= df["organization_id"] == organization_id

    df = df[conditions].drop_duplicates()

    return df


@expand_args
def create_train_test_split(
    df: pd.DataFrame,
    date_col: str,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    features: List[str],
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits the given DataFrame into training and testing sets based on the specified date ranges.
    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        date_col (str): The name of the column containing date information.
        train_start (str): The start date for the training set.
        train_end (str): The end date for the training set.
        test_start (str): The start date for the testing set.
        test_end (str): The end date for the testing set.
        features (List[str]): A list of column names to be used as features.
        target_col (str): The name of the column to be used as the target variable.
    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: A tuple containing the training features (X_train),
        training target (y_train), testing features (X_test), and testing target (y_test).
    """
    df = df.sort_values(by=date_col)

    df_train = df[
        (df["created_date_off"] >= train_start) & (df["created_date_off"] < train_end)
    ]
    df_test = df[
        (df["created_date_off"] >= test_start) & (df["created_date_off"] < test_end)
    ]

    logger.info(f"ranges: {train_start}, {train_end}, {test_start}, {test_end}")
    logger.info(f"Train data shape: {df_train.shape}")
    logger.info(f"Test data shape: {df_test.shape}")

    # TODO Check with Roman if we want this a hard check - I suggest so
    # features = list(set(features) & set(df.columns))
    # if missing_features := set(features) - set(df.columns):
    # logger.warning(f"The following features are missing in the dataset: {', '.join(missing_features)}")

    X_train = df_train[features]
    y_train = df_train[target_col]
    X_test = df_test[features]
    y_test = df_test[target_col]

    return X_train, y_train, X_test, y_test


@expand_args
def create_modelling_pipeline(
    target_encoding_features: List[str] | None = None,
    target_encoding: Dict[str, str] | None = None,
    ordinal_features: List[str] | None = None,
    ordinal_encoding: Dict[str, str] | None = None,
    other_features: List[str] | None = None,
    model: Dict[str, str] | None = None,
) -> Pipeline:
    """
    Create a modelling pipeline with optional preprocessing steps for categorical and ordinal features.
    Args:
        categorical_features (List[str] | None): List of categorical feature names. Default is None.
        target_encoding (Dict[str, str] | None): Parameters for target encoding. Default is None.
        ordinal_features (List[str] | None): List of ordinal feature names. Default is None.
        ordinal_encoding (Dict[str, str] | None): Parameters for ordinal encoding. Default is None.
        model (Dict[str, str] | None): Parameters for the LGBMClassifier model. Default is None.
    Returns:
        Pipeline: A scikit-learn pipeline with the specified preprocessing and model.
    """
    column_transformers = list()

    if target_encoding_features:
        column_transformers.append(
            (
                TargetEncoder(**target_encoding),
                target_encoding_features,
            )
        )

    if ordinal_features:
        column_transformers.append(
            (
                OrdinalEncoder(**ordinal_encoding),
                ordinal_features,
            )
        )

    if other_features:
        column_transformers.append(("passthrough", other_features))

    preprocessor = make_pipeline(
        make_column_transformer(
            *column_transformers,
            remainder="drop",
        )
    )
    pipeline = make_pipeline(preprocessor, LGBMClassifier(**model))

    if model is None:
        model = {}

    pipeline.set_output(transform="pandas")
    return pipeline


@expand_args
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pipeline: Pipeline,
    cv_conf: Dict[str, Any],
    scoring: str,
    sampler_conf: Dict[str, Any],
    study_conf: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Train a machine learning model using a pipeline and hyperparameter optimization with Optuna.
    Args:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target data.
        pipeline (Pipeline): A scikit-learn pipeline object.
        cv_conf (Dict[str, Any]): Configuration dictionary for cross-validation.
        scoring (str): Scoring metric to evaluate the model.
        sampler_conf (Dict[str, Any]): Configuration dictionary for the Optuna sampler.
        study_conf (Dict[str, Any]): Configuration dictionary for the Optuna study.
    Returns:
        Tuple[Any, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
            - Trained pipeline with the best parameters.
            - Best hyperparameters found by Optuna.
            - Best score achieved by the model.
            - Optuna study object containing the optimization results.
    """

    def objective(trial):
        # TODO update parameters
        params = {
            "lgbmclassifier__learning_rate": trial.suggest_float(
                "lgbmclassifier__learning_rate", 0.01, 0.1, log=True
            ),
            "lgbmclassifier__max_depth": trial.suggest_int(
                "lgbmclassifier__max_depth", 3, 6
            ),
            "lgbmclassifier__n_estimators": trial.suggest_int(
                "lgbmclassifier__n_estimators", 10, 500
            ),
            "lgbmclassifier__min_split_gain": trial.suggest_float(
                "lgbmclassifier__min_split_gain", 0.01, 1.0, log=True
            ),
        }

        # Tune the parameters of the pipeline
        pipeline.set_params(**params)
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=TimeSeriesSplit(**cv_conf),
            n_jobs=-1,
            scoring=scoring,
            error_score="raise",
        )
        # scoring functions in sklearn are by default maximization functions, so we need to negate the score
        return -scores.mean()

    sampler = TPESampler(**sampler_conf)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, **study_conf)

    # Set the best parameters to the pipeline
    pipeline.set_params(**study.best_params)
    pipeline.fit(X_train, y_train)

    return pipeline, study.best_params, study.best_value, study


@expand_args
def plot_feature_importance(fitted_pipeline: Pipeline) -> Figure:
    """
    Plots the feature importance of a fitted machine learning pipeline.

    Args:
        fitted_pipeline (Pipeline): A scikit-learn pipeline object that has been fitted to the data.
                                    The last step of the pipeline should be a model that supports
                                    feature importance plotting.

    Returns:
        Figure: A matplotlib Figure object containing the feature importance plot.
    """

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax = plot_importance(fitted_pipeline[-1], ax=ax, importance_type="gain")
    plt.tight_layout()
    return fig


@expand_args
def plot_training_metric(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    fitted_pipeline: Pipeline,
    test_size: float = 0.3,
) -> Figure:
    """
    Plots the training metric for a given pipeline.
    Args:
        X_train (pd.DataFrame): The training feature data.
        y_train (pd.Series): The training target data.
        fitted_pipeline (Pipeline): The machine learning pipeline that has been fitted.
        test_size (float, optional): The proportion of the data to be used as validation set. Defaults to 0.3.
    Returns:
        Figure: The matplotlib figure object containing the plot of the training metric.
    """
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=test_size, shuffle=False
    )

    # Train model with validation & OOT tracking
    X_train_t = fitted_pipeline[0].transform(X_train)
    X_train_v = fitted_pipeline[0].transform(X_valid)
    eval_set = [(X_train_t, y_train), (X_train_v, y_valid)]

    fitted_pipeline = fitted_pipeline.fit(
        X_train,
        y_train,
        lgbmclassifier__eval_set=eval_set,
        lgbmclassifier__eval_names=["training", "validation"],
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax = plot_metric(fitted_pipeline[-1], ax=ax)
    plt.tight_layout()
    return fig
