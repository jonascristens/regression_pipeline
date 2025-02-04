"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.11
"""

import logging

import pandas as pd

from pricing.utils import expand_args

logger = logging.getLogger(__name__)


@expand_args
def standardize_dataset(
    df: pd.DataFrame,
    rename_columns: dict[str, str],
    keep_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Standardizes the dataset by renaming columns.

    Args:
        df (pd.DataFrame): The input DataFrame to be standardized.
        rename_columns (dict[str, str]): A dictionary mapping old column names
            to new column names.

    Returns:
        pd.DataFrame: The standardized DataFrame with renamed columns.
    """
    df = df.rename(columns=rename_columns)

    if keep_columns:
        df = df[keep_columns]
    return df


@expand_args
def convert_dtype_columns(
    df: pd.DataFrame,
    to_numeric: list[str],
    to_category: list[str],
    to_datetime: list[str],
) -> pd.DataFrame:
    """
    Convert the data types of specified columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame whose columns need to be converted.
    to_numeric (list[str]): list of column names to be converted to numeric type.
    to_category (list[str]): list of column names to be converted to category type.
    to_datetime (list[str]): list of column names to be converted to datetime type.

    Returns:
    pd.DataFrame: The DataFrame with the specified columns converted to the
        desired data types.
    """

    # TODO Check with Roman if we want this a hard check - I suggest so
    # columns = df.columns
    # if missing_num_cols := set(to_numeric) - set(columns):
    #     logging.warning(f"The following numeric columns are missing in the dataset: {', '.join(missing_num_cols)}") # noqa
    # to_numeric = list(set(columns) & set(to_numeric))

    # if missing_cat_cols := set(to_category) - set(columns):
    #     logging.warning(f"The following categorical columns are missing in the dataset: {', '.join(missing_cat_cols)}") # noqa
    # to_category = list(set(columns) & set(to_category))

    # if missing_cat_cols := set(to_category) - set(columns):
    #     logging.warning(f"The following datetime columns are missing in the dataset: {', '.join(missing_cat_cols)}") # noqa
    # to_datetime = list(set(columns) & set(to_datetime))

    df[to_numeric] = df[to_numeric].apply(pd.to_numeric)

    for column in to_category:
        if column in df.columns:
            df[column] = df[column].astype("category")

    for column in to_datetime:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column])
    return df


@expand_args
def clean_dataset(
    df: pd.DataFrame,
    organization_id: str | None = None,
) -> pd.DataFrame:
    """
    Cleans the dataset by removing rows with missing values.

    Args:
        df (pd.DataFrame): The input DataFrame to be cleaned.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """

    if organization_id is not None:
        df = df[df["organization_id"] == organization_id]

    return df.drop_duplicates()
