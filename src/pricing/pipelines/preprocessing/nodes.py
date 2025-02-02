"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.11
"""

import logging
from typing import Dict, List

import pandas as pd

from pricing.utils import expand_args

logger = logging.getLogger(__name__)


@expand_args
def standardize_dataset(
    df: pd.DataFrame, rename_columns: Dict[str, str]
) -> pd.DataFrame:
    """
    Standardizes the dataset by renaming columns.

    Args:
        df (pd.DataFrame): The input DataFrame to be standardized.
        rename_columns (Dict[str, str]): A dictionary mapping old column names to new column names.

    Returns:
        pd.DataFrame: The standardized DataFrame with renamed columns.
    """
    df = df.rename(columns=rename_columns)
    return df


@expand_args
def convert_dtype_columns(
    df: pd.DataFrame,
    to_numeric: List[str],
    to_category: List[str],
    to_datetime: List[str],
) -> pd.DataFrame:
    """
    Convert the data types of specified columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame whose columns need to be converted.
    to_numeric (List[str]): List of column names to be converted to numeric type.
    to_category (List[str]): List of column names to be converted to category type.
    to_datetime (List[str]): List of column names to be converted to datetime type.

    Returns:
    pd.DataFrame: The DataFrame with the specified columns converted to the desired data types.
    """

    # TODO Check with Roman if we want this a hard check - I suggest so
    # columns = df.columns
    # if missing_num_cols := set(to_numeric) - set(columns):
    #     logging.warning(f"The following numeric columns are missing in the dataset: {', '.join(missing_num_cols)}")
    # to_numeric = list(set(columns) & set(to_numeric))

    # if missing_cat_cols := set(to_category) - set(columns):
    #     logging.warning(f"The following categorical columns are missing in the dataset: {', '.join(missing_cat_cols)}")
    # to_category = list(set(columns) & set(to_category))

    # if missing_cat_cols := set(to_category) - set(columns):
    #     logging.warning(f"The following datetime columns are missing in the dataset: {', '.join(missing_cat_cols)}")
    # to_datetime = list(set(columns) & set(to_datetime))

    df[to_numeric] = df[to_numeric].apply(pd.to_numeric)

    for column in to_category:
        if column in df.columns:
            df[column] = df[column].astype("category")

    for column in to_datetime:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column])
    return df
