import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dython.nominal import associations
from IPython.display import display
from scipy.stats import chi2_contingency, f_oneway


# Data Overview Function
def data_overview(df: pd.DataFrame) -> None:
    """Provides a high-level overview of the dataset."""
    display("--- Data Overview ---")
    display(f"Number of rows: {df.shape[0]}")
    display(f"Number of columns: {df.shape[1]}")
    # display(f"Date range: {df['act_date'].nunique()}")
    display("Data Types:", df.dtypes)
    display("Summary Statistics for Numerical Variables:", df.describe().transpose())
    display(
        "Summary Statistics for Categorical Variables:",
        df.describe(include=["object", "category"]).transpose(),
    )


# Data Quality Checks
def data_quality_summary(df: pd.DataFrame) -> None:
    """displays data dimensions, types, and summary of missing values."""
    display(f"Dataset contains {df.shape[0]} observations and {df.shape[1]} features.")
    display("Data types:")
    display(df.dtypes)
    display("Missing values per column:")
    missing_values = df.isnull().sum()
    display(missing_values[missing_values > 0])
    display("Describe pandas DataFrame:")
    display(df.describe(include="all").T)


# Univariate Analysis
def univariate_analysis(
    df: pd.DataFrame,
    numerical_vars: list | None = None,
    categorical_vars: list | None = None,
) -> None:
    """Conducts univariate analysis for numerical and categorical variables."""
    if numerical_vars is None:
        numerical_vars = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if categorical_vars is None:
        categorical_vars = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

    # Plotting distributions for numerical variables
    for col in numerical_vars:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        # plt.show()

    # Plotting frequencies for categorical variables
    for col in categorical_vars:
        plt.figure(figsize=(8, 4))
        sns.countplot(y=df[col])
        plt.title(f"Frequency of {col}")
        # plt.show()


# Bivariate Analysis with Target Variable
def bivariate_analysis(
    df: pd.DataFrame,
    target_var: str,
    numerical_vars: list | None = None,
    categorical_vars: list | None = None,
) -> None:
    """Conducts bivariate analysis with the target variable."""
    if numerical_vars is None:
        numerical_vars = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if categorical_vars is None:
        categorical_vars = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

    if df[target_var].dtype in ["int64", "float64"]:
        for col in numerical_vars:
            if col != target_var:
                plt.figure(figsize=(10, 4))
                sns.scatterplot(x=df[col], y=df[target_var])
                plt.title(f"{col} vs. {target_var}")
                # plt.show()
                display(f"""Correlation between {col} and {target_var}:
                        {df[[col, target_var]].corr().iloc[0, 1]:.2f}""")

        for col in categorical_vars:
            plt.figure(figsize=(10, 4))
            sns.boxplot(x=df[col], y=df[target_var])
            plt.xticks(rotation=90)
            plt.title(f"{col} vs. {target_var}")
            plt.show()
            groups = [
                df[df[col] == cat][target_var].dropna() for cat in df[col].unique()
            ]
            if len(groups) > 1:
                anova_result = f_oneway(*groups)
                display(f"""ANOVA test for {col} vs. {target_var}:
                        F={anova_result.statistic:.2f},
                        p-value={anova_result.pvalue:.4f}""")

    elif df[target_var].dtype in ["object", "category", "bool"]:
        for col in numerical_vars:
            plt.figure(figsize=(10, 4))
            sns.boxplot(x=df[target_var], y=df[col])
            plt.xticks(rotation=90)
            plt.title(f"{col} vs. {target_var}")
            # plt.show()

            groups = [
                df[df[target_var] == cat][col].dropna()
                for cat in df[target_var].unique()
            ]
            if len(groups) > 1:
                anova_result = f_oneway(*groups)
                display(f"""ANOVA test for {col} vs. {target_var}:
                        F={anova_result.statistic:.2f},
                        p-value={anova_result.pvalue:.4f}""")

        for col in categorical_vars:
            if col != target_var:
                contingency_table = pd.crosstab(df[col], df[target_var])
                display(f"Contingency Table for {col} vs. {target_var}:")
                display(contingency_table)
                chi2_result = chi2_contingency(contingency_table)
                display(f"""Chi-Square test for {col} vs. {target_var}:
                        Chi2={chi2_result[0]:.2f},
                        p-value={chi2_result[1]:.4f}""")
                plt.figure(figsize=(10, 4))
                sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu")
                plt.title(f"Contingency Table Heatmap for {col} vs. {target_var}")
                # plt.show()


# Main function to run the EDA pipeline
def generate_eda_report(
    df: pd.DataFrame,
    target_var: str,
    numerical_vars: list | None = None,
    categorical_vars: list | None = None,
    include_data_overview: bool = True,
    include_data_quality_summary: bool = True,
    include_univariate_analysis: bool = True,
    include_association_analysis: bool = True,
    include_bivariate_analysis: bool = True,
):
    """
    Generates an exploratory data analysis (EDA) report for the given DataFrame.
    Args:
        df (pandas.DataFrame): The DataFrame containing the data to analyze.
        target_var (str): The name of the target variable for bivariate analysis.
        numerical_vars (list of str, optional): List of numerical variable names.
            Defaults to None.
        categorical_vars (list of str, optional): List of categorical variable names.
            Defaults to None.
    Returns:
        None
    This function performs the following steps:
        1. Data Overview: Provides an overview of the data.
        2. Data Quality Summary: Displays a summary of data quality.
        3. Univariate Analysis: Conducts univariate analysis.
        4. Association Analysis: Computes and plots the association matrix.
        5. Bivariate Analysis with Target: Performs bivariate analysis with the target.
    """

    # Data Overview
    if include_data_overview:
        data_overview(df)

    # Data Quality Summary
    if include_data_quality_summary:
        display("--- Data Quality Summary ---")
        data_quality_summary(df)

    # Univariate Analysis
    if include_univariate_analysis:
        display("--- Univariate Analysis ---")
        univariate_analysis(df, numerical_vars, categorical_vars)

    # Compute and Plot Association Matrix
    if include_association_analysis:
        display("--- Association Analysis ---")
        associations(df)

    # Bivariate Analysis with Target
    if include_bivariate_analysis:
        display("--- Bivariate Analysis with Target Variable ---")
        bivariate_analysis(df, target_var, numerical_vars, categorical_vars)


def _get_conversion_rate(df, start, end, target_col, date_col):
    return df[(df[date_col] >= start) & (df[date_col] <= end)][target_col].mean()


def plot_train_test_split(
    df: pd.DataFrame,
    target_col: str,
    splitting_dates: dict[str, str],
    date_col: str = "created_date_off",
    country: str = "all",
) -> None:
    """
    Plots the conversion rate by month and marks the train-test split dates.
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        target_col (str): The name of the target column to plot.
        splitting_dates (dict[str, str]): A dictionary containing the train and
            test split dates.
            Expected keys are "train_start", "train_end", "test_start", and "test_end".
        date_col (str, optional): The name of the date column in the dataframe.
            Defaults to "created_date_off".
        country (str, optional): The country name to include in the plot title.
            Defaults to "all".
    Returns:
        None
    """

    ax = (
        df.groupby(df[date_col].dt.to_period("M"))[target_col]
        .mean()
        .plot(
            title="Conversion rate by month for "
            f"country {' '.join(country.split('_')).title()}",
            ylabel="Conversion rate",
            xlabel="Month-year",
        )
    )
    ax.axvline(
        pd.to_datetime(splitting_dates["train_start"]).to_period("M"),
        color="g",
        linestyle="--",
        label="Train start",
        alpha=0.5,
    )
    ax.axvline(
        pd.to_datetime(splitting_dates["train_end"]).to_period("M"),
        color="r",
        linestyle="--",
        label="Train end",
        alpha=0.5,
    )
    ax.axvline(
        pd.to_datetime(splitting_dates["test_start"]).to_period("M"),
        color="g",
        linestyle=":",
        label="Test start",
        alpha=0.5,
    )
    ax.axvline(
        pd.to_datetime(splitting_dates["test_end"]).to_period("M"),
        color="r",
        linestyle=":",
        label="Test end",
        alpha=0.5,
    )
    plt.legend()

    avg_conversion_rate = pd.Series()

    avg_conversion_rate["train_set"] = _get_conversion_rate(
        df,
        splitting_dates["train_start"],
        splitting_dates["train_end"],
        target_col,
        date_col,
    )
    avg_conversion_rate["test_set"] = _get_conversion_rate(
        df,
        splitting_dates["test_start"],
        splitting_dates["test_end"],
        target_col,
        date_col,
    )

    display("Average conversion rate for train and test sets:")
    display(avg_conversion_rate)
