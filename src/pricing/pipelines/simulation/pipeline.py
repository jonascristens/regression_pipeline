"""
This is a boilerplate pipeline 'simulation'
generated using Kedro 0.19.11
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import (
    simulate_discount_effect,
    find_optimal_discount,
    plot_simulation_results,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                simulate_discount_effect,
                inputs=["df", "fitted_pipeline", "params:discount"],
                outputs="discount_simulation",
                name="compute_price_dependent_cols_node",
            ),
            node(
                find_optimal_discount,
                inputs=["discount_simulation", "params:optimize"],
                outputs="optimal_discount",
                name="update_financial_metrics_node",
            ),
            node(
                plot_simulation_results,
                inputs=[
                    "discount_simulation",
                    "params:plot_results",
                ],
                outputs="simulation_results",
                name="plot_simulation_results_node",
            ),
        ],
        namespace="simulation",
        inputs={
            "df": "modelling.df_test",
            "fitted_pipeline": "modelling.fitted_pipeline",
        },
    )
