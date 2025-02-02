"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.11
"""

from kedro.pipeline import Pipeline, node, pipeline  # noqa

from .nodes import convert_dtype_columns, standardize_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                standardize_dataset,
                inputs=["raw_dataset", "params:standardize"],
                outputs="standardized_dataset",
                name="standardize_dataset_node",
            ),
            node(
                func=convert_dtype_columns,
                inputs=[
                    "standardized_dataset",
                    "params:dtypes",
                ],
                outputs="master_dataset",
                name="convert_dtype_columns_node",
            ),
        ],
        namespace="preprocessing",
    )
