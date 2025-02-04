"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.11
"""

from kedro.pipeline import Pipeline, node, pipeline  # noqa

from .nodes import convert_dtype_columns, standardize_dataset, clean_dataset


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
                outputs="dtype_converted_dataset",
                name="convert_dtype_columns_node",
            ),
            node(
                func=clean_dataset,
                inputs=["dtype_converted_dataset", "params:clean"],
                outputs="master_dataset",
                name="clean_dataset_node",
            ),
        ],
        namespace="preprocessing",
    )
