"""This is a boilerplate pipeline 'inference'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import predict


def create_pipeline(**kwargs) -> Pipeline:
    """Inference pipeline."""
    return pipeline(
        [
            node(
                predict,
                inputs=["master_dataset", "fitted_pipeline", "params:predict"],
                outputs="inference_predictions",
                name="predict_node",
            ),
        ],
        namespace="inference",
        inputs={
            "fitted_pipeline": "modelling.fitted_pipeline",
            "master_dataset": "preprocessing.master_dataset",
        },
    )
