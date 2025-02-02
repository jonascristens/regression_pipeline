"""This is a boilerplate pipeline 'modellingling'
generated using Kedro 0.19.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (create_modelling_pipeline, create_train_test_split,
                    filter_dataset, fix_target, plot_feature_importance,
                    plot_training_metric, train_model)


def create_pipeline(**kwargs) -> Pipeline:
    """modelling pipeline."""
    return pipeline(
        [
            node(
                func=fix_target,
                inputs=[
                    "master_dataset",
                    "params:target",
                ],
                outputs="fix_target_dataset",
                name="fix_target_node",
            ),
            node(
                func=filter_dataset,
                inputs=["fix_target_dataset", "params:filter_dataset"],
                outputs="modelling_dataset",
                name="filter_dataset_node",
            ),
            node(
                create_train_test_split,
                inputs=["modelling_dataset", "params:split"],
                outputs=[
                    "x_train",
                    "y_train",
                    "x_test",
                    "y_test",
                ],
                name="create_train_test_split_node",
            ),
            node(
                create_modelling_pipeline,
                inputs="params:pipeline",
                outputs="pipeline",
                name="create_modelling_pipeline_node",
            ),
            node(
                train_model,
                inputs=["x_train", "y_train", "pipeline", "params:train"],
                outputs=[
                    "fitted_pipeline",
                    "best_params",
                    "best_value",
                    "study",
                ],
                name="train_model_node",
            ),
            node(
                plot_feature_importance,
                inputs=["fitted_pipeline"],
                outputs="feature_importance_plot",
                name="plot_feature_importance_node",
            ),
            node(
                plot_training_metric,
                inputs=["x_train", "y_train", "fitted_pipeline", "params:train_val"],
                outputs="validation_metric_plot",
                name="plot_metric_node",
            ),
        ],
        namespace="modelling",
        inputs={"master_dataset": "preprocessing.master_dataset"},
    )
