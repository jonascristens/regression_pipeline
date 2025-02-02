"""
This is a boilerplate pipeline 'reporting'
generated using Kedro 0.19.11
"""

from kedro.pipeline import Pipeline, node, pipeline  # noqa

from .nodes import (evaluate_model_metrics, generate_shap_values,
                    plot_actual_vs_expected, plot_confusion_matrix,
                    plot_precision_recall, plot_roc_auc, plot_shap_beeswarm,
                    plot_shap_feature_importance, plot_sklearn_pdp, predict)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=predict,
                inputs=["x_test", "fitted_pipeline"],
                outputs=["y_pred", "y_pred_proba"],
                name="predict_node",
            ),
            node(
                func=evaluate_model_metrics,
                inputs=["y_test", "y_pred", "y_pred_proba"],
                outputs="model_metrics",
                name="evaluate_model_metrics_node",
            ),
            node(
                func=plot_roc_auc,
                inputs=["y_test", "y_pred_proba"],
                outputs="roc_auc_plot",
                name="plot_roc_auc_node",
            ),
            node(
                func=plot_precision_recall,
                inputs=["y_test", "y_pred"],
                outputs="precision_recall_plot",
                name="plot_precision_recall_node",
            ),
            node(
                func=plot_confusion_matrix,
                inputs=["y_test", "y_pred"],
                outputs="confusion_matrix_plot",
                name="plot_confusion_matrix_node",
            ),
            node(
                func=plot_actual_vs_expected,
                inputs=["x_test", "y_test", "y_pred"],
                outputs="actual_vs_predicted_plot",
                name="plot_actual_vs_expected_node",
            ),
            node(
                func=generate_shap_values,
                inputs=["fitted_pipeline", "x_test"],
                outputs="shap_values",
                name="generate_shap_values_node",
            ),
            node(
                func=plot_shap_feature_importance,
                inputs=["shap_values", "x_test"],
                outputs="shap_feature_importance_plot",
                name="plot_shap_feature_importance_node",
            ),
            node(
                func=plot_shap_beeswarm,
                inputs=["shap_values", "x_test"],
                outputs="shap_beeswarm_plot",
                name="plot_shap_beeswarm_node",
            ),
            node(
                func=plot_sklearn_pdp,
                inputs=["fitted_pipeline", "x_test", "params:pdp_plot"],
                outputs="pdp_plot",
                name="plot_pdp_node",
            ),
        ],
        namespace="reporting",
        inputs={
            "fitted_pipeline": "modelling.fitted_pipeline",
            "y_test": "modelling.y_test",
            "x_test": "modelling.x_test",
        },
    )
