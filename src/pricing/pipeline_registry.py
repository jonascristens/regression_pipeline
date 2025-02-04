"""Project pipelines."""

from __future__ import annotations

from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import find_pipelines, settings
from kedro.pipeline import Pipeline, pipeline

from pricing.pipelines.modelling.pipeline import create_pipeline as modelling_pipeline
from pricing.pipelines.preprocessing.pipeline import (
    create_pipeline as preprocessing_pipeline,
)
from pricing.pipelines.reporting.pipeline import create_pipeline as reporting_pipeline
from pricing.pipelines.simulation.pipeline import create_pipeline as simulation_pipeline


def _load_globals_key(key: str = "countries") -> list[str]:
    """Load the key from the global configuration.

    Args:
        key (str): The key to retrieve from the global configuration.

    Returns:
        list[str]: The list of values associated with the given key.

    Raises:
        KeyError: If the key is missing from the global configuration.
    """
    project_path = Path().cwd()

    while project_path.name != "postal-pricing":
        if project_path == Path("/"):
            raise FileNotFoundError(
                "Could not find the project root directory. "
                "Please ensure that you are running this function "
                "from within the project directory."
            )
        project_path = project_path.parent

    conf_path = str(project_path / settings.CONF_SOURCE)
    conf_loader = OmegaConfigLoader(
        conf_source=conf_path, base_env="base", default_run_env="local"
    )

    try:
        return conf_loader["globals"][key]
    except KeyError as e:
        raise KeyError(
            f"Missing required key '{key}' in global configuration. "
            "Ensure that the key is defined in your `globals.yml` "
            "or relevant config file."
        ) from e


def _apply_namespaces_to_pipelines(
    pipes: dict[str, Pipeline], namespaces: list[str]
) -> dict[str, Pipeline]:
    """Apply namespaces to the pipelines."""
    return {
        pipeline_name: sum(
            pipeline(pipe, namespace=namespace) for namespace in namespaces
        )
        for pipeline_name, pipe in pipes.items()
    }


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to `Pipeline` objects.
    """
    # define the pipelines
    pipelines = find_pipelines()
    default_pipeline = (
        preprocessing_pipeline()
        # + feature_engineering_pipeline() # to implement once identified
        + modelling_pipeline()
        + reporting_pipeline()
        + simulation_pipeline()
    )
    pipelines["__default__"] = default_pipeline  # Assign default pipeline

    # apply namespace the pipelines
    countries = _load_globals_key(key="countries")
    namespaced_pipelines = _apply_namespaces_to_pipelines(pipelines, countries)

    return namespaced_pipelines
