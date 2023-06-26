"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from player_rating_system.pipelines.data_processing.pipeline import create_pipeline as de_pipeline
from player_rating_system.pipelines.data_science.pipeline import create_pipeline as ds_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())

    de = de_pipeline()
    ds = ds_pipeline()

    return {"__default__": de+ds,"de": de, "ds":ds}
