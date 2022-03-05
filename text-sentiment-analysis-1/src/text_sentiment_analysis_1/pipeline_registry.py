"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline
from text_sentiment_analysis_1.pipelines import data_preprocessing as dpp


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_preprocessing_pipeline = dpp.create_pipeline()


    return {
        "__default__": data_preprocessing_pipeline,
        "dpp": data_preprocessing_pipeline,
    }
