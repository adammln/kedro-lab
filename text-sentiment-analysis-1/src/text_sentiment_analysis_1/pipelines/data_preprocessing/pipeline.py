"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_labelled_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_labelled_data,
                inputs="labelled_data",
                outputs="preprocessed_labelled_data",
                name="preprocess_labelled_data_node",
            ),
        ],
        namespace="data_preprocessing", # TODO: to revise into relevant name
        inputs=["labelled_data"],
        outputs="preprocessed_labelled_data", # TODO: to revise into relevant name
    )
