"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_and_convert_labelled_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_and_convert_labelled_data,
                inputs="labelled_data",
                outputs="converted_labelled_data",
                name="load_and_convert_labelled_data_node",
            ),
            # TODO: add pre-processing node for text cleaning
        ],
        namespace="data_preprocessing", 
        inputs=["labelled_data"],
        outputs="converted_labelled_data",
    )
