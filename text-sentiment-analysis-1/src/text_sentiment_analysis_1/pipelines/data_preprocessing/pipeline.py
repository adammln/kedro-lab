"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import extract_and_convert_labelled_data, preprocess_labelled_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=extract_and_convert_labelled_data,
                inputs="labelled_data",
                outputs="converted_labelled_data",
                name="extract_and_convert_labelled_data_node",
            ),
            node(
                func=preprocess_labelled_data,
                inputs=["converted_labelled_data", "stopwords_custom"],
                outputs="preprocessed_labelled_data",
                name="preprocess_labelled_data_node",
            ),
        ],
        namespace="data_preprocessing", 
        inputs=["labelled_data", "stopwords_custom"],
        outputs=["converted_labelled_data", "preprocessed_labelled_data"],
    )
