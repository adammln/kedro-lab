"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import extract_and_convert_labelled_data, extract_and_convert_testing_data, preprocess_text_column, preprocess_gold_standard, create_testing_data_table

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
                func=preprocess_text_column,
                inputs=["converted_labelled_data", "stopwords_custom"],
                outputs="preprocessed_labelled_data",
                name="preprocess_labelled_data_node",
            ),
            node(
                func=extract_and_convert_testing_data,
                inputs="testing_data",
                outputs="converted_testing_data",
                name="extract_and_convert_testing_data_node",
            ),
            node(
                func=preprocess_text_column,
                inputs=["converted_testing_data", "stopwords_custom"],
                outputs="preprocessed_testing_data",
                name="preprocess_testing_data_node",
            ),
            node(
                func=preprocess_gold_standard,
                inputs="gold_standard",
                outputs="testing_data_labels",
                name="preprocess_gold_standard_node",
            ),
            node(
                func=create_testing_data_table,
                inputs=["preprocessed_testing_data", "testing_data_labels"],
                outputs="testing_data_table",
                name="create_testing_data_table_node",
            )
        ],
        namespace="data_preprocessing", 
        inputs=[
            "labelled_data",
            "testing_data",
            "stopwords_custom",
            "gold_standard"
        ],
        outputs=[
            "preprocessed_labelled_data",
            "testing_data_table",
        ],
    )