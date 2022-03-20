"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    create_agreed_aspect_level_table,
    create_agreed_polarity_level_table,
    create_gold_standard_table, 
    create_labelled_data_table,
    create_unlabelled_data_table,
    extract_and_convert_labelled_data, 
    extract_and_convert_xml_data, 
    extract_train_test_features_from_texts, 
    preprocess_gold_standard, 
    preprocess_text_column, 
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=extract_and_convert_labelled_data,
                inputs="labelled_data",
                outputs="typed_labelled_data",
                name="extract_and_convert_labelled_data_node",
            ),
            node(
                func=extract_and_convert_xml_data,
                inputs="testing_data",
                outputs="typed_testing_data",
                name="extract_and_convert_testing_data_node",
            ),
            node(
                func=extract_and_convert_xml_data,
                inputs="unlabelled_data",
                outputs="typed_unlabelled_data",
                name="extract_and_convert_unlabelled_data_node",
            ),
            node(
                func=preprocess_text_column,
                inputs=["typed_labelled_data", "stopwords_custom"],
                outputs="preprocessed_labelled_data",
                name="preprocess_labelled_data_node",
            ),
            node(
                func=preprocess_text_column,
                inputs=["typed_testing_data", "stopwords_custom"],
                outputs="preprocessed_testing_data",
                name="preprocess_testing_data_node",
            ),
            node(
                func=preprocess_text_column,
                inputs=["typed_unlabelled_data", "stopwords_custom"],
                outputs="preprocessed_unlabelled_data",
                name="preprocess_unlabelled_data_node",
            ),
            node(
                func=preprocess_gold_standard,
                inputs="gold_standard",
                outputs="gold_standard_labels",
                name="preprocess_gold_standard_node",
            ),
            node(
                func=create_unlabelled_data_table,
                inputs=[
                    "preprocessed_unlabelled_data", 
                    "preprocessed_testing_data",
                    "gold_standard_labels",
                ],
                outputs="unlabelled_data_table",
                name="create_unlabelled_data_table_node",
            ),
            node(
                func=create_gold_standard_table,
                inputs=["preprocessed_testing_data", "gold_standard_labels"],
                outputs="gold_standard_table",
                name="create_gold_standard_table_node",
            ),
            node(
                func=create_labelled_data_table,
                inputs="preprocessed_labelled_data",
                outputs="labelled_data_table",
                name="create_labelled_data_table_node",
            ),
            node(
                func=create_agreed_aspect_level_table,
                inputs="labelled_data_table",
                outputs="agreed_aspect_table",
                name="create_agreed_aspect_level_table_node",
            ),
            node(
                func=create_agreed_polarity_level_table,
                inputs="labelled_data_table",
                outputs="agreed_polarity_table",
                name="create_agreed_polarity_level_table_node",
            ),
            # node(
            #     func=extract_train_test_features_from_texts,
            #     inputs=[
            #         "preprocessed_labelled_data", 
            #         "testing_data_table"
            #     ],
            #     outputs=[
            #         "feature_tf_train",
            #         "feature_tf_idf_train",
            #         "feature_tf_test",
            #         "feature_tf_idf_test",
            #         "count_vectorizer",
            #         "tfidf_transformer"
            #     ]
            # )
        ],
        namespace="data_preprocessing", 
        inputs=[
            "unlabelled_data",
            "labelled_data",
            "testing_data",
            "stopwords_custom",
            "gold_standard"
        ],
        outputs=[
            "labelled_data_table",
            "typed_unlabelled_data",
            "typed_labelled_data",
            "typed_testing_data",
            "unlabelled_data_table",
            "gold_standard_table",
            "agreed_aspect_table",
            "agreed_polarity_table"
            # "converted_testing_data",
            # "feature_tf_train",
            # "feature_tf_idf_train",
            # "feature_tf_test",
            # "feature_tf_idf_test",
            # "count_vectorizer",
            # "tfidf_transformer"
        ],
    )