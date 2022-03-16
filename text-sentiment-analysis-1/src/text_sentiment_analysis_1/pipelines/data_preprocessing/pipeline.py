"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import extract_and_convert_labelled_data, extract_and_convert_testing_data, preprocess_text_column, preprocess_gold_standard, create_testing_data_table, extract_train_test_features_from_texts

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
                func=extract_and_convert_testing_data,
                inputs="testing_data",
                outputs="typed_testing_data",
                name="extract_and_convert_testing_data_node",
            ),
            # node(
            #     func=preprocess_text_column,
            #     inputs=["converted_labelled_data", "stopwords_custom"],
            #     outputs="preprocessed_labelled_data",
            #     name="preprocess_labelled_data_node",
            # ),
            # node(
            #     func=preprocess_text_column,
            #     inputs=["converted_testing_data", "stopwords_custom"],
            #     outputs="preprocessed_testing_data",
            #     name="preprocess_testing_data_node",
            # ),
            # node(
            #     func=preprocess_gold_standard,
            #     inputs="gold_standard",
            #     outputs="testing_data_labels",
            #     name="preprocess_gold_standard_node",
            # ),
            # node(
            #     func=create_testing_data_table,
            #     inputs=["preprocessed_testing_data", "testing_data_labels"],
            #     outputs="testing_data_table",
            #     name="create_testing_data_table_node",
            # ),
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
            # "unlabelled_data",
            "labelled_data",
            "testing_data",
            # "stopwords_custom",
            # "gold_standard"
        ],
        outputs=[
            # "preprocessed_labelled_data",
            # "typed_unlabelled_data",
            "typed_labelled_data",
            "typed_testing_data",
            # "testing_data_table",
            # "converted_testing_data",
            # "feature_tf_train",
            # "feature_tf_idf_train",
            # "feature_tf_test",
            # "feature_tf_idf_test",
            # "count_vectorizer",
            # "tfidf_transformer"
        ],
    )