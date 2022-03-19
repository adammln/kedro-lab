"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.17.7
"""
from lxml import etree
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import nltk
import numpy as np
import pandas as pd
import re, string, uuid
import xml.etree.ElementTree as ET

nltk.download('stopwords')
_stopwords_dirty = stopwords.words("english") + stopwords.words("indonesian")

def _remove_xml_special_character_on_labelled_data(content: str) -> str:
    lines = content.split("\n")
    clean_lines = []
    chars_to_remove = "<>&\"\'"
    table = dict((ord(c), None) for c in chars_to_remove)
    for line in lines:
        if (line[0] != "<") or line[:2] == "< ":
            line = line.translate(table)
        clean_lines.append(line)
    return "\n".join(clean_lines)

def _extract_xml_tree_from_text(content: str) -> ET.ElementTree:
    parser = etree.XMLParser(recover=False)
    return ET.ElementTree(ET.fromstring(content, parser=parser))

def _convert_testing_data_tree_to_dataframe(tree: ET.ElementTree) -> pd.DataFrame:
    root = tree.getroot()
    columns = ["review_id", "text"]
    rows = []

    for node in root:
        review_id = int(node.find("index").text)
        text = node.find("review").text
        row = {"review_id": review_id, "text": text}
        rows.append(row)
    
    dataframe = pd.DataFrame(rows, columns = columns)
    return dataframe

def _convert_labelled_data_tree_to_dataframe(tree: ET.ElementTree) -> pd.DataFrame:
    root = tree.getroot()
    columns = [
        "review_id", "text", 
        "food_0", "price_0", "ambience_0", "service_0", 
        "food_1", "price_1", "ambience_1", "service_1"
    ] 
    rows = []

    for node in root:
        review_id = int(node.attrib.get("rid"))
        review_text = node.find("text").text if node is not None else None
        reviewers = node.findall("aspects")      
        row_payload = {
            "review_id": review_id, "text": review_text,
            "food_0": "unknown", "price_0": "unknown", "ambience_0": "unknown", "service_0": "unknown",
            "food_1": "unknown", "price_1": "unknown", "ambience_1": "unknown", "service_1": "unknown"
        }

        for reviewer in reviewers:
            temp_id = ''
            temp_polarity = ''

            for aspect in reviewer:
                temp_id = "_".join(
                    [
                        aspect.attrib.get("category").lower(),
                        reviewer.attrib['id']
                    ]
                )
                temp_polarity = aspect.attrib.get("polarity").lower()    
                row_payload[temp_id] = temp_polarity
        rows.append(row_payload)
    dataframe = pd.DataFrame(rows, columns = columns)
    return dataframe

def _normalize_text(x: str) -> str:
    x = x.strip()
    x = x.lower()
    x = re.sub(r'\s+', ' ', x) # replace trailing spaces into a single space
    x = re.sub(r'[^\w\s]', '', x) # remove punctuations
    x = re.sub(r'\d', '', x) # remove digits
    x = re.sub(r'(.+?)\1+', r'\1\1', x) # replace repeating character (hahaha -> haha, heyyyy -> heyy)
    x = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b|\\b[A-Za-z]\\b)', '', x) # remove single letter
    return x

def _normalize_text_column(x: pd.Series) -> pd.Series:
    x = x.apply(_normalize_text)
    return x

STOPWORDS = set(_normalize_text(' '.join(_stopwords_dirty)).split(' '))

def _remove_stopwords(x: str) -> str:
    x = ' '.join([word for word in x.split() if word not in STOPWORDS])
    return x

def _remove_stopwords_in_text_column(x: pd.Series) -> pd.Series:
    x = x.apply(_remove_stopwords)
    return x

def _remove_custom_stopwords(text: str, stopwords_text: str):
    stopwords = set(' '.join(stopwords_text.split('\n')))
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

def _remove_custom_stopwords_in_text_column(
        data_series: pd.Series,
        stopwords_text: str
    ) -> pd.Series:

    def _sw_removal_helper(x: str) -> str:
            x = _remove_custom_stopwords(x, stopwords_text)
            return x

    data_series = data_series.apply(_sw_removal_helper)
    return data_series

def _map_label(x: str, mapper: dict) -> str:
    return mapper[x]

def _convert_aspect_level_to_binary(label_column: pd.Series) -> pd.Series:
    label_column = label_column.apply(lambda x: x != "unknown")
    return label_column

def _convert_polarity_level_to_binary(label_column: pd.Series) -> pd.Series:
    label_column = label_column.apply(lambda x: x == "positive")
    return label_column

def _count_labels(df: pd.DataFrame, columns_of_aspect_labels_per_person:list, labels: list, aspect_name:str) -> pd.DataFrame:
    df_t = df[columns_of_aspect_labels_per_person].T
    label_counts = pd.DataFrame(columns=labels)
    for subject in df_t:
        c = df_t[subject].value_counts()
        label_counts = label_counts.append(dict(c), True)
    label_counts = label_counts.fillna(0)
    new_column_names = {}
    for label in labels:
        new_column_names[label] = aspect_name + "_" + label
    label_counts.rename(columns=new_column_names, inplace=True)
    for column in label_counts.columns:
        df[column] = label_counts[column]
    df = df.join(label_counts)
    return df

def _transform_aspect_label_columns_to_label_counts(df: pd.DataFrame) -> pd.DataFrame:
    labels = ['unknown', 'positive', 'negative']
    aspects = ['food', 'price', 'ambience', 'service']
    rater_ids = [0, 1]
    for aspect in aspects:
        aspect_labels_per_person = []
        for rater_id in rater_ids:
            aspect_labels_per_person.append(aspect+"_"+str(rater_id))
            # labelled_data[aspect+"_presence_"+str(rater_id)] = _convert_aspect_level_to_binary(labelled_data[aspect+"_"+str(rater_id)])
            # labelled_data[aspect+"_positive_"+str(rater_id)] = _convert_polarity_level_to_binary(labelled_data[aspect+"_"+str(rater_id)])
        df = _count_labels(
            df=df,
            columns_of_aspect_labels_per_person=aspect_labels_per_person,
            labels= labels,
            aspect_name=aspect
        )
    return df

def extract_and_convert_labelled_data(xml_content: str) -> pd.DataFrame:
    """ Extract content of XML file of labelled data 
        and convert to pandas Dataframe
    
    Args:
        xml_content: full content of xml file represented in string

    Returns:
        Content of xml file of labelled data represented in pandas dataframe
        with renamed column
    """
    xml_content = _remove_xml_special_character_on_labelled_data(xml_content)
    tree = _extract_xml_tree_from_text(xml_content)
    dataframe = _convert_labelled_data_tree_to_dataframe(tree)
    return dataframe

def extract_and_convert_xml_data(xml_content: str) -> pd.DataFrame:
    """ Extract content of XML file 
        and convert to pandas Dataframe
    
    Args:
        xml_content: full content of xml file represented in string

    Returns:
        Content of xml file of testing data represented in pandas dataframe
        with renamed column
    """
    tree = _extract_xml_tree_from_text(xml_content)
    dataframe = _convert_testing_data_tree_to_dataframe(tree)
    return dataframe

def preprocess_text_column(
        dataframe: pd.DataFrame,
        stopwords_custom: str,
    ) -> pd.DataFrame:
    """Preprocess 'text' column in dataframe 
        includes:
        - regular expression
        - stopwords removal
        - custom stopwords removal
        TODO: add lemmatization
    
    Args:
        dataframe: customer reviews data
        accompanied with sentiment & aspect labels

    Returns:
        dataframe with preprocessed 'text' column
    """
    dataframe["text"] = _normalize_text_column(dataframe["text"])
    dataframe["text"] = _remove_stopwords_in_text_column(dataframe["text"])
    dataframe["text"] = _remove_custom_stopwords_in_text_column(
        dataframe["text"], 
        stopwords_custom
    )
    return dataframe

def preprocess_gold_standard(dataframe: pd.DataFrame) -> pd.DataFrame:
    """ Preprocess gold_standard (labels of testing_data)
        includes: 
        - column renaming 
        - label name conversion (with label_mapper)

    Args:
        dataframe: dataframe of sentiment labels 
                for all aspects (food, ambience, service, price)

    Returns:
        preprocessed dataframe
    """
    dataframe = dataframe.rename(columns={
        'ID':'review_id', 
        "FOOD": "food", 
        "SERVICE":"service", 
        "AMBIENCE":"ambience", 
        "PRICE": "price"
    })
    label_mapper = {
        "-": "unknown",
        "NEGATIVE": "negative",
        "POSITIVE": "positive",
    }

    def _map_label_helper(x:str) -> str:
        return _map_label(x, label_mapper)

    dataframe["food"] = dataframe["food"].apply(_map_label_helper)
    dataframe["price"] = dataframe["price"].apply(_map_label_helper)
    dataframe["service"] = dataframe["service"].apply(_map_label_helper)
    dataframe["ambience"] = dataframe["ambience"].apply(_map_label_helper)
    return dataframe

def create_unlabelled_data_table(
        unlabelled_data: pd.DataFrame,
        testing_data: pd.DataFrame,
        gold_standard: pd.DataFrame,
    ) -> pd.DataFrame:
    """ Create unlabelled_data_table 
        by merging testing_data and unlabelled_data
        where testing_data rows don't have gold_standard labels

        Args:
            unlabelled_data: unlabelled customer reviews of restaurants
                        extracted from unlabelled_data.xml
            testing_data: unlabelled customer reviews of restaurants 
                        extracted from testing_data.xml
            gold_standard: gold_standard labels, for a subset of testing_data (named t1)
        
        Returns:
            a merged dataframe with rows comes from all unlabelled_data 
            and testing_data excludes t1
    """
    unlabelled_testing_data = testing_data.loc[
        ~testing_data.review_id.isin(gold_standard.review_id)
    ]
    merged_unlabelled_data = pd.concat([unlabelled_data,unlabelled_testing_data])
    merged_unlabelled_data["review_id"] = merged_unlabelled_data["review_id"].astype(str)
    merged_unlabelled_data["review_id"] = [str(uuid.uuid4()) for _ in range(len(merged_unlabelled_data.index))]
    return merged_unlabelled_data

def create_gold_standard_table(
        testing_data: pd.DataFrame,
        gold_standard: pd.DataFrame,
    ) -> pd.DataFrame:
    """ Join testing_data and gold_standard (as labels of testing_data)
        based on review_id

    Args:
        reviews: dataframe (from testing_data) containing id and review texts
        labels: dataframe containing sentiment labels for all 4 aspects
            referencing to 'reviews' (testing_data)
    
    Return:
        complete testing data table containing review text and it's label
    """
    merged = testing_data.merge(
        gold_standard, 
        left_on="review_id", 
        right_on="review_id",
    )
    merged = merged.dropna()
    merged["review_id"] = merged["review_id"].astype(str)
    merged["review_id"] = [str(uuid.uuid4()) for _ in range(len(merged.index))]
    return merged

def create_labelled_data_table(
        labelled_data: pd.DataFrame,
    ) -> pd.DataFrame:
    """ Create labelled_data_table by dropping review_id column

    Args:
        labelled_data: preprocessed labelled data
    
    Return:
        labelled data table
    """
    labelled_data["review_id"] = labelled_data["review_id"].astype(str)
    labelled_data["review_id"] = [str(uuid.uuid4()) for _ in range(len(labelled_data.index))]
    labelled_data = _transform_aspect_label_columns_to_label_counts(labelled_data)
    return labelled_data

def _n_gram_fit_transform(texts: pd.Series):
    """ Train Vectorizer & Extract N-Gram features on training data
        with N ranging from 1-3
    """
    vectorizer = CountVectorizer(ngram_range=(1,3), max_features=5000)
    features = np.asarray(vectorizer.fit_transform(np.array(texts)).todense()) # Train vectorizer
    feature_names = vectorizer.get_feature_names_out()
    return features, vectorizer

def _tfidf_fit_transform(vectors: np.ndarray):
    """ Train TF-IDF (Term Frequency — Inverse Document Frequency) 
        Transformer & Extract TF-IDF features on training data
    """
    transformer = TfidfTransformer()
    features = transformer.fit_transform(vectors).toarray()
    return features, transformer

def _extract_n_gram_tfidf_test_data(
        texts: pd.Series, 
        vectorizer: CountVectorizer, 
        tfidf_transformer: TfidfTransformer
    ):
    tf_features = np.asarray(vectorizer.transform(np.array(texts)).todense())
    tfidf_features = tfidf_transformer.transform(tf_features).toarray()
    return tf_features, tfidf_features

def _extract_word2vec(texts: pd.Series) -> pd.Series:
    # TODO: create Word2Vec feature extraction
    """ Extract word embeddings feature using Word2Vec
    """
    return texts

def _extract_bert(texts: pd.Series) -> pd.Series:
    # TODO: create BERT feature extraction
    """ Extract word embeddings feature 
        using Bi-Directional Encoder Representations from Transformer (BERT)
    """
    return texts

def extract_train_test_features_from_texts(
        train_dataframe: pd.DataFrame,
        test_dataframe: pd.DataFrame,
    ):
    # Generate features from train dataset
    train_tf_feature, count_vectorizer = _n_gram_fit_transform(train_dataframe["text"])
    train_tfidf_feature, tfidf_transformer = _tfidf_fit_transform(train_tf_feature)
    
    # Reuse vectorizer & transformer to create features from test dataset
    test_tf_feature, test_tfidf_feature = _extract_n_gram_tfidf_test_data(
        test_dataframe["text"],
        count_vectorizer,
        tfidf_transformer
    )

    # Save features into dictionaries with review_id mapping (as a key)
    ## training data
    train_data_keys = train_dataframe["review_id"]
    train_tf_feature_out = dict(zip(train_data_keys, train_tf_feature.tolist()))
    train_tfidf_feature_out = dict(zip(train_data_keys, train_tfidf_feature.tolist()))

    ## testing data
    test_data_keys = test_dataframe["review_id"]
    test_tf_feature_out = dict(zip(test_data_keys, test_tf_feature.tolist()))
    test_tfidf_feature_out = dict(zip(test_data_keys, test_tfidf_feature.tolist()))

    return train_tf_feature_out, train_tfidf_feature_out, test_tf_feature_out, test_tfidf_feature_out, count_vectorizer, tfidf_transformer


