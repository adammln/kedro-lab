"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.17.7
"""
from lxml import etree
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import nltk
import pandas as pd
import re, string
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

def extract_and_convert_testing_data(xml_content: str) -> pd.DataFrame:
    """ Extract content of XML file of testing data 
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

def create_testing_data_table(
        reviews: pd.DataFrame,
        labels: pd.DataFrame,
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
    labelled_testing_data = reviews.merge(
        labels, 
        left_on="review_id", 
        right_on="review_id",
    )
    testing_data_table = labelled_testing_data.dropna()
    return testing_data_table

def _n_gram_fit_transform(texts: pd.Series) -> 
        np.matrix, 
        np.ndarray,
        CountVectorizer
    :
    """ Train Vectorizer & Extract N-Gram features on training data
        with N ranging from 1-3
    """
    vectorizer = CountVectorizer(ngram_range=(1,3), max_features=5000)
    features = vectorizer.fit_transform(np.array(texts)).todense() # Train vectorizer
    feature_names = vectorizer.get_feature_names_out()
    return features, vectorizer

def _tfidf_fit_transform(vectors: np.ndarray) -> pd.Series:
    """ Train TF-IDF (Term Frequency — Inverse Document Frequency) 
        Transformer & Extract TF-IDF features on training data
    """
    transformer = TfidfTransformer()
    features = transformer.fit_transform(vectors)
    return features, transformer

def _extract_n_gram_tfidf_test_data(
        texts: pd.Series, 
        vectorizer: CountVectorizer, 
        tfidf_transformer: TfidfTransformer
    ) -> np.matrix, np.matrix:
    tf_features = vectorizer.transform(np.array(texts)).todense()
    tfidf_features = tfidf_transformer.transform(tf_features)
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