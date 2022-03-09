"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.17.7
"""
from lxml import etree
from nltk.corpus import stopwords

import nltk
import pandas as pd
import re
import xml.etree.ElementTree as ET

nltk.download('stopwords')
_stopwords_dirty = stopwords.words("english") + stopwords.words("indonesian")

def _extract_xml_tree_from_text(content: str) -> ET.ElementTree:
    parser = etree.XMLParser(recover=True) # recover=T -> skip broken form
    return ET.ElementTree(ET.fromstring(content, parser=parser))

def _convert_testing_data_tree_to_dataframe(tree: ET.ElementTree) -> pd.DataFrame:
    root = tree.getroot()
    columns = ["review_id", "text"]
    rows = []

    for node in root:
        review_id = node.find("index").text
        text = node.find("review").text
        row = {"review_id": review_id, "text": text}
    
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
        review_id = node.attrib.get("rid")
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
    x = re.sub(r's+', ' ', x) # replace trailing spaces into a single space
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

def extract_and_convert_labelled_data(xml_content: str) -> pd.DataFrame:
    """ Extract content of XML file of labelled data 
        and convert to pandas Dataframe
    
    Args:
        xml_content: full content of xml file represented in string

    Returns:
        Content of xml file of labelled data represented in pandas dataframe
        with renamed column
    """
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
    
    Args:
        dataframe: customer reviews data
        accompanied with sentiment & aspect labels

    Returns:
        dataframe with preprocessed 'text' column
        preprocessing includes:
        - regular expression
        - stopwords removal
        - custom stopwords removal
        TODO: add lemmatization
    """
    dataframe["text"] = _normalize_text_column(dataframe["text"])
    dataframe["text"] = _remove_stopwords_in_text_column(dataframe["text"])
    dataframe["text"] = _remove_custom_stopwords_in_text_column(
        dataframe["text"], 
        stopwords_custom
    )
    return dataframe
