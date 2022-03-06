"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.17.7
"""
from lxml import etree

import pandas as pd
import re
import xml.etree.ElementTree as ET

def _load_xml_tree_from_text(content: str) -> ET.ElementTree:
    parser = etree.XMLParser(recover=True) # recover=T -> skip broken form
    return ET.ElementTree(ET.fromstring(content, parser=parser))

def _convert_labelled_data_tree_to_dataframe(tree: ET.ElementTree) -> pd.DataFrame:
    root = tree.getroot()
    columns = [
        "rid", "text", "food_0", "price_0", "ambience_0", 
        "service_0", "food_1", "price_1", "ambience_1", 
        "service_1"
    ] 
    rows = []

    for node in root:
        review_id = node.attrib.get("rid")
        review_text = node.find("text").text if node is not None else None
        reviewers = node.findall("aspects")      
        row_payload = {
            "rid": review_id, "text": review_text,
            "food_0": "unknown", "price_0": "unknown", 
            "ambience_0": "unknown", "service_0": "unknown",
            "food_1": "unknown", "price_1": "unknown", 
            "ambience_1": "unknown", "service_1": "unknown"
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

def load_and_convert_labelled_data(xml_content: str) -> pd.DataFrame:
    """Load XML file of labelled data and convert to pandas Dataframe
    
    Args:
        xml_content: full content of xml file represented in string

    Returns:
        Content of xml file represented in pandas dataframe
        with renamed column
    """
    tree = _load_xml_tree_from_text(xml_content)
    dataframe = _convert_labelled_data_tree_to_dataframe(tree)
    return dataframe

# TODO: use pre-processing/text normalization
#       for "converted_labelled_data"

def preprocess_labelled_data(labelled_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess text data of reviews in labelled_data dataframe 
    
    Args:
        labelled_data: customer reviews data 
        accompanied with sentiment & aspect labels

    Returns:
        same data structure of labelled_data 
        but with review text in the normalized form
    """
    labelled_data["text"] = _normalize_text_column(labelled_data["text"])
    return labelled_data