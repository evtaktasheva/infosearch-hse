import numpy as np
import json
import pymorphy2
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional


def lemmatize(text: str, model: pymorphy2.analyzer.MorphAnalyzer) -> str:
    """
    Tokenize and lemmatize text data,
    delete stopwords and non-alphabetical characters
    :param text: text to lemmatize
    :param model: tokenizer + lemmatizer model
    :return: lemmatized text
    """
    stops = set(stopwords.words('russian'))
    text = [
        model.parse(word)[0].normal_form for word in wordpunct_tokenize(text)
    ]
    return ' '.join(text)


def sort_answers(answers: List[dict]) -> List[Dict]:
    """
    Sort the answers by the author rating
    :param answers: answers to sort
    :return: items sorted in the descending order of the author rating
    """
    # check for '' values
    answers = [answer for answer in answers if answer['author_rating']['value']]
    return sorted(answers,
                  key=lambda x: int(x['author_rating']['value']),
                  reverse=True)


def get_corpora_content(corpora: List[str],
                        preprocess: bool = True) -> Tuple[np.array, np.array, np.array]:
    """
    Extract corpora contents
    :param corpora: filename
    :return: lemmatized texts from the corpora
    """
    # load tokenizer
    morph = pymorphy2.MorphAnalyzer()
    documents = []
    questions = []
    originals = []
    for doc in tqdm(corpora, total=len(corpora), desc='Reading files'):
        doc = json.loads(doc)
        answer = doc['answers']
        if answer:  # там есть вопросы без ответа
            text = sort_answers(answer)[0]['text']
            originals.append(text)
            if preprocess:
                text = lemmatize(text, morph)
            documents.append(text)
            questions.append(doc['question'])
    return np.array(documents), np.array(questions), np.array(originals)


def read_files(path: str, preprocess: bool = True, n=None) -> Tuple[np.array, np.array, np.array]:
    """
    Read all the files in a given directory
    :param path: path to directory
    :return: list of lemmatized documents
    """
    # load json file
    with open(path, 'r') as f:
        corpus = list(f)
    if n:
        corpus = corpus[:n]
    # extract contents
    documents, questions, originals = get_corpora_content(corpora=corpus,
                                                          preprocess=preprocess)
    return documents, questions, originals
