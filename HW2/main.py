import numpy as np
import os
import pandas as pd
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.language import Language
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Union


def lemmatize(text: str, model: Language) -> str:
    """
    Tokenize and lemmatize text data,
    delete stopwords and non-alphabetical characters
    :param text: text to lemmatize
    :param model: tokenizer + lemmatizer model
    :return: lemmatized text
    """
    stops = set(stopwords.words('russian'))
    text = [word.lemma_ for word in model(text) if (word.lemma_.isalpha() and
                                                    word.lemma_ not in stops)]
    return ' '.join(text)


def get_file_content(file: str, model: Language) -> str:
    """
    Extract file contents
    :param file: filename
    :param model: tokenizer + lemmatizer model
    :return: lemmatized text from the file
    """
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
    text = ' '.join(text.splitlines())
    return lemmatize(text, model)


def read_files(path: str) -> Dict[str, str]:
    """
    Read all the files in a given directory
    :param path: path to directory
    :return: list of lemmatized documents
    """
    # load tokenizer
    nlp = spacy.load("ru_core_news_sm")
    documents = {}
    # check dirs
    for directory in tqdm(os.listdir(path)):
        dir_path = os.path.join(path, directory)
        # check if dir
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                # check if text file
                if not file.endswith('.txt'):
                    continue
                file_path = os.path.join(dir_path, file)
                documents[file] = get_file_content(file_path, nlp)
    assert len(documents) == 165

    return documents


def get_term_frequency(corpus: List[str],
                       vectorizer: Union[CountVectorizer,
                                         TfidfVectorizer]) -> np.array:
    """
    Extract the term-frequency matrix
    :param corpus: list of document contents
    :param vectorizer: vectorizer to use
    :return: term-freq matrix
    """
    X = vectorizer.fit_transform(corpus)
    return X.toarray()


def get_index(path: str = 'friends-data',
              vectorizer: str = 'CountVectorizer'
              ) -> Tuple[pd.DataFrame,
                         Union[CountVectorizer, TfidfVectorizer]]:
    """
    Create the reversed index from the text corpora
    :param path: path to directory with text files
    :param vectorizer: vectorizer to use
    :return: reversed index matrix, word features, document names
    """
    # extract corpora
    documents = read_files(path)
    # get reversed_index
    if vectorizer == 'TfidfVectorizer':
        vectorizer = TfidfVectorizer(analyzer='word')
    elif vectorizer == 'CountVectorizer':
        vectorizer = CountVectorizer(analyzer='word')
    reverse_index = get_term_frequency(list(documents.values()), vectorizer)
    # get the list of features
    feature_names = vectorizer.get_feature_names()
    reverse_index = pd.DataFrame(reverse_index,
                                 index=list(documents.keys()),
                                 columns=feature_names)
    return reverse_index, vectorizer


def get_query_vector(input_query: str,
                     vectorizer: Union[CountVectorizer,
                                       TfidfVectorizer]) -> np.array:
    """
    Vectorize input
    :param input_query: search query
    :param vectorizer: vectorizer used for index
    :return: vectorized input
    """
    # load lemmatizer
    nlp = spacy.load("ru_core_news_sm")
    # vectorize query
    query_vector = vectorizer.transform([
        lemmatize(text=input_query,
                  model=nlp)])
    return query_vector


def calc_similarity(index: np.array, query_vector: np.array) -> np.array:
    """
    Calculate cosine similarity between the
    query and the documents in the corpora
    :param index: index
    :param query_vector: search query
    :return: np.array of the similarity between document i and query
    """
    return cosine_similarity(index, query_vector).squeeze(1)


def query(input_query: str,
          index: pd.DataFrame,
          vectorizer: Union[CountVectorizer,
                            TfidfVectorizer],
          n: int = 10) -> List[str]:
    """
    Search query. Sort documents by their cosine similarity to the query
    :param input_query: string to search
        will require to type the string if not specified
    :param index: reversed_index
    :param vectorizer: vectorizer used for the index
    :param n: number of documents to print
        print all the documents if n = -1
    :return: list of suitable documents sorted be similarity to query
    """
    while not input_query:
        input_query = input("Enter the query: ")
    else:
        print(f"\nSearch query: {input_query}")
    # get query vector
    query_vector = get_query_vector(input_query, vectorizer)
    # calculate similarity
    similarity = calc_similarity(index.to_numpy(), query_vector)
    # sort documents by descending similarity
    ids = np.argsort(similarity)[::-1]
    output = index.index[ids]
    # output results on the screen for now
    print(f'\nTop {(n if n != -1 else "")} documents:')
    print('\n'.join(output[:(n if n != -1 else len(output))]))
    return output


def main():
    # extract reversed index
    reverse_index, vectorizer = get_index(vectorizer='TfidfVectorizer')
    # test some queries just to showcase
    output = query('', reverse_index, vectorizer, n=-1)
    output2 = query('Привет, Росс! Как дела?', reverse_index, vectorizer)


if __name__ == '__main__':
    main()



