import numpy as np
import os
import spacy
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from spacy.language import Language
from tqdm.auto import tqdm
from typing import List, Tuple


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


def read_files(path: str) -> List[str]:
    """
    Read all the files in a given directory
    :param path: path to directory
    :return: list of lemmatized documents
    """
    # load tokenizer
    nlp = spacy.load("ru_core_news_sm")
    documents = []
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
                documents.append(get_file_content(file_path, nlp))

    assert len(documents) == 165

    return documents


def get_term_frequency(corpus: List[str],
                       vectorizer: CountVectorizer) -> np.array:
    """
    Extract the term-frequency matrix
    :param corpus: list of document contents
    :param vectorizer: vectorizer to use
    :return: term-freq matrix
    """
    X = vectorizer.fit_transform(corpus)
    return X.toarray()


def get_index(path: str = 'friends-data') -> Tuple[np.array, List[str]]:
    """
    Create the reversed index from the text corpora
    :param path: path to directory with text files
    :return: reversed index matrix, word features
    """
    # extract corpora
    documents = read_files(path)
    # get reversed_index
    vectorizer = CountVectorizer(analyzer='word')
    reverse_index = get_term_frequency(documents, vectorizer)
    # get the list of features
    feature_names = vectorizer.get_feature_names()
    return reverse_index, feature_names


def freq_word(index: np.array, features: List[str]):
    """
    Print the most and the least common words
    :param index: reversed index matrix
    :param features: list of words
    :return:
    """
    # extract frequency matrix
    matrix_freq = np.asarray(index.sum(axis=0)).ravel()
    # align word and its frequency and sort
    word_freq = sorted(list(zip(features, matrix_freq.tolist())),
                       key=lambda x: x[1],
                       reverse=True)
    # print results
    print(f'Most common word: {word_freq[0][0]} ({word_freq[0][1]} entries)')
    print(f'Least common word: {word_freq[-1][0]} ({word_freq[-1][1]} entries)',
          end='\n\n')


def word_in_all_docs(index: np.array, features: List[str]):
    """
    Print the n words, that occur in all documents
    :param index: reversed index matrix
    :param features: words
    :return:
    """
    common = np.array(features)[np.where(index == 0, False, True).all(axis=0)]
    print(f'Words in all documents:')
    print(', '.join(common), end='\n\n')


def most_popular_character(index: np.array, features: List[str]):
    """
    Find the most frequent character
    :param index: reversed index matrix
    :param features: words
    :return:
    """
    # dict of all the names and their variants
    names = {'Моника': ['моника', 'мон'],
             'Рэйчел': ['рэйчел', 'рейч'],
             'Чендлер': ['чендлер', 'чэндлер', 'чен'],
             'Росс': ['росс'],
             'Джоуи': ['джоуи', 'джои', 'джо']
             }
    # calculate character frequency
    counter = Counter()
    # get word frequency
    matrix_freq = np.asarray(index.sum(axis=0)).ravel()
    word_freq = dict(zip(features, matrix_freq.tolist()))

    # check each name
    for name in names:
        counter[name] = sum([word_freq[n] for n in names[name]
                             if n in word_freq])
    # print results
    print(f'Most popular character: {counter.most_common()[0][0]} ' +
          f'({counter.most_common()[0][1]} entries)', end='\n\n')


def test(index: np.array, features: List[str]):
    """
    run test tasks
    :param index: reversed index matrix
    :param features: words
    :return:
    """
    # a-b) most and least common words
    freq_word(index, features)
    # c) words in all the documents
    word_in_all_docs(index, features)
    # d) most frequent character
    most_popular_character(index, features)


def main():
    # install the requirements
    os.system('sh install_tools.sh')
    # extract reversed index
    reverse_index, feature_names = get_index()
    # run tests
    test(reverse_index, feature_names)


if __name__ == '__main__':
    main()


