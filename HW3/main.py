import numpy as np
import json
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from scipy import sparse
import pymorphy2
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Union


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


def get_corpora_content(corpora: List[str]) -> Tuple[List[str], np.array]:
    """
    Extract corpora contents
    :param corpora: filename
    :return: lemmatized texts from the corpora
    """
    # load tokenizer
    morph = pymorphy2.MorphAnalyzer()
    documents = []
    original_docs = []
    for doc in tqdm(corpora, total=len(corpora), desc='Reading files'):
        doc = json.loads(doc)
        answer = doc['answers']
        if answer:  # там есть вопросы без ответа
            text = sort_answers(answer)[0]['text']
            lemmatized = lemmatize(text, morph)
            documents.append(lemmatized)
            original_docs.append(text)
    return documents, np.array(original_docs)


def read_files(path: str) -> Tuple[List[str], np.array]:
    """
    Read all the files in a given directory
    :param path: path to directory
    :return: list of lemmatized documents
    """
    # load json file
    with open(path, 'r') as f:
        corpus = list(f)[:50000] # долго ждать...
    # extract contents
    documents, original = get_corpora_content(corpus)
    return documents, original


def sparse_matrix_add(matrix: sparse.csr_matrix,
                      vector: np.array) -> sparse.csr_matrix:
    """
    Addition for sparse matrices
    :param matrix: sparse matrix
    :param vector: vector to add
    :return: result of addition
    """
    for i, j in zip(*matrix.nonzero()):
        matrix[i, j] = matrix[i, j] + vector[i]
    return matrix


def calculate_bm25(corpus: List[str]) -> Tuple[sparse.csr_matrix,
                                               CountVectorizer]:
    """
    Realization of Okapi BM25
    :param corpus: text corpora
    :return: index, fitted count_vectorizer
    """
    # set constant values
    k = 2
    b = 0.75

    # initialize vectorizers
    tf_vectorizer = TfidfVectorizer(analyzer='word', use_idf=False)
    tfidf_vectorizer = TfidfVectorizer(analyzer='word')
    count_vectorizer = CountVectorizer(analyzer='word')

    # calculate tf
    tf = tf_vectorizer.fit_transform(corpus)
    # calculate idf
    _ = tfidf_vectorizer.fit_transform(corpus)
    # idf = np.expand_dims(tfidf_vectorizer.idf_, axis=0)
    idf = tfidf_vectorizer.idf_
    # calculate BoW
    bow = count_vectorizer.fit_transform(corpus)

    len_d = bow.sum(axis=1)
    avgdl = len_d.mean()
    constant = (k * (1 - b + (b * len_d / avgdl)))

    # я так понимаю, что от нас хотят такого:
    for i, j in tqdm(zip(*tf.nonzero()), desc="Calculating index"):
        tf[i, j] = (tf[i, j] * (k+1) * idf[j]) / (tf[i, j] + constant[i])
    return tf, count_vectorizer

    # # но вообще так тоже работает без перехода из спарсованных матриц,
    # # можно ли так делать?
    # # numerator part of the BM25 formula
    # numerator = tf.multiply(idf) * (k+1)  # sparse coo matrix
    # assert type(numerator) is sparse.coo_matrix
    # numerator = numerator.tocsr()
    #
    # # denumerator part of the BM25 formula
    # denumerator = sparse_matrix_add(tf, constant)
    # assert type(denumerator) is sparse.csr_matrix
    # final_matrix = numerator.multiply(denumerator.power(-1))
    # assert type(final_matrix) is sparse.csr_matrix
    # print(final_matrix.toarray())
    # return final_matrix, count_vectorizer


def get_index(path: str = 'questions_about_love.jsonl') -> Tuple[
                                                            sparse.csr_matrix,
                                                            np.array,
                                                            CountVectorizer]:
    """
    Create the reversed index from the text corpora
    :param path: path to directory with text files
    :return: reversed index matrix,  documents, count vectorizer
    """
    # extract corpora
    documents, original = read_files(path)
    # get index
    index, vectorizer = calculate_bm25(documents)
    return index, original, vectorizer


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
    morph = pymorphy2.MorphAnalyzer()
    # vectorize query
    input_query = lemmatize(text=input_query, model=morph)
    query_vector = vectorizer.transform([input_query])
    return query_vector


def search_query(input_query: str,
                 index: sparse.csr_matrix,
                 vectorizer: CountVectorizer,
                 n: int = 10):
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
    similarity = np.dot(index, query_vector.T)
    # get similar documents' indexes
    documents = similarity.nonzero()[0]
    # sort documents by descending similarity
    similarity = similarity[similarity.nonzero()].A.ravel()
    ids = np.argsort(similarity)[::-1]
    output = documents[ids]
    return output[:n] if n != -1 else output


def print_results(ids: np.array, documents: np.array):
    """
    Print search results
    :param ids: ids of the documents
    :param documents: corpors
    :return: None
    """
    # output results on the screen
    print(f'\nTop {len(ids)} documents:')
    print('\n'.join(
        [str(doc_index)+'\t'+documents[doc_index] for doc_index in ids])
    )


def main():
    # extract reversed index
    reverse_index, documents, vectorizer = get_index()
    # test some queries just to showcase
    output1 = search_query('', reverse_index, vectorizer)
    print_results(output1, documents)
    output2 = search_query('как найти любовь?', reverse_index, vectorizer)
    print_results(output2, documents)


if __name__ == '__main__':
    main()



