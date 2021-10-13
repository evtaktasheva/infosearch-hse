import torch
from abc import abstractmethod
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import FastTextKeyedVectors
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from utilities import *
import numpy as np
import pymorphy2
from tqdm.auto import tqdm
from typing import Tuple, Optional, Union


class Index(object):
    def __init__(self,
                 filepath: str = 'questions_about_love.jsonl',
                 n: Optional[int] = None):
        """
        Initialize index
        :param filepath: path to text files
        :param n: number of documents in corpora
        """
        self.data, self.questions, self.originals = read_files(filepath, n=n)
        self.index, self.vectorizer = self.get_index()

    @abstractmethod
    def get_index(self):
        """
        Create index
        :return: index and vectorizer
        """
        pass

    @abstractmethod
    def get_query_matrix(self, data: Union[list, np.array]):
        """
        Vectorize the query
        :param data: query data
        :return: vectorized query
        """
        pass

    def similarity(self, data: Union[list, np.array]) -> np.array:
        """
        Calsulate similarity of two matrices
        :param data: query data
        :return: sorted ids
        """
        query_matrix = self.get_query_matrix(data)
        similarity = cosine_similarity(self.index, query_matrix)
        idx = np.argsort(similarity, axis=0)[::-1]
        return idx

    def top_k(self, data: Union[list, np.array], k: int = 5) -> np.array:
        """
        Find top k most similar documents
        :param data: query data
        :param k: number of documents to find
        :return: top k document ids
        """
        similar_idx = self.similarity(data)[:k]
        return similar_idx

    def score(self, k: int = 5) -> float:
        """
        Calculate index score
        :param k: number of most similar documents to check
        :return: index score
        """
        pred = self.top_k(self.questions, k)
        mask = np.indices(pred.shape)[1]
        metric = np.sum(pred == mask) / pred.shape[1]
        return metric

    def search_query(self, k: int = 5):
        """
        Search query. Sort documents by their cosine similarity to the query
        :param k: number of documents to output
        :return: None
        """
        input_query = input('Enter your query: ')
        top_answers = self.top_k([input_query], k)
        self.print_results(top_answers)

    def print_results(self, ids: np.array):
        """
        Print search results
        :param ids: ids of the documents
        :return: None
        """
        # output results on the screen
        print(f'\nTop {len(ids)} documents:')
        print('\n'.join(self.originals[ids].T[0]))


class TfidfCorpora(Index):
    def __init__(self,
                 filepath: str = 'questions_about_love.jsonl',
                 n: Optional[int] = None):
        """
        Initialize index
        :param filepath: path to text files
        :param n: number of documents in corpora
        """
        super().__init__(filepath, n)

    def get_index(self) -> Tuple[sparse.csr_matrix, TfidfVectorizer]:
        """
        Create index
        :return: index and vectorizer
        """
        vectorizer = TfidfVectorizer(analyzer='word')
        index = vectorizer.fit_transform(self.data)
        return index, vectorizer

    def get_query_matrix(self,
                         data: Union[list, np.array]) -> sparse.csr_matrix:
        """
        Vectorize the query
        :param data: query data
        :return: vectorized query
        """
        morph = pymorphy2.MorphAnalyzer()
        input_query = [lemmatize(text=sent, model=morph) for sent in
                       data]
        query_matrix = self.vectorizer.transform(input_query)
        return query_matrix


class BoWCorpora(Index):
    def __init__(self,
                 filepath: str = 'questions_about_love.jsonl',
                 n: Optional[int] = None):
        """
        Initialize index
        :param filepath: path to text files
        :param n: number of documents in corpora
        """
        super().__init__(filepath, n)

    def get_index(self) -> Tuple[sparse.csr_matrix, CountVectorizer]:
        """
        Create index
        :return: index and vectorizer
        """
        vectorizer = CountVectorizer(analyzer='word')
        index = vectorizer.fit_transform(self.data)
        return index, vectorizer

    def get_query_matrix(self,
                         data: Union[list, np.array]) -> sparse.csr_matrix:
        """
        Vectorize the query
        :param data: query data
        :return: vectorized query
        """
        morph = pymorphy2.MorphAnalyzer()
        input_query = [lemmatize(text=sent, model=morph) for sent in
                       data]
        query_matrix = self.vectorizer.transform(input_query)
        return query_matrix


class FastTextCorpora(Index):
    def __init__(self,
                 filepath: str = 'questions_about_love.jsonl',
                 n: Optional[int] = None):
        """
        Initialize index
        :param filepath: path to text files
        :param n: number of documents in corpora
        """
        super().__init__(filepath, n)

    @staticmethod
    def fasttext_pool(text: str, model: FastTextKeyedVectors) -> np.array:
        """
        Extract text embedding
        :param text: text to embed
        :param model: fasttext model
        :return: text embedding
        """
        text = text.split()
        vector = np.zeros((300,))
        for word in text:
            vector += model[word]
        return np.divide(vector, len(text))

    def get_index(self) -> Tuple[np.array, FastTextKeyedVectors]:
        """
        Create index
        :return: index and vectorizer
        """
        model_path = 'araneum_none_fasttextcbow_300_5_2018.model'
        model = KeyedVectors.load(model_path)
        index = np.array([self.fasttext_pool(text, model) for text in self.data])
        return index, model

    def get_query_matrix(self,
                         data: Union[list, np.array]) -> np.array:
        """
        Vectorize the query
        :param data: query data
        :return: vectorized query
        """
        morph = pymorphy2.MorphAnalyzer()
        input_query = [lemmatize(text=sent, model=morph) for sent in
                       data]
        query_matrix = [self.fasttext_pool(text, self.vectorizer) for text in
                        input_query]
        return np.array(query_matrix)


class OkapiBM25Corpus(Index):
    def __init__(self,
                 filepath: str = 'questions_about_love.jsonl',
                 n: Optional[int] = None):
        super().__init__(filepath, n)

    def get_index(self) -> Tuple[sparse.csr_matrix, CountVectorizer]:
        """
        Realization of Okapi BM25
        :return: index, fitted vectorizer
        """
        # set constant values
        k = 2
        b = 0.75

        # initialize vectorizers
        tf_vectorizer = TfidfVectorizer(analyzer='word', use_idf=False)
        tfidf_vectorizer = TfidfVectorizer(analyzer='word')
        count_vectorizer = CountVectorizer(analyzer='word')

        # calculate tf
        tf = tf_vectorizer.fit_transform(self.data)
        # calculate idf
        _ = tfidf_vectorizer.fit_transform(self.data)
        # idf = np.expand_dims(tfidf_vectorizer.idf_, axis=0)
        idf = tfidf_vectorizer.idf_
        # calculate BoW
        bow = count_vectorizer.fit_transform(self.data)

        len_d = bow.sum(axis=1)
        avgdl = len_d.mean()
        constant = (k * (1 - b + (b * len_d / avgdl)))

        for i, j in tqdm(zip(*tf.nonzero()),
                         total=tf.nonzero()[0].shape[0],
                         desc="Calculating index"):
            tf[i, j] = (tf[i, j] * (k + 1) * idf[j]) / (tf[i, j] + constant[i])
        return tf, count_vectorizer

    def get_query_matrix(self,
                         data: Union[list, np.array]) -> sparse.csr_matrix:
        """
        Vectorize the query
        :param data: query data
        :return: vectorized query
        """
        morph = pymorphy2.MorphAnalyzer()
        input_query = [lemmatize(text=sent, model=morph) for sent in
                       data]
        query_matrix = self.vectorizer.transform(input_query)
        return query_matrix


class BertCorpus(Index):
    def __init__(self,
                 filepath: str = 'questions_about_love.jsonl',
                 n: Optional[int] = None):
        """
        Initialize index
        :param filepath: path to text files
        :param n: number of documents in corpora
        """
        super().__init__(filepath, n)
        self.tokenizer, self.model = self.vectorizer

    def get_index(self) -> Tuple[np.array, Tuple[AutoTokenizer, AutoModel]]:
        """
        Create index
        :return: index, bert tokenizer and bert model
        """
        tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny')
        model = AutoModel.from_pretrained('cointegrated/rubert-tiny')
        model.eval()
        corpus = tokenizer(self.data.tolist(),
                           padding=True,
                           truncation=True)['input_ids']
        out = []
        for sent in tqdm(corpus, total=len(corpus), desc='Indexing corpus'):
            sent = torch.tensor([sent])
            with torch.no_grad():
                output = model(sent)['last_hidden_state'][:, 0, :][0]
                out.append(output.numpy())
        return np.array(out), (tokenizer, model)

    def get_query_matrix(self, data: Union[list, np.array]) -> np.array:
        """
        Vectorize the query
        :param data: query data
        :return: vectorized query
        """
        if type(data) is not list:
            data = data.tolist()
        input_query = self.tokenizer(data,
                                     padding=True,
                                     truncation=True)['input_ids']
        query_matrix = []
        for sent in tqdm(input_query,
                         total=len(input_query),
                         desc='Indexing query'):
            sent = torch.tensor([sent])
            with torch.no_grad():
                output = self.model(sent)['last_hidden_state'][:, 0, :][0]
                query_matrix.append(output.numpy())
        return np.array(query_matrix)

