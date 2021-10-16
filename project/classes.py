import os
import torch
import pickle
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
    def __init__(self, stype: str, vec: str):
        """
        Initialize index
        :param stype: search type
        :param vec: vectorizer type
        """
        self.vec = vec
        self.stype = stype
        self.index, self.vectorizer, self.answers = self.get_index()

    def get_index(self) -> tuple:
        """
        Create index
        :return: index and vectorizer
        """
        filepath = f'corpus/{self.vec}_{{}}_{self.stype}.pickle'
        path = os.path.join(os.curdir, filepath)
        answer_path = os.path.join(os.curdir, 'corpus/answers.pickle')

        with open(path.format('features'), 'rb') as f:
            index = pickle.load(f)
        with open(path.format('vectorizer'), 'rb') as f:
            vectorizer = pickle.load(f)
        with open(answer_path, 'rb') as f:
            answers = pickle.load(f)
        return index, vectorizer, answers

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

    def search_query(self, input_query: str, k: int = 5):
        """
        Search query. Sort documents by their cosine similarity to the query
        :param k: number of documents to output
        :return: None
        """
        top_answers = self.top_k([input_query], k)
        return self.answers[top_answers].tolist()

    def print_results(self, ids: np.array):
        """
        Print search results
        :param ids: ids of the documents
        :return: None
        """
        # output results on the screen
        print(f'\nTop {len(ids)} documents:')
        print('\n'.join(self.answers[ids].T[0]))


class TfidfCorpora(Index):
    def __init__(self, stype: str, vec: str = 'tfidf'):
        """
        Initialize index
        """
        super().__init__(stype, vec)
        self.vec = 'tfidf'

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
    def __init__(self, stype: str, vec: str = 'bow'):
        """
        Initialize index
        """
        super().__init__(stype, vec)

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
    def __init__(self, stype: str, vec: str = 'fasttext'):
        """
        Initialize index
        """
        super().__init__(stype, vec)

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
        return np.divide(vector, len(text)) if (vector != 0).any() else vector

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
    def __init__(self, stype: str, vec: str = 'bm25'):
        """
        Initialize index
        """
        super().__init__(stype, vec)

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
    def __init__(self, stype: str, vec: str = 'bert'):
        """
        Initialize index
        """
        super().__init__(stype, vec)
        self.tokenizer, self.model = self.vectorizer

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
        for sent in input_query:
            sent = torch.tensor([sent])
            with torch.no_grad():
                output = self.model(sent)['last_hidden_state'][:, 0, :][0]
                query_matrix.append(output.numpy())
        return np.array(query_matrix)

