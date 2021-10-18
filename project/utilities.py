import pymorphy2
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import nltk
nltk.download('stopwords')


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
