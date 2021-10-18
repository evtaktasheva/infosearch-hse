from classes import *
import streamlit as st
from datetime import datetime as time


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_models():
    bow_q = BoWCorpora(stype='q')
    tfidf_q = TfidfCorpora(stype='q')
    bm25_q = OkapiBM25Corpus(stype='q')
    ft_q = FastTextCorpora(stype='q')
    bert_q = BertCorpus(stype='q')
    bow_a = BoWCorpora(stype='a')
    tfidf_a = TfidfCorpora(stype='a')
    bm25_a = OkapiBM25Corpus(stype='a')
    ft_a = FastTextCorpora(stype='a')
    bert_a = BertCorpus(stype='a')
    return (bow_q, tfidf_q, bm25_q, ft_q, bert_q,
            bow_a, tfidf_a, bm25_a, ft_a, bert_a)


def choose_vectorizer(vec, stype, models):
    (bow_q, tfidf_q, bm25_q, ft_q, bert_q,
    bow_a, tfidf_a, bm25_a, ft_a, bert_a) = models
    if vec == 'CountVectorizer':
        if stype == 'Вопросы':
            return bow_q
        return bow_a
    elif vec == 'TfidfVectorizer':
        if stype == 'Вопросы':
            return tfidf_q
        return tfidf_a
    elif vec == 'Okapi BM25':
        if stype == 'Вопросы':
            return bm25_q
        return bm25_q
    elif vec == 'FastText':
        if stype == 'Вопросы':
            return ft_q
        return ft_a
    else:
        if stype == 'Вопросы':
            return bert_q
        return bert_a


def main():
    st.markdown("<h1 style='text-align: center; font-size: 700%'>✨🔮✨</h1>",
                unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>❤️ Я могу ответить на твои "
                "вопросы про любовь ❤️</h2>",  unsafe_allow_html=True)
    st.markdown('',  unsafe_allow_html=True)
    st.markdown("<p style='text=align: center;'>Но лучше меня не слушать, "
                "потому что я опираюсь на ответы мейл.ру, а это то еще "
                "место...</p>",  unsafe_allow_html=True)

    with st.spinner('✨ Обращаюсь к звездам... ✨'):
        models = load_models()

    vec = st.radio(
          "Какой поисковик использовать?",
          ('CountVectorizer', 'TfidfVectorizer',
           'Okapi BM25', 'FastText', 'BERT'))
    st.write('<style>div.row-widget.stRadio > div{'
             'flex-direction:row;justify-content: '
             'center;padding:10px;}</style>', unsafe_allow_html=True)

    search_type = st.radio(
        "По чему ты хочешь искать?",
        ('Вопросы', 'Ответы'))

    vectorizer = choose_vectorizer(vec, search_type, models)

    k = st.slider('Как много ответов ты хочешь получить?', 1, 50, 5)
    st.subheader('Задай мне вопрос:')
    query = st.text_input('', 'Как найти любовь?')

    time_start = time.now()
    results = vectorizer.search_query(query, k)
    time_end = time.now()

    for r, result in enumerate(results, 1):
        st.write(f'{r}. {result[0]}')

    st.write(f"<p style='opacity:.5'>Время поиска: {time_end-time_start}</p>",
             unsafe_allow_html=True)


if __name__ == '__main__':
    main()
