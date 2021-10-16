from classes import *
import streamlit as st
from datetime import datetime as time
import os


def main():
    st.write(os.getcwd())
    st.markdown("<h1 style='text-align: center; font-size: 700%'>✨🔮✨</h1>",
                unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>❤️ Я могу ответить на твои "
                "вопросы про любовь ❤️</h2>",  unsafe_allow_html=True)
    st.markdown('',  unsafe_allow_html=True)
    st.markdown("<p style='text=align: center;'>Но лучше меня не слушать, "
                "потому что я опираюсь на ответы мейл.ру, а это то еще "
                "место...</p>",  unsafe_allow_html=True)

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

    if search_type == 'Вопросы':
        stype = 'q'
    else:
        stype = 'a'

    if vec == 'CountVectorizer':
        vectorizer = BoWCorpora(stype)
    elif vec == 'TfidfVectorizer':
        vectorizer = TfidfCorpora(stype)
    elif vec == 'Okapi BM25':
        vectorizer = OkapiBM25Corpus(stype)
    elif vec == 'FastText':
        vectorizer = FastTextCorpora(stype)
    else:
        vectorizer = BertCorpus(stype)

    k = st.slider('Как много ответов ты хочешь получить?', 1, 50, 5)
    st.subheader('Задай мне вопрос:')
    query = st.text_input('', 'Как найти любовь?')

    time_start = time.now()
    results = vectorizer.search_query(query, k)
    time_end = time.now()
    for r, result in enumerate(results, 1):
        st.write(f'{r}. {result[0]}')

    st.write(f"<p style='opacity:.5'>Время поиска: {time_end-time_start}</p>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
