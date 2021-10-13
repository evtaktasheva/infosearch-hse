from classes import *


def main():
    print('Testing CountVectorizer index...')
    count = BoWCorpora(n=10000)
    print(count.score(k=5))

    print('Testing TfidfVectorizer index...')
    tfidf = TfidfCorpora(n=10000)
    print(tfidf.score(k=5))

    print('Testing Okapi BM25 index...')
    bm25 = OkapiBM25Corpus(n=10000)
    print(bm25.score(k=5))

    print('Testing FastTextVectorizer index...')
    ft = FastTextCorpora(n=10000)
    print(ft.score(k=5))

    print('Testing Bert index...')
    bert = BertCorpus(n=10000)
    print(bert.score(k=5))


if __name__ == '__main__':
    main()