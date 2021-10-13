from classes import *


def main():
    fasttext_ind = FastTextCorpora(n=100)
    fasttext_ind.search_query(k=3)

    bert_index = BertCorpus(n=100)
    bert_index.search_query()


if __name__ == '__main__':
    main()
