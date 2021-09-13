echo 'Installing requirements...'

python3 -m pip install -r requirements.txt
python3 -m spacy download ru_core_news_sm
