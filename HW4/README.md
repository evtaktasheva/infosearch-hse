## HW4

### How to use

1. Initialize class

```
index = TfidfCorpora(n=100)  # change corpus size to n=100 documents
```

2. Search

```
index.search_query(k=2)   # output top k=2 documents
```
```
Enter your query: Как дела?

>>>> Top 2 documents:
>>>> Зачем, просто от ворот поворот и все дела)
>>>> Пожелай им приятного вечера. Скажи что пора домой, домашние дела и уйди.
```

3. Check index accuracy

```
index.score(k=10)  # check top k=10 predictions
```

```
>>>> 0.42424242424242425
```

### Task 1

Run **task1.py**:

```
python run task1.py
```


### Task 2

Run **task2.py**:

```
python run task2.py
```
