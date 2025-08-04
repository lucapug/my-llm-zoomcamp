import marimo

__generated_with = "0.14.16"
app = marimo.App()


@app.cell
def _():
    import json

    with open('documents-with-ids.json', 'rt') as f_in:
        documents = json.load(f_in)
    return (documents,)


@app.cell
def _(documents):
    documents[0]
    return


@app.cell
def _():
    from elasticsearch import Elasticsearch

    es_client = Elasticsearch('http://localhost:9200') 

    # Test connessione
    print(es_client.info())
    return (es_client,)


@app.cell
def _(es_client):
    # create index 'course-questions'

    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "course": {"type": "keyword"},
                "id": {"type": "keyword"},
            }
        }
    }

    index_name = "course-questions"

    es_client.indices.delete(index=index_name, ignore_unavailable=True)
    es_client.indices.create(index=index_name, body=index_settings)
    return (index_name,)


@app.cell
def _(documents, es_client, index_name):
    from tqdm.auto import tqdm

    for doc in tqdm(documents):
        es_client.index(index=index_name, document=doc)
    return (tqdm,)


@app.cell
def _(es_client, index_name):
    def elastic_search(query, course):
        search_query = {
            "size": 5,
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query,
                            "fields": ["question^3", "text", "section"],
                            "type": "best_fields"
                        }
                    },
                    "filter": {
                        "term": {
                            "course": course
                        }
                    }
                }
            }
        }

        response = es_client.search(index=index_name, body=search_query)

        result_docs = []

        for hit in response['hits']['hits']:
            result_docs.append(hit['_source'])

        return result_docs
    return (elastic_search,)


@app.cell
def _(elastic_search):
    elastic_search(
        query="I just discovered the course. Can I still join?",
        course="data-engineering-zoomcamp"
    )
    return


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(pd):
    df_ground_truth = pd.read_csv('ground-truth-data.csv')
    return (df_ground_truth,)


@app.cell
def _(df_ground_truth):
    ground_truth = df_ground_truth.to_dict(orient='records')
    return (ground_truth,)


@app.cell
def _(elastic_search, ground_truth, tqdm):
    relevance_total = []
    for _q in tqdm(ground_truth):
        _doc_id = _q['document']
        _results = elastic_search(query=_q['question'], course=_q['course'])
        _relevance = [d['id'] == _doc_id for d in _results]
        relevance_total.append(_relevance)
    return (relevance_total,)


@app.cell
def _():
    example = [
        [True, False, False, False, False], # 1, 
        [False, False, False, False, False], # 0
        [False, False, False, False, False], # 0 
        [False, False, False, False, False], # 0
        [False, False, False, False, False], # 0 
        [True, False, False, False, False], # 1
        [True, False, False, False, False], # 1
        [True, False, False, False, False], # 1
        [True, False, False, False, False], # 1
        [True, False, False, False, False], # 1 
        [False, False, True, False, False],  # 1/3
        [False, False, False, False, False], # 0
    ]

    # 1 => 1
    # 2 => 1 / 2 = 0.5
    # 3 => 1 / 3 = 0.3333
    # 4 => 0.25
    # 5 => 0.2
    # rank => 1 / rank
    # none => 0
    return (example,)


@app.function
def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)


@app.function
def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)


@app.cell
def _(example):
    hit_rate(example)
    return


@app.cell
def _(example):
    mrr(example)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    - hit-rate (recall)
    - Mean Reciprocal Rank (mrr)
    """
    )
    return


@app.cell
def _(relevance_total):
    hit_rate(relevance_total), mrr(relevance_total)
    return


@app.cell
def _(documents):
    import minsearch

    index = minsearch.Index(
        text_fields=["question", "text", "section"],
        keyword_fields=["course", "id"]
    )

    index.fit(documents)
    return (index,)


@app.cell
def _(index):
    def minsearch_search(query, course):
        boost = {'question': 3.0, 'section': 0.5}
        _results = index.search(query=query, filter_dict={'course': course}, boost_dict=boost, num_results=5)
        return _results
    return (minsearch_search,)


@app.cell
def _(ground_truth, minsearch_search, tqdm):
    relevance_total_1 = []
    for _q in tqdm(ground_truth):
        _doc_id = _q['document']
        _results = minsearch_search(query=_q['question'], course=_q['course'])
        _relevance = [d['id'] == _doc_id for d in _results]
        relevance_total_1.append(_relevance)
    return (relevance_total_1,)


@app.cell
def _(relevance_total_1):
    (hit_rate(relevance_total_1), mrr(relevance_total_1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Compare with ES results:
    ```
    (0.7395720769397017, 0.6032418413658963)
    ```
    """
    )
    return


@app.cell
def _(tqdm):
    def evaluate(ground_truth, search_function):
        relevance_total = []
        for _q in tqdm(ground_truth):
            _doc_id = _q['document']
            _results = search_function(_q)
            _relevance = [d['id'] == _doc_id for d in _results]
            relevance_total.append(_relevance)
        return {'hit_rate': hit_rate(relevance_total), 'mrr': mrr(relevance_total)}
    return (evaluate,)


@app.cell
def _(elastic_search, evaluate, ground_truth):
    evaluate(ground_truth, lambda q: elastic_search(q['question'], q['course']))
    return


@app.cell
def _(df_ground_truth, mo):
    mo.ui.table(df_ground_truth)
    return


@app.cell
def _(evaluate, ground_truth, minsearch_search):
    evaluate(ground_truth, lambda q: minsearch_search(q['question'], q['course']))
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
