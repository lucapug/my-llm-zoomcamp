import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import requests
    import pandas as pd
    import minsearch

    from tqdm.auto import tqdm
    return minsearch, pd, requests, tqdm


@app.cell
def _(pd, requests):
    # load documents and ground truth data for search evaluation

    url_prefix = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/03-evaluation/'
    docs_url = url_prefix + 'search_evaluation/documents-with-ids.json'

    documents = requests.get(docs_url).json()

    ground_truth_url = url_prefix + 'search_evaluation/ground-truth-data.csv'
    df_ground_truth = pd.read_csv(ground_truth_url)

    ground_truth = df_ground_truth.to_dict(orient='records')
    return documents, ground_truth


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""####Here, documents contains the documents from the FAQ database with unique IDs, and ground_truth contains generated question-answer pairs.""")
    return


@app.cell
def _(documents, ground_truth):
    documents[0:3], ground_truth[0:3]  # show first 3 items of each list]
    return


@app.cell
def _(documents, ground_truth):
    len(documents), len(ground_truth)  # show the number of items in each list
    return


@app.cell
def _(tqdm):
    # define metrics and evaluation function

    def hit_rate(relevance_total):
        cnt = 0

        for line in relevance_total:
            if True in line:
                cnt = cnt + 1

        return cnt / len(relevance_total)

    def mrr(relevance_total):
        total_score = 0.0

        for line in relevance_total:
            for rank in range(len(line)):
                if line[rank] == True:
                    total_score = total_score + 1 / (rank + 1)

        return total_score / len(relevance_total)

    def evaluate(ground_truth, search_function):
        relevance_total = []

        for q in tqdm(ground_truth):
            doc_id = q['document']
            results = search_function(q)
            relevance = [d['id'] == doc_id for d in results]
            relevance_total.append(relevance)

        return {
            'hit_rate': hit_rate(relevance_total),
            'mrr': mrr(relevance_total),
        }
    return (evaluate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""###Text search (minsearch)""")
    return


@app.cell
def _(documents, minsearch):
    # create an index over the documents with minsearch

    index = minsearch.Index(
        text_fields=["question", "text", "section"],
        keyword_fields=["course", "id"]
    )

    index.fit(documents)
    return (index,)


@app.cell
def _(index):
    # define a search function that uses the index
    def minsearch_search(query, course):
        #boost = {'question': 3.0, 'section': 0.5}
        boost = {'question': 1.5, 'section': 0.1}

        results = index.search(
            query=query,
            filter_dict={'course': course},
            boost_dict=boost,
            num_results=5
        )

        return results
    return (minsearch_search,)


@app.cell
def _(evaluate, ground_truth, minsearch_search):
     # evaluate the search function with minsearch index
    course = 'data-engineering-course'
    evaluate(ground_truth, lambda q: minsearch_search(q['question'], q['course']))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""###Vector search (minsearch)""")
    return


@app.cell
def _():
    from minsearch import VectorSearch
    return (VectorSearch,)


@app.cell
def _():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import make_pipeline
    return TfidfVectorizer, TruncatedSVD, make_pipeline


@app.cell
def _(TfidfVectorizer, TruncatedSVD, documents, make_pipeline):
    # create embeddings for the question field
    texts = []

    for doc in documents:
        t = doc['question']
        texts.append(t)

    pipeline = make_pipeline(
        TfidfVectorizer(min_df=3),
        TruncatedSVD(n_components=128, random_state=1)
    )
    X = pipeline.fit_transform(texts)
    return X, pipeline


@app.cell
def _(X):
    X.shape  # shape of the embeddings matrix
    return


@app.cell
def _(X):
    X[0]
    return


@app.cell
def _(VectorSearch, X, documents):
    # create a vector search index with minsearch
    vindex = VectorSearch(keyword_fields={'course'})
    vindex.fit(X, documents)
    return (vindex,)


@app.cell
def _(evaluate, ground_truth, pipeline, vindex):
    # Funzione per trasformare la query in vettore e fare la ricerca
    def vector_search(q):
        # Trasforma la query in vettore usando lo stesso pipeline
        query_vector = pipeline.transform([q['question']])
        # Cerca usando il vettore della query
        results = vindex.search(
            query_vector=query_vector[0], 
            filter_dict={'course': q['course']},
            num_results=5
        )
        return results

    # Valuta la ricerca vettoriale
    evaluate(ground_truth, vector_search)
    return


if __name__ == "__main__":
    app.run()
