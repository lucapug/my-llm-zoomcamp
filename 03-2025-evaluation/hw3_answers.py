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
    from qdrant_client import QdrantClient, models
    #from fastembed import TextEmbedding
    return QdrantClient, minsearch, models, pd, requests, tqdm


@app.cell
def _(pd, requests):
    # load documents and ground truth data for search evaluation

    url_prefix = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/03-evaluation/'
    docs_url = url_prefix + 'search_evaluation/documents-with-ids.json'

    documents = requests.get(docs_url).json()

    ground_truth_url = url_prefix + 'search_evaluation/ground-truth-data.csv'
    df_ground_truth = pd.read_csv(ground_truth_url)

    ground_truth = df_ground_truth.to_dict(orient='records')
    return documents, ground_truth, url_prefix


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


@app.cell
def _(documents):
    # create combination of question and answer
    texts2 = []

    for doc2 in documents:
        t2 = doc2['question'] + ' ' + doc2['text']
        texts2.append(t2)
    return (texts2,)


@app.cell
def _(pipeline, texts2):
    # create embeddings for the combined question and answer

    # fit the pipeline on the combined texts
    X2 = pipeline.fit_transform(texts2)
    return (X2,)


@app.cell
def _(VectorSearch, X2, documents):
    vindex2 = VectorSearch(keyword_fields={'course'})
    vindex2.fit(X2, documents)
    return (vindex2,)


@app.cell
def _(evaluate, ground_truth, pipeline, vindex2):
    # Funzione per trasformare la query in vettore e fare la ricerca
    def vector_search2(q):
        # Trasforma la query in vettore usando lo stesso pipeline
        query_vector = pipeline.transform([q['question']])
        # Cerca usando il vettore della query
        results = vindex2.search(
            query_vector=query_vector[0], 
            filter_dict={'course': q['course']},
            num_results=5
        )
        return results

    # Valuta la ricerca vettoriale
    evaluate(ground_truth, vector_search2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""###Vector search (qdrant)###""")
    return


@app.cell
def _(QdrantClient):
    qd_client = QdrantClient("http://localhost:6333")
    return (qd_client,)


@app.cell
def _():
    EMBEDDING_DIMENSIONALITY = 512
    model_handle = "jinaai/jina-embeddings-v2-small-en"
    return EMBEDDING_DIMENSIONALITY, model_handle


@app.cell
def _():
    collection_name = "hw3-2025"
    return (collection_name,)


@app.cell
def _(collection_name, qd_client):
    # se già esiste la cancella per ricrearla
    qd_client.delete_collection(collection_name=collection_name)
    return


@app.cell
def _(EMBEDDING_DIMENSIONALITY, collection_name, models, qd_client):
    qd_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSIONALITY,
            distance=models.Distance.COSINE
        )
    )
    return


@app.cell
def _(collection_name, qd_client):
    qd_client.create_payload_index(
        collection_name=collection_name,
        field_name="course",
        field_schema="keyword"
    )
    return


@app.cell
def _(documents, model_handle, models):
    points = []

    for i, doc3 in enumerate(documents):
        text = doc3['question'] + ' ' + doc3['text']
        vector = models.Document(text=text, model=model_handle)
        point = models.PointStruct(
            id=i,
            vector=vector,
            payload=doc3
        )
        points.append(point)
    return (points,)


@app.cell
def _(collection_name, points, qd_client):
    qd_client.upsert(
        collection_name=collection_name,
        points=points
    )
    return


@app.cell
def _(collection_name, model_handle, models, qd_client):
    def vector_search3(q):  # q è un dizionario dal ground_truth
        #print('vector_search is used')

        # Estrai la domanda e il corso dal dizionario
        question = q['question']  # ✅ Estrai la stringa della domanda
        course = q['course']      # ✅ Usa il corso dal ground_truth invece di hardcodarlo

        query_points = qd_client.query_points(
            collection_name=collection_name,
            query=models.Document(
                text=question,  # ✅ Ora question è una stringa
                model=model_handle 
            ),
            query_filter=models.Filter( 
                must=[
                    models.FieldCondition(
                        key="course",
                        match=models.MatchValue(value=course)  # ✅ Usa il corso corretto
                    )
                ]
            ),
            limit=5,
            with_payload=True
        )

        results = []

        for point in query_points.points:
            results.append(point.payload)

        return results
    return (vector_search3,)


@app.cell
def _(evaluate, ground_truth, vector_search3):
    # Valuta la ricerca vettoriale
    evaluate(ground_truth, vector_search3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### RAG offline evaluation: cosine similarity ###""")
    return


@app.cell
def _():
    import numpy as np
    def cosine(u, v):
        u_norm = np.sqrt(u.dot(u))
        v_norm = np.sqrt(v.dot(v))
        return u.dot(v) / (u_norm * v_norm)
    return cosine, np


@app.cell
def _(pd, url_prefix):
    # load data from gpt-4o-mini evaluation (llm as a judge)

    results_url = url_prefix + 'rag_evaluation/data/results-gpt4o-mini.csv'
    df_results = pd.read_csv(results_url)
    return (df_results,)


@app.cell
def _(df_results):
    df_results.head(3)  # show first 3 rows of the dataframe
    return


@app.cell
def _(pipeline):
    pipeline
    return


@app.cell
def _(df_results, pipeline):
    X1 = pipeline.fit(df_results.answer_llm + ' ' + df_results.answer_orig + ' ' + df_results.question)
    return (X1,)


@app.cell
def _(df_results):
    results_gpt4o_mini = df_results.to_dict(orient='records')
    return (results_gpt4o_mini,)


@app.cell
def _(X1, cosine):
    def compute_similarity(record):
        answer_orig = record['answer_orig']
        answer_llm = record['answer_llm']

        v_llm = X1.transform([answer_llm])
        v_orig = X1.transform([answer_orig])

        return cosine(v_llm[0], v_orig[0])
    return (compute_similarity,)


@app.cell
def _(compute_similarity, results_gpt4o_mini, tqdm):
    similarity_4o_mini = []

    for record in tqdm(results_gpt4o_mini):
        sim = compute_similarity(record)
        similarity_4o_mini.append(sim)
    return (similarity_4o_mini,)


@app.cell
def _(similarity_4o_mini):
    similarity_4o_mini[:3]  # show first 3 similarity scores
    return


@app.cell
def _(df_results, similarity_4o_mini):
    df_results['cosine_similarity'] = similarity_4o_mini
    df_results.head(3)  # show first 3 rows of the dataframe with cosine similarity']
    return


@app.cell
def _(df_results):
    df_results['cosine_similarity'].mean()  # mean cosine similarity score
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Alternative text similarity: Rouge ###""")
    return


@app.cell
def _(df_results):
    from rouge import Rouge
    rouge_scorer = Rouge()

    r = df_results.iloc[10]
    scores = rouge_scorer.get_scores(r.answer_llm, r.answer_orig)[0]
    scores
    return rouge_scorer, scores


@app.cell
def _(scores):
    scores['rouge-1']['f']  # F1 score for Rouge-L
    return


@app.cell
def _(df_results, np, rouge_scorer):
    tot_scores = []

    for i1 in range(df_results.shape[0]):
        r1 = df_results.iloc[i1]
        scores1 = rouge_scorer.get_scores(r1.answer_llm, r1.answer_orig)[0]
        tot_scores.append(scores1['rouge-1']['f'])

    np.mean(tot_scores)  # mean Rouge-1 F1 score

    return


if __name__ == "__main__":
    app.run()
