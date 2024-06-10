# Module 1: Introduction

[](https://github.com/lucapug/llm-zoomcamp/tree/main/01-intro#module-1-introduction)

In this module, we will learn what LLM and RAG are and implement a simple RAG pipeline to answer questions about the FAQ Documents from our Zoomcamp courses

What we will do:

* Index Zoomcamp FAQ documents
  * DE Zoomcamp: [https://docs.google.com/document/d/19bnYs80DwuUimHM65UV3sylsCn2j1vziPOwzBwQrebw/edit](https://docs.google.com/document/d/19bnYs80DwuUimHM65UV3sylsCn2j1vziPOwzBwQrebw/edit)
  * ML Zoomcamp: [https://docs.google.com/document/d/1LpPanc33QJJ6BSsyxVg-pWNMplal84TdZtq10naIhD8/edit](https://docs.google.com/document/d/1LpPanc33QJJ6BSsyxVg-pWNMplal84TdZtq10naIhD8/edit)
  * MLOps Zoomcamp: [https://docs.google.com/document/d/12TlBfhIiKtyBv8RnsoJR6F72bkPDGEvPOItJIxaEzE0/edit](https://docs.google.com/document/d/12TlBfhIiKtyBv8RnsoJR6F72bkPDGEvPOItJIxaEzE0/edit)
* Create a Q&A system for answering questions about these documents

## 1.1 Introduction to LLM and RAG

[](https://github.com/lucapug/llm-zoomcamp/tree/main/01-intro#11-introduction-to-llm-and-rag)

Video

* LLM
* RAG
* RAG architecture
* Course outcome

## 1.2 Preparing the Environment

[](https://github.com/lucapug/llm-zoomcamp/tree/main/01-intro#12-preparing-the-environment)

Video - codespaces

* Installing libraries
* Alternative: installing anaconda or miniconda

```shell notranslate position-relative overflow-auto
pip install tqdm notebook==7.1.2 openai elasticsearch pandas scikit-learn
```

## 1.3 Retrieval

[](https://github.com/lucapug/llm-zoomcamp/tree/main/01-intro#13-retrieval)

Video

* We will use the search engine we build in the [build-your-own-search-engine workshop](https://github.com/alexeygrigorev/build-your-own-search-engine): [minsearch](https://github.com/alexeygrigorev/minsearch)
* Indexing the documents
* Peforming the search

## 1.4 Generation

[](https://github.com/lucapug/llm-zoomcamp/tree/main/01-intro#14-generation)

Video

* Invoking OpenAI API
* Building the prompt
* Getting the answer

## 1.5 Cleaned RAG flow

[](https://github.com/lucapug/llm-zoomcamp/tree/main/01-intro#15-cleaned-rag-flow)

Video

* Cleaning the code we wrote so far
* Making it modular

## 1.6 Searching with ElasticSearch

[](https://github.com/lucapug/llm-zoomcamp/tree/main/01-intro#16-searching-with-elasticsearch)

Video

* Run ElasticSearch with Docker
* Index the documents
* Replace MinSearch with ElasticSearch

Running ElasticSearch:

```shell notranslate position-relative overflow-auto
docker run -it \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3
```

Index settings:

```python notranslate position-relative overflow-auto
{
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"} 
        }
    }
}
```

Query:

```python notranslate position-relative overflow-auto
{
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
                    "course": "data-engineering-zoomcamp"
                }
            }
        }
    }
}
```

# Notes

[](https://github.com/lucapug/llm-zoomcamp/tree/main/01-intro#notes)

* Replace it with a link
* Did you take notes? Add them above this line
