import json
from elasticsearch import Elasticsearch, helpers
from transformers import AutoTokenizer, AutoModel
import torch
import re
import feedparser
from es_copilot_dave import generate_vector

# Function to escape non-text characters
def escape_nontext_characters(text):
    # Define a dictionary of characters to escape
    escape_dict = {
        '\n': '\\n',
        '\t': '\\t',
        '\r': '\\r',
        '\\': '\\\\',
        '\"': '\\"',
        '\'': '\\\''
    }
    # Use regex to replace non-text characters with their escaped versions
    escaped_text = re.sub(r'[\n\t\r\\\"\']', lambda match: escape_dict[match.group(0)], text)
    return escaped_text

# Function to index documents with embeddings
def index_documents(documents, index_name):
    actions = [
        {
            "_index": index_name,
            "_source": {
                "title": doc['title'],
                "embedding": doc['embedding'],
                "content": doc['content'],
            }
        }
        for doc in documents
    ]
    helpers.bulk(es, actions)
    print("Documents indexed successfully")

if __name__ == "__main__":
    # Initialize Elasticsearch client
    es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

    # Define the index name
    index_name = 'qa_data_index_2'

    # File path to save the JSON array
    file_path = 'data/qadata.json'

    # with open(file_path, 'r', encoding='utf-8') as file:
    #     content = file.read()
    # qadata = escape_nontext_characters(content)

    # Load the JSON file
    with open(file_path, 'r') as file:
       qadata = json.load(file)

    documents = []
    for doc in qadata:
        documents.append(
            {"title": doc["title"], "embedding": generate_vector(doc["title"]),"content": doc["content"]},
        )

    # Create an index
    # Check if the index exists
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        
    # Define the index settings and mappings
    index_settings = {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "similarity": {
                    "default": {
                        "type": "BM25",
                        "k1": 1.2,
                        "b": 0.75
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "title": {
                    "type": "text"
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": 768  # Adjust dimensions based on your model
                },
                "content": {
                    "type": "text",
                    "index": False
                },
            }
        }
    }
    
    es.indices.create(index=index_name, body=index_settings)
    print(f"Index '{index_name}' created successfully")
    
    # Index documents
    index_documents(documents,index_name=index_name)
    print(f"{len(documents)} entries from {file_path} indexed successfully")