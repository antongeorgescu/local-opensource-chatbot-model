import json
from elasticsearch import Elasticsearch, helpers
from transformers import AutoTokenizer, AutoModel
import torch
import re
import feedparser
from es_copilot_dave import generate_vector

# Function to index documents with embeddings
def index_documents(documents, index_name):
    actions = [
        {
            "_index": index_name,
            "_source": {
                "title": doc['title'],
                "embedding": doc['embedding'],
                "description": doc['description'],
                "articlelink": doc['articlelink']
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
    index_name = 'bbc_rss_index'

    # Fetch RSS feed data
    rss_url = "https://feeds.bbci.co.uk/news/world/rss.xml"
    feed = feedparser.parse(rss_url)
    feeditems = [{"title": entry.title, "description": entry.description, "link": entry.link} for entry in feed.entries]
    

    # File path to save the JSON array
    file_path = 'data/bbc_rss.json'

    # Save the JSON array to a file
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(feeditems, file, ensure_ascii=False, indent=4)
    print(f"RSS documents saved to {file_path}")

    documents = []
    for item in feeditems:
        documents.append(
            {"title": item["title"], "embedding": generate_vector(item["title"]),"description": item["description"],"articlelink": item["link"]},
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
                "description": {
                    "type": "text",
                    "index": False
                },
                "articlelink": {
                    "type": "text",
                    "index": False
                },
            }
        }
    }
    
    es.indices.create(index=index_name, body=index_settings)
    print(f"Index '{index_name}' created successfully")
    
    # Index documents
    index_documents(documents,index_name="bbc_rss_index")
    print(f"{len(documents)} articles from BBC News RSS feed indexed successfully")