import json
from elasticsearch import Elasticsearch, helpers
from transformers import AutoTokenizer, AutoModel
import torch
import re


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

# Create the index with appropriate mappings
def create_index():
    mappings = {
        "mappings": {
            "properties": {
                "content": {
                    "type": "text"
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": 768  # Adjust dimensions based on your model
                }
            }
        }
    }
    
    es.indices.create(index=index_name, body=mappings)
    print(f"Index '{index_name}' created successfully")

def generate_vector(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()[0]

# Function to index documents with embeddings
def index_documents(documents):
    actions = [
        {
            "_index": index_name,
            "_source": {
                "title": doc['title'],
                "embedding": doc['embedding']
            }
        }
        for doc in documents
    ]
    helpers.bulk(es, actions)
    print("Documents indexed successfully")

# Function to perform semantic search
def test_semantic_search(query_embedding, top_k=5):
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": query_embedding}
            }
        }
    }
    response = es.search(
        index=index_name,
        body={
            "size": top_k,
            "query": script_query
        }
    )
    return response['hits']['hits']

# Function to generate embeddings
def get_embeddings(text):
    embeddings = model.encode(text)
    return embeddings

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return escape_nontext_characters(content)

# Function to find a document by title
def get_entry_by_value(json_array, key, value):
    for entry in json_array:
        if entry.get(key) == value:
            return entry
    return None

if __name__ == "__main__":
    # Initialize Elasticsearch client
    es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

    # Define the index name
    index_name = 'qa_data_index'

    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer and model for generating vector representations
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)

    if es.indices.exists(index=index_name):
        print(f"Index '{index_name}' already exists. Will be re-used.")
    else:
        # Create the index
        create_index()

    # Index the existing documents

    # Path to the JSON file
    json_file_path = 'data/qadata.json'

    # Load the JSON file
    with open(json_file_path, 'r') as file:
       qadata = json.load(file)

    documents = []
    for doc in qadata:
        documents.append(
            {"title": doc["title"], "embedding": generate_vector(doc["title"])},
        )
    

    # Index the documents
    index_documents(documents)
    print("Data indexed successfully")

    # Example query embedding
    query_text = "What is RAP and how can I apply for it?"
    query_embedding = generate_vector(query_text)

    # Perform semantic search
    results = test_semantic_search(query_embedding)
    if len(results) == 0:
        print("No results found")
    else:
        for result in results:
            answer = get_entry_by_value(qadata, 'title', result['_source']['title'])
            if answer:
                print(f"Q: {result['_source']['title']}\nA: {answer}\n")
            else:
                print(f"Q: {result['_source']['title']}\nA: Answer not found\n")