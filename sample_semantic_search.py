import json
from elasticsearch import Elasticsearch, helpers
from transformers import AutoTokenizer, AutoModel
import torch

# Initialize Elasticsearch client
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# Define the index name
index_name = 'semantic_search_index'

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model for generating vector representations
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").to(device)

def generate_vector(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()[0]

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
    if es.indices.exists(index=index_name):
        print(f"Index '{index_name}' already exists. Will be deleted.")
        es.indices.delete(index=index_name)
    es.indices.create(index=index_name, body=mappings)
    print(f"Index '{index_name}' created successfully")
        

# Function to index documents with embeddings
def index_documents(documents):
    actions = [
        {
            "_index": index_name,
            "_source": {
                "content": doc['content'],
                "embedding": doc['embedding']
            }
        }
        for doc in documents
    ]
    helpers.bulk(es, actions)
    print("Documents indexed successfully")

# Function to perform semantic search
def semantic_search(query_embedding, top_k=5):
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

# Example usage
if __name__ == "__main__":
    # Create the index
    create_index()

    # Example documents with embeddings

    article1 = "Anton has a Subaru Forester car that is quite old. He does not want to buy an electric car."
    article2 = "Alex, who is Anton's son, has a Subaru Impreza car. He loves hybrid cars, because they are better fit to Canadian winters."
    documents = [
        {"content": article1, "embedding": generate_vector(article1)},
        {"content": article2,"embedding": generate_vector(article2)},
        # Add more documents as needed
    ]

    # Index the documents
    index_documents(documents)

    # Example query embedding
    query_text = "Which cars people think are better fit for Canadian winters?"
    query_embedding = generate_vector(query_text)

    # Perform semantic search
    results = semantic_search(query_embedding)
    for result in results:
        print(result['_source']['content'])