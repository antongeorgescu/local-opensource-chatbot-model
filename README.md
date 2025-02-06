# README.md

## Project Overview

This project provides a set of Python scripts for creating and querying an Elasticsearch index using BERT embeddings for semantic search. The main scripts are [`es_create_index.py`](es_create_index.py ) and [`es_query_loop.py`](es_query_loop.py ).

## Requirements

To install the required dependencies, run:

```sh
pip install -r requirements.txt

```markdown
# README.md

## Project Overview

This project provides a set of Python scripts for creating and querying an Elasticsearch index using BERT embeddings for semantic search. The main scripts are [`es_create_index.py`](es_create_index.py) and [`es_query_loop.py`](es_query_loop.py).

## Requirements

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

## Scripts

### 

es_create_index.py



This script is responsible for creating an Elasticsearch index and indexing documents with BERT embeddings.

#### Key Functions

- **`create_index()`**: Creates an Elasticsearch index with appropriate mappings for text and dense vector fields.
- **`generate_vector(text)`**: Generates BERT embeddings for the given text using a pre-trained BERT model.
- **`index_documents(documents)`**: Indexes a list of documents with their embeddings into Elasticsearch.
- **`test_semantic_search(query_embedding, top_k=5)`**: Performs a semantic search on the indexed documents using the provided query embedding.

#### Code Explanation

- **`create_index()`**: This function sets up the Elasticsearch index with the necessary mappings for text fields and dense vector fields to store BERT embeddings.
- **`generate_vector(text)`**: This function uses a pre-trained BERT model to convert input text into a dense vector representation.
- **`index_documents(documents)`**: This function takes a list of documents, generates their BERT embeddings, and indexes them into the Elasticsearch index.
- **`test_semantic_search(query_embedding, top_k=5)`**: This function performs a semantic search by comparing the query embedding with the indexed document embeddings and returns the top-k results.

#### Usage

To create the index and index documents, run:

```sh
python es_create_index.py
```

### 

es_query_loop.py



This script provides an interactive loop for querying the Elasticsearch index using semantic search.

#### Key Functions

- **`semantic_search(query_embedding, top_k=5)`**: Performs a semantic search on the indexed documents using the provided query embedding.
- **`generate_vector(text)`**: Generates BERT embeddings for the given text using a pre-trained BERT model.
- **`get_entry_by_value(json_array, key, value)`**: Finds a document in a JSON array by a specific key-value pair.

#### Code Explanation

- **`semantic_search(query_embedding, top_k=5)`**: This function performs a semantic search by comparing the query embedding with the indexed document embeddings and returns the top-k results.
- **`generate_vector(text)`**: This function uses a pre-trained BERT model to convert input text into a dense vector representation.
- **`get_entry_by_value(json_array, key, value)`**: This function searches through a JSON array to find a document that matches a specific key-value pair.

#### Usage

To start the interactive query loop, run:

```sh
python es_query_loop.py
```

## Example Usage

1. **Create the Index and Index Documents**:
   ```sh
   python es_create_index.py
   ```

2. **Query the Index**:
   ```sh
   python es_query_loop.py
   ```

## Configuration

- **Elasticsearch**: Ensure that Elasticsearch is running locally on `localhost:9200`.
- **BERT Model**: The scripts use the `bert-base-uncased` model from Hugging Face's Transformers library.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Elasticsearch](https://www.elastic.co/elasticsearch/)

---

For any issues or contributions, please open an issue or submit a pull request on the project's GitHub repository.
```