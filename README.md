# Local AI Copilot with free open-source software

## Project Overview

This project provides a couple of AI Copilots (<b>Dave</b>, <b>Sally</b>) built on a set of Python scripts for creating and querying an Elasticsearch index using BERT embeddings for semantic search. The main scripts for their respectyive copilots are:
* <b> Copilot Sally</b>, assisting with Q&A related to <i>programs of study loans</i>: 
   [`es_create_index_sally.py`](es_create_index_sally.py)
   [`es_copilot_sally.py`](es_copilot_sally.py)
* <b> Copilot Dave</b>, assisting with news collected from <i>BBC News RSS feed</i>:    
   [`es_create_index_dave.py`](es_create_index_dave.py)
   [`es_copilot_dave.py`](es_copilot_dave.py)

## Requirements

To install the required dependencies, run:

``` sh
pip install -r requirements.txt
```

## Scripts

### es\_create\_index.py

This script is responsible for creating an Elasticsearch index and indexing documents with BERT embeddings.

#### Key Functions

* <b>`create_index()`</b>: Creates an Elasticsearch index with appropriate mappings for text and dense vector fields.
* <b>`generate_vector(text)`</b>: Generates BERT embeddings for the given text using a pre-trained BERT model.
* <b>`index_documents(documents)`</b>: Indexes a list of documents with their embeddings into Elasticsearch.
* <b>`test_semantic_search(query_embedding, top_k=5)`</b>: Performs a semantic search on the indexed documents using the provided query embedding.

#### Code Explanation

* <b>`create_index()`</b>: This function sets up the Elasticsearch index with the necessary mappings for text fields and dense vector fields to store BERT embeddings.
* <b>`generate_vector(text)`</b>: This function uses a pre-trained BERT model to convert input text into a dense vector representation.
* <b>`index_documents(documents)`</b>: This function takes a list of documents, generates their BERT embeddings, and indexes them into the Elasticsearch index.
* <b>`test_semantic_search(query_embedding, top_k=5)`</b>: This function performs a semantic search by comparing the query embedding with the indexed document embeddings and returns the top-k results.

#### Usage

To create the index and index documents, run:

``` sh
python es_create_index_sally.py
python es_create_index_dave.py
```

### es\_copilot\_sally.py, es\_copilot\_dave.py

These two script provide an interactive loop for querying the Elasticsearch index using semantic search.

#### Key Functions

* <b>`semantic_search_cossim(query_embedding, top_k=5)`</b>: Performs a semantic search on the indexed documents using the provided query embedding anc cosineSimilarity similarity algorithm
* <b>`semantic_search_b25(query_embedding, top_k=5)`</b>: Performs a semantic search on the indexed documents using the provided query embedding anc B25 similarity algorithm
* <b>`generate_vector(text)`</b>: Generates BERT embeddings for the given text using a pre-trained BERT model.
* <b>`get_entry_by_value(json_array, key, value)`</b>: Finds a document in a JSON array by a specific key-value pair.

#### Code Explanation

* <b>`semantic_search_*(query_embedding, top_k=5)`</b>: This function performs a semantic search by comparing the query embedding with the indexed document embeddings and returns the top-k results.
* <b>`generate_vector(text)`</b>: This function uses a pre-trained BERT model to convert input text into a dense vector representation.
* <b>`get_entry_by_value(json_array, key, value)`</b>: This function searches through a JSON array to find a document that matches a specific key-value pair.

#### Usage

To start the interactive query loop, run:

``` sh
python es_copilot_sally.py
python es_copilot_dave.py
```

## Functional diagram

Below is the functional diagram that covers the <i>unified logical functionality</i> of the scripts es\_create\_index\_dave.py and  es\_copilot\_dave.py

``` mermaid
graph TD
    A[Start] --> B[Initialize Elasticsearch]
    B --> C[Fetch BBC RSS Feed]
    C --> D[Generate BERT Embeddings]
    D --> E[Index Documents in Elasticsearch]
    E --> F[Query Loop]
    F --> G{User Input}
    G -->|Query| H[Generate Query Embedding]
    H --> I[Perform Semantic Search]
    I --> J[Display Results]
    G -->|Exit| K[End]
```

## Example Usage

1. **Create the Index and Index Documents**:
   ```sh
   python es_create_index_sally.py
   python es_create_index_dave.py
   ```

2. **Query the Index**:

   ``` sh
   python es_copilot_sally.py
   python es_copilot_dave.py
   ```

## Configuration

* **Elasticsearch**: Ensure that Elasticsearch is running locally on `localhost:9200`.
* **BERT Model**: The scripts use the `bert-base-uncased` model from Hugging Face's Transformers library.
* **Environment**: A local `environment.py` file holds the environment configuration settings used across the *query* scripts.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

* [Hugging Face Transformers](https://github.com/huggingface/transformers)
* [Elasticsearch](https://www.elastic.co/elasticsearch/)

- - -

For any issues or contributions, please open an issue or submit a pull request on the project's GitHub repository.

```

```
