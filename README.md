# Local AI Copilot with free open-source software

## Project Overview

This project provides a couple of AI Copilots (<b>Dave</b>, <b>Sally</b>) built on a set of Python scripts for creating and querying an Elasticsearch index using BERT embeddings for semantic search. The main scripts for their respectyive copilots are:
* <b> Copilot Sally</b>, assisting with Q&A related to <i>programs of study loans</i>: 
   [`es_create_index_sally.py`](es_create_index_sally.py)
   [`es_copilot_sally.py`](es_copilot_sally.py)
* <b> Copilot Dave</b>, assisting with news collected from <i>BBC News RSS feed</i>:    
   [`es_create_index_dave.py`](es_create_index_dave.py)
   [`es_copilot_dave.py`](es_copilot_dave.py)
* <b> Copilot Phil</b>, assisting with Q&A related to <i>programs of study loans</i>:     
   [`es_create_index_phil.py`](es_create_index_dave.py)
   [`es_copilot_phil.py`](es_copilot_dave.py)

## Requirements

To install the required dependencies, run:

``` sh
pip install -r requirements.txt
```

## Scripts

### es\_create\_index\_sally.py, es\_create\_index\_dave.py, es\_copilot\_phil.py

Each script is responsible for creating an Elasticsearch index and indexing documents with BERT embeddings.

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
python es_create_index_phil.py
```

### es\_copilot\_sally.py, es\_copilot\_dave.py, es\_copilot\_phil.py

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
python es_copilot_phil.py
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
   python es_create_index_phil.py
   ```

2. **Query the Index**:

   ``` sh
   python es_copilot_sally.py
   python es_copilot_dave.py
   python es_copilot_phil.py
   ```

## Configuration

* **Elasticsearch**: Ensure that Elasticsearch is running locally on `localhost:9200`.
* **BERT Model**: The scripts use the `bert-base-uncased` model from Hugging Face's Transformers library.
* **Environment**: A local `environment.py` file holds the environment configuration settings used across the *query* scripts.
* **GPU enabled Ollama models**: To force Ollama to run on a GPU, you can follow the steps below.
> 1. **Ensure CUDA is Installed**: Make sure you have the CUDA Toolkit installed and properly configured on your system. You can download it from the NVIDIA website.
> 2. **Install Ollama**: If you haven't already, install the Ollama binary and the Python package as previously described.
> 3. **Configure Ollama to Use GPU**: 
>> (a) Open the Ollama configuration file, typically located in your user directory (e.g., ~/.ollama/config.yaml).
>> (b) Add or modify the configuration to enable GPU support. For example:
   >>> ```
   >>> gpu:
   >>> enabled: true
   >>> device: 0  # Specify the GPU device ID if you have multiple GPUs
   >>> ```
> 4. **Run Ollama with GPU**:
>>> At the beginning of Python script set OLLAMA_GPU environment variable to enable GPU support:
   >>> ```
   >>> import os
   >>> os.environ['OLLAMA_GPU'] = '1'
   >>> ```
> 5. **Verify GPU Usage**: You can verify that Ollama is using the GPU by monitoring GPU usage with nvidia-smi:
   >>> ```
   >>> nvidia-smi
   >>> ```
>> Or goto to Task Manager / Performance tab and monitor the GPU utilization (device 0, for our particluar example above)  

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

* [Hugging Face Transformers](https://github.com/huggingface/transformers)
* [Elasticsearch](https://www.elastic.co/elasticsearch/)
* [Ollama](https://www.ollama.com)

- - -

For any issues or contributions, please open an issue or submit a pull request on the project's GitHub repository.

```

```
