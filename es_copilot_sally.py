from elasticsearch import Elasticsearch, ConnectionError, TransportError
import ollama
import json, os
import re
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
import torch
# from transformers import AutoTokenizer, AutoModel
from environment import ES_HOST, ES_PORT, SEARCH_RESULTS_SIZE, COSSIM_SEARCH_SCORE_MIN, BM25_SEARCH_SCORE_MIN  # Import environment variables
import pickle

# Set OLLAMA_GPU environment variable to enable GPU support
os.environ['OLLAMA_GPU'] = '1'

# Function to perform semantic search
# Function to perform semantic search
def semantic_search_cossim(query_embedding, top_k=5):
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

def semantic_search_bm25(query, top_k=5):
    query_body = {
        "size": top_k,
        "query": {
            "match": {
                "title": query
            }
        }
    }
    response = es.search(index=index_name, body=query_body)
    return response['hits']['hits']

def generate_vector(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().tolist()[0]

# Function to find a document by title
def get_entry_by_value(json_array, key, value):
    for entry in json_array:
        if entry.get(key) == value:
            return entry
    return None

# Example usage
if __name__ == "__main__":
    # Initialize the console
    console = Console()

    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: {device}", style="bold green")

    # Define the index name
    index_name = 'qa_data_index_sally'

    # Load the tokenizer and model for generating vector representations
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # model = AutoModel.from_pretrained("bert-base-uncased").to(device)

    # Load the tokenizer from the saved directory
    with open('serialize_bert/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    # Load the model from the saved directory
    with open('serialize_bert/model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Path to the JSON file
    json_file_path = 'data/qadata.json'

    # Load the JSON file
    with open(json_file_path, 'r') as file:
       qadata = json.load(file)

    # Connect to the local Elasticsearch instance
    es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

    try:
        # Check if the connection is successful
        if es.ping():
            console.print("*** Connected to Elasticsearch", style="bold green")
        else:
            console.print("*** Could not connect to Elasticsearch", style="bold red")
    except ConnectionError as e:
        console.print(f"*** ConnectionError: {e}", style="bold red")
    except TransportError as e:
        console.print(f"*** TransportError: {e}", style="bold red")
    except Exception as e:
        console.print(f"*** An error occurred: {e}", style="bold red")

    # Infinite loop to keep prompting the user for queries
    while True:
        # Prompt the user to choose between grounded data and generic data
        model_name = Prompt.ask("Choose AI model", choices=["deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-v2"], default="deepseek-r1:1.5b")
        # Show thinking for a reasoning model
        show_thinking = Prompt.ask("Show thinking", choices=["yes", "no"], default="no")
        # Show thinking for a reasoning model
        show_results = Prompt.ask(f"Show the search results (max {SEARCH_RESULTS_SIZE})", choices=["yes", "no"], default="no")
        # Prompt the user to choose between grounded data and generic data
        data_type = Prompt.ask("Choose data type", choices=["student loans", "general"], default="student loans")
        # Prompt the user for a query
        QUERY = Prompt.ask("Enter your query (or type 'quit' to exit)",default="What is RAP and how can I apply for it?")

        # Check if the user wants to quit
        if QUERY.lower() == 'quit':
            console.print("Exiting...", style="bold red")
            break

        # Display the query and data type in a lined box
        console.print(Panel(f"Query: {QUERY}\nData Type: {data_type}", title="Query", border_style="green"))

        if data_type == "student loans":
            
            try:
                # Perform semantic search
                query_embedding = generate_vector(QUERY)

                answers = []

                # Perform semantic search with cosine similarity
                results = semantic_search_cossim(query_embedding)
                if len(results) == 0:
                    print("No results found")
                else:
                    search_results = ""
                    for result in results:
                        if result['_score'] > COSSIM_SEARCH_SCORE_MIN:
                            answer = get_entry_by_value(qadata, 'title', result['_source']['title'])
                            if answer:
                                answers.append(answer) 
                                search_results= search_results + f"Q [{result['_score']}]: {result['_source']['title']}\nA: {answer}\n"                       
                    if show_results == "yes":
                        console.print(Panel(search_results, title="Search Results [cossim]", border_style="red"))
                
                results_str = " ".join(answer['content'] for answer in answers if 'content' in answer)
                prompt = f"Summarize the following text in a good journalistic style, that makes sense, is concise and focused on the topic, and do not enrich the response:{results_str}"  

                # Generate a response using the DeepSeek selected model
                response = ollama.generate(model=model_name, prompt=prompt)
                
                if show_thinking == "no":
                    # Remove the string between <think> and </think> in the response
                    display_response = re.sub(r'<think>.*?</think>', '', response['response'], flags=re.DOTALL)
                else:
                    display_response = response['response']

                # Print the response
                console.print(Panel(display_response, title="Response [cossim]", border_style="red"))

                # Perform semantic search with BM25
                results = semantic_search_bm25(QUERY)
                if len(results) == 0:
                    print("No results found")
                else:
                    search_results = ""
                    for result in results:
                        if result['_score'] > BM25_SEARCH_SCORE_MIN:
                            answer = get_entry_by_value(qadata, 'title', result['_source']['title'])
                            if answer:
                                answers.append(answer) 
                                search_results= search_results + f"Q [{result['_score']}]: {result['_source']['title']}\nA: {answer}\n"                       
                    if show_results == "yes":
                        console.print(Panel(search_results, title="Search Results [BM25]", border_style="blue"))
                
                results_str = " ".join(answer['content'] for answer in answers if 'content' in answer)
                prompt = f"Summarize the following text in a good journalistic style, that makes sense, is concise and focused on the topic, and do not enrich the response:{results_str}"  

                # Generate a response using the DeepSeek selected model
                response = ollama.generate(model=model_name, prompt=prompt)
                
                if show_thinking == "no":
                    # Remove the string between <think> and </think> in the response
                    display_response = re.sub(r'<think>.*?</think>', '', response['response'], flags=re.DOTALL)
                else:
                    display_response = response['response']

                # Print the response
                console.print(Panel(display_response, title="Response [BM25]", border_style="blue"))

            except TransportError as e:
                console.print(f"*** TransportError: {e}", style="bold red")
            except Exception as e:
                console.print(f"*** An error occurred: {e}", style="bold red")
        else:
            try:
                # Generate a response using the DeepSeek model
                response = ollama.generate(model=model_name, prompt=QUERY)

                if show_thinking == "no":
                    # Remove the string between <think> and </think> in the response
                    display_response = re.sub(r'<think>.*?</think>', '', response['response'], flags=re.DOTALL)
                else:
                    display_response = response['response']

                # Print the response
                console.print(Panel(display_response, title="Response", border_style="blue"))
            except Exception as e:
                console.print(f"*** An error occurred: {e}", style="bold red")


