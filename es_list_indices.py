import requests

# Elasticsearch host
es_host = 'http://localhost:9200'

index_name = 'bbc_rss_index'

# List all indices
indices_response = requests.get(f'{es_host}/_cat/indices?v')
print(indices_response.text)

# List fields in a specific index
fields_response = requests.get(f'{es_host}/{index_name}/_mapping')
print(fields_response.json())