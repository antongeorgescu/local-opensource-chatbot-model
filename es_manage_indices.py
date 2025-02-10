import requests
from elasticsearch import Elasticsearch, helpers
import sys

if __name__ == "__main__":
    # Initialize Elasticsearch client
    es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

    if sys.argv[1] == "delete_all":
        # Update the cluster setting to allow wildcard deletions
        es.cluster.put_settings(
            body={
                "persistent": {
                    "action.destructive_requires_name": False
                }
            }
        )
        es.indices.delete(index="*")
    elif sys.argv[1] == "delete":
        es.indices.delete(index=sys.argv[2])
    elif sys.argv[1] == "list_all":
        # List all indices
        indices_response = requests.get(f'http://localhost:9200/_cat/indices?v')
        print(indices_response.text)




