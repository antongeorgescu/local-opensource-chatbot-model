import os

# Environment variables for Elasticsearch connection
ES_HOST = os.getenv('ES_HOST', 'localhost')
ES_PORT = os.getenv('ES_PORT', '9200')
COSSIM_SEARCH_SCORE_MIN = os.getenv('SEARCH_SCORE_MIN', 1.6)
BM25_SEARCH_SCORE_MIN = os.getenv('SEARCH_SCORE_MIN', 2.5)
SEARCH_RESULTS_SIZE = os.getenv('SEARCH_RESULTS_SIZE', 3)
SEARCH_RESULT_ACCURACY = os.getenv('SEARCH_RESULT_ACCURACY', 0.60)
