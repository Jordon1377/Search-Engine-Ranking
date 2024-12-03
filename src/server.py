# Basic REST API server that returns document scores for a given query
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from rank import rank_documents
import requests
import os
import cache

app = Flask(__name__)


# IPs for the other VMs. May need to change this later since we haven't decided how things talk to each other.
# Also, pretty sure not all of these teams need access to our API so maybe delete some of these?
ACT_TRANSFORM_IP = os.getenv('ACT_TRANSFORM_IP')
LINK_ANALYSIS_IP = os.getenv('LINK_ANALYSIS_IP')
DATA_EVAL_IP = os.getenv('DATA_EVAL_IP')
INDEX_RANKING_IP = os.getenv('INDEX_RANKING_IP')
QUERY_UI_IP = os.getenv('QUERY_UI_IP')

@app.route('/status', methods=['GET']) 
def healthCheck():
    return jsonify({"status": "healthy"})

@app.route('/getDocScores', methods=['GET'])
def getDocScores() -> list:
    """
    GET request to get document scores for a given query.

    Returns:
        list of dict
            Ranked documents with metadata and final scores.
    """

    if request.remote_addr not in ['127.0.0.1', ACT_TRANSFORM_IP, LINK_ANALYSIS_IP, DATA_EVAL_IP, INDEX_RANKING_IP, QUERY_UI_IP]:
        return jsonify({"error": "Unauthorized"}, 401)

    data = request.get_json()
    query = data.get('query')
    start = data.get('start')
    end = data.get('end')

    if not query:
        return jsonify({"error": "Query parameter is missing"}, 400)
    if start and not end:
        return jsonify({"error": "End parameter is missing"}, 400)
    if not start and end:
        return jsonify({"error": "Start parameter is missing"}, 400)
    if start and end and (not start.isdigit() or not end.isdigit()):
        return jsonify({"error": "Start and End parameters should be integers"}, 400)
    
    print(f"Query: {query}, Start: {start}, End: {end}")

    weights = {} # TODO: replace with actual weights, placeholder for now
    if redis_cache and redis_cache.exists(query):
        #Return type not correct
        return redis_cache.get(query)

    doc_scores = rank_documents(query, weights, fetchTotalDocStatistics, fetchRelevantDocs, fetchDocMetadata, fetchPageRank)
    if not doc_scores:
        return jsonify({"error": "Internal server error"}, 500)
    if start and end:
        doc_scores = doc_scores[start:end]

    if redis_cache:
        redis_cache.set(query, jsonify(doc_scores))

    return jsonify(doc_scores)

# TODO: This is probably not how this is supposed to be implemented
def fetchTotalDocStatistics(query: str) -> dict:
    """
    Fetches the total document statistics for a given query.

    Parameters:
        query: str The search query entered by the user.
    Returns:
        dict
            The total document statistics for a given query.
    """

    ip = INDEX_RANKING_IP
    port = '???' # TODO: fill in the port number
    endpoint = 'getTotalDocStatistics'
    endpoint_url = f'http://{ip}:{port}/{endpoint}'
    data = {"query": query}
    response = requests.get(endpoint_url, json=data)
    if response.status_code != 200:
        return None
    return response.json()

# TODO: This is probably not how this is supposed to be implemented
def fetchRelevantDocs(query: str) -> dict:
    """
    Fetches all relevant documents for a given query term.

    Parameters:
        query: str The search query entered by the user.
    Returns:
        dict
            All relevant documents for a given query term.
    """
        
    ip = INDEX_RANKING_IP
    port = '???' # TODO: fill in the port number
    endpoint = 'getDocsFromIndex'
    endpoint_url = f'http://{ip}:{port}/{endpoint}'
    # go through each term in the query and get the relevant docs
    for term in query.split():
        data = {"query": term}
        response = requests.get(endpoint_url, json=data)
        if response.status_code != 200:
            return None

def fetchDocMetadata(docID: int) -> dict:
    """
    Fetches metadata for a document given a doc id.

    Parameters:
        docID: int The document id.
    Returns:
        dict
            Metadata for a document given a doc id.
    """

    ip = INDEX_RANKING_IP
    port = '???' # TODO: fill in the port number
    endpoint = 'getDocMetadata'
    endpoint_url = f'http://{ip}:{port}/{endpoint}'
    data = {"docID": docID}
    response = requests.get(endpoint_url, json=data)
    if response.status_code != 200:
        return None
    return response.json()    

def fetchPageRank(url: str) -> dict:
    """
    Fetches pagerank score for a document given a doc id.

    Parameters:
        url: str The URL of the document.
    Returns:
        dict
            Pagerank score for a document given a doc id.
    """

    ip = LINK_ANALYSIS_IP
    port = '???' # TODO: fill in the port number
    endpoint = 'getPageRank'
    endpoint_url = f'http://{ip}:{port}/{endpoint}'
    data = {"url": url}
    response = requests.get(endpoint_url, json=data)
    if response.status_code != 200:
        return None
    return response.json()

def sendMetrics():
    # send metrics to the data evaluation team
    ip = DATA_EVAL_IP
    port = '???' # TODO: fill in the port number
    endpoint = 'updateMetrics'
    endpoint_url = f'http://{ip}:{port}/{endpoint}'
    data = {"metrics": "???"} # TODO: fill in the metrics

    try:
        response = requests.post(endpoint_url, json=data)
        print(f"Data sent. Response: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error sending metrics: {e}")

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(sendMetrics, 'interval', hours=24)
    scheduler.start()
    app.run(debug=True, port=42069)
    redis_cache = cache.createCache()
