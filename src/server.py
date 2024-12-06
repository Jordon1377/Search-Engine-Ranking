# Basic REST API server that returns document scores for a given query
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from rank import rank_documents
from cache import createCache
import requests
import os

# IPs for the other VMs. May need to change this later since we haven't decided how things talk to each other.
# Also, pretty sure not all of these teams need access to our API so maybe delete some of these?
ACT_TRANSFORM_IP = os.getenv('ACT_TRANSFORM_IP')
LINK_ANALYSIS_IP = os.getenv('LINK_ANALYSIS_IP')
DATA_EVAL_IP = os.getenv('DATA_EVAL_IP')
INDEX_RANKING_IP = os.getenv('INDEX_RANKING_IP')
QUERY_UI_IP = os.getenv('QUERY_UI_IP')

class QueryMetrics:
    def __init__(self, query: str):
        self.query = query
        self.totalDocs = 0
        self.parsedDocs = 0
        self.returnedDocs = 0
        self.timeToRank = 0
        self.inCache = False

    def __init__(self, query: str, totalDocs: int, parsedDocs: int, returnedDocs: int, timeToRank: float, inCache: bool):
        self.query = query
        self.totalDocs = totalDocs
        self.parsedDocs = parsedDocs
        self.returnedDocs = returnedDocs
        self.timeToRank = timeToRank
        self.inCache = inCache

    def __str__(self):
        return f"Query: {self.query}, Total Docs: {self.totalDocs}, Parsed Docs: {self.parsedDocs}, Returned Docs: {self.returnedDocs}, Time to Rank: {self.timeToRank}, In Cache: {self.inCache}"
    
    def __repr__(self):
        return self.__str__()
    
    def toDict(self):
        return {
            "label": "ranked_docs",
            "value": {
                "query": self.query,
                "totalDocs": self.totalDocs,
                "parsedDocs": self.parsedDocs,
                "returnedDocs": self.returnedDocs,
                "timeToRank": self.timeToRank,
                "inCache": self.inCache
            }
        }

app = Flask(__name__)
app.config['ENV'] = 'production'
app.config['DEBUG'] = False
app.config['TESTING'] = False

redis_cache = None

queryMetricsList: list[QueryMetrics] = []

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

    # if request.remote_addr not in ['127.0.0.1', '0.0.0.0', ACT_TRANSFORM_IP, LINK_ANALYSIS_IP, DATA_EVAL_IP, INDEX_RANKING_IP, QUERY_UI_IP]:
        # return jsonify({"error": "Unauthorized"}, 401)

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

    weights = {'bm25_params': {
            'k1': 3.3564610481262207,
            'b': 0.6601634621620178
        }, 
        'bm25': 0.8437421917915344, 
        'pageRank': {
           'pageRank': -2.149700549125555e-06,
           'inLink': 1.035431068885373e-05, 
           'outLink': -0.01162341237068176
        }, 
       'metadata': {
           'freshness': 0.7459535598754883
       }
    }
    
    try:
        if redis_cache and redis_cache.exists(query):
            return jsonify(redis_cache.get(query))
    except Exception as e:
        print(f"Error accessing cache for query: {query} - {e}")
    
    doc_scores = rank_documents(query, weights, fetchTotalDocStatistics, fetchRelevantDocs, fetchDocMetadata, fetchPageRank)
    if not doc_scores:
        return jsonify({"error": "Internal server error"}, 500)
    if start and end:
        doc_scores = doc_scores[start:end]

    if redis_cache:
        redis_cache.set(query, jsonify(doc_scores))

    queryMetricsList.append(QueryMetrics(query, 0, 0, len(doc_scores), 0, False))

    return jsonify(doc_scores)

# TODO: This is probably not how this is supposed to be implemented
def fetchTotalDocStatistics(query: str) -> list:
    """
    Fetches the total document statistics for a given query.

    Parameters:
        query: str The search query entered by the user.
    Returns:
        list
            Total document statistics for a given query.
    Component:
        Indexing
    """

    ip = INDEX_RANKING_IP
    port = 8080 # TODO: fill in the port number
    endpoint = 'getTotalDocStatistics'
    endpoint_url = f'http://{ip}:{port}/{endpoint}'
    data = {"query": query}
    response = requests.get(endpoint_url, json=data)
    if response.status_code != 200:
        return None
    return response.json()

# TODO: This is probably not how this is supposed to be implemented
def fetchRelevantDocs(query: str) -> list:
    """
    Fetches all relevant documents for a given query term.

    Parameters:
        query: str The search query entered by the user.
    Returns:
        list
            All relevant documents for a given query term.
    Component: 
        Indexing
    """
        
    ip = INDEX_RANKING_IP
    port = 8080 # TODO: fill in the port number
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
    Component:
        Indexing
    """

    ip = INDEX_RANKING_IP
    port = 8080 # TODO: fill in the port number
    endpoint = 'getDocumentMetadata'
    endpoint_url = f'http://{ip}:{port}/{endpoint}'
    data = {"docID": docID}
    response = requests.get(endpoint_url, json=data)
    if response.status_code != 200:
        return None
    return response.json()    

def fetchPageRank(url: str) -> float:
    """
    Fetches pagerank score for a document given a doc id.

    Parameters:
        url: str The URL of the document.
    Returns:
        float
            Pagerank score for a document given a doc id.
    Component:
        Link Analysis
    """

    ip = LINK_ANALYSIS_IP
    port = 1234 # TODO: fill in the port number
    endpoint = 'ranking/score'
    endpoint_url = f'http://{ip}:{port}/{endpoint}/{url}'
    response = requests.get(endpoint_url)
    if response.status_code != 200:
        return None
    
    rank = response.json().get('page_rank')
    if not rank:
        return None
    return rank

def reportMetrics():
    """
    Send metrics to the data evaluation team. Called every 24 hours by the scheduler.

    Component:
        Data Evaluation
    """
    # send metrics to the data evaluation team
    ip = DATA_EVAL_IP
    port = '8080' # TODO: fill in the port number
    endpoint = '/v0/ReportMetrics'
    endpoint_url = f'http://{ip}:{port}/{endpoint}'
    query_metrics = [metric.toDict() for metric in queryMetricsList]
    data = {"metrics": query_metrics}

    try:
        response = requests.post(endpoint_url, json=data)
        print(f"Data sent. Response: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error sending metrics: {e}")

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(reportMetrics, 'interval', hours=24)
    scheduler.start()
    redis_cache = createCache()
    app.run(debug=True)
