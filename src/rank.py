from typing import Callable
from bm25 import bm25_score, get_bm25_params
from combine import combine_scores

def rank_documents(query: str, avg_doc_len: float, weights: dict, 
                   fetch_relevant_docs: Callable[[str], list], 
                   fetch_doc_metadata: Callable[[str], dict], 
                   fetch_pagerank: Callable[[str], float]) -> list:
    """
    Full ranking algorithm for retrieving and ranking documents.

    Parameters:
        query: str
            The search query entered by the user.
        avg_doc_len: float
            Pre-computed average document length for BM25 scoring.
        weights: dict
            Weights for scoring components, including BM25, metadata, and link analysis.
        fetch_relevant_docs: Callable[[str], list]
            API call that returns all relevant documents for a given query term.
        fetch_doc_metadata: Callable[[str], dict]
            API call that returns metadata for a document given a doc id.
        fetch_pagerank: Callable[[str], float]
            API call that returns pagerank score for a document given a doc id.
    Returns:
        list of dict
            Ranked documents with metadata and final scores.
    """
    # Tokenize query and retrieve relevant documents
    query_terms = tokenize_query(query)
    relevant_docs = fetch_relevant_docs(query_terms)
    metadata = fetch_doc_metadata(relevant_docs)
    k1, b, avg_doc_len = get_bm25_params(weights)

    # Process and score each document
    ranked_results = []
    for doc_id, doc_data in relevant_docs.items():
        document_metadata = metadata.get(doc_id, {})
        pagerank_score = fetch_pagerank(metadata.get(doc_id, {}))

        # Compute Scores
        bm25_score_value = bm25_score(
            query_terms, doc_data, inverted_index, k1, b, avg_doc_len
        )
        combined_score = combine_scores(
            bm25_score_value, document_metadata, pagerank_score, weights
        )
        ranked_results.append({
            "docID": doc_id,
            "score": combined_score,
            "metadata": document_metadata
        })

    return sort_ranked_results(ranked_results)

def tokenize_query(query: str):
    """
    Tokenize and preprocess the query.
    Includes unigrams, bigrams, and trigrams.
    """
    def generate_ngrams(words: list, n: int):
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

    words = query.lower().split()  # Tokenize and convert to lowercase
    unigrams = words
    bigrams = generate_ngrams(words, 2)
    trigrams = generate_ngrams(words, 3)
    
    return unigrams + bigrams + trigrams

def sort_ranked_results(ranked_results: list):
    """Sort ranked documents by score in descending order."""
    return sorted(ranked_results, key=lambda x: x["score"], reverse=True)
