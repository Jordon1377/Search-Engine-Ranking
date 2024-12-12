from typing import Callable

from bm25 import bm25_score, get_bm25_params
from combine import combine_scores

def rank_documents(query: str, weights: dict, doc_stats: list,
                   fetch_relevant_docs: Callable[[str], list],
                   fetch_doc_metadata: Callable[[str], dict],
                   fetch_pagerank: Callable[[str], float]) -> list:
    """
    Full ranking algorithm for retrieving and ranking documents.

    Parameters:
        query: str The search query entered by the user.
        weights: dict Weights for scoring components.
        doc_stats: document stats from indexing.
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
    avg_doc_len, total_docs = doc_stats
    k1, b = get_bm25_params(weights)
    query_terms = tokenize_query(query)

    # Process and score each document
    ranked_results = {}
    num_parsed = 0
    for term in query_terms:
        relevant_docs = fetch_relevant_docs(term)
        if not relevant_docs: continue
        num_parsed += len(relevant_docs["index"])

        for doc in relevant_docs["index"]:
            doc_id = doc["docID"]
            metadata = fetch_doc_metadata(doc_id)["metadata"]
            pagerank_score = fetch_pagerank(metadata["URL"])

            # Compute Scores
            doc_occurances = len(relevant_docs["index"])
            bm25_value = bm25_score(
                term, doc, metadata, k1, b, avg_doc_len, total_docs, doc_occurances
            )
            combined_score = combine_scores(
                bm25_value, metadata, pagerank_score, weights
            )

            ranked_results.setdefault(doc_id, {
                "docID": doc_id,
                "score": 0,
                "metadata": metadata
            })
            ranked_results[doc_id]["score"] += combined_score

    return [sort_ranked_results(list(ranked_results.values())), num_parsed]

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
    results = sorted(ranked_results, key=lambda x: x["score"], reverse=True)

    # Add rank into return
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank

    return results
