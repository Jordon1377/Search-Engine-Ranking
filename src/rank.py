from bm25 import bm25_score, get_bm25_params
from combine import combine_score

def rank_documents(query, avg_doc_len, inverted_index, metadata, link_analysis, weights):
    """
    Full ranking algorithm for retrieving and ranking documents.

    Parameters:
        query: str
            The search query entered by the user.
        avg_doc_len: float
            Pre-computed average document length for BM25 scoring.
        inverted_index: dict
            Inverted index providing term-document mappings and term frequencies.
        metadata: dict
            Pre-fetched metadata for documents (keyed by docID).
        link_analysis: dict
            Pre-fetched link analysis data for documents (keyed by docID).
        weights: dict
            Weights for scoring components, including BM25, metadata, and link analysis.

    Returns:
        list of dict
            Ranked documents with metadata and final scores.
    """
    # Tokenize query and retrieve relevant documents
    query_terms = tokenize_query(query)
    relevant_docs = fetch_relevant_documents(query_terms, inverted_index)
    k1, b = get_bm25_params(weights)

    # Process and score each document
    ranked_results = []
    for doc_id, doc_data in relevant_docs.items():
        document_metadata = metadata.get(doc_id, {})
        link_data = link_analysis.get(doc_id, {})

        # Compute Scores
        bm25_score_value = bm25_score(
            query_terms, doc_data, inverted_index, k1, b, avg_doc_len
        )
        combined_score = combine_scores(
            query_terms, bm25_score_value, document_metadata, link_data, weights
        )
        ranked_results.append({
            "docID": doc_id,
            "score": combined_score,
            "metadata": document_metadata
        })

    return sort_ranked_results(ranked_results)

def tokenize_query(query):
    """
    Tokenize and preprocess the query.
    """
    # TODO: Add Bigrams Trigram
    return query.lower().split()

def fetch_relevant_documents(query_terms, inverted_index):
    """Retrieve relevant documents from the inverted index."""
    relevant_docs = {}
    for term in query_terms:
        if term not in inverted_index:
            continue
        for entry in inverted_index[term]["index"]:
            doc_id = entry["docID"]
            if doc_id not in relevant_docs:
                relevant_docs[doc_id] = {"term_frequency": {}, "docLength": None}
            relevant_docs[doc_id]["term_frequency"][term] = entry["frequency"]
    return relevant_docs

def sort_ranked_results(ranked_results):
    """Sort ranked documents by score in descending order."""
    return sorted(ranked_results, key=lambda x: x["score"], reverse=True)
