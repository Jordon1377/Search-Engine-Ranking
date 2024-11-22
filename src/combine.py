from datetime import datetime

def combine_scores(bm25_score, document_metadata, pagerank_score, weights):
    """
    Combines scores for a single document.

    Parameters:
        query_terms: list
            List of query terms.
        bm25_score: float
            BM25 score for the document.
        document_metadata: dict
            Metadata for the document.
        pagerank_score: float
            Link analysis data for the document.
        weights: dict
            Tunable weights for scoring components and subcomponents.

    Returns:
        float
            Final combined score for the document.
    """
    bm25_weighted = weights.get("bm25", 1.0) * bm25_score
    pagerank_weighted = weights.get("pagerank", 1.0) * pagerank_score
    metadata_score = calculate_metadata_score(
        document_metadata,
        weights.get("metadata", {})
    )
    
    # Combine scores
    final_score = bm25_weighted + metadata_score + pagerank_weighted

    return final_score

def calculate_metadata_score(metadata, weights):
    """
    Calculate a metadata-based score for a document.
    """
    if not metadata:
        return 0

    freshness_weight = weights.get("freshness", 1.0)
    doc_type_weights = weights.get("docType", {})

    # Freshness based on timeLastUpdated
    freshness_score = 0
    if "timeLastUpdated" in metadata:
        last_update_time = datetime.fromisoformat(metadata["timeLastUpdated"])
        time_diff = datetime.now() - last_update_time
        freshness_score = freshness_weight / max(1, time_diff.days)

    # Document type influence
    doc_type_score = doc_type_weights.get(metadata.get("docType", ""), 0.5)

    return freshness_score + doc_type_score
