from datetime import datetime

def combine_scores(query_terms, bm25_score, document_metadata, link_data, weights):
    """
    Combines scores for a single document.

    Parameters:
        query_terms: list
            List of query terms.
        bm25_score: float
            BM25 score for the document.
        document_metadata: dict
            Metadata for the document.
        link_data: dict
            Link analysis data for the document.
        weights: dict
            Tunable weights for scoring components and subcomponents.

    Returns:
        float
            Final combined score for the document.
    """
    bm25_weighted = weights.get("bm25", 1.0) * bm25_score
    metadata_score = calculate_metadata_score(
        document_metadata,
        weights.get("metadata", {})
    )
    link_score = calculate_link_score(
        link_data,
        weights.get("link_analysis", {})
    )

    # Combine scores
    final_score = bm25_weighted + metadata_score + link_score

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

def calculate_link_score(link_data, weights):
    """
    Calculate link-based score.
    """
    if not link_data:
        return 0

    pagerank_weight = weights.get("pageRank", 1.0)
    inlink_weight = weights.get("inLinkCount", 0.5)
    outlink_weight = weights.get("outLinkCount", -0.1)

    pagerank_score = pagerank_weight * link_data.get("pageRank", 0)
    inlink_score = inlink_weight * link_data.get("inLinkCount", 0)
    outlink_score = outlink_weight * link_data.get("outLinkCount", 0)

    return pagerank_score + inlink_score + outlink_score
