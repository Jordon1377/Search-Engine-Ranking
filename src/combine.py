from datetime import datetime, timezone

def combine_scores(bm25_score: float, document_metadata: dict,
                   pagerank: dict, weights: dict) -> float:
    """
    Combines scores for a single document.

    Parameters:
        bm25_score: float BM25 score for the document.
        document_metadata: dict Metadata for the document.
        pagerank_score: float Pagerank score for the document.
        weights: dict Tunable weights for scoring components and subcomponents.

    Returns:
        float Final combined score for the document.
    """
    bm25_weighted = weights.get("bm25", 1.0) * bm25_score
    pagerank_score = calculate_pagerank_score(
        pagerank,
        weights.get("pagerank", {})
    )
    metadata_score = calculate_metadata_score(
        document_metadata,
        weights.get("metadata", {})
    )

    # Combine scores
    final_score = bm25_weighted + metadata_score + pagerank_score

    return final_score

def calculate_pagerank_score(data: dict, weights: dict) -> float:
    """Calculate pagerank score based on its data"""
    if not data:
        return 0

    pagerank_weighted = weights.get("pageRank", 1.0) * data.get("pageRank", 0.0)
    in_link_weighted = weights.get("inLink", 1.0) * data.get("inLinkCount", 0.0)
    out_link_weighted = weights.get("outLink", 1.0) * data.get("outLinkCount", 0.0)

    return pagerank_weighted + in_link_weighted + out_link_weighted


def calculate_metadata_score(metadata: dict, weights: dict) -> float:
    """Calculate a metadata-based score for a document."""
    if not metadata:
        return 0

    freshness_weight = weights.get("freshness", 1.0)
    doc_type_weights = weights.get("docType", {})

    # Freshness based on timeLastUpdated
    freshness_score = 0
    if "timeLastUpdated" in metadata:
        last_update_time = datetime.fromisoformat(metadata["timeLastUpdated"])
        time_diff = datetime.now(timezone.utc) - last_update_time
        freshness_score = freshness_weight / max(1, time_diff.days)

    # Document type influence
    # FIXME: this
    doc_type_score = doc_type_weights.get(metadata.get("docType", ""), 0.5)

    return freshness_score + doc_type_score
