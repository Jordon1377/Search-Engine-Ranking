def bm25_score(query_terms, document, idf, k1, b, avg_doc_len):
    """
    Simulates BM25 score computation for a document given query terms.
    """
    score = 0.0
    doc_length = document["docLength"]
    for term in query_terms:
        if term in document["term_frequency"]:
            term_freq = document["term_frequency"][term]
            idf_term = idf.get(term, 0)
            numerator = term_freq * (k1 + 1)
            denominator = term_freq + k1 * (1 - b + b * (doc_length / avg_doc_len))
            score += idf_term * (numerator / denominator)
    return score

def get_bm25_params(weights):
    """
    Extract BM25 parameters (k1, b, avg_doc_len) from weights.
    Defaults are provided if parameters are not explicitly set.

    Parameters:
        weights: dict
            Weights dictionary containing tuning parameters for BM25.

    Returns:
        tuple
            BM25 parameters: (k1, b, avg_doc_len)
    """
    k1 = weights.get("bm25_params", {}).get("k1", 1.5)
    b = weights.get("bm25_params", {}).get("b", 0.75)
    avg_doc_len = weights.get("bm25_params", {}).get("avg_doc_len", 1500)
    return k1, b, avg_doc_len
