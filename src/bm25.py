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
