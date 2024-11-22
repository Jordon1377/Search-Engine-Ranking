import math
from typing import List, Dict, Tuple

def bm25_score(query_terms: List[str], document: Dict[str, any], k1: float,
               b: float, avg_doc_len: float, total_docs: int,
               num_docs_with_term: int) -> float:
    """
    Calculates the BM25 score for a document given query terms.

    Parameters:
        query_terms (List[str]): List of terms in the query.
        document (Dict[str, any]): Document representation containing:
            - "docLength" (int): The length of the document.
            - "term_frequency" (Dict[str, int]): Term frequency in the document.
        k1 (float): BM25 tuning parameter controlling term frequency saturation.
        b (float): BM25 tuning parameter controlling document length normalization.
        avg_doc_len (float): Average document length across the corpus.
        total_docs (int): Total number of documents in the corpus.
        num_docs_with_term (Dict[str, int]): A mapping of terms to the number
        of documents they appear in.

    Returns:
        float: BM25 score for the document.
    """
    score = 0.0
    doc_length = document["docLength"]
    for term in query_terms:
        if term in document["term_frequency"]:
            term_freq = document["term_frequency"][term]
            idf_term = calculate_idf(total_docs, num_docs_with_term[term])
            numerator = term_freq * (k1 + 1)
            denominator = term_freq + k1 * (1 - b + b * (doc_length / avg_doc_len))
            score += idf_term * (numerator / denominator)
    return score

def calculate_idf(N: int, nt: int) -> float:
    """
    Calculates the Inverse Document Frequency (IDF) for a term.

    Parameters:
        N (int): Total number of documents in the corpus.
        nt (int): Number of documents containing the term.

    Returns:
        float: The IDF value for the term.
    """
    return math.log((N - nt + 0.5) / (nt + 0.5) + 1)

def get_bm25_params(weights: Dict[str, any]) -> Tuple[float, float, float]:
    """
    Extracts BM25 parameters (k1, b, avg_doc_len) from weights.

    Parameters:
        weights (Dict[str, any]): Dictionary containing BM25 tuning parameters.

    Returns:
        Tuple[float, float, float]: BM25 parameters (k1, b, avg_doc_len).
    """
    k1 = weights.get("bm25_params", {}).get("k1", 1.5)
    b = weights.get("bm25_params", {}).get("b", 0.75)
    avg_doc_len = weights.get("bm25_params", {}).get("avg_doc_len", 1500)
    return k1, b, avg_doc_len
