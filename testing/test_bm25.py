import unittest
import sys
import os
from typing import List, Dict

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from bm25 import bm25_score, calculate_idf, get_bm25_params

class TestBM25Score(unittest.TestCase):
    def test_bm25_single_term(self):
        query_terms = ["data"]
        document = {
            "docID": "12345",
            "docLength": 2450,
            "term_frequency": {"data": 5}
        }
        num_docs_with_term = {"data": 10}
        total_docs = 100
        avgdl = 2000
        k1, b = 1.5, 0.75
        idf_data = calculate_idf(total_docs, num_docs_with_term["data"])
        expected_score = idf_data * ((5 * (k1 + 1)) \
            / (5 + k1 * (1 - b + b * (2450 / avgdl))))
        calculated_score = bm25_score(
            query_terms, document, k1, b, avgdl, total_docs, num_docs_with_term
        )
        self.assertAlmostEqual(calculated_score, expected_score, places=2)

    def test_bm25_multiple_terms(self):
        query_terms = ["data", "science"]
        document = {
            "docID": "12345",
            "docLength": 2450,
            "term_frequency": {"data": 5, "science": 2}
        }
        num_docs_with_term = {"data": 10, "science": 20}
        total_docs = 100
        avgdl = 2000
        k1, b = 1.5, 0.75
        idf_data = calculate_idf(total_docs, num_docs_with_term["data"])
        idf_science = calculate_idf(total_docs, num_docs_with_term["science"])
        expected_score = (
            idf_data * ((5 * (k1 + 1)) / (5 + k1 * (1 - b + b * (2450 / avgdl)))) +
            idf_science * ((2 * (k1 + 1)) / (2 + k1 * (1 - b + b * (2450 / avgdl))))
        )
        calculated_score = bm25_score(
            query_terms, document, k1, b, avgdl, total_docs, num_docs_with_term
        )
        self.assertAlmostEqual(calculated_score, expected_score, places=2)

    def test_bm25_no_matching_terms(self):
        query_terms = ["math"]
        document = {
            "docID": "12345",
            "docLength": 2450,
            "term_frequency": {"data": 5, "science": 2}
        }
        num_docs_with_term = {"math": 0}
        total_docs = 100
        avgdl = 2000
        k1, b = 1.5, 0.75
        expected_score = 0.0
        calculated_score = bm25_score(
            query_terms, document, k1, b, avgdl, total_docs, num_docs_with_term
        )
        self.assertAlmostEqual(calculated_score, expected_score, places=2)

    def test_bm25_edge_case_zero_term_frequency(self):
        query_terms = ["data"]
        document = {
            "docID": "12345",
            "docLength": 2450,
            "term_frequency": {"data": 0}
        }
        num_docs_with_term = {"data": 10}
        total_docs = 100
        avgdl = 2000
        k1, b = 1.5, 0.75
        expected_score = 0.0
        calculated_score = bm25_score(
            query_terms, document, k1, b, avgdl, total_docs, num_docs_with_term
        )
        self.assertAlmostEqual(calculated_score, expected_score, places=2)

    def test_bm25_short_document(self):
        query_terms = ["data"]
        document = {
            "docID": "12345",
            "docLength": 500,
            "term_frequency": {"data": 3}
        }
        num_docs_with_term = {"data": 10}
        total_docs = 100
        avgdl = 2000
        k1, b = 1.5, 0.75
        idf_data = calculate_idf(total_docs, num_docs_with_term["data"])
        expected_score = idf_data * ((3 * (k1 + 1)) \
            / (3 + k1 * (1 - b + b * (500 / avgdl))))
        calculated_score = bm25_score(
            query_terms, document, k1, b, avgdl, total_docs, num_docs_with_term
        )
        self.assertAlmostEqual(calculated_score, expected_score, places=2)

class TestCalculateIDF(unittest.TestCase):
    def test_idf_high_document_frequency(self):
        """IDF is low when term appears in many documents."""
        self.assertGreater(calculate_idf(100, 90), 0)

    def test_idf_low_document_frequency(self):
        """IDF is high when term appears in few documents."""
        self.assertGreater(calculate_idf(100, 1), 2)

    def test_idf_no_document_matches(self):
        """IDF is high when term matches no documents."""
        self.assertGreater(calculate_idf(100, 0), 4)

    def test_idf_term_in_all_documents(self):
        """IDF is near zero when term matches all documents."""
        self.assertGreater(0.01, calculate_idf(100, 100))

class TestGetBM25Params(unittest.TestCase):
    def test_bm25_params_with_all_defaults(self):
        """Test BM25 parameter extraction with no explicit weights."""
        weights = {}
        expected_k1, expected_b, expected_avg_doc_len = 1.5, 0.75, 1500
        self.assertEqual(
            get_bm25_params(weights),
            (expected_k1, expected_b, expected_avg_doc_len)
        )

    def test_bm25_params_with_partial_weights(self):
        """Test BM25 parameter extraction with some custom values."""
        weights = {"bm25_params": {"k1": 1.2}}
        expected_k1, expected_b, expected_avg_doc_len = 1.2, 0.75, 1500
        self.assertEqual(
            get_bm25_params(weights),
            (expected_k1, expected_b, expected_avg_doc_len)
        )

    def test_bm25_params_with_all_custom_weights(self):
        """Test BM25 parameter extraction with all custom values."""
        weights = {"bm25_params": {
            "k1": 1.8, "b": 0.65, "avg_doc_len": 1800}
        }
        expected_k1, expected_b, expected_avg_doc_len = 1.8, 0.65, 1800
        self.assertEqual(
            get_bm25_params(weights),
            (expected_k1, expected_b, expected_avg_doc_len)
        )

    def test_bm25_params_with_nested_structure(self):
        """Test BM25 parameter extraction with unexpected structures."""
        weights = {"bm25_params": {
            "k1": 2.0, "nested": {"b": 0.7}
        }, "avg_doc_len": 1600}
        expected_k1, expected_b, expected_avg_doc_len = 2.0, 0.75, 1500
        self.assertEqual(
            get_bm25_params(weights),
            (expected_k1, expected_b, expected_avg_doc_len)
        )

if __name__ == "__main__":
    unittest.main()
