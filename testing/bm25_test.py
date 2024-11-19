import unittest
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Now you can import bm25
import bm25


class TestBM25Score(unittest.TestCase):
    def test_bm25_single_term(self):
        query_terms = ["data"]
        document = {
            "docID": "12345",
            "docLength": 2450,
            "term_frequency": {"data": 5}
        }
        idf = {"data": 1.2}
        avgdl = 2000
        k1, b = 1.5, 0.75
        expected_score = 1.2 * ((5 * (k1 + 1)) / (5 + k1 * (1 - b + b * (2450 / avgdl))))
        calculated_score = bm25.bm25_score(query_terms, document, idf, k1, b, avgdl)
        self.assertAlmostEqual(calculated_score, expected_score, places=2)

    def test_bm25_multiple_terms(self):
        query_terms = ["data", "science"]
        document = {
            "docID": "12345",
            "docLength": 2450,
            "term_frequency": {"data": 5, "science": 2}
        }
        idf = {"data": 1.2, "science": 0.8}
        avgdl = 2000
        k1, b = 1.5, 0.75
        expected_score = (
            1.2 * ((5 * (k1 + 1)) / (5 + k1 * (1 - b + b * (2450 / avgdl)))) +
            0.8 * ((2 * (k1 + 1)) / (2 + k1 * (1 - b + b * (2450 / avgdl))))
        )
        calculated_score = bm25.bm25_score(query_terms, document, idf, k1, b, avgdl)
        self.assertAlmostEqual(calculated_score, expected_score, places=2)

    def test_bm25_no_matching_terms(self):
        query_terms = ["math"]
        document = {
            "docID": "12345",
            "docLength": 2450,
            "term_frequency": {"data": 5, "science": 2}
        }
        idf = {"math": 1.5}
        avgdl = 2000
        k1, b = 1.5, 0.75
        expected_score = 0.0
        calculated_score = bm25.bm25_score(query_terms, document, idf, k1, b, avgdl)
        self.assertAlmostEqual(calculated_score, expected_score, places=2)

    def test_bm25_edge_case_zero_term_frequency(self):
        query_terms = ["data"]
        document = {
            "docID": "12345",
            "docLength": 2450,
            "term_frequency": {"data": 0}
        }
        idf = {"data": 1.2}
        avgdl = 2000
        k1, b = 1.5, 0.75
        expected_score = 0.0
        calculated_score = bm25.bm25_score(query_terms, document, idf, k1, b, avgdl)
        self.assertAlmostEqual(calculated_score, expected_score, places=2)

    def test_bm25_short_document(self):
        query_terms = ["data"]
        document = {
            "docID": "12345",
            "docLength": 500,
            "term_frequency": {"data": 3}
        }
        idf = {"data": 1.2}
        avgdl = 2000
        k1, b = 1.5, 0.75
        expected_score = 1.2 * ((3 * (k1 + 1)) / (3 + k1 * (1 - b + b * (500 / avgdl))))
        calculated_score = bm25.bm25_score(query_terms, document, idf, k1, b, avgdl)
        self.assertAlmostEqual(calculated_score, expected_score, places=2)

if __name__ == "__main__":
    unittest.main()
