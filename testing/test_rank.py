import unittest
from unittest.mock import MagicMock
from typing import Callable
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from rank import rank_documents, tokenize_query, sort_ranked_results

class TestTokenizeQuery(unittest.TestCase):
    def test_tokenize_unigrams(self):
        query = "data science"
        tokens = tokenize_query(query)
        # Assert unigrams are included
        self.assertIn("data", tokens)
        self.assertIn("science", tokens)

    def test_tokenize_bigrams(self):
        query = "data science is fun"
        tokens = tokenize_query(query)
        # Assert bigrams are included
        self.assertIn("data science", tokens)
        self.assertIn("science is", tokens)
        self.assertIn("is fun", tokens)

    def test_tokenize_trigrams(self):
        query = "data science is fun"
        tokens = tokenize_query(query)
        # Assert trigrams are included
        self.assertIn("data science is", tokens)
        self.assertIn("science is fun", tokens)

    def test_tokenize_edge_case_single_word(self):
        query = "data"
        tokens = tokenize_query(query)
        # Only unigrams should exist
        self.assertIn("data", tokens)
        self.assertEqual(len(tokens), 1)

    def test_tokenize_edge_case_empty_query(self):
        query = ""
        tokens = tokenize_query(query)
        # No tokens should be produced
        self.assertEqual(len(tokens), 0)

class TestSortRankedResults(unittest.TestCase):
    def test_sort_basic(self):
        ranked_results = [
            {"docID": "1", "score": 2.5},
            {"docID": "2", "score": 3.7},
            {"docID": "3", "score": 1.2},
        ]
        sorted_results = sort_ranked_results(ranked_results)

        self.assertEqual(sorted_results[0]["score"], 3.7)
        self.assertEqual(sorted_results[0]["rank"], 1)
        self.assertEqual(sorted_results[1]["score"], 2.5)
        self.assertEqual(sorted_results[1]["rank"], 2)
        self.assertEqual(sorted_results[2]["score"], 1.2)
        self.assertEqual(sorted_results[2]["rank"], 3)

    def test_sort_ties(self):
        ranked_results = [
            {"docID": "1", "score": 2.5},
            {"docID": "2", "score": 2.5},
            {"docID": "3", "score": 3.0},
        ]
        sorted_results = sort_ranked_results(ranked_results)

        self.assertEqual(sorted_results[0]["score"], 3.0)
        self.assertEqual(sorted_results[0]["rank"], 1)
        self.assertIn({"docID": "1", "score": 2.5, "rank": 2}, sorted_results[1:])
        self.assertIn({"docID": "2", "score": 2.5, "rank": 3}, sorted_results[1:])

    def test_sort_empty_list(self):
        ranked_results = []
        sorted_results = sort_ranked_results(ranked_results)
        self.assertEqual(sorted_results, [])

    def test_sort_single_element(self):
        ranked_results = [{"docID": "1", "score": 4.2}]
        sorted_results = sort_ranked_results(ranked_results)

        self.assertEqual(sorted_results, [{"docID": "1", "score": 4.2, "rank": 1}])

class TestRankDocuments(unittest.TestCase):
    def setUp(self):
        # Mock the API calls
        self.fetch_total_doc_statistics = [798.8730, 4567876]

        self.fetch_relevant_docs = MagicMock(side_effect=lambda term: {
            "data": {
                "term": "data",
                "index": [
                    {
                        "docID": "12345",
                        "frequency": 5,
                        "positions": [4, 15, 28, 102, 204]
                    },
                    {
                        "docID": "54321",
                        "frequency": 2,
                        "positions": [81, 1005]
                    }
                ]
            },
            "science": {
                "term": "science",
                "index": [
                    {
                        "docID": "1",
                        "frequency": 1,
                        "positions": [1]
                    },
                    {
                        "docID": "213098109872",
                        "frequency": 12,
                        "positions": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                    },
                    {
                        "docID": "12345",
                        "frequency": 2,
                        "positions": [2, 16]
                    }
                ]
            }
        }.get(term, None))

        self.fetch_doc_metadata = MagicMock(side_effect=lambda doc_id: {
            "12345": {
                "docID": "12345",
                "metadata": {
                    "docLength": 2450,
                    "timeLastUpdated": "2024-11-09T15:30:00Z",
                    "docType": "PDF",
                    "docTitle": "Introduction to Data Science",
                    "URL": "https://example.com/documents/12345"
                }
            },
            "54321": {
                "docID": "54321",
                "metadata": {
                    "docLength": 1500,
                    "timeLastUpdated": "2024-10-15T10:00:00Z",
                    "docType": "HTML",
                    "docTitle": "Data Analytics Overview",
                    "URL": "https://example.com/documents/54321"
                }
            },
            "1": {
                "docID": "1",
                "metadata": {
                    "docLength": 300,
                    "timeLastUpdated": "2024-11-01T12:00:00Z",
                    "docType": "Blog",
                    "docTitle": "Science in Daily Life",
                    "URL": "https://example.com/documents/1"
                }
            },
            "213098109872": {
                "docID": "213098109872",
                "metadata": {
                    "docLength": 5000,
                    "timeLastUpdated": "2024-08-20T09:30:00Z",
                    "docType": "PDF",
                    "docTitle": "Advanced Science Techniques",
                    "URL": "https://example.com/documents/213098109872"
                }
            }
        }.get(doc_id, None))

        self.fetch_pagerank = MagicMock(side_effect=lambda url: {
            "https://example.com/documents/12345": {
                "pageRank": 63.379,
                "inLinkCount": 10,
                "outLinkCount": 25
            },
            "https://example.com/documents/54321": {
                "pageRank": 45.211,
                "inLinkCount": 5,
                "outLinkCount": 15
            },
            "https://example.com/documents/1": {
                "pageRank": 12.042,
                "inLinkCount": 2,
                "outLinkCount": 5
            },
            "https://example.com/documents/213098109872": {
                "pageRank": 78.123,
                "inLinkCount": 20,
                "outLinkCount": 30
            }
        }.get(url, {"pageRank": 0.0, "inLinkCount": 0, "outLinkCount": 0}))

    def test_rank_documents_basic(self):
        """Test with multiple query terms (data science)."""
        query = "data science"

        ranked_results, num_parsed = rank_documents(
            query=query,
            weights={},
            doc_stats=self.fetch_total_doc_statistics,
            fetch_relevant_docs=self.fetch_relevant_docs,
            fetch_doc_metadata=self.fetch_doc_metadata,
            fetch_pagerank=self.fetch_pagerank
        )

        # Ensure results are returned
        self.assertTrue(len(ranked_results) > 0)

        # Assert the top-ranked document
        self.assertEqual(ranked_results[0]["docID"], "12345")
        self.assertIn("rank", ranked_results[0])
        self.assertEqual(ranked_results[0]["rank"], 1)
        self.assertIn("metadata", ranked_results[0])
        self.assertEqual(
            ranked_results[0]["metadata"]["docTitle"],
            "Introduction to Data Science"
        )

    def test_rank_documents_single_term(self):
        """Test with a single query term (data)."""
        query = "data"

        ranked_results, num_parsed = rank_documents(
            query=query,
            weights={},
            doc_stats=self.fetch_total_doc_statistics,
            fetch_relevant_docs=self.fetch_relevant_docs,
            fetch_doc_metadata=self.fetch_doc_metadata,
            fetch_pagerank=self.fetch_pagerank
        )

        # Ensure results are returned
        self.assertTrue(len(ranked_results) > 0)

        # Assert the top-ranked document
        self.assertEqual(ranked_results[0]["docID"], "12345")
        self.assertIn("rank", ranked_results[0])
        self.assertEqual(ranked_results[0]["rank"], 1)
        self.assertIn("metadata", ranked_results[0])
        self.assertEqual(
            ranked_results[0]["metadata"]["docTitle"],
            "Introduction to Data Science"
        )

        # Ensure the second document is correctly ranked
        self.assertEqual(ranked_results[1]["docID"], "54321")
        self.assertEqual(ranked_results[1]["rank"], 2)

    def test_rank_documents_without_weights(self):
        """Test ranking without weights."""
        query = "data science"

        ranked_results, num_parsed = rank_documents(
            query=query,
            weights={},  # No weights applied
            doc_stats=self.fetch_total_doc_statistics,
            fetch_relevant_docs=self.fetch_relevant_docs,
            fetch_doc_metadata=self.fetch_doc_metadata,
            fetch_pagerank=self.fetch_pagerank
        )

        # Assert that documents are ranked correctly
        ranked_doc_ids = [result["docID"] for result in ranked_results]
        expected_order = ['12345', '213098109872', '54321', '1']
        self.assertEqual(ranked_doc_ids, expected_order)

    def test_rank_documents_with_weights(self):
        """Test ranking with specific weights."""
        query = "data science"
        weights = {
            "bm25_params": {"k1": 3.5, "b": 0.5},
            "bm25": 1.0, # Booooooost
            "pageRank": {
                "pageRank": 0.2, # Screw this metric
                "inLink": 0.1,
                "outLink": 0.1,
            },
            "metadata": {
                "freshness": 1.0,
                "docType": {
                    "PDF": 1.0,
                    "HTML": 0.8,
                    "Blog": 0.5
                }
            }
        }

        ranked_results, num_parsed = rank_documents(
            query=query,
            weights=weights,
            doc_stats=self.fetch_total_doc_statistics,
            fetch_relevant_docs=self.fetch_relevant_docs,
            fetch_doc_metadata=self.fetch_doc_metadata,
            fetch_pagerank=self.fetch_pagerank
        )

        # Assert that documents are ranked correctly with weights applied
        ranked_doc_ids = [result["docID"] for result in ranked_results]
        expected_order = ['12345', '213098109872', '54321', '1']
        self.assertEqual(ranked_doc_ids, expected_order)

if __name__ == '__main__':
    unittest.main()
