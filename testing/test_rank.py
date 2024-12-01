import unittest
from unittest.mock import Mock, patch
from typing import Callable
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from rank import rank_documents, tokenize_query, sort_ranked_results

class TestRankingAlgorithm(unittest.TestCase):
    def setUp(self):
        # Setup common test data
        self.mock_fetch_relevant_docs = Mock()
        self.mock_fetch_doc_metadata = Mock()
        self.mock_fetch_pagerank = Mock()
        self.test_weights = {
            'bm25': 0.6,
            'metadata': 0.2,
            'pagerank': 0.2
        }
        self.avg_doc_len = 500.0

    def test_tokenize_query_basic_properties(self):
        """Test basic properties that must hold for tokenize_query"""
        test_cases = [
            "test query",
            "single",
            "long query with multiple words",
            "",  # Empty query
            "A B C D"  # Multiple single characters
        ]
        
        for query in test_cases:
            tokens = tokenize_query(query)
            
            # Properties that must be true
            self.assertIsInstance(tokens, list)
            
            # If query is empty, result should be empty
            if not query:
                self.assertEqual(len(tokens), 0)
                
            # If query has words, result should contain at least the number of words
            if query:
                words_count = len(query.split())
                self.assertGreaterEqual(len(tokens), words_count)
                
            # All tokens should be strings
            for token in tokens:
                self.assertIsInstance(token, str)
                
            # All tokens should be lowercase
            for token in tokens:
                self.assertEqual(token.lower(), token)

    def test_tokenize_query_ngram_properties(self):
        """Test n-gram specific properties of tokenize_query"""
        query = "word1 word2 word3"
        tokens = tokenize_query(query)
        
        # Should contain original words
        self.assertTrue("word1" in tokens)
        self.assertTrue("word2" in tokens)
        self.assertTrue("word3" in tokens)
        
        # Should contain bigrams
        self.assertTrue("word1 word2" in tokens)
        self.assertTrue("word2 word3" in tokens)
        
        # Should contain trigrams
        self.assertTrue("word1 word2 word3" in tokens)

    def test_sort_ranked_results_properties(self):
        """Test properties that must be true for sort_ranked_results"""
        test_results = [
            {"docID": "1", "score": 0.5, "metadata": {}},
            {"docID": "2", "score": 0.8, "metadata": {}},
            {"docID": "3", "score": 0.2, "metadata": {}},
            {"docID": "4", "score": 0.9, "metadata": {}}
        ]
        
        sorted_results = sort_ranked_results(test_results)
        
        # Test length preservation
        self.assertEqual(len(sorted_results), len(test_results))
        
        # Test sorting order (descending)
        for i in range(len(sorted_results) - 1):
            self.assertGreaterEqual(
                sorted_results[i]["score"],
                sorted_results[i + 1]["score"]
            )
        
        # Test that all documents are preserved
        original_ids = {r["docID"] for r in test_results}
        sorted_ids = {r["docID"] for r in sorted_results}
        self.assertEqual(original_ids, sorted_ids)

    def test_rank_documents_basic_properties(self):
        """Test basic properties that must be true for rank_documents"""
        # Setup mock returns
        self.mock_fetch_relevant_docs.return_value = {
            "doc1": {"text": "test document"},
            "doc2": {"text": "another document"}
        }
        self.mock_fetch_doc_metadata.return_value = {
            "doc1": {"title": "Test"},
            "doc2": {"title": "Another"}
        }
        self.mock_fetch_pagerank.return_value = 0.5

        result = rank_documents(
            "test query",
            self.avg_doc_len,
            self.test_weights,
            self.mock_fetch_relevant_docs,
            self.mock_fetch_doc_metadata,
            self.mock_fetch_pagerank
        )

        # Test basic properties
        self.assertIsInstance(result, list)
        
        # Test that each result has required fields
        for item in result:
            self.assertIn("docID", item)
            self.assertIn("score", item)
            self.assertIn("metadata", item)
            
        # Test that scores are between 0 and 1
        for item in result:
            self.assertGreaterEqual(item["score"], 0)
            self.assertLessEqual(item["score"], 1)
            
        # Test that results are sorted
        for i in range(len(result) - 1):
            self.assertGreaterEqual(
                result[i]["score"],
                result[i + 1]["score"]
            )

if __name__ == '__main__':
    unittest.main()
