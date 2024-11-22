import unittest
import sys
import os
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from scoring_params import ScoringParams
from combine import combine_scores

class TestCombineScores(unittest.TestCase):
    def test_combine_single_document(self):
        # Inputs for the test
        query_terms = ["data", "science"]
        bm25_score = 3.5
        document_metadata = {
            "docLength": 1200,
            "timeLastUpdated": (datetime.now()).isoformat(),
            "docType": "PDF"
        }
        link_data = {
            "pageRank": 50,
            "inLinkCount": 10,
            "outLinkCount": 2
        }
        weights = {
            "bm25": 1.2,
            "metadata": {
                "freshness": 0.8,
                "docType": {
                    "PDF": 1.0,
                    "HTML": 1.2
                }
            },
            "link_analysis": {
                "pageRank": 1.0,
                "inLinkCount": 0.5,
                "outLinkCount": -0.1
            }
        }

        # Test setup
        params = ScoringParams(query_terms, bm25_score, document_metadata, link_data, weights)
        score = combine_scores(params)

        # Assertions
        self.assertGreater(score, 0)  # Score should always be positive
        self.assertGreaterEqual(score, bm25_score * weights["bm25"])
        self.assertGreaterEqual(score, weights["metadata"]["freshness"])
        rank_cont = weights["link_analysis"]["pageRank"] * link_data["pageRank"]
        self.assertGreaterEqual(score, rank_cont)

    def test_missing_metadata(self):
        query_terms = ["data"]
        bm25_score = 2.5
        document_metadata = {}  # Missing metadata
        link_data = {
            "pageRank": 30,
            "inLinkCount": 5,
            "outLinkCount": 1
        }
        weights = {
            "bm25": 1.0,
            "metadata": {
                "freshness": 0.8
            },
            "link_analysis": {
                "pageRank": 1.0,
                "inLinkCount": 0.5,
                "outLinkCount": -0.1
            }
        }

        # Test setup
        params = ScoringParams(query_terms, bm25_score, document_metadata, link_data, weights)
        score = combine_scores(params)

        # Assertions
        self.assertGreater(score, 0)  # Score should still be positive
        self.assertGreaterEqual(score, bm25_score * weights["bm25"])
        rank_cont = weights["link_analysis"]["pageRank"] * link_data["pageRank"]
        self.assertGreaterEqual(score, rank_cont)


    def test_high_outlink_penalty(self):
        query_terms = ["data"]
        bm25_score = 4.0
        document_metadata = {
            "docLength": 1000,
            "timeLastUpdated": (datetime.now()).isoformat(),
            "docType": "HTML"
        }
        link_data = {
            "pageRank": 40,
            "inLinkCount": 20,
            "outLinkCount": 100  # High outlink count
        }
        weights = {
            "bm25": 1.0,
            "metadata": {
                "freshness": 0.8,
                "docType": {
                    "HTML": 1.2
                }
            },
            "link_analysis": {
                "pageRank": 1.0,
                "inLinkCount": 0.5,
                "outLinkCount": -0.5 # High penalty
            }
        }

        # Test setup
        params = ScoringParams(query_terms, bm25_score, document_metadata, link_data, weights)
        score = combine_scores(params)

        # Assertions
        self.assertGreater(score, 0)  # Score should remain positive
        self.assertLess(score, bm25_score + link_data["pageRank"])  # Outlink penalty reduces score


    def test_zero_bm25(self):
        # Test with zero BM25 score
        query_terms = ["example"]
        bm25_score = 0.0
        document_metadata = {
            "docLength": 1500,
            "timeLastUpdated": (datetime.now()).isoformat(),
            "docType": "PDF"
        }
        link_data = {
            "pageRank": 25,
            "inLinkCount": 5,
            "outLinkCount": 2
        }
        weights = {
            "bm25": 1.0,
            "metadata": {
                "freshness": 0.8,
                "docType": {
                    "PDF": 1.0
                }
            },
            "link_analysis": {
                "pageRank": 1.0,
                "inLinkCount": 0.5,
                "outLinkCount": -0.1
            }
        }

        # Test setup
        params = ScoringParams(query_terms, bm25_score, document_metadata, link_data, weights)
        score = combine_scores(params)

        # Assertions
        self.assertGreater(score, 0)  # Metadata and link analysis should contribute
        self.assertLess(score, 50)  # Ensure BM25 absence affects total score

if __name__ == "__main__":
    unittest.main()
