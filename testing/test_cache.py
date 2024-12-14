from unittest.mock import MagicMock


import unittest
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from rank import rank_documents

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import cache

class TestCache(unittest.TestCase):
    def setUp(self):
        self.cache = cache.createCache()
        self.fetch_total_doc_statistics = MagicMock(return_value={
            "avgDocLength": 798.8730,
            "docCount": 4567876
        })

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

    def test_set(self):
        query = "data science"
        ranking = 1
        ex = 100
        self.assertTrue(self.cache.set(query, ranking, ex))
        self.assertTrue(self.cache.exists(query))

    def test_get(self):
        query = "data science"
        ranking = [{"docID": 1, "score": 0, "metadata": {"date": "2016"}}, {"docID": 2, "score": 0, "metadata": {"date": "2017"}}]
        ex = 100
        self.assertTrue(self.cache.set(query, ranking, ex))
        self.assertEqual(self.cache.get(query), ranking)

    def test_rename(self):
        old_key = "data science"
        new_key = "data engineering"
        self.assertTrue(self.cache.set(old_key, 1))
        self.assertTrue(self.cache.rename(old_key, new_key))
        self.assertFalse(self.cache.exists(old_key))
        self.assertTrue(self.cache.exists(new_key))

    def test_delete(self):
        key = "data science"
        self.assertTrue(self.cache.set(key, 1))
        self.assertEqual(self.cache.delete(key), 1)
        self.assertFalse(self.cache.exists(key))

    def test_keys(self):
        keys = ["data science", "data engineering", "data analytics"]
        for key in keys:
            self.assertTrue(self.cache.set(key, 1))
        self.assertCountEqual(self.cache.keys(), keys)

    def test_flush(self):
        keys = ["data science", "data engineering", "data analytics"]
        for key in keys:
            self.assertTrue(self.cache.set(key, 1))
        self.assertTrue(self.cache.flush())
        self.assertFalse(self.cache.keys())

    def test_exists(self):
        key = "data science"
        self.assertFalse(self.cache.exists(key))
        self.assertTrue(self.cache.set(key, 1))
        self.assertTrue(self.cache.exists(key))

    def test_cache_real_data(self):
        """Test cache with a single repeated query term (data)."""

        self.cache.flush()
        query = "data"

        ranked_results = rank_documents(
            query=query,
            weights={},
            doc_stats=[798.8730, 4567876],
            fetch_relevant_docs=self.fetch_relevant_docs,
            fetch_doc_metadata=self.fetch_doc_metadata,
            fetch_pagerank=self.fetch_pagerank
        )

        output = json.dumps(ranked_results)
        self.cache.set(query, output)

        self.assertEqual(self.cache.get(query),json.dumps(ranked_results))

if __name__ == '__main__':
    unittest.main(verbosity=2)