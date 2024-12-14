import unittest
import random
import time
from collections import Counter, defaultdict
from wordfreq import top_n_list, word_frequency
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from rank import rank_documents, tokenize_query

# ----------------------------
# Data Generation & Mocks
# ----------------------------

def get_frequency_weighted_vocabulary(vocab_size=10000):
    vocab = top_n_list('en', vocab_size)
    word_frequencies = [word_frequency(w, 'en') for w in vocab]
    total_freq = sum(word_frequencies) or 1.0
    normalized_freq = [f / total_freq for f in word_frequencies]
    return vocab, normalized_freq

def generate_documents(num_docs=10000, vocab_size=10000, min_doc_len=50, max_doc_len=500):
    vocab, vocab_probs = get_frequency_weighted_vocabulary(vocab_size)

    documents = {}
    doc_lengths = []
    for doc_id in range(1, num_docs + 1):
        doc_length = random.randint(min_doc_len, max_doc_len)
        words = random.choices(vocab, weights=vocab_probs, k=doc_length)
        freq_map = Counter(words)
        documents[doc_id] = {
            "metadata": {
                "URL": f"http://example.com/doc{doc_id}",
                "docLength": doc_length,
                "term_frequency": dict(freq_map)
            }
        }
        doc_lengths.append(doc_length)
    return documents, doc_lengths

def build_inverted_index(documents):
    inverted_index = defaultdict(lambda: {"index": []})
    # Populate the inverted index with frequency information
    for doc_id, doc_data in documents.items():
        term_freq_map = doc_data["metadata"]["term_frequency"]
        for term, freq in term_freq_map.items():
            inverted_index[term]["index"].append({"docID": doc_id, "frequency": freq})
    return dict(inverted_index)

def compute_total_doc_stats(doc_lengths):
    total_docs = len(doc_lengths)
    avg_doc_len = sum(doc_lengths) / total_docs
    return (avg_doc_len, total_docs)

def generate_pagerank_scores(documents):
    pagerank_map = {}
    for doc_id, doc_data in documents.items():
        url = doc_data["metadata"]["URL"]
        # Return empty dict or actual data as needed
        pagerank_map[url] = {
            # Update as needed if you want actual pagerank data
        }
    return pagerank_map

def mock_fetch_relevant_docs(inverted_index):
    def inner(term: str):
        return inverted_index.get(term, {"index": []})
    return inner

def mock_fetch_doc_metadata(documents):
    def inner(docID: int):
        return documents.get(docID)
    return inner

def mock_fetch_pagerank(pagerank_map):
    def inner(url: str):
        return pagerank_map.get(url, {"pageRank": 0.0, "inLinkCount": 0, "outLinkCount": 0})
    return inner

# ----------------------------
# Test Case
# ----------------------------


class TestRankingSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.NUM_DOCS = 200000
        cls.VOCAB_SIZE = 20000
        cls.MIN_DOC_LEN = 250
        cls.MAX_DOC_LEN = 1500
        cls.documents, cls.doc_lengths = generate_documents(
            num_docs=cls.NUM_DOCS,
            vocab_size=cls.VOCAB_SIZE,
            min_doc_len=cls.MIN_DOC_LEN,
            max_doc_len=cls.MAX_DOC_LEN
        )
        cls.inverted_index = build_inverted_index(cls.documents)
        cls.total_doc_stats = compute_total_doc_stats(cls.doc_lengths)
        cls.pagerank_map = generate_pagerank_scores(cls.documents)
        cls.weights = {'bm25_params': {
                'k1': 3.3564610481262207,
                'b': 0.6601634621620178
            }, 
            'bm25': 0.8437421917915344, 
            'pageRank': {
               'pageRank': -2.149700549125555e-06,
               'inLink': 1.035431068885373e-05, 
               'outLink': -0.01162341237068176
            }, 
           'metadata': {
               'freshness': 0.7459535598754883
           }
        }

    def _run_test_ranking(self, query: str):
        fetch_relevant_docs = mock_fetch_relevant_docs(self.inverted_index)
        fetch_doc_metadata = mock_fetch_doc_metadata(self.documents)
        fetch_pagerank = mock_fetch_pagerank(self.pagerank_map)

        start_time = time.time()
        results, num_parsed = rank_documents(
            query=query,
            weights=self.weights,
            doc_stats=self.total_doc_stats,
            fetch_relevant_docs=fetch_relevant_docs,
            fetch_doc_metadata=fetch_doc_metadata,
            fetch_pagerank=fetch_pagerank
        )
        end_time = time.time()
        elapsed = end_time - start_time

        # Print info for demonstration
        print(f"\n---------------------------------------------")
        print(f"Query: {query}")
        print(f"Time taken: {elapsed:.4f} seconds")
        print(f"Number of parsed docs: {num_parsed}")

        # Assertions can be added here if proper testing is implemented
        self.assertIsInstance(results, list)
        return results

    def test_query_fish(self):
        self._run_test_ranking("fish")

    def test_query_tropical_fish(self):
        self._run_test_ranking("tropical fish")

    def test_query_uncommon_query_vernacular(self):
        self._run_test_ranking("uncommon query vernacular")

    def test_query_campus_map(self):
        self._run_test_ranking("campus map")

    def test_query_minute_maid(self):
        self._run_test_ranking("minute maid lemonade")

    def test_query_the(self):
        self._run_test_ranking("the")

if __name__ == "__main__":
    unittest.main()
