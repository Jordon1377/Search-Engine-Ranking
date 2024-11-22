import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import cache

class TestCache(unittest.TestCase):
    def setUp(self):
        self.cache = cache.createCache()

    def test_set(self):
        query = "data science"
        ranking = 1
        ex = 100
        self.assertTrue(self.cache.set(query, ranking, ex))
        self.assertTrue(self.cache.exists(query))

    def test_get(self):
        query = "data science"
        ranking = {"data": 1, "science": 2, "data science": 3, "science data": 4}
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

if __name__ == '__main__':
    unittest.main(verbosity=2)