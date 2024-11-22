import redis
import json
from redis.commands.json.path import Path

class Cache:
    def __init__(self):
        self.cache = redis.Redis(host='localhost', port=6379, decode_responses=True)

    def get(self, query: str) -> json:
        """
        Get the ranking of a query from the cache

        `query` query string

        Returns:
            json: ranking of the query in json format
        """
        return self.cache.json().get(query)

    def set(self, query: str, ranking: json, ex:int = None) -> bool:
        """ 
        Add a query and its ranking to the cache
        
        `query` query string
        
        `ranking` ranking of the query in json format
        
        `ex` expiration time in seconds (default: None)

        Returns:
            bool: True if the query is added to the cache, False otherwise
        """
        res = self.cache.json().set(query, Path.root_path(), ranking)
        if ex is not None:
            self.cache.expire(query, ex)
        return res
    
    def rename(self, old_key: str, new_key: str) -> bool:
        """
        Rename a key in the cache

        `old_key` old key name

        `new_key` new key name

        Returns:
            bool: True if the key is renamed, False otherwise
        """
        return self.cache.rename(old_key, new_key)

    def delete(self, key: str) -> int:
        """
        Delete a key from the cache

        `key` key to be deleted

        Returns:
            int: number of keys deleted (0 or 1)
        """
        return self.cache.delete(key)

    def keys(self) -> list:
        """
        Get all keys in the cache

        Returns:
            list: list of all keys in the cache
        """
        return self.cache.keys()

    def flush(self) -> bool:
        """
        Delete all keys in the cache

        Returns:
            bool: True if the cache is flushed, False otherwise
        """
        return self.cache.flushall()

    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache

        `key` key to be checked

        Returns:
            bool: True if the key exists, False otherwise
        """
        return self.cache.exists(key)
    
    def get_expiration(self, key: str) -> int:
        """
        Returns the number of seconds until the key `key` will expire

        `key` key to be checked

        Returns:
            int: expiration time in seconds
        """
        return self.cache.ttl(key)

def createCache():
    return Cache()
