# src/cache_manager.py - Local Caching System

import json
import os
from datetime import datetime, timedelta
import logging
from typing import Any, Optional, Dict
import sqlite3

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheManager:
    """Manages local data caching for fast access and offline capability."""
    
    def __init__(self, cache_dir: str = config.CACHE_DIR):
        self.cache_dir = cache_dir
        self.db_path = os.path.join(cache_dir, "cache.db")
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for caching."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expiry_time DATETIME
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    last_updated DATETIME,
                    version TEXT,
                    source TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Cache database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize cache database: {e}")
    
    def set(self, key: str, value: Any, ttl_hours: int = config.CACHE_EXPIRY_HOURS):
        """
        Store data in cache with optional TTL.
        
        Args:
            key: Cache key
            value: Data to cache
            ttl_hours: Time-to-live in hours
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            json_value = json.dumps(value, default=str)
            expiry_time = datetime.now() + timedelta(hours=ttl_hours)
            
            cursor.execute('''
                INSERT OR REPLACE INTO cache (key, value, expiry_time)
                VALUES (?, ?, ?)
            ''', (key, json_value, expiry_time))
            
            conn.commit()
            conn.close()
            logger.info(f"Cached data with key: {key}")
            return True
        except Exception as e:
            logger.error(f"Cache set operation failed: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve data from cache if not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if expired/not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT value, expiry_time FROM cache WHERE key = ?
            ''', (key,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                value, expiry_time = result
                if datetime.fromisoformat(expiry_time) > datetime.now():
                    logger.info(f"Retrieved cached data for key: {key}")
                    return json.loads(value)
                else:
                    self.delete(key)
                    logger.info(f"Cache expired for key: {key}")
                    return None
            
            logger.info(f"No cache found for key: {key}")
            return None
        except Exception as e:
            logger.error(f"Cache get operation failed: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete cache entry."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM cache WHERE key = ?', (key,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Cache delete failed: {e}")
            return False
    
    def clear_expired(self):
        """Remove all expired cache entries."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM cache WHERE expiry_time < ?
            ''', (datetime.now(),))
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            logger.info(f"Cleared {deleted} expired cache entries")
            return deleted
        except Exception as e:
            logger.error(f"Failed to clear expired cache: {e}")
            return 0
    
    def set_metadata(self, key: str, source: str, version: str = config.CACHE_VERSION):
        """Store metadata about cached data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO metadata (key, last_updated, version, source)
                VALUES (?, ?, ?, ?)
            ''', (key, datetime.now(), version, source))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to set metadata: {e}")
    
    def get_metadata(self, key: str) -> Optional[Dict]:
        """Retrieve metadata about cached data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT last_updated, version, source FROM metadata WHERE key = ?
            ''', (key,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    "last_updated": result[0],
                    "version": result[1],
                    "source": result[2]
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get metadata: {e}")
            return None
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about cache usage."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM cache')
            total_entries = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT COUNT(*) FROM cache WHERE expiry_time < ?
            ''', (datetime.now(),))
            expired_entries = cursor.fetchone()[0]
            
            active_entries = total_entries - expired_entries
            
            conn.close()
            
            return {
                "total_entries": total_entries,
                "active_entries": active_entries,
                "expired_entries": expired_entries,
                "cache_db_path": self.db_path
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}


class JsonCacheManager:
    """Alternative JSON-based caching for simpler scenarios."""
    
    def __init__(self, cache_file: str = config.CACHE_FILE):
        self.cache_file = cache_file
        self._ensure_cache_file()
    
    def _ensure_cache_file(self):
        """Ensure cache file exists."""
        if not os.path.exists(self.cache_file):
            with open(self.cache_file, 'w') as f:
                json.dump({}, f)
    
    def set(self, key: str, value: Any, ttl_hours: int = config.CACHE_EXPIRY_HOURS):
        """Store data in JSON cache."""
        try:
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)
            
            expiry_time = (datetime.now() + timedelta(hours=ttl_hours)).isoformat()
            cache[key] = {
                "value": value,
                "expiry_time": expiry_time,
                "created_at": datetime.now().isoformat()
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache, f, indent=2, default=str)
            
            logger.info(f"Cached (JSON) data with key: {key}")
            return True
        except Exception as e:
            logger.error(f"JSON cache set failed: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve data from JSON cache."""
        try:
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)
            
            if key in cache:
                expiry_time = datetime.fromisoformat(cache[key]["expiry_time"])
                if expiry_time > datetime.now():
                    logger.info(f"Retrieved cached data (JSON) for key: {key}")
                    return cache[key]["value"]
                else:
                    del cache[key]
                    with open(self.cache_file, 'w') as f:
                        json.dump(cache, f, indent=2, default=str)
                    logger.info(f"JSON cache expired for key: {key}")
                    return None
            
            return None
        except Exception as e:
            logger.error(f"JSON cache get failed: {e}")
            return None
    
    def clear(self):
        """Clear entire JSON cache."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump({}, f)
            logger.info("JSON cache cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear JSON cache: {e}")
            return False
