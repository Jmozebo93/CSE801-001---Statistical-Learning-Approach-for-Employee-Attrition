"""
Simple file-based caching for Gemini API responses to save quota.

Cache is stored in .cache/ directory with hash-based filenames.
Each cache entry stores the LLM response text.
"""

import hashlib
import os
import json
from pathlib import Path


CACHE_DIR = Path(".cache")


def _ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(exist_ok=True)


def _get_cache_key(prompt: str, model: str = "gemini-2.0-flash") -> str:
    """Generate a cache key from prompt and model name."""
    content = f"{model}:{prompt}"
    return hashlib.sha256(content.encode()).hexdigest()


def get_cached_response(prompt: str, model: str = "gemini-2.0-flash") -> str | None:
    """
    Retrieve a cached LLM response if it exists.
    Returns None if cache miss or cache file doesn't exist.
    """
    _ensure_cache_dir()
    cache_key = _get_cache_key(prompt, model)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
                return data.get("response")
        except Exception:
            return None
    return None


def cache_response(prompt: str, response: str, model: str = "gemini-2.0-flash") -> None:
    """Store an LLM response in the cache."""
    _ensure_cache_dir()
    cache_key = _get_cache_key(prompt, model)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    try:
        with open(cache_file, "w") as f:
            json.dump({"response": response, "model": model}, f)
    except Exception:
        pass  # Fail silently; caching is optional


def clear_cache() -> None:
    """Clear all cached responses."""
    if CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
