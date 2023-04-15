"""
A simple mechanism to help speed up repeated of slow functions by caching the results to hard drive.

Tips for using this:
- the main use case for this cache is to help reduce repeated runs of the same, slow functions during development and testing.
- don't use this for performance optimization in production code--it's not very efficient or secure (the cache is in the form of pickle files), so 
- the arguments to the function whose results are to be cached must be hashable
- the cached function itself must be pure—i.e. the same arguments to the function must produce the exact same results on different runs.
"""

import functools
from logging import getLogger
import pickle
import os
from typing import Any, Callable

from lib.utilities import stable_hash, package_func_args

logger = getLogger(__name__)


class SimpleCache:
    """
    A simple dictionary-like cache for storing and retrieving data.

    Parameters
    ----------
    cache_dir : str
        The directory to store the cache in.

    Notes
    -----
    The cache is stored as pickled files in the cache directory.
    """

    def __init__(self, cache_dir: str):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir: str = cache_dir
        self.hashes: "list[str]" = os.listdir(cache_dir)

    def __getitem__(self, key: Any):
        key_hash = stable_hash(key)
        # if key_hash not in self.hashes
        if not self.has_key(key):
            raise KeyError(key)
        with open(os.path.join(self.cache_dir, key_hash), "rb") as f:
            return pickle.load(f)

    def __setitem__(self, key: Any, value: Any):
        key_hash = stable_hash(key)
        if key_hash not in self.hashes:
            self.hashes.append(key_hash)
        with open(os.path.join(self.cache_dir, key_hash), "wb") as f:
            pickle.dump(value, f)

    def has_key(self, key: Any) -> bool:
        """Check if the cache has a key."""
        key_hash = stable_hash(key)
        return key_hash in self.hashes

    def fetch_all(self) -> "dict[str, list]":
        """Fetch all data from the cache."""
        all_data = []
        failed_hashes = []
        for hash_val in self.hashes:
            with open(os.path.join(self.cache_dir, hash_val), "rb") as file:
                try:
                    all_data.append(pickle.load(file))
                except OSError:
                    failed_hashes.append(hash_val)
        return {
            "all_data": all_data,
            "failed_hashes": failed_hashes,
        }


def add_simple_cache(
    func: Callable,
    cache: SimpleCache,
    use_cached_values: bool = True,
    write_to_cache: bool = True,
    override_existing: bool = False,
) -> Callable:
    """
    Attach a simple cache to an asynchronous function. The result of the function
    is stored in the cache and retrieved from the cache if the function is called
    with the same arguments.

    Parameters
    ----------
    func : Callable
        The function to attach the cache to.
    cache : SimpleCache
        The cache to use.
    use_cached_values : bool, optional
        Whether to use cached values, by default True
    write_to_cache : bool, optional
        Whether to write to the cache, by default True
    override_existing : bool, optional
        Whether to override existing cache entries, by default False

    Returns
    -------
    Callable
        The function with the cache attached.
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        packaged_args = package_func_args(func, args, kwargs)

        def in_cache():
            return cache.has_key(packaged_args)

        result = (
            cache[packaged_args]
            if use_cached_values and in_cache()
            else await func(*args, **kwargs)
        )
        should_write = write_to_cache and (override_existing or not in_cache())
        if should_write:
            cache[packaged_args] = result
        return result

    return wrapper
