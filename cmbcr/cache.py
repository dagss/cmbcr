"""
Load from given FITS file, but if file has already been loaded in running process, return it
from cache instead. Use a seperte module so that reload() seldom/never needs to touch it.
"""
import logging
from functools import wraps
import joblib


memory = joblib.Memory(cachedir='cache')


_cache = {}

def cached(func, copy=lambda x: x.copy()):
    @wraps(func)
    def replacement(filename):
        x = _cache.get(filename, None)
        if x is None:
            logging.info('Loading {}'.format(filename))
            x = _cache[filename] = func(filename)
        return x
    return replacement
