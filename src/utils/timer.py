import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def execution_timer(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        result = func(self, *args, **kwargs)
        end = time.perf_counter()
        logger.info(f"{func.__name__}() took {(end - start) :.2f} s or {(end - start) / 60 :.2f} min!")
        return result
    return wrapper
