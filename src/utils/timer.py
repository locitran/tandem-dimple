import time
import logging
from functools import wraps
import json

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

class Timer:
    """
    Record the time taken by functions.

    This class tracks execution times for decorated functions. It maintains a detailed
    dictionary of the function names, their execution times, call counts, and average
    execution times. Optionally logs this information and saves it to a file.

    Example usage:
    @timer.track
    def example_function():
        pass

    At the end of the program, call timer.report() to view the recorded statistics.
    """
        
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.timers = {}

    def track(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()

            elapsed_time = end - start
            func_name = func.__name__

            if func_name not in self.timers:
                self.timers[func_name] = {
                    "n_calls": 0,
                    "execution_time": [],
                    "average_execution_time": 0.0,
                    "total_execution_time": 0.0
                }

            timer_data = self.timers[func_name]
            timer_data["n_calls"] += 1
            timer_data["execution_time"].append(elapsed_time)
            timer_data["average_execution_time"] = sum(timer_data["execution_time"]) / timer_data["n_calls"]
            timer_data["total_execution_time"] += elapsed_time

            message = f"{func_name}() took {elapsed_time:.2f} s"
            if self.verbose:
                print(message)
            logger.info(message)

            return result

        return wrapper

    def report(self, to_file=None):
        if self.verbose:
            print(f"Timing Report:\n{self.timers}")
        if to_file:
            with open(to_file, "w") as f:
                json.dump(self.timers, f, indent=4)
        return self.timers

# Global registry for named timers
_named_timers = {}

def getTimer(name, verbose=False):
    """
    Retrieve or create a named Timer instance.

    Args:
        name (str): The name of the Timer instance.
        to_file (str): Path to save timing logs.
        verbose (bool): Whether to print timing details to stdout.
        logger (logging.Logger): Optional logger instance for the Timer.

    Returns:
        Timer: The Timer instance corresponding to the given name.
    """
    global _named_timers
    if name not in _named_timers:
        _named_timers[name] = Timer(verbose=verbose)
    return _named_timers[name]