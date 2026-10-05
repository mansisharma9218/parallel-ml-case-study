"""A tiny context manager for wall-clock timing of a code block."""

import time
from contextlib import contextmanager


class _TimerResult:
    elapsed = None


@contextmanager
def timer():
    """Usage:

        with timer() as t:
            do_work()
        print(t.elapsed)  # seconds, set once the block exits
    """
    result = _TimerResult()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed = time.perf_counter() - start
