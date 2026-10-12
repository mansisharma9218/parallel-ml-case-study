"""Peak USS memory (main process + live worker children) sampled by a
background thread during one timed run. Caveat: USS ignores pages shared
between processes, so the shared_memory dataset is largely invisible here.
"""
import threading
import time

import psutil


def _uss(proc):
    try:
        return proc.memory_full_info().uss
    except (AttributeError, psutil.AccessDenied):
        try:
            return proc.memory_info().rss
        except psutil.Error:
            return 0
    except psutil.Error:
        return 0


class MemorySampler:
    def __init__(self, interval=0.02):
        self.interval = interval
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._main = psutil.Process()

    def _sample(self):
        procs = [self._main]
        try:
            procs += self._main.children(recursive=True)
        except psutil.Error:
            pass
        total = sum(_uss(p) for p in procs)
        self.peak_mb = max(self.peak_mb, total / (1024**2))

    def _loop(self):
        while not self._stop.is_set():
            self._sample()
            time.sleep(self.interval)

    def __enter__(self):
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._sample()
