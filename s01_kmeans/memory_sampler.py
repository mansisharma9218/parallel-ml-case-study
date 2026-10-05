"""Per-run peak memory across the main process and any live worker
children, via a background-thread psutil sampler. common/memory.py's
peak_memory_mb() only tracks the main process's cumulative RSS since
start, which isn't useful for comparing individual runs or seeing
worker memory. Uses USS so the shared_memory dataset isn't
double-counted per worker (tradeoff: the dataset's own footprint is
then mostly invisible too, since it's shared rather than uniquely
owned — worth noting as a limitation, not an exact measurement).
"""

import threading
import time

import psutil


class MemorySampler:
    """Usage:

        with MemorySampler() as sampler:
            run_something()
        sampler.peak_mb  # max combined USS of this process + live children
    """

    def __init__(self, poll_interval=0.05):
        self._process = psutil.Process()
        self._poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread = None
        self.peak_mb = 0.0

    def _sample_once(self):
        total = self._process_memory(self._process)
        for child in self._process.children(recursive=True):
            total += self._process_memory(child)
        return total / (1024**2)

    @staticmethod
    def _process_memory(proc):
        try:
            return proc.memory_full_info().uss
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            try:
                return proc.memory_info().rss
            except psutil.NoSuchProcess:
                return 0

    def _run(self):
        while not self._stop_event.is_set():
            self.peak_mb = max(self.peak_mb, self._sample_once())
            time.sleep(self._poll_interval)

    def __enter__(self):
        self.peak_mb = self._sample_once()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self.peak_mb = max(self.peak_mb, self._sample_once())
        return False
