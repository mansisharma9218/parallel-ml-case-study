"""Per-run peak memory and CPU utilization, via a background-thread psutil
sampler (same approach and USS caveat as s01_kmeans/memory_sampler.py:
the shared-memory buffers are mostly invisible to USS).

Polls every 0.2 s rather than S01's 0.05 s: the sampler thread shares the
GIL with the main process, and the sequential deskew stage is a Python
loop, so frequent polling measurably slowed the baseline (~10-20% at
0.05 s on a quick test) and would inflate parallel speedups.

CPU utilization is system-wide (psutil.cpu_percent over the run), so it
also counts any other load on the machine -- run experiments on an
otherwise idle machine.
"""

import threading

import psutil


class ResourceSampler:
    """Usage:

        with ResourceSampler() as rs:
            run_something()
        rs.peak_mb, rs.cpu_percent
    """

    def __init__(self, poll_interval=0.2):
        self._process = psutil.Process()
        self._poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread = None
        self.peak_mb = 0.0
        self.cpu_percent = None

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
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return 0

    def _run(self):
        while not self._stop_event.is_set():
            self.peak_mb = max(self.peak_mb, self._sample_once())
            self._stop_event.wait(self._poll_interval)  # wakes at once on exit

    def __enter__(self):
        psutil.cpu_percent(interval=None)  # starts the measurement window
        self.peak_mb = self._sample_once()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self.cpu_percent = psutil.cpu_percent(interval=None)
        self.peak_mb = max(self.peak_mb, self._sample_once())
        return False
