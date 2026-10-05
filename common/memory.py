"""Peak resident-set-size measurement for the Memory_Use column."""

import platform

try:
    import resource
except ImportError:  # Windows has no `resource` module
    resource = None


def peak_memory_mb():
    """Peak RSS of the current process so far, in MB. Returns None if no
    measurement method is available (no `resource` and no psutil).
    """
    if resource is not None:
        usage_kb_or_b = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports ru_maxrss in KB, macOS in bytes.
        if platform.system() == "Darwin":
            return usage_kb_or_b / (1024**2)
        return usage_kb_or_b / 1024
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024**2)
    except ImportError:
        return None
