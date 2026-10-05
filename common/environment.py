"""Collects a snapshot of the hardware/software environment once per
experiment run, for the reproducibility records the rubric requires.
"""

import json
import os
import platform
import sys
from pathlib import Path


def _total_ram_gb():
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        pass
    try:
        if platform.system() in ("Darwin", "Linux"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round(pages * page_size / (1024**3), 2)
    except (ValueError, OSError, AttributeError):
        pass
    return None


def collect_environment(extra=None):
    import numpy

    data = {
        "cpu_model": platform.processor() or platform.machine(),
        "cpu_cores_logical": os.cpu_count(),
        "ram_gb": _total_ram_gb(),
        "os": f"{platform.system()} {platform.release()}",
        "os_version_detail": platform.version(),
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "numpy_version": numpy.__version__,
    }
    if extra:
        data.update(extra)
    return data


def write_environment_json(path, extra=None):
    data = collect_environment(extra=extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data
