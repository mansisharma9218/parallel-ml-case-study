# common/

Shared harness for S01 (K-means), S02 (k-NN), S03 (preprocessing).

**Edit policy:** changes land only via a PR all three teammates approve.
Each sub-problem imports from here but never edits it directly.

## What's here

- `csv_schema.py` — the canonical `COLUMNS` list for raw result rows. Every
  sub-problem writes rows with exactly these keys (missing ones, e.g.
  `Speedup`/`Efficiency` in raw files, are written empty).
- `csv_writer.py` — `append_row(csv_path, row_dict)` appends one row,
  writing the header on first use.
- `timing.py` — `timer()` context manager for wall-clock timing a block:
  `with timer() as t: ...` then `t.elapsed` holds seconds.
- `environment.py` — `write_environment_json(path)` records CPU model,
  logical core count, RAM, OS, Python/NumPy versions once per run.
- `memory.py` — `peak_memory_mb()` returns the current process's peak RSS.

## Adding a dependency

Add it to the root `requirements.txt` in its own small PR — don't bundle
a new dependency into a sub-problem's feature PR.
