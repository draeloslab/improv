"""Bounded-memory, crash-safe per-step logs.

A long run used to keep every step's 2D/3D keypoints in RAM and write them only in stop() (GBs per hour at 7
cameras; a crash or kill lost the run). A ChunkLog keeps at most `every` rows in memory; each full block is
written to <folder>/.parts/<name>/NNNNN.npy by one background thread per process, so the step never waits on
the disk. stop() calls array() for the whole series and saves it as before, then drop_parts().

After a crash, assemble what was written:  python -m actors.chunk_log <run folder>
"""
import queue
import shutil
import sys
import threading
from pathlib import Path

import numpy as np

_q = None


def _writer():
    """Background thread: write queued blocks to disk."""
    while True:
        path, block = _q.get()
        try:
            np.save(path, block)
        except Exception as e:           # noqa: BLE001 - a lost block must not kill the actor
            print(f"chunk_log: could not write {path}: {e}", file=sys.stderr)
        finally:
            _q.task_done()


def _enqueue(path, block):
    """Queue a block for the writer thread (started on first use)."""
    global _q
    if _q is None:
        _q = queue.Queue()
        threading.Thread(target=_writer, daemon=True, name='chunk-log-writer').start()
    _q.put((path, block))


class ChunkLog:
    """A list-like log that spills every `every` rows to <folder>/.parts/<name>/NNNNN.npy, so long runs keep memory
    bounded and a crashed run can be recovered (python -m actors.chunk_log <run folder>)."""

    def __init__(self, folder, name, every=900):
        self.dir = Path(folder) / '.parts' / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.every, self.rows, self.n_parts, self.n = every, [], 0, 0

    def append(self, row):
        """Add one row (one step's array)."""
        self.rows.append(row)
        self.n += 1
        if len(self.rows) >= self.every:
            _enqueue(self.dir / f'{self.n_parts:05d}.npy', np.asarray(self.rows))
            self.rows, self.n_parts = [], self.n_parts + 1

    def __len__(self):
        return self.n

    def __bool__(self):
        return self.n > 0

    def array(self):
        """Every row so far, as one array (waits for pending block writes)."""
        if _q is not None:
            _q.join()
        blocks = [np.load(self.dir / f'{k:05d}.npy', allow_pickle=True) for k in range(self.n_parts)]
        if self.rows:
            blocks.append(np.asarray(self.rows))
        return np.concatenate(blocks) if blocks else np.zeros((0,))

    def drop_parts(self):
        """Delete the spilled blocks (after array() has been saved)."""
        shutil.rmtree(self.dir, ignore_errors=True)
        try:
            self.dir.parent.rmdir()          # .parts, once the last log is gone
        except OSError:
            pass


def recover(run_folder):
    """Assemble the parts a killed run left behind into <name>.npy (existing files are not overwritten)."""
    parts = Path(run_folder) / '.parts'
    for d in sorted(p for p in parts.glob('*') if p.is_dir()):
        out = Path(run_folder) / f'{d.name}.npy'
        files = sorted(d.glob('*.npy'))
        if out.exists() or not files:
            continue
        np.save(out, np.concatenate([np.load(f, allow_pickle=True) for f in files]))
        print(f'{out}: {sum(len(np.load(f, mmap_mode="r", allow_pickle=True)) for f in files)} rows from {len(files)} parts')


if __name__ == '__main__':
    recover(sys.argv[1])
