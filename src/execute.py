"""Sandboxed execution of generated programs and answer comparison.

Programs run in forked worker processes (chunk of programs per worker) with a per-program SIGALRM timeout and
a memory limit. A worker that hangs inside C code (e.g. `10 ** 10 ** 10`) or dies is killed/replaced, the
program it was running gets None and the rest of its chunk is re-queued.
"""
import contextlib
import io
import math
import multiprocessing as mp
import os
import resource
import signal
import time
from collections import Counter
from multiprocessing.connection import wait

PRELUDE = "import math\n"


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def normalize(x):
    """Answer -> float, or None if it is not a finite number."""
    try:
        if isinstance(x, bool):
            return None
        if hasattr(x, "item") and not isinstance(x, (list, tuple, dict, str)):
            x = x.item()
        if isinstance(x, str):
            x = float(x.strip().replace(",", "").replace("$", "").rstrip("%"))
        if isinstance(x, int) and abs(x) > 2 ** 1023:
            return None
        x = float(x)
        return None if math.isnan(x) or math.isinf(x) else x
    except BaseException:
        return None


def _run_one(code, timeout):
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        env = {"__name__": "__sandbox__"}
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(PRELUDE + code, env)
            res = env["solve"]()
        signal.setitimer(signal.ITIMER_REAL, 0)
        return normalize(res)
    except BaseException:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _worker(conn, codes, timeout, mem_bytes):
    try:
        # the limit is added to the address space inherited from the parent (torch/CUDA map a lot)
        with open("/proc/self/status") as f:
            vm = next(int(line.split()[1]) * 1024 for line in f if line.startswith("VmSize:"))
        resource.setrlimit(resource.RLIMIT_AS, (vm + mem_bytes, vm + mem_bytes))
    except Exception:
        pass
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    for code in codes:
        conn.send(_run_one(code, timeout))
    conn.close()
    os._exit(0)


def run_programs(codes, timeout=1.0, n_workers=None, chunk=None, mem_mb=1024):
    """Executes every program and returns the normalized results of solve() (None on any failure)."""
    ctx = mp.get_context("fork")
    n_workers = n_workers or os.cpu_count() or 1
    if chunk is None:  # few large chunks: forking a large parent process is expensive
        chunk = min(4096, max(64, len(codes) // (4 * n_workers)))
    results = [None] * len(codes)
    pending = [(s, min(s + chunk, len(codes))) for s in range(0, len(codes), chunk)]
    running = {}  # connection -> [process, next index, end index, time of the last result]

    def start(s, e):
        r, w = ctx.Pipe(duplex=False)
        p = ctx.Process(target=_worker, args=(w, codes[s:e], timeout, mem_mb * 2 ** 20), daemon=True)
        p.start()
        w.close()
        running[r] = [p, s, e, time.monotonic()]

    def requeue_rest(st):
        if st[1] + 1 < st[2]:
            pending.append((st[1] + 1, st[2]))

    while pending or running:
        while pending and len(running) < n_workers:
            start(*pending.pop())
        now = time.monotonic()
        for conn in wait(list(running), timeout=0.05):
            st = running[conn]
            try:
                while conn.poll():
                    results[st[1]] = conn.recv()
                    st[1] += 1
                    st[3] = now
            except (EOFError, OSError):  # the worker has exited, normally or on program st[1]
                del running[conn]
                st[0].join()
                if st[1] < st[2]:
                    requeue_rest(st)
        for conn, st in list(running.items()):
            if st[1] < st[2] and now - st[3] > timeout + 1.0:
                st[0].kill()
                st[0].join()
                del running[conn]
                requeue_rest(st)
    return results


def answers_equal(a, b, tol=1e-6):
    """Same check as the grader: isclose(a, b, rtol=1e-6, atol=1e-6)."""
    if a is None or b is None:
        return False
    return abs(a - b) <= tol + tol * abs(b)


def majority_vote(candidates, answers):
    """Returns (program, answer) with the most frequent non-None answer; ties go to the earliest candidate."""
    keys = [None if a is None else round(a, 6) for a in answers]
    counts = Counter(k for k in keys if k is not None)
    if not counts:
        return candidates[0], answers[0]
    best = max(counts.values())
    for c, k, a in zip(candidates, keys, answers):
        if k is not None and counts[k] == best:
            return c, a
