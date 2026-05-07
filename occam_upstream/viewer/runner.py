"""Multiprocessing wrapper: runs solver in child process, events via Queue."""
from __future__ import annotations

import asyncio
import time as _time
from multiprocessing import Process, Queue


def _solver_worker(queue: Queue, mode: str, game_filter: str | None, max_actions: int):
    """Entry point for the solver child process."""
    import sys
    import os

    # Ensure occam package is importable in child process
    occam_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if occam_root not in sys.path:
        sys.path.insert(0, occam_root)

    def callback(event: dict):
        try:
            queue.put(event, timeout=1)
        except Exception as e:
            print(f"QUEUE ERROR: {e} for event {event.get('type', '?')}", file=__import__('sys').stderr, flush=True)

    # Give server 2 seconds to bind before starting solver
    _time.sleep(2)

    try:
        from solver.benchmark import BenchmarkRunner
    except ImportError as e:
        callback({"type": "solver_error", "data": {"reason": f"Import failed: {e}"}, "timestamp": _time.time()})
        return

    try:
        import logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
        runner = BenchmarkRunner(
            event_callback=callback,
            mode=mode,
            game_filter=game_filter,
            max_actions=max_actions,
        )
        callback({"type": "solver_starting", "data": {"mode": mode, "game_filter": game_filter}, "timestamp": _time.time()})

        asyncio.run(runner.run())
        callback({"type": "solver_complete", "data": {}, "timestamp": _time.time()})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        callback({"type": "solver_error", "data": {"reason": str(e), "traceback": tb}, "timestamp": _time.time()})
        # Also write to stderr so it shows in server logs
        import sys
        print(f"SOLVER ERROR: {e}\n{tb}", file=sys.stderr, flush=True)


class SolverProcess:
    """Manages a solver running in a child process."""

    def __init__(self, queue: Queue, mode: str = "full", game_filter: str | None = None, max_actions: int = 2000):
        self.queue = queue
        self._proc = Process(
            target=_solver_worker,
            args=(queue, mode, game_filter, max_actions),
            daemon=True,
        )

    def start(self):
        self._proc.start()

    def is_alive(self) -> bool:
        return self._proc.is_alive()

    def join(self, timeout: float | None = None):
        self._proc.join(timeout=timeout)

    def terminate(self):
        self._proc.terminate()
