import multiprocessing
import queue
import sys
import time
from unittest.mock import patch, AsyncMock, MagicMock

from viewer.runner import SolverProcess, _solver_worker


def drain_queue(q: multiprocessing.Queue, timeout: float = 0.5) -> list:
    """Drain a multiprocessing.Queue, waiting up to timeout for items to appear."""
    events = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            events.append(q.get(timeout=0.05))
        except queue.Empty:
            # If we got something already, assume we're done
            if events:
                break
    return events


def test_solver_process_instantiation():
    """SolverProcess can be created with a queue."""
    q = multiprocessing.Queue()
    sp = SolverProcess(queue=q, mode="full", game_filter=None, max_actions=2000)
    assert sp.queue is q
    assert not sp.is_alive()


def test_solver_process_default_args():
    """SolverProcess accepts keyword defaults."""
    q = multiprocessing.Queue()
    sp = SolverProcess(queue=q)
    assert sp.queue is q


def test_solver_worker_emits_error_on_import_failure():
    """_solver_worker puts a solver_error event when solver.benchmark is missing."""
    q = multiprocessing.Queue()

    with patch("viewer.runner._time") as mock_time:
        mock_time.sleep = lambda _: None
        mock_time.time = time.time

        with patch.dict("sys.modules", {"solver.benchmark": None}):  # type: ignore[dict-item]
            _solver_worker(q, "full", None, 2000)

    events = drain_queue(q)
    assert len(events) >= 1
    assert events[0]["type"] == "solver_error"
    assert "reason" in events[0]["data"]


def test_solver_worker_runs_benchmark_and_emits_events():
    """_solver_worker calls BenchmarkRunner and forwards events via the queue."""
    q = multiprocessing.Queue()

    mock_runner_instance = MagicMock()
    mock_runner_instance.run = AsyncMock(return_value={})

    mock_benchmark_module = MagicMock()

    def init_side_effect(*args, **kwargs):
        cb = kwargs.get("event_callback")
        if cb:
            cb({"type": "benchmark_start", "data": {"n_games": 1}, "timestamp": 0.0})
            cb({"type": "benchmark_complete", "data": {"mean_rhae": 50.0}, "timestamp": 1.0})
        return mock_runner_instance

    mock_benchmark_module.BenchmarkRunner.side_effect = init_side_effect

    with patch("viewer.runner._time") as mock_time:
        mock_time.sleep = lambda _: None
        mock_time.time = time.time

        with patch.dict("sys.modules", {"solver.benchmark": mock_benchmark_module}):
            _solver_worker(q, "full", None, 2000)

    events = drain_queue(q)
    types = [e["type"] for e in events]
    assert "benchmark_start" in types
    assert "benchmark_complete" in types


def test_solver_worker_handles_runtime_exception():
    """_solver_worker emits solver_error when BenchmarkRunner.run() raises."""
    q = multiprocessing.Queue()

    mock_runner_instance = MagicMock()
    mock_runner_instance.run = AsyncMock(side_effect=RuntimeError("boom"))

    mock_benchmark_module = MagicMock()
    mock_benchmark_module.BenchmarkRunner.return_value = mock_runner_instance

    with patch("viewer.runner._time") as mock_time:
        mock_time.sleep = lambda _: None
        mock_time.time = time.time

        with patch.dict("sys.modules", {"solver.benchmark": mock_benchmark_module}):
            _solver_worker(q, "full", None, 2000)

    events = drain_queue(q)
    assert len(events) == 1
    assert events[0]["type"] == "solver_error"
    assert "boom" in events[0]["data"]["reason"]
