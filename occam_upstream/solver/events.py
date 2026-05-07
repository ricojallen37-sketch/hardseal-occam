"""Event system for solver instrumentation.

The solver emits events via EventEmitter.emit(). The callback is a simple
callable that receives a dict. The WebSocket server provides its own
callback that puts events onto a multiprocessing.Queue.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable


@dataclass
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)


# Typed event constructors for clarity in the solver code
def BenchmarkStartEvent(n_games: int, mode: str) -> Event:
    return Event(type="benchmark_start", data={"n_games": n_games, "mode": mode})


def GameStartEvent(game_id: str, title: str, total_levels: int) -> Event:
    return Event(type="game_start", data={"game_id": game_id, "title": title, "total_levels": total_levels})


def GameCompleteEvent(game_id: str, score: float, levels_completed: int, total_levels: int, time_s: float) -> Event:
    return Event(type="game_complete", data={"game_id": game_id, "score": score, "levels_completed": levels_completed, "total_levels": total_levels, "time_s": time_s})


def BenchmarkCompleteEvent(mean_rhae: float, games_solved: int, total_games: int, total_time_s: float) -> Event:
    return Event(type="benchmark_complete", data={"mean_rhae": mean_rhae, "games_solved": games_solved, "total_games": total_games, "total_time_s": total_time_s})


def PhaseChangeEvent(phase: str, strategy: str = "") -> Event:
    return Event(type="phase_change", data={"phase": phase, "strategy": strategy})


def ProbeEvent(action: int, effective: bool, diff_pixels: int = 0) -> Event:
    return Event(type="probe", data={"action": action, "effective": effective, "diff_pixels": diff_pixels})


def StateDiscoveredEvent(hash: str, depth: int, total_states: int) -> Event:
    return Event(type="state_discovered", data={"hash": hash, "depth": depth, "total_states": total_states})


def BfsStepEvent(from_state: str, action: int, to_state: str, is_new: bool) -> Event:
    return Event(type="bfs_step", data={"from_state": from_state, "action": action, "to_state": to_state, "is_new": is_new})


def ResetEvent(count: int, replay_prefix: list[int]) -> Event:
    return Event(type="reset", data={"count": count, "replay_prefix": replay_prefix})


def LevelSolvedEvent(level: int, actions: int, rhae: float) -> Event:
    return Event(type="level_solved", data={"level": level, "actions": actions, "rhae": rhae})


def LevelFailedEvent(level: int, reason: str) -> Event:
    return Event(type="level_failed", data={"level": level, "reason": reason})


def FrameDiffEvent(changes: list[tuple[int, int, int]]) -> Event:
    """changes = list of (x, y, color) for changed cells."""
    return Event(type="frame_diff", data={"changes": changes})


class EventEmitter:
    """Fire-and-forget event emitter. Callback is optional (noop if None)."""

    def __init__(self, callback: Callable[[dict], None] | None = None):
        self._callback = callback

    @property
    def callback(self):
        return self._callback

    def emit(self, event: Event) -> None:
        if self._callback is None:
            return
        payload = asdict(event)
        payload["timestamp"] = time.time()
        self._callback(payload)
