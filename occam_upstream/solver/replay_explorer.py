"""ReplayExplorer -- BFS tree search with env reset+replay.

Implements the core technique from the 3rd-place ARC-AGI-3 "Just Explore"
solution: systematically explore game states by resetting the environment
and replaying action prefixes to branch from any previously visited state.

This turns an irreversible game into a reversible one: any state can be
revisited by replaying the action sequence that first reached it.

Algorithm:
  1. Start from initial state, try all actions (BFS frontier)
  2. For each new state discovered, record the action prefix that reached it
  3. Pick the shallowest unexplored state (BFS)
  4. Reset env, replay the prefix to reach that state
  5. Try all untested actions from that state
  6. Repeat until WIN found or budget exhausted

Optimisation: incremental replay skips prefix steps when the current env
state already matches a prefix of the target state's action sequence.
"""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from typing import Any, Callable

import numpy as np

from solver.events import (
    EventEmitter,
    StateDiscoveredEvent,
    ResetEvent,
    BfsStepEvent,
    FrameDiffEvent,
)

logger = logging.getLogger("occam.solver.replay_explorer")


def _frame_hash(frame: np.ndarray, mask: np.ndarray | None = None) -> str:
    """Hash a frame, optionally masking counter/UI pixels."""
    if mask is not None:
        frame = frame.copy()
        frame[mask] = 0
    return hashlib.md5(frame.tobytes()).hexdigest()[:12]


class ReplayExplorer:
    """BFS tree search using env reset+replay for branching.

    For each state in the BFS queue:
    1. Reset env to initial state
    2. Replay the action prefix to reach the target state
    3. Try each untested action from that state
    4. Record new states and their prefixes
    """

    def __init__(
        self,
        env: Any,
        n_actions: int,
        max_total_steps: int = 5000,
        counter_mask: np.ndarray | None = None,
        action_list: list[int] | None = None,
        max_unique_states: int = 0,
        step_modulus: int = 0,
        heuristic_fn: callable | None = None,
        event_callback: Callable | None = None,
    ) -> None:
        self.env = env
        self.n_actions = n_actions
        self.max_total_steps = max_total_steps
        self.counter_mask = counter_mask
        self.max_unique_states = max_unique_states  # 0 = no limit
        self.step_modulus = max(0, int(step_modulus))
        self.heuristic_fn = heuristic_fn
        # If action_list provided, only try those actions (pruned set)
        # Otherwise try all 0..n_actions-1
        self.action_list = action_list if action_list is not None else list(range(n_actions))
        self.total_steps = 0
        self.resets = 0
        # Track which actions ever produced new states (for pruning)
        self._effective_actions: set[int] = set()
        # Event emitter for visualization
        self.events = EventEmitter(callback=event_callback)
        # Frame diff tracking for viewer
        self._prev_frame: np.ndarray | None = None
        self._new_state_count: int = 0

    def _emit_frame(self, frame: np.ndarray) -> None:
        """Emit frame diff event for viewer rendering."""
        if frame.ndim == 3 and frame.shape[-1] == 3:
            channel = frame[:, :, 0]
        elif frame.ndim == 3:
            channel = frame[0]
        else:
            channel = frame

        if self._prev_frame is None:
            changes = []
            for r in range(min(channel.shape[0], 64)):
                for c in range(min(channel.shape[1], 64)):
                    val = int(channel[r, c])
                    if val != 0:
                        changes.append((r, c, val))
            self._prev_frame = channel.copy()
            if changes:
                self.events.emit(FrameDiffEvent(changes=changes[:500]))
            return

        diff_mask = channel != self._prev_frame
        if not diff_mask.any():
            return
        positions = np.argwhere(diff_mask)
        changes = [(int(r), int(c), int(channel[r, c])) for r, c in positions[:200]]
        self._prev_frame = channel.copy()
        if changes:
            self.events.emit(FrameDiffEvent(changes=changes))

    def _hash(self, frame: np.ndarray, depth: int = 0) -> str:
        base_hash = _frame_hash(frame, self.counter_mask)
        if self.step_modulus > 1:
            return f"{base_hash}|m{self.step_modulus}:{depth % self.step_modulus}"
        return base_hash

    def _reset(self) -> np.ndarray:
        self.resets += 1
        frame = self.env.reset()
        # Emit reset event (throttled: every 10th reset)
        if self.resets % 10 == 0:
            self.events.emit(ResetEvent(count=self.resets, replay_prefix=[]))
        return frame

    def _replay(self, actions: list[int]) -> tuple[np.ndarray, bool]:
        """Reset and replay an action sequence. Returns (frame, done)."""
        frame = self._reset()
        for a in actions:
            frame, _, done, info = self.env.step(a)
            self.total_steps += 1
            if done:
                return frame, True
            if self.total_steps >= self.max_total_steps:
                return frame, True
        return frame, False

    def _replay_incremental(
        self, actions: list[int], current_prefix: list[int] | None, current_frame: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Replay from current state if possible, otherwise from scratch."""
        if (
            current_prefix is not None
            and len(actions) > len(current_prefix)
            and actions[:len(current_prefix)] == current_prefix
        ):
            frame = current_frame
            for a in actions[len(current_prefix):]:
                frame, _, done, info = self.env.step(a)
                self.total_steps += 1
                if done:
                    return frame, True
                if self.total_steps >= self.max_total_steps:
                    return frame, True
            return frame, False

        return self._replay(actions)

    def explore(self, initial_known_states: dict[str, list[int]] | None = None) -> dict:
        """Run BFS exploration with optimized DFS-order replay.

        Uses iterative deepening: explores all states at depth d before d+1,
        but within each depth processes states in DFS order (sorted by prefix)
        to maximize incremental replay. This guarantees BFS optimality while
        achieving DFS-like replay efficiency.
        """
        import heapq
        initial_frame = self._reset()
        initial_hash = self._hash(initial_frame, depth=0)

        # State tracking: hash -> action prefix that reaches it
        state_prefix: dict[str, list[int]] = {initial_hash: []}

        # Track current env state for incremental replay
        current_prefix: list[int] | None = []
        current_frame = initial_frame

        # Emit initial frame for viewer rendering
        self._prev_frame = None
        self._new_state_count = 0
        self._emit_frame(initial_frame)

        # Emit initial state discovered
        self.events.emit(StateDiscoveredEvent(hash=initial_hash, depth=0, total_states=1))

        # Merge in any externally known states
        if initial_known_states:
            for h, prefix in initial_known_states.items():
                if h not in state_prefix:
                    state_prefix[h] = prefix

        if self.heuristic_fn:
            # A* search
            h = self.heuristic_fn(initial_frame)
            queue = [(h, 0, initial_hash, [])]
            logger.info("ReplayExplorer: starting A* search (initial h=%.1f)", h)

            while queue and self.total_steps < self.max_total_steps and (
                self.max_unique_states == 0 or len(state_prefix) < self.max_unique_states
            ):
                prio, depth, state_hash, prefix = heapq.heappop(queue)

                if prefix:
                    frame, done = self._replay_incremental(prefix, current_prefix, current_frame)
                    if done or self.total_steps >= self.max_total_steps:
                        current_prefix = None
                        break
                    actual_hash = self._hash(frame, depth=len(prefix))
                    if actual_hash != state_hash:
                        current_prefix = None
                        continue
                else:
                    frame = initial_frame

                current_prefix = prefix
                current_frame = frame

                for i, action in enumerate(self.action_list):
                    if self.total_steps >= self.max_total_steps:
                        break
                    if i > 0:
                        frame, done = self._replay_incremental(prefix, current_prefix, current_frame)
                        if done or self.total_steps >= self.max_total_steps:
                            current_prefix = None
                            break
                        current_prefix = prefix
                        current_frame = frame

                    new_frame, reward, done, info = self.env.step(action)
                    self.total_steps += 1
                    current_prefix = prefix + [action]
                    current_frame = new_frame

                    if info.get("solved"):
                        self._emit_frame(new_frame)  # Emit winning frame
                        sol = prefix + [action]
                        logger.info("Solution found! %d actions, %d steps, %d resets", len(sol), self.total_steps, self.resets)
                        return {"solved": True, "solution_actions": sol, "solution_length": len(sol),
                                "total_steps": self.total_steps, "resets": self.resets,
                                "unique_states": len(state_prefix), "discovered_states": state_prefix}

                    if done:
                        current_prefix = None
                        continue

                    new_prefix = prefix + [action]
                    new_hash = self._hash(new_frame, depth=len(new_prefix))
                    is_new = new_hash not in state_prefix
                    if is_new:
                        state_prefix[new_hash] = new_prefix
                        self._effective_actions.add(action)
                        self._new_state_count += 1
                        # Emit frame every 10th new state to avoid flooding
                        if self._new_state_count % 10 == 0:
                            self._emit_frame(new_frame)
                        h = self.heuristic_fn(new_frame)
                        heapq.heappush(queue, (len(new_prefix) + h, len(new_prefix), new_hash, new_prefix))
                        self.events.emit(StateDiscoveredEvent(
                            hash=new_hash, depth=len(new_prefix), total_states=len(state_prefix)))

                    # Emit BFS step (throttled: every 10th step)
                    if self.total_steps % 10 == 0:
                        self.events.emit(BfsStepEvent(
                            from_state=state_hash, action=action,
                            to_state=new_hash if is_new else self._hash(new_frame, depth=len(new_prefix)),
                            is_new=is_new))
        else:
            # -- Depth-batched BFS with DFS-order replay --
            current_depth_states = [(initial_hash, [])]
            if initial_known_states:
                depth_groups: dict[int, list] = {}
                for h, prefix in state_prefix.items():
                    depth_groups.setdefault(len(prefix), []).append((h, prefix))
                current_depth_states = []
                for d in sorted(depth_groups.keys()):
                    current_depth_states.extend(depth_groups[d])

            while current_depth_states and self.total_steps < self.max_total_steps and (
                self.max_unique_states == 0 or len(state_prefix) < self.max_unique_states
            ):
                current_depth_states.sort(key=lambda x: x[1])
                next_depth_states: list[tuple[str, list[int]]] = []

                for state_hash, prefix in current_depth_states:
                    if self.total_steps >= self.max_total_steps:
                        break
                    if self.max_unique_states > 0 and len(state_prefix) >= self.max_unique_states:
                        break

                    if prefix:
                        frame, done = self._replay_incremental(prefix, current_prefix, current_frame)
                        if done or self.total_steps >= self.max_total_steps:
                            current_prefix = None
                            break
                        actual_hash = self._hash(frame, depth=len(prefix))
                        if actual_hash != state_hash:
                            current_prefix = None
                            continue
                    else:
                        frame = initial_frame

                    current_prefix = prefix
                    current_frame = frame

                    for i, action in enumerate(self.action_list):
                        if self.total_steps >= self.max_total_steps:
                            break
                        if i > 0:
                            frame, done = self._replay_incremental(prefix, current_prefix, current_frame)
                            if done or self.total_steps >= self.max_total_steps:
                                current_prefix = None
                                break
                            current_prefix = prefix
                            current_frame = frame

                        new_frame, reward, done, info = self.env.step(action)
                        self.total_steps += 1
                        current_prefix = prefix + [action]
                        current_frame = new_frame

                        if info.get("solved"):
                            self._emit_frame(new_frame)  # Emit winning frame
                            sol = prefix + [action]
                            logger.info("Solution found! %d actions, %d steps, %d resets", len(sol), self.total_steps, self.resets)
                            return {"solved": True, "solution_actions": sol, "solution_length": len(sol),
                                    "total_steps": self.total_steps, "resets": self.resets,
                                    "unique_states": len(state_prefix), "discovered_states": state_prefix}

                        if done:
                            current_prefix = None
                            continue

                        new_prefix = prefix + [action]
                        new_hash = self._hash(new_frame, depth=len(new_prefix))
                        is_new = new_hash not in state_prefix
                        if is_new:
                            state_prefix[new_hash] = new_prefix
                            self._effective_actions.add(action)
                            self._new_state_count += 1
                            # Emit frame every 10th new state to avoid flooding
                            if self._new_state_count % 10 == 0:
                                self._emit_frame(new_frame)
                            next_depth_states.append((new_hash, new_prefix))
                            self.events.emit(StateDiscoveredEvent(
                                hash=new_hash, depth=len(new_prefix), total_states=len(state_prefix)))

                        # Emit BFS step (throttled: every 10th step)
                        if self.total_steps % 10 == 0:
                            self.events.emit(BfsStepEvent(
                                from_state=state_hash, action=action,
                                to_state=new_hash if is_new else self._hash(new_frame, depth=len(new_prefix)),
                                is_new=is_new))

                current_depth_states = next_depth_states

        return {
            "solved": False,
            "solution_actions": None,
            "solution_length": 0,
            "total_steps": self.total_steps,
            "resets": self.resets,
            "unique_states": len(state_prefix),
            "discovered_states": state_prefix,
        }
