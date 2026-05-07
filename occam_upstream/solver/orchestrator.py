"""Occam Game Orchestrator — plays unknown ARC-AGI-3 games.

Coordinates probe-based discovery, BFS state exploration, combo search,
navigation solving, and RHAE scoring into a two-phase loop:

  DISCOVER    -> probe actions, identify effective ones, build state graph
  PLAN+EXECUTE -> combo search, navigation solver, replay BFS, deepcopy BFS
"""

from __future__ import annotations

import hashlib
import logging
import random
from typing import Any, Callable

import numpy as np

from solver.action_ranker import ActionRanker
from solver.context_compression import (
    ObjectTracker,
    compress_diff,
    compress_l1,
    compress_l2,
    detect_objects,
)
from solver.priority_tiers import get_priority_click_targets
from solver.replay_explorer import ReplayExplorer
from solver.rhae import compute_rhae, should_give_up, weighted_game_score
from solver.state_graph import StateGraph
from solver.events import (
    EventEmitter,
    PhaseChangeEvent,
    ProbeEvent,
    LevelSolvedEvent,
    LevelFailedEvent,
    FrameDiffEvent,
    GameStartEvent,
)

logger = logging.getLogger("occam.solver.orchestrator")



# ---------------------------------------------------------------------------
# Environment adapter -- normalises Sokoban vs ARC-AGI-3 interfaces
# ---------------------------------------------------------------------------


def _is_arc_env(env: Any) -> bool:
    """Detect ARC-AGI-3 environment via duck typing."""
    return hasattr(env, "available_actions") and hasattr(env, "levels_completed")


def _frame_to_numpy(raw: Any) -> np.ndarray:
    """Convert any frame representation to numpy uint8 array."""
    if isinstance(raw, np.ndarray):
        return raw.astype(np.uint8)
    # ARC-AGI-3 FrameDataRaw -- has .frame attribute
    if hasattr(raw, "frame"):
        return np.asarray(raw.frame, dtype=np.uint8)
    return np.asarray(raw, dtype=np.uint8)


def _env_step(env: Any, action: int | dict) -> tuple[np.ndarray, float, bool, dict]:
    """Step the environment, normalising both interfaces."""
    if _is_arc_env(env):
        if isinstance(action, dict):
            result = env.step(action["action_id"], data=action.get("data"))
        else:
            result = env.step(action)
        frame = _frame_to_numpy(result)
        from enum import IntEnum

        done = False
        solved = False
        if hasattr(env, "state"):
            state_val = env.state
            if hasattr(state_val, "value"):
                done = state_val.value != 0
                solved = state_val.value == 1
            else:
                done = state_val != 0
                solved = state_val == 1
        info = {"solved": solved, "levels_completed": getattr(env, "levels_completed", 0)}
        reward = 1.0 if solved else 0.0
        return frame, reward, done, info
    else:
        frame, reward, done, info = env.step(action)
        return _frame_to_numpy(frame), reward, done, info


def _env_reset(env: Any) -> np.ndarray:
    return _frame_to_numpy(env.reset())


def _available_actions(env: Any) -> list[int]:
    if _is_arc_env(env):
        return list(env.available_actions)
    if hasattr(env, "get_available_actions"):
        return env.get_available_actions()
    return list(range(env.n_actions))


_COLOR_NAMES = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "grey", 6: "magenta", 7: "orange", 8: "light-blue",
    9: "dark-red", 10: "dark-blue", 11: "light-purple",
    12: "dark-green", 13: "purple", 14: "pink", 15: "white",
}


def _get_channel(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[-1] == 3:
        return frame[:, :, 0]
    elif frame.ndim == 3:
        return frame[0]
    return frame


def _ranker_frame(frame: np.ndarray) -> np.ndarray:
    """Extract single channel and clamp to 0-15 for action ranking."""
    ch = _get_channel(frame)
    return np.clip(ch, 0, 15).astype(np.uint8)


def _build_ui_mask(frame: np.ndarray, border_px: int = 3) -> np.ndarray:
    """Build a boolean mask for UI elements (status bars, counters, borders)."""
    h, w = frame.shape[:2] if frame.ndim >= 2 else (64, 64)
    if frame.ndim == 3 and frame.shape[-1] == 3:
        mask = np.zeros(frame.shape, dtype=bool)
        mask[:border_px, :, :] = True
        mask[-border_px:, :, :] = True
        mask[:, :border_px, :] = True
        mask[:, -border_px:, :] = True
    elif frame.ndim == 3:
        mask = np.zeros(frame.shape, dtype=bool)
        mask[:, :border_px, :] = True
        mask[:, -border_px:, :] = True
        mask[:, :, :border_px] = True
        mask[:, :, -border_px:] = True
    else:
        mask = np.zeros(frame.shape, dtype=bool)
        mask[:border_px, :] = True
        mask[-border_px:, :] = True
        mask[:, :border_px] = True
        mask[:, -border_px:] = True
    return mask


def _frame_hash(frame: np.ndarray, mask: np.ndarray | None = None) -> str:
    """Hash a frame, optionally masking out counter/UI pixels."""
    if mask is not None:
        frame = frame.copy()
        frame[mask] = 0
    return hashlib.md5(frame.tobytes()).hexdigest()[:12]


def detect_counter_mask(env: Any, frame: np.ndarray, actions: list[int]) -> np.ndarray:
    """Detect step-counter pixels by taking two actions and finding always-changing pixels."""
    if frame.ndim == 3 and frame.shape[-1] == 3:
        get_ch = lambda f: f[:, :, 0]
    elif frame.ndim == 3:
        get_ch = lambda f: f[0]
    else:
        get_ch = lambda f: f

    frames_ch = [get_ch(frame)]
    current_frame = frame
    actions_taken = []

    for i in range(min(3, len(actions))):
        a = actions[i % len(actions)]
        new_frame, _, done, _ = _env_step(env, a)
        frames_ch.append(get_ch(new_frame))
        actions_taken.append(a)
        current_frame = new_frame
        if done:
            break

    if len(frames_ch) < 3:
        return np.zeros(frame.shape, dtype=bool)

    always_change = np.ones(frames_ch[0].shape, dtype=bool)
    for i in range(len(frames_ch) - 1):
        always_change &= (frames_ch[i] != frames_ch[i + 1])

    if frame.ndim == 3 and frame.shape[-1] == 3:
        mask = np.stack([always_change] * 3, axis=-1)
    elif frame.ndim == 3:
        mask = np.stack([always_change] * frame.shape[0], axis=0)
    else:
        mask = always_change

    n_masked = int(always_change.sum())
    if n_masked > 0:
        logger.info("Detected %d counter/UI pixels to mask", n_masked)

    return mask


# ---------------------------------------------------------------------------
# Click-action expansion -- discretize (x,y) clicks into virtual actions
# ---------------------------------------------------------------------------

CLICK_ACTION_ID = 6
FRAME_SIZE = 64


def expand_click_actions(
    raw_actions: list[int],
    grid_size: int = 8,
    frame: np.ndarray | None = None,
) -> tuple[list[int], dict[int, int | dict]]:
    """Expand click actions into a grid of virtual actions."""
    action_map: dict[int, int | dict] = {}
    idx = 0

    is_click_only = all(a == CLICK_ACTION_ID for a in raw_actions)
    effective_grid = grid_size * 2 if is_click_only else grid_size

    for a in raw_actions:
        if a == CLICK_ACTION_ID:
            click_positions = _compute_click_positions(effective_grid, frame)
            for x, y in click_positions:
                action_map[idx] = {
                    "action_id": CLICK_ACTION_ID,
                    "data": {"x": int(x), "y": int(y)},
                }
                idx += 1
        else:
            action_map[idx] = a
            idx += 1

    return list(range(idx)), action_map


def _compute_click_positions(
    grid_size: int, frame: np.ndarray | None
) -> list[tuple[int, int]]:
    """Compute click positions, prioritizing non-background regions."""
    positions: list[tuple[int, int]] = []

    if frame is not None:
        try:
            priority_targets = get_priority_click_targets(frame, max_targets=grid_size * 2)
            for y, x in priority_targets:
                if 0 <= x < FRAME_SIZE and 0 <= y < FRAME_SIZE and (x, y) not in positions:
                    positions.append((x, y))
        except Exception:
            pass

        if frame.ndim == 3 and frame.shape[-1] == 3:
            channel = frame[:, :, 0]
        elif frame.ndim == 3:
            channel = frame[0]
        else:
            channel = frame

        unique, counts = np.unique(channel, return_counts=True)
        bg_val = unique[np.argmax(counts)]

        fg_mask = channel != bg_val
        fg_positions = np.argwhere(fg_mask)

        if len(fg_positions) > 0:
            from solver.context_compression import detect_objects

            objects = detect_objects(frame)
            for obj in objects[:grid_size * grid_size // 2]:
                cx = (obj["bbox"][0] + obj["bbox"][2]) // 2
                cy = (obj["bbox"][1] + obj["bbox"][3]) // 2
                positions.append((cx, cy))

            if len(fg_positions) > 0:
                min_r, min_c = fg_positions.min(axis=0)
                max_r, max_c = fg_positions.max(axis=0)
                n_grid = max(4, grid_size // 2)
                r_step = max(1, (max_r - min_r) // n_grid)
                c_step = max(1, (max_c - min_c) // n_grid)
                for gr in range(n_grid):
                    for gc in range(n_grid):
                        y = min_r + gr * r_step + r_step // 2
                        x = min_c + gc * c_step + c_step // 2
                        if 0 <= x < FRAME_SIZE and 0 <= y < FRAME_SIZE:
                            positions.append((x, y))

    budget = grid_size * grid_size
    if len(positions) < budget:
        step = max(1, (FRAME_SIZE - 1) // (grid_size - 1)) if grid_size > 1 else FRAME_SIZE
        for gy in range(grid_size):
            for gx in range(grid_size):
                x = min(gx * step, FRAME_SIZE - 1)
                y = min(gy * step, FRAME_SIZE - 1)
                if (x, y) not in positions:
                    positions.append((x, y))

    if frame is not None:
        if frame.ndim == 3 and frame.shape[-1] == 3:
            ch = frame[:, :, 0]
        elif frame.ndim == 3:
            ch = frame[0]
        else:
            ch = frame
        bg_color = int(np.bincount(ch.ravel()).argmax())
        h, w = ch.shape[:2]
        pruned = []
        for col, row in positions:
            r1, r2 = max(0, row - 2), min(h, row + 3)
            c1, c2 = max(0, col - 2), min(w, col + 3)
            region = ch[r1:r2, c1:c2]
            if np.any(region != bg_color):
                pruned.append((col, row))
        if pruned:
            positions = pruned

    seen: set[tuple[int, int]] = set()
    unique_positions: list[tuple[int, int]] = []
    for p in positions:
        if p not in seen:
            seen.add(p)
            unique_positions.append(p)
    max_clicks = min(budget, 32)
    return unique_positions[:max_clicks]


# ---------------------------------------------------------------------------
# Virtual-action env wrapper for ReplayExplorer
# ---------------------------------------------------------------------------


class _VirtualActionEnv:
    """Wraps an env to translate virtual action indices to real actions."""

    def __init__(self, env: Any, action_map: dict[int, int | dict]) -> None:
        self._env = env
        self._action_map = action_map

    def reset(self) -> np.ndarray:
        return _env_reset(self._env)

    def step(self, virtual_action: int) -> tuple[np.ndarray, float, bool, dict]:
        real_action = self._action_map[virtual_action]
        return _env_step(self._env, real_action)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


class GameOrchestrator:
    """Occam orchestrator -- plays unknown games via BFS-only strategies."""

    def __init__(
        self,
        max_actions_per_level: int = 100000,
        event_callback: Callable | None = None,
        skip_navigation: bool = False,
        skip_combo: bool = False,
        skip_deepcopy: bool = False,
    ) -> None:

        self.max_actions_per_level = max_actions_per_level
        self._configured_max_actions_per_level = max_actions_per_level
        self.skip_navigation = skip_navigation
        self.skip_combo = skip_combo
        self.skip_deepcopy = skip_deepcopy

        # Sub-modules
        self.tracker = ObjectTracker()
        self.state_graph = StateGraph(n_actions=0)
        self.action_ranker = ActionRanker()

        # Event emitter for visualization hooks
        self.events = EventEmitter(callback=event_callback)
        self._event_callback = event_callback

        # Frame diff tracking for viewer rendering
        self._prev_frame: np.ndarray | None = None
        self._frame_emit_count: int = 0

        # State
        self.action_log: list[dict] = []
        self._winning_combos: list[tuple[int, ...]] = []
        self._winning_raw_actions: list[list[dict | int]] = []
        self._action_map: dict[int, int | dict] = {}
        self._cached_nav: dict | None = None

    # ------------------------------------------------------------------
    # Frame diff emission for viewer rendering
    # ------------------------------------------------------------------

    def _emit_frame(self, frame: np.ndarray) -> None:
        """Emit frame diff event for viewer rendering."""
        channel = _get_channel(frame)
        if self._prev_frame is None:
            # Full frame on first render -- emit ALL non-black cells (4096 max)
            changes = []
            for r in range(min(channel.shape[0], 64)):
                for c in range(min(channel.shape[1], 64)):
                    val = int(channel[r, c])
                    if val != 0:
                        changes.append((r, c, val))
            self._prev_frame = channel.copy()
            if changes:
                self.events.emit(FrameDiffEvent(changes=changes))
            return

        diff_mask = channel != self._prev_frame
        if not diff_mask.any():
            return
        positions = np.argwhere(diff_mask)
        changes = [(int(r), int(c), int(channel[r, c])) for r, c in positions]
        self._prev_frame = channel.copy()
        if changes:
            self.events.emit(FrameDiffEvent(changes=changes))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def play_game(self, env: Any, baselines: list[int] | None = None) -> dict:
        """Play one game (possibly multi-level). Returns result dict."""
        if baselines is None:
            baselines = [self.max_actions_per_level]

        total_levels = len(baselines)
        levels_completed = 0
        level_scores: list[float] = []
        total_actions = 0
        total_phases: dict[str, int] = {"discover": 0, "execute": 0}
        last_result: dict = {}

        # Emit game start event
        self.events.emit(PhaseChangeEvent(phase="game_start", strategy="bfs_only"))

        # Reset per-game state
        self._winning_combos = []
        self._winning_raw_actions = []
        self._action_map = {}
        self._cached_nav = None

        for level_idx in range(total_levels):
            baseline = baselines[level_idx]

            # Reset per-level state
            self.state_graph = StateGraph(n_actions=0)
            self.action_log = []
            self.action_ranker.reset()

            dynamic_budget = max(baseline * baseline * 200, 1000000)
            self.max_actions_per_level = min(
                dynamic_budget,
                self._configured_max_actions_per_level,
            )

            # -- COMBO SHORT-CIRCUIT: try cached combos before discovery --
            shortcut_solved = False

            # Try raw action sequences first
            for prev_raw in self._winning_raw_actions:
                if not prev_raw:
                    continue
                _env_reset(env)
                prev_levels = env.levels_completed if _is_arc_env(env) else 0
                for action in prev_raw:
                    _, _, _, info = _env_step(env, action)
                cur_levels = info.get("levels_completed", 0) if not _is_arc_env(env) else env.levels_completed
                if (_is_arc_env(env) and cur_levels > prev_levels) or info.get("solved"):
                    levels_completed += 1
                    actions_used = len(prev_raw) + 1
                    total_actions += actions_used
                    rhae = compute_rhae(actions_used, baseline)
                    level_scores.append(rhae)
                    logger.info(
                        "SHORT-CIRCUIT: Level %d solved with cached raw combo (length %d, RHAE %.1f%%)",
                        level_idx, len(prev_raw), rhae * 100,
                    )
                    self.events.emit(LevelSolvedEvent(level=level_idx, actions=actions_used, rhae=rhae))
                    shortcut_solved = True
                    break

            # Try virtual-action combos
            if not shortcut_solved and self._winning_combos and self._action_map:
                for prev_combo in self._winning_combos:
                    if not prev_combo:
                        continue
                    _env_reset(env)
                    prev_levels = env.levels_completed if _is_arc_env(env) else 0
                    for action in prev_combo:
                        real_action = self._action_map.get(action, action)
                        _, _, _, info = _env_step(env, real_action)
                    cur_levels = info.get("levels_completed", 0) if not _is_arc_env(env) else env.levels_completed
                    if (_is_arc_env(env) and cur_levels > prev_levels) or info.get("solved"):
                        levels_completed += 1
                        actions_used = len(prev_combo) + 1
                        total_actions += actions_used
                        rhae = compute_rhae(actions_used, baseline)
                        level_scores.append(rhae)
                        logger.info(
                            "SHORT-CIRCUIT: Level %d solved with cached combo (length %d, RHAE %.1f%%)",
                            level_idx, len(prev_combo), rhae * 100,
                        )
                        self.events.emit(LevelSolvedEvent(level=level_idx, actions=actions_used, rhae=rhae))
                        shortcut_solved = True
                        break

            if shortcut_solved:
                continue

            result = await self._play_level(env, baselines, level_idx)
            last_result = result

            actions_used = result.get("total_actions", 0)
            total_actions += actions_used

            phases = result.get("phases", {})
            for k in total_phases:
                total_phases[k] += phases.get(k, 0)

            if result.get("levels_completed", 0) > 0:
                levels_completed += 1
                rhae = compute_rhae(actions_used, baseline)
                level_scores.append(rhae)
                self.events.emit(LevelSolvedEvent(level=level_idx, actions=actions_used, rhae=rhae))
            else:
                self.events.emit(LevelFailedEvent(level=level_idx, reason="budget_exhausted"))
                break

        mean_rhae = weighted_game_score(level_scores, total_levels)

        return {
            "total_actions": total_actions,
            "levels_completed": levels_completed,
            "total_levels": total_levels,
            "level_scores": level_scores,
            "mean_rhae": mean_rhae,
            "game_score": mean_rhae,
            "cost_usd": 0.0,
            "phases": total_phases,
            "graph_nodes": last_result.get("graph_nodes", 0),
            "graph_edges": last_result.get("graph_edges", 0),
            "graph_frontier": last_result.get("graph_frontier", 0),
            "graph_exploration_pct": last_result.get("graph_exploration_pct", 0.0),
            "resets": last_result.get("resets", 0),
            "solution_length": last_result.get("solution_length", 0),
        }

    async def _play_level(self, env: Any, baselines: list[int] | None = None, level_idx: int = 0) -> dict:
        """Play a single level."""
        baselines = baselines or [50]
        frame = _env_reset(env)
        self._prev_frame = None  # Reset for new level
        self._frame_emit_count = 0
        self._emit_frame(frame)  # Emit initial frame for viewer
        raw_available = _available_actions(env)

        virtual_actions, self._action_map = expand_click_actions(
            raw_available, frame=frame
        )
        all_virtual_actions = list(virtual_actions)
        all_action_map = dict(self._action_map)
        n_virtual = len(virtual_actions)

        total_actions = 0
        levels_completed = 0
        level_scores: list[float] = []
        phase_log: dict[str, int] = {"discover": 0, "execute": 0}

        baseline = baselines[min(level_idx, len(baselines) - 1)]

        # Reset per-level state
        self.tracker = ObjectTracker()
        self.state_graph = StateGraph(n_actions=n_virtual)
        self.action_log = []

        self._counter_mask = np.zeros(frame.shape, dtype=bool)
        self.tracker.update(detect_objects(frame))

        # -- PHASE 1: DISCOVER --
        # Skip probing if we have cached navigation — try nav first, fall back to full discovery
        if self._cached_nav is not None and not self.skip_navigation:
            self.events.emit(PhaseChangeEvent(phase="discover", strategy="cached_nav_skip"))
            remaining_budget = self.max_actions_per_level
            reactive_nav = self._solve_reactive_navigation(env, virtual_actions, remaining_budget)
            total_actions += reactive_nav["actions_used"]
            phase_log["execute"] += reactive_nav["actions_used"]
            if reactive_nav["solved"]:
                levels_completed += 1
                level_scores.append(compute_rhae(total_actions, baseline))
                logger.info("Cached nav solved level %d in %d actions!", level_idx, reactive_nav["actions_used"])
                return {
                    "total_actions": total_actions,
                    "levels_completed": levels_completed,
                    "level_scores": level_scores,
                    "phases": phase_log,
                    "graph_nodes": 0, "graph_edges": 0,
                    "graph_frontier": 0, "graph_exploration_pct": 0,
                    "resets": 0, "solution_length": len(reactive_nav.get("solution_actions", [])),
                }
            else:
                # Cached nav failed — clear cache and fall through to full discovery
                self._cached_nav = None
                logger.info("Cached nav failed, falling back to full discovery")

        self.events.emit(PhaseChangeEvent(phase="discover", strategy="probe_all_actions"))
        effective_actions, virtual_actions, n_virtual, probes, discover_actions = \
            self._discover_and_prune(env, frame, virtual_actions)
        phase_log["discover"] = len(probes)
        total_actions += discover_actions

        frame = probes[-1]["frame"] if probes else frame

        # -- PHASE 2: STATE-DEPENDENT CLICK SOLVER --
        self.events.emit(PhaseChangeEvent(phase="execute", strategy="plan_and_execute"))
        explore_result = {"solved": False, "total_steps": 0, "resets": 0, "unique_states": 0, "solution_length": 0}
        if (levels_completed == 0
                and getattr(self, '_click_only_game', False)
                and len(virtual_actions) <= 2):
            remaining_budget = self.max_actions_per_level - total_actions
            reactive_result = self._solve_reactive_click(env, remaining_budget)
            total_actions += reactive_result["actions_used"]
            phase_log["execute"] += reactive_result["actions_used"]
            if reactive_result["solved"]:
                levels_completed += 1
                solution_len = reactive_result["solution_length"]
                level_scores.append(compute_rhae(total_actions, baseline))
                raw_actions = reactive_result.get("solution_raw_actions")
                if raw_actions:
                    self._winning_raw_actions.append(raw_actions)
                logger.info("Reactive click solved level %d in %d actions!", level_idx, solution_len)

        # -- PHASE 3a.5: NAVIGATION SOLVER --
        if levels_completed == 0 and 3 <= len(virtual_actions) <= 7 and not self.skip_navigation:
            remaining_budget = self.max_actions_per_level - total_actions

            # Try reactive navigation first (pixel-wise greedy)
            reactive_nav = self._solve_reactive_navigation(env, virtual_actions, remaining_budget)
            if reactive_nav["solved"]:
                total_actions += reactive_nav["actions_used"]
                phase_log["execute"] += reactive_nav["actions_used"]
                levels_completed += 1
                level_scores.append(compute_rhae(total_actions, baseline))
                if reactive_nav.get("solution_actions"):
                    self._winning_combos.append(tuple(reactive_nav["solution_actions"]))
                logger.info("Reactive navigation solved level %d in %d actions!", level_idx, reactive_nav["actions_used"])
            else:
                total_actions += reactive_nav["actions_used"]
                phase_log["execute"] += reactive_nav["actions_used"]
                # Fallback to standard navigation solver
                nav_result = self._try_navigation_solve(env, frame, virtual_actions, self.max_actions_per_level - total_actions)
                if nav_result and nav_result.get("actions_used", 0) > 0:
                    total_actions += nav_result["actions_used"]
                    phase_log["execute"] += nav_result["actions_used"]
                if nav_result and nav_result["solved"]:
                    levels_completed += 1
                    level_scores.append(compute_rhae(total_actions, baseline))
                    logger.info("Navigation solver solved level %d in %d actions!", level_idx, nav_result["actions_used"])
                    if nav_result.get("solution_actions"):
                        self._winning_combos.append(tuple(nav_result["solution_actions"]))
                        raw = [self._action_map.get(a, a) for a in nav_result["solution_actions"]]
                        self._winning_raw_actions.append(raw)

        # -- PHASE 3b: COMBO SEARCH + BFS EXPLORATION --
        if levels_completed == 0 and len(virtual_actions) <= 10 and not self.skip_combo:
            remaining_budget = self.max_actions_per_level - total_actions
            n_eff = len(virtual_actions)
            depth = 20 if n_eff <= 2 else 15 if n_eff <= 3 else 12 if n_eff <= 5 else 10
            if n_eff <= 2:
                combo_budget = min(remaining_budget, remaining_budget // 4)
            elif n_eff <= 3:
                combo_budget = min(remaining_budget, remaining_budget // 5)
            elif n_eff <= 6:
                combo_budget = min(remaining_budget, max(50000, remaining_budget // 6))
            else:
                combo_budget = 0
            combo_result = self._execute_combo_search(
                env, virtual_actions, combo_budget, max_depth=depth,
            )
            total_actions += combo_result["actions_used"]
            phase_log["execute"] += combo_result["actions_used"]
            if combo_result["solved"]:
                levels_completed += 1
                explore_result = combo_result["explore_result"]

        # -- PHASE 3b.5: COMBO SEARCH WITH UNPRUNED RAW ACTIONS --
        raw_non_click = [a for a in raw_available if a != CLICK_ACTION_ID]
        n_raw = len(raw_non_click)
        if (levels_completed == 0
                and n_raw > len(virtual_actions)
                and n_raw <= 8
                and not self.skip_combo):
            remaining_budget = self.max_actions_per_level - total_actions
            unpruned_map: dict[int, int | dict] = {i: a for i, a in enumerate(raw_non_click)}
            unpruned_va = list(range(n_raw))
            unpruned_depth = 7 if n_raw <= 6 else 5
            unpruned_budget = min(remaining_budget // 4, 30000)
            if unpruned_budget > 100:
                logger.info(
                    "Combo search (unpruned): trying %d raw non-click actions %s up to depth %d",
                    n_raw, raw_non_click, unpruned_depth,
                )
                saved_action_map = self._action_map
                self._action_map = unpruned_map
                combo_result2 = self._execute_combo_search(
                    env, unpruned_va, unpruned_budget, max_depth=unpruned_depth,
                )
                if not combo_result2["solved"]:
                    self._action_map = saved_action_map
                total_actions += combo_result2["actions_used"]
                phase_log["execute"] += combo_result2["actions_used"]
                if combo_result2["solved"]:
                    levels_completed += 1
                    explore_result = combo_result2["explore_result"]
                    virtual_actions = unpruned_va
                    all_virtual_actions = unpruned_va

        # BFS fallback
        if levels_completed == 0:
            remaining_budget = self.max_actions_per_level - total_actions
            if len(virtual_actions) <= 8 and n_raw > len(virtual_actions):
                deepcopy_reserve = min(15000, remaining_budget // 5)
                bfs_budget = remaining_budget - deepcopy_reserve
            else:
                bfs_budget = remaining_budget
            bfs_result = self._execute_bfs_fallback(
                env, virtual_actions, n_virtual, bfs_budget,
                probes=probes, initial_frame=frame, all_virtual_actions=all_virtual_actions,
            )
            total_actions += bfs_result["actions_used"]
            phase_log["execute"] += bfs_result["actions_used"]
            explore_result = bfs_result["explore_result"]
            if bfs_result["solved"]:
                levels_completed += 1
                solution_len = bfs_result["explore_result"]["solution_length"]
                level_scores.append(compute_rhae(total_actions, baseline))
                sol_actions = bfs_result["explore_result"].get("solution_actions")
                if sol_actions:
                    self._winning_combos.append(tuple(sol_actions))
                    raw = [self._action_map.get(a, a) for a in sol_actions]
                    self._winning_raw_actions.append(raw)
                logger.info(
                    "BFS solved level %d in %d actions (%d resets, %d states)",
                    level_idx, solution_len, bfs_result["explore_result"]["resets"],
                    bfs_result["explore_result"]["unique_states"],
                )

        # -- PHASE 3d: DEEPCOPY BFS FALLBACK --
        dc_actions = list(range(n_raw)) if n_raw <= 8 else virtual_actions
        dc_action_map = {i: a for i, a in enumerate(raw_non_click)} if n_raw <= 8 else self._action_map
        should_try_deepcopy = n_raw > len(virtual_actions) and n_raw <= 8
        if levels_completed == 0 and should_try_deepcopy and not self.skip_deepcopy:
            remaining_budget = self.max_actions_per_level - total_actions
            dc_budget = min(remaining_budget, 15000)
            if dc_budget > 100:
                logger.info(
                    "Deepcopy BFS: trying %d actions (raw=%s) with budget %d",
                    len(dc_actions), raw_non_click[:8] if n_raw > len(virtual_actions) else "pruned", dc_budget,
                )
                saved_action_map2 = self._action_map
                self._action_map = dc_action_map
                dc_result = self._execute_deepcopy_bfs(
                    env, dc_actions, dc_budget,
                )
                if not dc_result["solved"]:
                    self._action_map = saved_action_map2
                total_actions += dc_result["actions_used"]
                phase_log["execute"] += dc_result["actions_used"]
                if dc_result["solved"]:
                    levels_completed += 1
                    solution_len = len(dc_result["solution_actions"])
                    level_scores.append(compute_rhae(total_actions, baseline))
                    sol_actions = dc_result["solution_actions"]
                    self._winning_combos.append(tuple(sol_actions))
                    raw = [self._action_map.get(a, a) for a in sol_actions]
                    self._winning_raw_actions.append(raw)
                    logger.info(
                        "Deepcopy BFS solved level %d in %d actions (%d steps explored)",
                        level_idx, solution_len, dc_result["actions_used"],
                    )

        return {
            "total_actions": total_actions,
            "levels_completed": levels_completed,
            "level_scores": level_scores,
            "game_score": sum(level_scores) / max(len(level_scores), 1),
            "cost_usd": 0.0,
            "phases": phase_log,
            "graph_nodes": len(self.state_graph.nodes),
            "graph_edges": sum(len(e) for e in self.state_graph._edges.values()),
            "graph_frontier": len([n for n in self.state_graph.nodes
                                   if self.state_graph.untried_actions(n)]),
            "graph_exploration_pct": (
                len(self.state_graph.nodes) / max(explore_result.get("unique_states", 1), 1)
            ),
            "resets": explore_result.get("resets", 0),
            "solution_length": explore_result.get("solution_length", 0),
        }

    # ------------------------------------------------------------------
    # Navigation solver
    # ------------------------------------------------------------------

    def _solve_reactive_navigation(
        self, env: Any, virtual_actions: list[int], remaining_budget: int
    ) -> dict:
        """Reactive navigation solver: pixel-wise moves toward targets and interacts."""
        actions_used = 0
        solution_actions = []
        max_steps = min(500, remaining_budget // 2)

        # Fast path: reuse cached navigation parameters from previous level
        cached = getattr(self, '_cached_nav', None)
        if cached and cached["cursor_color"] is not None:
            cursor_color = cached["cursor_color"]
            bg_color = cached["bg_color"]
            cached_threshold = cached.get("threshold", 1.0)

            # Use real actions directly — they survive action remapping between levels
            arrow_real = cached.get("arrow_map_real", {})
            interact_real = cached.get("interact_real")

            # Fall back to virtual-action cache if real actions not available
            if not arrow_real:
                arrow_map = cached.get("arrow_map", {})
                interact_va = cached.get("interact_va")
                if len(arrow_map) < 2:
                    pass  # Will fall through to full probing
                else:
                    arrow_real = {k: self._action_map.get(v, v) for k, v in arrow_map.items()}
                    interact_real = self._action_map.get(interact_va) if interact_va is not None else None

            if len(arrow_real) >= 2:
                logger.info("Reactive Nav (cached): cursor=%d, arrows=%s, interact=%s, threshold=%.1f",
                             cursor_color, list(arrow_real.keys()), interact_real, cached_threshold)
                curr_f = _env_reset(env)
                actions_used += 1
                info = {}
                for step in range(max_steps):
                    grid = _get_channel(curr_f)
                    pos_cursor = np.argwhere(grid == cursor_color)
                    if len(pos_cursor) == 0:
                        break
                    cursor_pos = pos_cursor.mean(axis=0)
                    unique_g, counts_g = np.unique(grid, return_counts=True)
                    non_bg = [c for c in unique_g if c != bg_color and c != 0 and c != cursor_color]
                    if not non_bg:
                        break
                    target_color = min(non_bg, key=lambda c: counts_g[np.where(unique_g == c)[0][0]])
                    targets = np.argwhere(grid == target_color)
                    dists = np.sum(np.abs(targets - cursor_pos), axis=1)
                    target = targets[np.argmin(dists)]
                    dr, dc = target[0] - cursor_pos[0], target[1] - cursor_pos[1]
                    within = abs(dr) < cached_threshold and abs(dc) < cached_threshold
                    if within and interact_real is not None:
                        curr_f, reward, done, info = _env_step(env, interact_real)
                        actions_used += 1
                    else:
                        # Keep moving toward target
                        if abs(dr) >= abs(dc):
                            action_name = "up" if dr < 0 else "down"
                        else:
                            action_name = "left" if dc < 0 else "right"
                        if action_name in arrow_real:
                            curr_f, reward, done, info = _env_step(env, arrow_real[action_name])
                            actions_used += 1
                        else:
                            break
                    if info.get("solved") or done:
                        return {"solved": True, "actions_used": actions_used,
                                "solution_actions": solution_actions}
                return {"solved": False, "actions_used": actions_used}

        # 1. Probe to find arrows and interaction (first level only)
        frame_before = _env_reset(env)
        actions_used += 1
        ch_before = _get_channel(frame_before)
        unique, counts = np.unique(ch_before, return_counts=True)
        bg_color = int(unique[np.argmax(counts)])

        arrow_map: dict[str, int] = {}
        cursor_color = None
        _step_sizes: list[float] = []

        for va in virtual_actions:
            _env_reset(env)
            actions_used += 1
            frame_after, _, _, _ = _env_step(env, self._action_map[va])
            actions_used += 1
            ch_after = _get_channel(frame_after)
            diff = (ch_before != ch_after)
            if not diff.any():
                continue

            candidate_colors = set(ch_before[diff].tolist()) | set(ch_after[diff].tolist())
            candidate_colors.discard(bg_color)
            candidate_colors.discard(0)

            best_score = -1
            best_color = None
            best_dy, best_dx = 0, 0
            for c in candidate_colors:
                p0 = np.argwhere(ch_before == c)
                p1 = np.argwhere(ch_after == c)
                if len(p0) == 0 or len(p1) == 0:
                    continue
                c0 = p0.mean(axis=0)
                c1 = p1.mean(axis=0)
                dy, dx = c1[0] - c0[0], c1[1] - c0[1]
                dist = abs(dy) + abs(dx)
                score = dist / (1.0 + len(p0))
                if score > best_score:
                    best_score = score
                    best_color = c
                    best_dy, best_dx = dy, dx

            if best_color is not None and best_score > 0.01:
                if cursor_color is None:
                    cursor_color = best_color
                if cursor_color == best_color:
                    if abs(best_dy) > abs(best_dx):
                        dir_name = "up" if best_dy < 0 else "down"
                    else:
                        dir_name = "left" if best_dx < 0 else "right"
                    if dir_name not in arrow_map:
                        arrow_map[dir_name] = va
                        _step_sizes.append(abs(best_dy) + abs(best_dx))

        interact_va = None
        for va in virtual_actions:
            if va not in arrow_map.values():
                interact_va = va
                break

        if len(arrow_map) < 2 or cursor_color is None:
            return {"solved": False, "actions_used": actions_used}

        _step_size = float(np.median(_step_sizes)) if _step_sizes else 1.0
        wide_threshold = max(1.1, _step_size / 2.0 + 0.1)

        # Adaptive threshold: only use wide threshold when there's an interact action
        # Games without interact (cursor-collision games) need pixel-precise proximity
        use_threshold = wide_threshold if (_step_size > 2.0 and interact_va is not None) else 1.0
        threshold_validated = False

        # Cache navigation parameters for reuse on subsequent levels
        self._cached_nav = {
            "cursor_color": cursor_color,
            "arrow_map": dict(arrow_map),
            "interact_va": interact_va,
            "bg_color": bg_color,
            "step_size": _step_size,
            "threshold": use_threshold,  # Will be updated after validation
        }

        logger.info("Reactive Nav: cursor=%d, arrows=%s, interact=%s, step=%.1f, threshold=%.1f",
                     cursor_color, list(arrow_map.keys()), interact_va, _step_size, use_threshold)

        # 2. Reactive loop: move toward rarest-color targets, interact on arrival
        curr_f = _env_reset(env)
        actions_used += 1
        for step in range(max_steps):
            grid = _get_channel(curr_f)
            pos_cursor = np.argwhere(grid == cursor_color)
            if len(pos_cursor) == 0:
                break
            cursor = pos_cursor.mean(axis=0)

            unique_g, counts_g = np.unique(grid, return_counts=True)
            non_bg = [c for c in unique_g if c != bg_color and c != 0 and c != cursor_color]
            if not non_bg:
                break

            target_color = min(non_bg, key=lambda c: counts_g[np.where(unique_g == c)[0][0]])
            targets = np.argwhere(grid == target_color)
            dists = np.sum(np.abs(targets - cursor), axis=1)
            target = targets[np.argmin(dists)]

            dr, dc = target[0] - cursor[0], target[1] - cursor[1]

            # Decide whether to interact or keep moving
            within_threshold = abs(dr) < use_threshold and abs(dc) < use_threshold
            if within_threshold and interact_va is not None:
                # Save state before interaction to detect if it worked
                pre_interact_hash = _frame_hash(_get_channel(curr_f))
                curr_f, reward, done, info = _env_step(env, self._action_map[interact_va])
                actions_used += 1
                solution_actions.append(interact_va)

                # Validate threshold on first interaction
                if not threshold_validated and use_threshold > 1.0:
                    post_hash = _frame_hash(_get_channel(curr_f))
                    if post_hash == pre_interact_hash and not info.get("solved") and not done:
                        use_threshold = 1.0
                        self._cached_nav["threshold"] = 1.0
                        logger.info("Adaptive threshold: wide failed, falling back to 1.0")
                    else:
                        threshold_validated = True
            else:
                # Not within threshold, OR within threshold but no interact — keep moving
                if abs(dr) >= abs(dc):
                    action_name = "up" if dr < 0 else "down"
                else:
                    action_name = "left" if dc < 0 else "right"

                if action_name in arrow_map:
                    va = arrow_map[action_name]
                    curr_f, reward, done, info = _env_step(env, self._action_map[va])
                    actions_used += 1
                    solution_actions.append(va)
                else:
                    break

            if info.get("solved") or done:
                return {"solved": True, "actions_used": actions_used,
                        "solution_actions": solution_actions}

        return {"solved": False, "actions_used": actions_used}

    def _try_navigation_solve(
        self, env: Any, frame: np.ndarray, virtual_actions: list[int],
        remaining_budget: int,
    ) -> dict:
        """Detect cursor and target, then pathfind with BFS on logical grid."""
        from collections import deque

        FAIL = {"solved": False, "actions_used": 0}
        actions_used = 0

        arrow_map: dict[int, int] = {}
        for va in virtual_actions:
            real = self._action_map.get(va)
            if isinstance(real, int) and 1 <= real <= 4:
                arrow_map[real] = va
        if len(arrow_map) < 2:
            return FAIL

        frame_before = _env_reset(env)
        actions_used += 1
        ch_before = _get_channel(frame_before)
        bg_color = int(np.argmax(np.bincount(ch_before.flatten(), minlength=16)))

        directions: dict[int, tuple[int, int]] = {}
        step_sizes: list[float] = []
        cursor_color = None

        for real_action in sorted(arrow_map.keys()):
            va = arrow_map[real_action]
            _env_reset(env)
            actions_used += 1
            frame_after, _, _, _ = _env_step(env, self._action_map[va])
            actions_used += 1
            ch_after = _get_channel(frame_after)

            diff_mask = ch_before != ch_after
            if not diff_mask.any():
                continue

            candidate_colors = set(ch_before[diff_mask].tolist()) | set(ch_after[diff_mask].tolist())
            candidate_colors.discard(bg_color)

            best_displacement = 0.0
            best_color = None
            best_dy = 0.0
            best_dx = 0.0

            for color in candidate_colors:
                pos_before = np.argwhere(ch_before == color)
                pos_after = np.argwhere(ch_after == color)
                if len(pos_before) == 0 or len(pos_after) == 0:
                    continue
                cb = pos_before.mean(axis=0)
                ca = pos_after.mean(axis=0)
                dy = ca[0] - cb[0]
                dx = ca[1] - cb[1]
                dist = abs(dy) + abs(dx)
                if dist > best_displacement:
                    best_displacement = dist
                    best_color = color
                    best_dy = dy
                    best_dx = dx

            if best_displacement < 0.5 or best_color is None:
                continue

            mag = max(abs(best_dy), abs(best_dx), 1.0)
            directions[va] = (int(round(best_dy / mag)), int(round(best_dx / mag)))
            step_sizes.append(best_displacement)
            if cursor_color is None:
                cursor_color = best_color

        if len(directions) < 2 or cursor_color is None:
            return {"solved": False, "actions_used": actions_used}

        logger.info("Navigation solver: cursor color=%d, %d directions: %s", cursor_color, len(directions), directions)

        verify_va = next(iter(directions))
        _env_reset(env)
        actions_used += 1
        positions_seq: list[np.ndarray] = []
        for _ in range(3):
            vf, _, _, _ = _env_step(env, self._action_map[verify_va])
            actions_used += 1
            ch_v = _get_channel(vf)
            cp = np.argwhere(ch_v == cursor_color)
            if len(cp) > 0:
                positions_seq.append(cp.mean(axis=0))
        if len(positions_seq) >= 3:
            d01 = np.linalg.norm(positions_seq[1] - positions_seq[0])
            d12 = np.linalg.norm(positions_seq[2] - positions_seq[1])
            if d01 < 0.5 and d12 < 0.5:
                return {"solved": False, "actions_used": actions_used}

        avg_step = np.mean(step_sizes) if step_sizes else 4.0
        tile_size = max(2, int(round(avg_step)))
        grid_h = 64 // tile_size
        grid_w = 64 // tile_size

        init_frame = _env_reset(env)
        actions_used += 1
        ch_init = _get_channel(init_frame)

        grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
        for gr in range(grid_h):
            for gc in range(grid_w):
                tile = ch_init[gr * tile_size:(gr + 1) * tile_size,
                               gc * tile_size:(gc + 1) * tile_size]
                if tile.size > 0:
                    grid[gr, gc] = int(np.argmax(np.bincount(tile.flatten(), minlength=16)))

        cursor_positions = np.argwhere(ch_init == cursor_color)
        if len(cursor_positions) == 0:
            return {"solved": False, "actions_used": actions_used}
        cursor_center = cursor_positions.mean(axis=0)
        cursor_grid = (
            min(int(cursor_center[0]) // tile_size, grid_h - 1),
            min(int(cursor_center[1]) // tile_size, grid_w - 1),
        )

        non_bg_colors = set(grid.flatten().tolist()) - {bg_color, cursor_color}
        color_counts: dict[int, int] = {}
        for c in non_bg_colors:
            color_counts[c] = int((grid == c).sum())

        pixel_colors: dict[int, int] = {}
        for c in range(16):
            if c in (bg_color, cursor_color):
                continue
            cnt = int((ch_init == c).sum())
            if 0 < cnt < 100:
                pixel_colors[c] = cnt

        target_color = None
        target_pos = None

        if color_counts:
            for tc, cnt in sorted(color_counts.items(), key=lambda x: x[1]):
                if cnt >= 1:
                    positions = np.argwhere(grid == tc)
                    target_color = tc
                    target_pos = tuple(positions[len(positions) // 2])
                    break

        if target_pos is None and pixel_colors:
            for tc, cnt in sorted(pixel_colors.items(), key=lambda x: x[1]):
                pos = np.argwhere(ch_init == tc)
                if len(pos) > 0:
                    center = pos.mean(axis=0)
                    target_color = tc
                    target_pos = (
                        min(int(center[0]) // tile_size, grid_h - 1),
                        min(int(center[1]) // tile_size, grid_w - 1),
                    )
                    break

        if target_pos is None:
            return {"solved": False, "actions_used": actions_used}

        if cursor_grid == target_pos:
            return {"solved": False, "actions_used": actions_used}

        wall_colors = set()
        for c in set(grid.flatten().tolist()):
            if c not in (bg_color, cursor_color) and c != target_color:
                if int((grid == c).sum()) > max(grid_h * grid_w * 0.2, 3):
                    wall_colors.add(c)

        walkable = np.ones((grid_h, grid_w), dtype=bool)
        for gr in range(grid_h):
            for gc in range(grid_w):
                if grid[gr, gc] in wall_colors:
                    walkable[gr, gc] = False

        dir_to_va: dict[tuple[int, int], int] = {}
        for va, d in directions.items():
            dir_to_va[d] = va

        queue: deque[tuple[tuple[int, int], list[int]]] = deque()
        queue.append((cursor_grid, []))
        visited: set[tuple[int, int]] = {cursor_grid}

        solution_path: list[int] | None = None
        while queue:
            pos, path = queue.popleft()
            if pos == target_pos:
                solution_path = path
                break
            if len(path) > grid_h * grid_w * 2:
                break
            for direction, va in dir_to_va.items():
                nr = pos[0] + direction[0]
                nc = pos[1] + direction[1]
                if 0 <= nr < grid_h and 0 <= nc < grid_w and (nr, nc) not in visited:
                    if walkable[nr, nc]:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [va]))

        if solution_path is None:
            return {"solved": False, "actions_used": actions_used}

        nav_va_set = set(directions.keys())
        extra_actions = [va for va in virtual_actions if va not in nav_va_set]

        _env_reset(env)
        actions_used += 1
        self._prev_frame = None  # Reset for nav solve visualization
        solved = False
        for i, va in enumerate(solution_path):
            new_frame, reward, done, info = _env_step(env, self._action_map[va])
            actions_used += 1
            # Emit frame every 3rd step for viewer
            if i % 3 == 0 or i == len(solution_path) - 1:
                self._emit_frame(new_frame)
            if info.get("solved") or reward > 0:
                self._emit_frame(new_frame)  # Final winning frame
                solved = True
                full_solution = solution_path[:i + 1]
                return {"solved": True, "actions_used": actions_used, "solution_actions": full_solution}
            if done:
                break

        if not solved and extra_actions:
            for extra_va in extra_actions:
                _env_reset(env)
                actions_used += 1
                for va in solution_path:
                    _env_step(env, self._action_map[va])
                    actions_used += 1
                _, _, _, info = _env_step(env, self._action_map[extra_va])
                actions_used += 1
                if info.get("solved") or info.get("levels_completed", 0) > 0:
                    full_solution = solution_path + [extra_va]
                    return {"solved": True, "actions_used": actions_used, "solution_actions": full_solution}

        return {"solved": False, "actions_used": actions_used}

    # ------------------------------------------------------------------
    # Extracted sub-phases for _play_level
    # ------------------------------------------------------------------

    def _discover_and_prune(
        self,
        env: Any,
        frame: np.ndarray,
        virtual_actions: list[int],
    ) -> tuple[list[int], list[int], int, list[dict], int]:
        """Probe all virtual actions and prune to only effective ones."""
        n_virtual = len(virtual_actions)
        actions_used = 0

        probes = self._probe_virtual_actions(env, frame, virtual_actions)
        actions_used += len(probes)

        # Emit probe events
        for p in probes:
            self.events.emit(ProbeEvent(
                action=p["virtual_action"],
                effective=p.get("effective", False),
                diff_pixels=int(p.get("diff_pct", 0) * 64 * 64),
            ))

        initial_frame_for_pruning = _env_reset(env)
        actions_used += 1
        # Re-emit full frame after probing so viewer shows the actual game
        self._prev_frame = None
        self._emit_frame(initial_frame_for_pruning)

        has_click = any(
            isinstance(self._action_map[va], dict) and self._action_map[va].get("action_id") == CLICK_ACTION_ID
            for va in virtual_actions
        )

        if has_click:
            # -- DENSE CLICK SCAN --
            dense_scan_step = 2

            sample_positions = [(0, 0), (60, 0), (0, 60), (60, 60), (32, 0), (0, 32)]
            sample_frames = []
            for sx, sy in sample_positions:
                _env_reset(env)
                actions_used += 1
                new_f, _, _, _ = _env_step(env, {"action_id": CLICK_ACTION_ID, "data": {"x": sx, "y": sy}})
                actions_used += 1
                sample_frames.append(new_f)

            if len(sample_frames) >= 3:
                ch_init = _get_channel(initial_frame_for_pruning)
                all_changed = None
                for f in sample_frames:
                    ch_f = _get_channel(f)
                    changed = ch_init != ch_f
                    all_changed = changed if all_changed is None else (all_changed & changed)
                if all_changed is not None:
                    n_counter = int(all_changed.sum())
                    if 0 < n_counter < all_changed.size * 0.20:
                        logger.info("Adaptive counter detection: %d pixels change on every action", n_counter)
                        ref = initial_frame_for_pruning
                        if ref.ndim == 3 and ref.shape[-1] == 3:
                            counter_mask_2d = np.stack([all_changed] * 3, axis=-1)
                        elif ref.ndim == 3:
                            counter_mask_2d = np.stack([all_changed] * ref.shape[0], axis=0)
                        else:
                            counter_mask_2d = all_changed
                        self._counter_mask = self._counter_mask | counter_mask_2d

            initial_hash_for_pruning = _frame_hash(initial_frame_for_pruning, self._counter_mask)

            any_corner_effective = False
            for f in sample_frames:
                sh = _frame_hash(f, self._counter_mask)
                if sh != initial_hash_for_pruning:
                    any_corner_effective = True
                    break

            has_non_click = any(
                not (isinstance(self._action_map[va], dict) and self._action_map[va].get("action_id") == CLICK_ACTION_ID)
                for va in virtual_actions
            )
            skip_full_scan = not any_corner_effective and has_non_click

            click_results: dict[str, tuple[int, int]] = {}
            if not skip_full_scan:
                for sy in range(0, FRAME_SIZE, dense_scan_step):
                    for sx in range(0, FRAME_SIZE, dense_scan_step):
                        _env_reset(env)
                        actions_used += 1
                        new_f, _, _, info = _env_step(env, {"action_id": CLICK_ACTION_ID, "data": {"x": sx, "y": sy}})
                        actions_used += 1
                        rh = _frame_hash(new_f, self._counter_mask)
                        if rh != initial_hash_for_pruning and rh not in click_results:
                            click_results[rh] = (sx, sy)
                        if info.get("solved") or info.get("levels_completed", 0) > 0:
                            logger.info("DENSE SCAN SOLVED during probe! Click (%d,%d)", sx, sy)
            else:
                logger.info("Dense click scan: skipping (non-click game with no corner effects)")

            scanned = (FRAME_SIZE // dense_scan_step) ** 2 if not skip_full_scan else 0
            logger.info(
                "Dense click scan: %d unique click outcomes from %d positions",
                len(click_results), scanned,
            )

            non_click_effective: list[tuple[int, int | dict]] = []
            for va in virtual_actions:
                real = self._action_map[va]
                if isinstance(real, dict) and real.get("action_id") == CLICK_ACTION_ID:
                    continue
                _env_reset(env)
                actions_used += 1
                new_f, _, _, _ = _env_step(env, real)
                actions_used += 1
                rh = _frame_hash(new_f, self._counter_mask)
                if rh != initial_hash_for_pruning:
                    non_click_effective.append((va, real))

            new_action_map: dict[int, int | dict] = {}
            new_virtual: list[int] = []
            idx = 0
            for _, real in non_click_effective:
                new_action_map[idx] = real
                new_virtual.append(idx)
                idx += 1
            for rh, (cx, cy) in click_results.items():
                new_action_map[idx] = {"action_id": CLICK_ACTION_ID, "data": {"x": cx, "y": cy}}
                new_virtual.append(idx)
                idx += 1

            self._action_map = new_action_map
            virtual_actions = new_virtual
            n_virtual = len(virtual_actions)
            effective_actions = list(virtual_actions)

            self._click_only_game = len(non_click_effective) == 0 and len(click_results) > 0
            self._dense_scan_step = dense_scan_step

            logger.info(
                "After dense scan: %d effective actions (%d click, %d non-click)",
                n_virtual, len(click_results), len(non_click_effective),
            )
        else:
            # -- Non-click games: original per-action probing --
            self._click_only_game = False
            self._dense_scan_step = 4

            independent_frames: list[np.ndarray] = []
            for va in virtual_actions:
                _env_reset(env)
                actions_used += 1
                real_action = self._action_map[va]
                new_frame, _, _, _ = _env_step(env, real_action)
                actions_used += 1
                independent_frames.append(new_frame)

            if len(independent_frames) >= 3:
                ch_init = _get_channel(initial_frame_for_pruning)
                all_changed = None
                for f in independent_frames:
                    ch_f = _get_channel(f)
                    changed = ch_init != ch_f
                    all_changed = changed if all_changed is None else (all_changed & changed)
                if all_changed is not None:
                    n_counter = int(all_changed.sum())
                    if 0 < n_counter < all_changed.size * 0.20:
                        logger.info("Adaptive counter detection: %d pixels change on every action", n_counter)
                        ref = initial_frame_for_pruning
                        if ref.ndim == 3 and ref.shape[-1] == 3:
                            counter_mask_2d = np.stack([all_changed] * 3, axis=-1)
                        elif ref.ndim == 3:
                            counter_mask_2d = np.stack([all_changed] * ref.shape[0], axis=0)
                        else:
                            counter_mask_2d = all_changed
                        self._counter_mask = self._counter_mask | counter_mask_2d

            initial_hash_for_pruning = _frame_hash(initial_frame_for_pruning, self._counter_mask)

            effective_actions = []
            for va_idx, va in enumerate(virtual_actions):
                nh = _frame_hash(independent_frames[va_idx], self._counter_mask)
                if nh != initial_hash_for_pruning:
                    effective_actions.append(va)

            if effective_actions:
                logger.info(
                    "Action pruning: %d/%d actions are effective (%.0f%% reduction)",
                    len(effective_actions), n_virtual,
                    (1 - len(effective_actions) / n_virtual) * 100,
                )

                if len(effective_actions) > 6:
                    seen_hashes: dict[str, int] = {}
                    deduped: list[int] = []
                    for va in effective_actions:
                        _env_reset(env)
                        actions_used += 1
                        new_frame, _, _, _ = _env_step(env, self._action_map[va])
                        actions_used += 1
                        result_hash = _frame_hash(new_frame, self._counter_mask)
                        if result_hash not in seen_hashes:
                            seen_hashes[result_hash] = va
                            deduped.append(va)
                    if len(deduped) < len(effective_actions):
                        logger.info(
                            "Action dedup: %d -> %d unique outcomes",
                            len(effective_actions), len(deduped),
                        )
                        effective_actions = deduped

                virtual_actions = effective_actions
                n_virtual = len(virtual_actions)
            else:
                logger.info("Action pruning: no effective actions found, keeping all %d", n_virtual)

        initial_hash = _frame_hash(frame, self._counter_mask)

        # -- DEPTH-2 FALLBACK --
        original_virtual_actions = list(virtual_actions)
        if not effective_actions and len(original_virtual_actions) <= 32:
            logger.info("Depth-2 probing: no single-action effects found, trying pairs...")
            initial_frame_d2 = _env_reset(env)
            initial_hash_d2 = _frame_hash(initial_frame_d2, self._counter_mask)
            actions_used += 1

            depth2_effective: set[int] = set()
            for a in original_virtual_actions:
                if len(depth2_effective) >= 10:
                    break
                for b in original_virtual_actions:
                    _env_reset(env)
                    actions_used += 1
                    _env_step(env, self._action_map[a])
                    actions_used += 1
                    new_frame, _, _, info = _env_step(env, self._action_map[b])
                    actions_used += 1
                    new_hash = _frame_hash(new_frame, self._counter_mask)
                    if new_hash != initial_hash_d2:
                        depth2_effective.add(a)
                        depth2_effective.add(b)
                    if info.get("solved") or info.get("levels_completed", 0) > 0:
                        logger.info("DEPTH-2 SOLVED! Pair (%d, %d) solved the level!", a, b)

            if depth2_effective:
                effective_actions = sorted(depth2_effective)
                virtual_actions = effective_actions
                n_virtual = len(virtual_actions)
                logger.info("Depth-2 probing found %d effective actions: %s", len(effective_actions), effective_actions[:10])

        self.state_graph = StateGraph(n_actions=n_virtual)
        self.state_graph.set_initial_state(initial_hash)
        prev_f = frame
        for p in probes:
            if p["virtual_action"] in virtual_actions:
                ph = _frame_hash(prev_f, self._counter_mask)
                nh = _frame_hash(p["frame"], self._counter_mask)
                self.state_graph.add_transition(ph, p["virtual_action"], nh)
            prev_f = p["frame"]

        return effective_actions, virtual_actions, n_virtual, probes, actions_used

    def _solve_reactive_click(
        self,
        env: Any,
        remaining_budget: int,
    ) -> dict:
        """Solve state-dependent click games by re-scanning for targets each step."""
        actions_used = 0
        step = getattr(self, "_dense_scan_step", 4)
        solution_path: list[tuple[int, int]] = []
        max_solution_depth = 200

        frame = _env_reset(env)
        actions_used += 1

        for depth in range(max_solution_depth):
            scan_cost = (FRAME_SIZE // step) ** 2 * (2 + len(solution_path))
            if actions_used + scan_cost > remaining_budget:
                break

            current_hash = _frame_hash(frame, self._counter_mask)

            best_click: tuple[int, int] | None = None
            best_diff = 0

            for sy in range(0, FRAME_SIZE, step):
                for sx in range(0, FRAME_SIZE, step):
                    _env_reset(env)
                    actions_used += 1
                    for px, py in solution_path:
                        _env_step(
                            env,
                            {"action_id": CLICK_ACTION_ID, "data": {"x": px, "y": py}},
                        )
                        actions_used += 1

                    new_f, _, _, info = _env_step(
                        env,
                        {"action_id": CLICK_ACTION_ID, "data": {"x": sx, "y": sy}},
                    )
                    actions_used += 1

                    if info.get("solved") or info.get("levels_completed", 0) > 0:
                        self._emit_frame(new_f)  # Emit winning frame
                        solution_path.append((sx, sy))
                        raw_actions = [
                            {"action_id": CLICK_ACTION_ID, "data": {"x": px, "y": py}}
                            for px, py in solution_path
                        ]
                        return {
                            "solved": True,
                            "actions_used": actions_used,
                            "solution_length": len(solution_path),
                            "solution_raw_actions": raw_actions,
                        }

                    new_hash = _frame_hash(new_f, self._counter_mask)
                    if new_hash != current_hash:
                        ch_cur = _get_channel(frame)
                        ch_new = _get_channel(new_f)
                        diff_pixels = int((ch_cur != ch_new).sum())
                        if diff_pixels > best_diff:
                            best_diff = diff_pixels
                            best_click = (sx, sy)

            if best_click is None:
                break

            solution_path.append(best_click)
            _env_reset(env)
            actions_used += 1
            for px, py in solution_path:
                frame, _, _, info = _env_step(
                    env,
                    {"action_id": CLICK_ACTION_ID, "data": {"x": px, "y": py}},
                )
                actions_used += 1

            if info.get("solved") or info.get("levels_completed", 0) > 0:
                return {
                    "solved": True,
                    "actions_used": actions_used,
                    "solution_length": len(solution_path),
                }

        return {
            "solved": False,
            "actions_used": actions_used,
            "solution_length": len(solution_path),
        }

    def _execute_combo_search(
        self,
        env: Any,
        virtual_actions: list[int],
        remaining_budget: int,
        max_depth: int = 10,
    ) -> dict:
        """Search action combos up to max_depth to solve the level.

        Optimizations:
        1. Prefix pruning: skip combos whose prefix produces no state change
        2. Start from winning depth: if previous level solved at depth D, start at D-1
        3. Winning combo variations: try substitutions/truncations before full search
        """
        max_depth = min(max_depth, remaining_budget // (len(virtual_actions) + 1))
        use_exhaustive = len(virtual_actions) <= 6
        from itertools import product
        replay_env = _VirtualActionEnv(env, self._action_map)
        combo_steps = 0
        starting_levels_completed = getattr(env, "levels_completed", 0)
        levels_completed = 0
        explore_result = {"solved": False, "total_steps": 0, "resets": 0, "unique_states": 0, "solution_length": 0}

        # Build prefix pruning set: test which single actions are no-ops
        dead_actions: set[int] = set()
        if use_exhaustive and len(virtual_actions) >= 2:
            replay_env.reset()
            combo_steps += 1
            init_hash = _frame_hash(_get_channel(_env_reset(env)))
            combo_steps += 1
            for va in virtual_actions:
                replay_env.reset()
                combo_steps += 1
                new_frame, _, _, _ = replay_env.step(va)
                combo_steps += 1
                if _frame_hash(_get_channel(new_frame)) == init_hash:
                    dead_actions.add(va)
            if dead_actions:
                logger.info("Combo prefix pruning: %d/%d actions are no-ops", len(dead_actions), len(virtual_actions))

        def _try_combo(combo: tuple[int, ...]) -> bool:
            nonlocal combo_steps, levels_completed, explore_result
            if combo_steps + len(combo) + 1 > remaining_budget:
                return False
            replay_env.reset()
            combo_steps += 1
            prev_levels = starting_levels_completed
            for action in combo:
                new_frame, reward, done, info = replay_env.step(action)
                combo_steps += 1
                cur_levels = info.get("levels_completed", 0)
                if info.get("solved") or cur_levels > prev_levels:
                    levels_completed += 1
                    self._emit_frame(new_frame)
                    logger.info("COMBO SOLVED! depth=%d, actions=%s, levels=%d", len(combo), combo, cur_levels)
                    explore_result = {
                        "solved": True,
                        "solution_actions": list(combo),
                        "solution_length": len(combo),
                        "total_steps": combo_steps,
                        "resets": len(combo),
                        "unique_states": 0,
                    }
                    self._winning_combos.append(combo)
                    raw = [self._action_map.get(a, a) for a in combo]
                    self._winning_raw_actions.append(raw)
                    return True
                if done:
                    return False
            return False

        def _combo_has_dead_prefix(combo: tuple[int, ...]) -> bool:
            """Skip combos that start with a known no-op action."""
            return len(dead_actions) > 0 and len(combo) > 0 and combo[0] in dead_actions

        # Phase 1: Try exact winning combos
        for prev_combo in self._winning_combos:
            if all(a in virtual_actions for a in prev_combo):
                if _try_combo(prev_combo):
                    break

        # Phase 2: Try winning combo variations (substitutions + truncations)
        if levels_completed == 0 and self._winning_combos:
            for prev_combo in self._winning_combos:
                if levels_completed > 0:
                    break
                if not all(a in virtual_actions for a in prev_combo):
                    continue
                # Try truncations (the solution might be shorter this level)
                for trunc_len in range(max(1, len(prev_combo) - 2), len(prev_combo)):
                    if levels_completed > 0:
                        break
                    truncated = prev_combo[:trunc_len]
                    if _try_combo(truncated):
                        break
                # Try single-action substitutions
                if levels_completed == 0:
                    for pos in range(len(prev_combo)):
                        if levels_completed > 0:
                            break
                        for alt_action in virtual_actions:
                            if alt_action == prev_combo[pos]:
                                continue
                            variant = prev_combo[:pos] + (alt_action,) + prev_combo[pos+1:]
                            if _try_combo(variant):
                                break

        # Phase 3: Determine starting depth
        # If we have a winning combo, start search near that depth
        start_depth = 1
        if self._winning_combos:
            prev_depth = len(self._winning_combos[-1])
            start_depth = max(1, prev_depth - 2)

        # Phase 4: Full iterative deepening search with prefix pruning
        if levels_completed == 0:
            for depth in range(start_depth, max_depth + 1):
                if levels_completed > 0:
                    break
                if combo_steps + depth + 1 > remaining_budget:
                    break
                if use_exhaustive:
                    for combo in product(virtual_actions, repeat=depth):
                        if combo_steps + depth + 1 > remaining_budget:
                            break
                        if _combo_has_dead_prefix(combo):
                            continue
                        if _try_combo(combo):
                            break
                else:
                    n_possible = len(virtual_actions) ** depth
                    n_samples = min(2000, n_possible)
                    seen: set[tuple[int, ...]] = set()
                    for _ in range(n_samples):
                        if combo_steps + depth + 1 > remaining_budget:
                            break
                        combo = tuple(random.choices(virtual_actions, k=depth))
                        if combo in seen:
                            continue
                        seen.add(combo)
                        if _combo_has_dead_prefix(combo):
                            continue
                        if _try_combo(combo):
                            break

        return {
            "solved": levels_completed > 0,
            "actions_used": combo_steps,
            "explore_result": explore_result,
        }

    def _execute_deepcopy_bfs(
        self,
        env: Any,
        available_actions: list[int],
        budget: int,
    ) -> dict:
        """Deepcopy-based BFS fallback for games where replay BFS fails."""
        import copy
        import time as _time
        from collections import deque

        replay_env = _VirtualActionEnv(env, self._action_map)

        init_frame = replay_env.reset()
        init_hash = _frame_hash(init_frame, self._counter_mask)
        starting_levels = getattr(env, '_prev_levels', 0) if hasattr(env, '_prev_levels') else 0

        queue: deque[tuple[str, list[int], Any]] = deque()
        queue.append((init_hash, [], copy.deepcopy(replay_env)))
        visited: set[str] = {init_hash}
        steps = 0
        t_start = _time.monotonic()
        time_limit = 120.0

        while queue and steps < budget and (_time.monotonic() - t_start) < time_limit:
            state_hash, path, current_env = queue.popleft()

            for a in available_actions:
                if steps >= budget or (_time.monotonic() - t_start) > time_limit:
                    break
                try:
                    branch_env = copy.deepcopy(current_env)
                    f, r, d, info = branch_env.step(a)
                except Exception:
                    steps += 1
                    continue
                steps += 1

                cur_levels = info.get("levels_completed", 0)
                if info.get("solved") or cur_levels > starting_levels:
                    self._emit_frame(f)  # Emit winning frame
                    solution = path + [a]
                    replay_env.reset()
                    for act in solution:
                        replay_env.step(act)
                    return {
                        "solved": True,
                        "actions_used": steps,
                        "solution_actions": solution,
                    }

                nhash = _frame_hash(f, self._counter_mask)
                if nhash not in visited:
                    visited.add(nhash)
                    queue.append((nhash, path + [a], branch_env))

        return {"solved": False, "actions_used": steps, "solution_actions": []}

    def _get_navigation_heuristic(self, frame: np.ndarray, cursor_color: int, target_centers: list[tuple[float, float]]) -> float:
        """Manhattan distance from cursor to nearest target center."""
        from solver.context_compression import detect_objects, _extract_single_channel
        grid = _extract_single_channel(frame)

        coords = np.argwhere(grid == cursor_color)
        if coords.size == 0:
            return 100.0

        cursor_center = coords.mean(axis=0)

        min_dist = 100.0
        for tc in target_centers:
            dist = abs(cursor_center[0] - tc[0]) + abs(cursor_center[1] - tc[1])
            if dist < min_dist:
                min_dist = dist

        return min_dist

    def _execute_bfs_fallback(
        self,
        env: Any,
        virtual_actions: list[int],
        n_virtual: int,
        remaining_budget: int,
        probes: list[dict] | None = None,
        initial_frame: np.ndarray | None = None,
        all_virtual_actions: list[int] | None = None,
    ) -> dict:
        """Run BFS exploration via ReplayExplorer as fallback."""
        explore_result = {"solved": False, "total_steps": 0, "resets": 0, "unique_states": 0, "solution_length": 0}
        if remaining_budget <= 50:
            return {"solved": False, "actions_used": 0, "explore_result": explore_result}

        replay_env = _VirtualActionEnv(env, self._action_map)
        n_actions_max = max(virtual_actions) + 1 if virtual_actions else n_virtual
        full_action_set = list(all_virtual_actions) if all_virtual_actions is not None else list(virtual_actions)

        known_states: dict[str, list[int]] = {}
        for node_hash in self.state_graph.nodes:
            path = self.state_graph.path_to(node_hash)
            if path is not None:
                known_states[node_hash] = path

        total_actions_used = 0

        from itertools import combinations
        action_subsets: list[list[int]] = []
        n_eff = len(virtual_actions)

        # -- NAVIGATION DETECTION & PRIORITIZATION --
        has_nav_subset = False
        heuristic_fn = None
        nav_actions: dict[str, int] = {}
        if initial_frame is not None and n_eff <= 8:
            ref_frame = _env_reset(env)
            total_actions_used += 1
            ref_ch = _get_channel(ref_frame)
            unique_vals, counts = np.unique(ref_ch, return_counts=True)
            bg = int(unique_vals[np.argmax(counts)])

            for va in virtual_actions:
                _env_reset(env)
                total_actions_used += 1
                new_f, _, _, _ = _env_step(env, self._action_map[va])
                total_actions_used += 1
                new_ch = _get_channel(new_f)
                diff = (new_ch != ref_ch)
                if not np.any(diff):
                    continue
                old_fg = np.argwhere(diff & (ref_ch != bg))
                new_fg = np.argwhere(diff & (new_ch != bg))
                if old_fg.size > 0 and new_fg.size > 0:
                    was_c = old_fg.mean(axis=0)
                    is_c = new_fg.mean(axis=0)
                    dr = is_c[0] - was_c[0]
                    dc = is_c[1] - was_c[1]
                    dist = abs(dr) + abs(dc)
                    if 0.5 < dist < 30.0:
                        if abs(dr) > abs(dc):
                            d = "up" if dr < 0 else "down"
                        else:
                            d = "left" if dc < 0 else "right"
                        if d not in nav_actions:
                            nav_actions[d] = va

            if nav_actions:
                nav_subset = sorted(list(set(nav_actions.values())))
                action_subsets.append(nav_subset)

                click_actions = [va for va in virtual_actions
                                if isinstance(self._action_map.get(va), dict)]
                if click_actions:
                    nav_click_subset = sorted(list(set(nav_subset + click_actions)))
                    if nav_click_subset != nav_subset:
                        action_subsets.append(nav_click_subset)

                logger.info("BFS: prioritized navigation subset %s (detected %s)", nav_subset, nav_actions)
                has_nav_subset = True

                from solver.context_compression import detect_objects, _extract_single_channel
                grid = _extract_single_channel(initial_frame)
                unique, counts = np.unique(grid, return_counts=True)
                background = int(unique[np.argmax(counts)])

                color_movements = {}
                for p_idx, p in enumerate(probes):
                    if p.get("effective"):
                        initial_objs = detect_objects(initial_frame)
                        new_objs = detect_objects(p["frame"])
                        for iobj in initial_objs:
                            for nobj in new_objs:
                                if iobj["color"] == nobj["color"] and abs(iobj["size"] - nobj["size"]) <= max(2, iobj["size"] // 10):
                                    dr = nobj["center"][0] - iobj["center"][0]
                                    dc = nobj["center"][1] - iobj["center"][1]
                                    if abs(dr) + abs(dc) > 0.1:
                                        color_movements[(p_idx, iobj["color"])] = (dr, dc)

                camera_shifts = {}
                for p_idx in range(len(probes)):
                    moves = [m for (idx, c), m in color_movements.items() if idx == p_idx]
                    if moves:
                        mdr = float(np.median([m[0] for m in moves]))
                        mdc = float(np.median([m[1] for m in moves]))
                        camera_shifts[p_idx] = (mdr, mdc)

                cursor_scores = {}
                all_colors = {c for (idx, c) in color_movements.keys()}
                for c in all_colors:
                    deviation = 0.0
                    for p_idx, shift in camera_shifts.items():
                        move = color_movements.get((p_idx, c), (0.0, 0.0))
                        deviation += abs(move[0] - shift[0]) + abs(move[1] - shift[1])
                    cursor_scores[c] = deviation

                cursor_color = None
                if cursor_scores:
                    cursor_color = max(cursor_scores.items(), key=lambda x: x[1])[0]

                if cursor_color is not None:
                    objs = detect_objects(initial_frame)
                    moving_colors = {c for c, dev in cursor_scores.items() if dev > 1.0}
                    moving_colors.add(cursor_color)

                    candidates = [o for o in objs
                                 if o["color"] != background
                                 and o["color"] != 0
                                 and o["color"] not in moving_colors
                                 and o["size"] < initial_frame.size // 20]

                    if candidates:
                        color_counts = {}
                        for o in candidates:
                            color_counts[o["color"]] = color_counts.get(o["color"], 0) + 1

                        rarest_color = min(color_counts.items(), key=lambda x: x[1])[0]
                        target_centers = [o["center"] for o in candidates if o["color"] == rarest_color]

                        heuristic_fn = lambda frame: self._get_navigation_heuristic(frame, cursor_color, target_centers)

        is_click_game = getattr(self, "_click_only_game", False)
        if 3 <= n_eff <= 6 and not has_nav_subset and not is_click_game:
            pair_scores: list[tuple[int, list[int]]] = []
            probe_budget = min(2000, remaining_budget // (n_eff * 5))
            for pair in combinations(virtual_actions, 2):
                probe_explorer = ReplayExplorer(
                    env=replay_env,
                    n_actions=n_actions_max,
                    max_total_steps=probe_budget,
                    counter_mask=self._counter_mask,
                    action_list=list(pair),
                    event_callback=self._event_callback,
                )
                probe_result = probe_explorer.explore()
                total_actions_used += probe_result["total_steps"]
                n_states = probe_result["unique_states"]
                if probe_result["solved"]:
                    explore_result = probe_result
                    sol_actions = probe_result.get("solution_actions", [])
                    if sol_actions:
                        replay_env.reset()
                        total_actions_used += 1
                        for a in sol_actions:
                            replay_env.step(a)
                            total_actions_used += 1
                    break
                pair_scores.append((n_states, list(pair)))

            if explore_result.get("solved"):
                pass
            else:
                pair_scores.sort(key=lambda x: x[0])
                for _, pair in pair_scores:
                    action_subsets.append(pair)

                if n_eff >= 4:
                    for triple in combinations(virtual_actions, 3):
                        action_subsets.append(list(triple))

        if not explore_result.get("solved"):
            action_subsets.append(virtual_actions)

        for subset_idx, action_subset in enumerate(action_subsets):
            if explore_result.get("solved"):
                break
            budget_left = remaining_budget - total_actions_used
            if budget_left <= 100:
                break

            subset_budget = budget_left
            if len(action_subset) < n_eff:
                max_states = 15000
                if len(action_subset) <= 2 and n_eff >= 4:
                    continue
                else:
                    subset_budget = min(
                        subset_budget,
                        max(4000, remaining_budget // max(len(action_subsets), 1)),
                    )
            else:
                max_states = 0

            if subset_budget <= 100:
                continue

            modulus = 0
            if (self._counter_mask is not None
                    and self._counter_mask.any()):
                modulus = 6

            explorer = ReplayExplorer(
                env=replay_env,
                n_actions=n_actions_max,
                max_total_steps=subset_budget,
                counter_mask=self._counter_mask,
                action_list=action_subset,
                max_unique_states=max_states,
                heuristic_fn=heuristic_fn,
                step_modulus=modulus,
                event_callback=self._event_callback,
            )

            seed_states = None
            explore_result = explorer.explore(
                initial_known_states=seed_states,
            )
            total_actions_used += explore_result["total_steps"]

            if explore_result["solved"]:
                sol_actions = explore_result.get("solution_actions", [])
                if sol_actions:
                    replay_env.reset()
                    total_actions_used += 1
                    for a in sol_actions:
                        replay_env.step(a)
                        total_actions_used += 1
                break

        should_try_hidden_state = (
            not explore_result.get("solved")
            and full_action_set
            and len(full_action_set) <= 6
            and (len(full_action_set) > len(virtual_actions) or len(virtual_actions) <= 1)
        )
        if should_try_hidden_state:
            for step_modulus in (3, 2, 4):
                budget_left = remaining_budget - total_actions_used
                if budget_left <= 100:
                    break
                if step_modulus == 3:
                    hidden_budget = budget_left
                else:
                    hidden_budget = min(budget_left, max(12000, remaining_budget // 4))
                explorer = ReplayExplorer(
                    env=replay_env,
                    n_actions=n_actions_max,
                    max_total_steps=hidden_budget,
                    counter_mask=None,
                    action_list=full_action_set,
                    step_modulus=step_modulus,
                    event_callback=self._event_callback,
                )
                hidden_result = explorer.explore()
                total_actions_used += hidden_result["total_steps"]
                if hidden_result["solved"]:
                    explore_result = hidden_result
                    sol_actions = hidden_result.get("solution_actions", [])
                    if sol_actions:
                        replay_env.reset()
                        total_actions_used += 1
                        for a in sol_actions:
                            replay_env.step(a)
                            total_actions_used += 1
                    break

        discovered = explore_result.get("discovered_states", {})
        for state_hash, prefix in discovered.items():
            if len(prefix) >= 1:
                parent_prefix = prefix[:-1]
                parent_key = None
                for h, p in discovered.items():
                    if p == parent_prefix:
                        parent_key = h
                        break
                if parent_key is not None:
                    self.state_graph.add_transition(
                        parent_key, prefix[-1], state_hash,
                    )

        return {
            "solved": explore_result["solved"],
            "actions_used": total_actions_used,
            "explore_result": explore_result,
        }

    # ------------------------------------------------------------------
    # Probing
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_counter_from_frames(frames: list[np.ndarray]) -> np.ndarray | None:
        """Detect counter/UI pixels from a sequence of frames."""
        if len(frames) < 3:
            return None

        def get_ch(f: np.ndarray) -> np.ndarray:
            if f.ndim == 3 and f.shape[-1] == 3:
                return f[:, :, 0]
            elif f.ndim == 3:
                return f[0]
            return f

        channels = [get_ch(f) for f in frames]
        n_pairs = len(channels) - 1

        change_count = np.zeros(channels[0].shape, dtype=np.int32)
        for i in range(n_pairs):
            change_count += (channels[i] != channels[i + 1]).astype(np.int32)

        threshold = max(1, int(n_pairs * 0.6))
        counter_pixels = change_count >= threshold

        n_masked = int(counter_pixels.sum())
        total_pixels = counter_pixels.size
        if n_masked == 0 or n_masked > total_pixels * 0.05:
            return None

        ref = frames[0]
        if ref.ndim == 3 and ref.shape[-1] == 3:
            mask = np.stack([counter_pixels] * 3, axis=-1)
        elif ref.ndim == 3:
            mask = np.stack([counter_pixels] * ref.shape[0], axis=0)
        else:
            mask = counter_pixels

        return mask

    def _probe_virtual_actions(
        self,
        env: Any,
        frame: np.ndarray,
        virtual_actions: list[int],
    ) -> list[dict]:
        """Probe each virtual action independently from the initial state."""
        probes: list[dict] = []
        initial_frame = frame.copy()

        for va in virtual_actions:
            _env_reset(env)
            prev_frame = initial_frame.copy()

            real_action = self._action_map[va]
            new_frame, reward, done, info = _env_step(env, real_action)

            diff_desc = compress_diff(prev_frame, new_frame, self.tracker)
            diff_pct = float(np.mean(prev_frame != new_frame))

            prev_hash = _frame_hash(prev_frame)
            new_hash = _frame_hash(new_frame)
            self.state_graph.add_transition(prev_hash, va, new_hash)

            probes.append({
                "virtual_action": va,
                "real_action": real_action,
                "diff": diff_desc,
                "diff_pct": diff_pct,
                "reward": reward,
                "effective": diff_pct > 0.001,
                "frame": new_frame,
                "change": {},
            })

            frame_changed = diff_pct > 0.001
            # Don't emit frame diffs during probing — probes test individual
            # actions from reset and produce partial/misleading frames

            self.action_ranker.record(_ranker_frame(prev_frame), va, frame_changed)

            self.tracker.update(detect_objects(new_frame))

        return probes

    def _describe_action(self, virtual_action: int) -> str:
        """Human-readable description of a virtual action."""
        real = self._action_map.get(virtual_action)
        if real is None:
            return f"action {virtual_action}"
        if isinstance(real, dict):
            x = real["data"]["x"]
            y = real["data"]["y"]
            return f"click ({x}, {y})"
        action_names = {1: "up", 2: "down", 3: "left", 4: "right", 5: "action5", 6: "click", 7: "undo"}
        return action_names.get(real, f"action {real}")

    def _resolve_action(self, action_desc: str, virtual_actions: list[int]) -> int | None:
        """Resolve an action description back to a virtual action index."""
        desc = action_desc.lower().strip()

        import re
        click_match = re.search(r"click\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?", desc)
        if click_match:
            target_x, target_y = int(click_match.group(1)), int(click_match.group(2))
            best_va = None
            best_dist = float("inf")
            for va in virtual_actions:
                real = self._action_map.get(va)
                if isinstance(real, dict):
                    x, y = real["data"]["x"], real["data"]["y"]
                    dist = abs(x - target_x) + abs(y - target_y)
                    if dist < best_dist:
                        best_dist = dist
                        best_va = va
            return best_va

        direction_map = {"up": 1, "down": 2, "left": 3, "right": 4, "undo": 7}
        for word, action_id in direction_map.items():
            if word in desc:
                for va in virtual_actions:
                    if self._action_map.get(va) == action_id:
                        return va

        num_match = re.search(r"action\s*(\d+)", desc)
        if num_match:
            target = int(num_match.group(1))
            for va in virtual_actions:
                if self._action_map.get(va) == target:
                    return va

        return None

    def _heuristic_action(
        self,
        frame: np.ndarray,
        available_actions: list[int],
        memory: dict[int, float],
    ) -> int:
        """Pick next action using state graph when possible, else weighted random."""
        if not available_actions:
            return 0

        frame_hash = _frame_hash(frame)
        suggested = self.state_graph.suggest_action(frame_hash)
        if suggested is not None and suggested in available_actions:
            return suggested

        if self.action_ranker is not None:
            try:
                ranked = self.action_ranker.rank(_ranker_frame(frame), available_actions)
                if ranked:
                    if random.random() < 0.7:
                        return ranked[0]
            except Exception:
                pass

        if memory:
            weights = [memory.get(a, 0.1) + 0.05 for a in available_actions]
        else:
            weights = [1.0] * len(available_actions)

        total = sum(weights)
        weights = [w / total for w in weights]

        r = random.random()
        cumulative = 0.0
        for action, w in zip(available_actions, weights):
            cumulative += w
            if r <= cumulative:
                return action
        return available_actions[-1]

    def _effective_actions(self) -> dict[int, float]:
        """Build action effectiveness map from action log."""
        from collections import defaultdict

        counts: dict[int, int] = defaultdict(int)
        effects: dict[int, float] = defaultdict(float)
        for entry in self.action_log:
            a = entry["action"]
            counts[a] += 1
            effects[a] += abs(entry.get("reward", 0.0))

        result: dict[int, float] = {}
        for a in counts:
            result[a] = effects[a] / counts[a] if counts[a] > 0 else 0.0
        return result
