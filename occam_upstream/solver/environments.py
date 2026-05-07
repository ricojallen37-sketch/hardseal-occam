"""ARC-AGI-3 environment adapter.

Adapts the arcengine LocalEnvironmentWrapper to the gym-like interface
expected by GameOrchestrator.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("occam.solver.environments")


class ArcEnvAdapter:
    """Adapt arc-agi LocalEnvironmentWrapper to GameOrchestrator's gym interface.

    The orchestrator's _is_arc_env() check looks for `available_actions` and
    `levels_completed` as direct env attributes. We deliberately do NOT expose
    those at class level so the orchestrator uses the standard gym path
    (frame, reward, done, info) rather than trying to read FrameDataRaw objects.
    """

    def __init__(self, arc_env: Any) -> None:
        self.arc_env = arc_env
        self.arc_env.include_frame_data = True
        self._obs: Any = None

    @staticmethod
    def _normalize_frame(frame: np.ndarray) -> np.ndarray:
        """Normalize frame to (64, 64, 3) HWC uint8 regardless of input channels.

        ARC-AGI-3 games can return (N, 64, 64) with N varying between steps.
        We always take the first channel and expand to 3-channel for consistency.
        """
        if frame.ndim == 3 and frame.shape[0] < frame.shape[-1]:
            # CHW format -- take first channel, expand to 3-ch HWC
            single = frame[0]  # (64, 64)
        elif frame.ndim == 3 and frame.shape[-1] <= 5:
            # HWC format -- take first channel
            single = frame[:, :, 0]
        elif frame.ndim == 2:
            single = frame
        else:
            # Unknown format -- flatten to 2D
            single = frame.reshape(64, 64) if frame.size == 4096 else frame[0]

        # Expand to 3-channel HWC for consistency
        return np.stack([single, single, single], axis=-1).astype(np.uint8)

    def reset(self) -> np.ndarray:
        self._obs = self.arc_env.reset()
        self._prev_levels = self._obs.levels_completed
        return self._normalize_frame(np.array(self._obs.frame, dtype=np.uint8))

    def step(self, action: int | dict, data: dict | None = None) -> tuple:
        """Step the environment. Accepts int or dict action.

        Handles:
        - Click actions (action 6) that require x,y data
        - Variable-channel frames (some games change channels mid-game)
        - Both int and dict action formats from the orchestrator
        """
        # Determine action_id and click data
        action_id = action
        click_data = data
        if isinstance(action, dict):
            action_id = action.get("action_id", action.get("id", 6))
            click_data = action.get("data", data)

        # Click actions (action 6) MUST have x,y data
        if action_id == 6 and click_data is None:
            click_data = {"x": 32, "y": 32}  # Default to center if no coords

        try:
            if click_data:
                self._obs = self.arc_env.step(action_id, data=click_data)
            else:
                self._obs = self.arc_env.step(action_id)
        except Exception as e:
            # Some games crash on certain actions -- return unchanged state
            logger.debug("Step error (action %s): %s", action_id, e)
            if self._obs is None:
                raise
            # Return current state as if nothing happened
            frame = self._normalize_frame(np.array(self._obs.frame, dtype=np.uint8))
            return frame, 0.0, False, {"solved": False, "levels_completed": 0, "available_actions": self.get_available_actions()}

        frame = self._normalize_frame(np.array(self._obs.frame, dtype=np.uint8))

        from arcengine.enums import GameState
        done = self._obs.state in (GameState.WIN, GameState.GAME_OVER)
        # Detect level completion via levels_completed increase OR WIN state
        cur_levels = self._obs.levels_completed
        prev_levels = getattr(self, '_prev_levels', 0)
        solved = self._obs.state == GameState.WIN or cur_levels > prev_levels
        self._prev_levels = cur_levels
        reward = 1.0 if solved else 0.0
        # If a level was just completed, mark as done so orchestrator can advance
        if cur_levels > prev_levels:
            done = True
        info = {
            "solved": solved,
            "levels_completed": self._obs.levels_completed,
            "state": self._obs.state,
            "available_actions": list(self._obs.available_actions),
        }
        return frame, reward, done, info

    @property
    def n_actions(self) -> int:
        """Number of available actions (used by orchestrator for action range)."""
        if self._obs and self._obs.available_actions:
            return max(self._obs.available_actions) + 1
        return 8

    def get_available_actions(self) -> list[int]:
        """Return current available action list."""
        if self._obs and self._obs.available_actions:
            return list(self._obs.available_actions)
        return [1, 2, 3, 4]
