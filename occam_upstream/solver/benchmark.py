"""BenchmarkRunner — runs all 25 ARC-AGI-3 public games sequentially."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

import arc_agi

from solver.environments import ArcEnvAdapter
from solver.events import (
    EventEmitter, BenchmarkStartEvent, BenchmarkCompleteEvent,
    GameStartEvent, GameCompleteEvent,
)
from solver.orchestrator import GameOrchestrator

logger = logging.getLogger("occam.benchmark")

QUICK_DEMO_GAMES = {"SK48", "LF52", "LS20", "CD82", "SP80"}


class BenchmarkRunner:
    def __init__(
        self,
        event_callback: Callable[[dict], None] | None = None,
        mode: str = "full",
        game_filter: str | None = None,
        max_actions: int = 2000,
        results_dir: Path | None = None,
        skip_navigation: bool = False,
        skip_combo: bool = False,
        skip_deepcopy: bool = False,
    ):
        self.events = EventEmitter(callback=event_callback)
        self.mode = mode
        self.game_filter = game_filter
        self.max_actions = max_actions
        self.results_dir = results_dir or Path("results")
        self.skip_navigation = skip_navigation
        self.skip_combo = skip_combo
        self.skip_deepcopy = skip_deepcopy

    async def run(self) -> dict:
        import random
        import numpy as np
        random.seed(42)
        np.random.seed(42)

        arcade = arc_agi.Arcade()
        all_envs = arcade.get_environments()

        if self.mode == "single" and self.game_filter:
            envs = [
                e for e in all_envs
                if self.game_filter.lower() in e.game_id.lower()
                or self.game_filter.upper() in e.title.upper()
            ]
        elif self.mode == "quick":
            envs = [e for e in all_envs if e.title.upper() in QUICK_DEMO_GAMES]
        else:
            envs = all_envs

        n_games = len(envs)
        self.events.emit(BenchmarkStartEvent(n_games=n_games, mode=self.mode))

        orchestrator = GameOrchestrator(
            max_actions_per_level=self.max_actions,
            skip_navigation=self.skip_navigation,
            skip_combo=self.skip_combo,
            skip_deepcopy=self.skip_deepcopy,
            event_callback=self.events.callback,
        )

        results = []
        start_time = time.time()

        for env_info in envs:
            game_start = time.time()
            self.events.emit(GameStartEvent(
                game_id=env_info.game_id, title=env_info.title,
                total_levels=len(env_info.baseline_actions),
            ))

            raw_env = arcade.make(env_info.game_id)
            if raw_env is None:
                continue

            env = ArcEnvAdapter(raw_env)
            result = await orchestrator.play_game(env, env_info.baseline_actions)

            game_time = time.time() - game_start
            result.update({
                "game_id": env_info.game_id,
                "title": env_info.title,
                "total_levels": len(env_info.baseline_actions),
                "baseline_actions": env_info.baseline_actions,
                "n_available_actions": len(env_info.available_actions) if hasattr(env_info, 'available_actions') else None,
                "time_s": round(game_time, 1),
            })
            results.append(result)

            self.events.emit(GameCompleteEvent(
                game_id=env_info.game_id,
                score=result.get("game_score", 0.0),
                levels_completed=result.get("levels_completed", 0),
                total_levels=len(env_info.baseline_actions),
                time_s=game_time,
            ))

        total_time = time.time() - start_time
        games_solved = sum(1 for r in results if r.get("levels_completed", 0) > 0)
        scores = [r.get("game_score", 0.0) for r in results]
        mean_rhae = sum(scores) / len(scores) * 100 if scores else 0.0

        self.events.emit(BenchmarkCompleteEvent(
            mean_rhae=mean_rhae, games_solved=games_solved,
            total_games=n_games, total_time_s=total_time,
        ))

        # Capture ARC-AGI scorecard UUID for independent verification
        scorecard_id = getattr(arcade, '_default_scorecard_id', None)

        self.results_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": self.mode,
            "n_games": n_games,
            "mean_rhae_pct": round(mean_rhae, 4),
            "games_solved": games_solved,
            "total_time_s": round(total_time, 1),
            "scorecard_id": scorecard_id,
            "random_seed": 42,
            "max_actions_per_level": self.max_actions,
            "games": results,
        }
        out_path = self.results_dir / f"occam_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out_path.write_text(json.dumps(summary, indent=2, default=str))
        logger.info("Results saved to %s", out_path)
        if scorecard_id:
            logger.info("ARC-AGI scorecard ID: %s", scorecard_id)

        return summary
