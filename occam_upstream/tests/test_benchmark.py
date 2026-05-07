from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
from solver.benchmark import BenchmarkRunner

def test_runner_emits_start_and_complete():
    events = []
    runner = BenchmarkRunner(event_callback=lambda e: events.append(e), mode="full")

    with patch("solver.benchmark.arc_agi") as mock_arc:
        mock_env_info = MagicMock()
        mock_env_info.game_id = "test01"
        mock_env_info.title = "TEST"
        mock_env_info.baseline_actions = [10]
        mock_arc.Arcade.return_value.get_environments.return_value = [mock_env_info]
        mock_raw_env = MagicMock()
        mock_arc.Arcade.return_value.make.return_value = mock_raw_env

        with patch("solver.benchmark.GameOrchestrator") as mock_orch:
            mock_orch.return_value.play_game = AsyncMock(return_value={
                "total_actions": 50, "levels_completed": 1,
                "level_scores": [0.8], "game_score": 0.8,
                "phases": {"discover": 10, "execute": 40},
            })
            asyncio.run(runner.run())

    types = [e["type"] for e in events]
    assert "benchmark_start" in types
    assert "game_start" in types
    assert "game_complete" in types
    assert "benchmark_complete" in types

def test_runner_quick_mode_limits_games():
    runner = BenchmarkRunner(event_callback=None, mode="quick")
    assert runner.mode == "quick"
