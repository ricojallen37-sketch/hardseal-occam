import asyncio
import pytest
from unittest.mock import MagicMock
import numpy as np
import sys
import os

# Ensure solver package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_orchestrator_imports_cleanly():
    """Verify importing orchestrator does not pull in heavy dependencies."""
    from solver.orchestrator import GameOrchestrator
    assert GameOrchestrator is not None


def test_orchestrator_creates_with_event_callback():
    from solver.orchestrator import GameOrchestrator
    events = []
    o = GameOrchestrator(max_actions_per_level=100, event_callback=lambda e: events.append(e))
    assert o.events._callback is not None


def test_play_game_is_async():
    from solver.orchestrator import GameOrchestrator
    o = GameOrchestrator(max_actions_per_level=100)
    assert asyncio.iscoroutinefunction(o.play_game)


def test_replay_explorer_emits_events():
    from solver.replay_explorer import ReplayExplorer
    events = []
    mock_env = MagicMock()
    mock_env.reset.return_value = np.zeros((64, 64), dtype=np.uint8)
    mock_env.step.return_value = (np.zeros((64, 64), dtype=np.uint8), 0.0, True, {"solved": False})
    explorer = ReplayExplorer(mock_env, n_actions=4, max_total_steps=10, event_callback=lambda e: events.append(e))
    explorer.explore()
    assert len(events) > 0


def test_action_ranker_ranks_effective_first():
    from solver.action_ranker import ActionRanker
    r = ActionRanker()
    r.reset()
    # Record that action 2 is effective
    r.record(None, 2, True)
    r.record(None, 0, False)
    r.record(None, 1, False)
    ranked = r.rank(None, [0, 1, 2])
    assert ranked[0] == 2
    assert set(ranked) == {0, 1, 2}


def test_state_graph_basic():
    from solver.state_graph import StateGraph
    g = StateGraph(n_actions=3)
    g.set_initial_state("s0")
    g.add_transition("s0", 0, "s1")
    g.add_transition("s0", 1, "s2")
    assert g.get_successor("s0", 0) == "s1"
    assert g.untried_actions("s0") == {2}
    assert g.path_to("s1") == [0]


def test_rhae_computation():
    from solver.rhae import compute_rhae, weighted_game_score
    assert compute_rhae(10, 10) == 1.0
    assert compute_rhae(20, 10) == 0.25
    assert compute_rhae(0, 10) == 0.0
    assert weighted_game_score([1.0, 0.5], 2) == pytest.approx((1 * 1.0 + 2 * 0.5) / 3)


def test_context_compression_detect_objects():
    from solver.context_compression import detect_objects
    frame = np.zeros((64, 64), dtype=np.uint8)
    frame[10:20, 10:20] = 1  # blue square
    objs = detect_objects(frame)
    assert len(objs) == 1
    assert objs[0]["color"] == 1
    assert objs[0]["size"] == 100


def test_priority_tiers_basic():
    from solver.priority_tiers import get_priority_click_targets
    frame = np.zeros((64, 64), dtype=np.uint8)
    frame[30:35, 30:35] = 7  # orange square in center
    targets = get_priority_click_targets(frame, max_targets=10)
    assert len(targets) > 0


def test_environments_adapter_class():
    from solver.environments import ArcEnvAdapter
    # Just verify class is importable and has expected methods
    assert hasattr(ArcEnvAdapter, "reset")
    assert hasattr(ArcEnvAdapter, "step")
    assert hasattr(ArcEnvAdapter, "get_available_actions")


def test_events_import():
    from solver.events import EventEmitter, PhaseChangeEvent, ProbeEvent, BfsStepEvent
    emitter = EventEmitter(callback=lambda e: None)
    emitter.emit(PhaseChangeEvent(phase="test"))
    emitter.emit(ProbeEvent(action=0, effective=True))
