# SPDX-License-Identifier: CC0-1.0 OR MIT
"""Bridges Occam's EventEmitter to hardseal-trace SealedReasoningTracer.

Pattern: pass an instance of HardsealOccamCallback as `event_callback` into
either BenchmarkRunner(...) or GameOrchestrator(...). The callback receives
asdict-serialized Event payloads from Occam and emits hash-chained,
HMAC-signed sealed traces via hardseal_trace.SealedReasoningTracer.

Design brief reference: HARDSEAL_WALKER_V0.4_DESIGN_BRIEF.md §5.
Event-to-seal mapping table: §4.

Chain-link discipline: hardseal-trace v0.2.0 captures `prev_post_hash` at
seal_pre_action time from the LAST trace in chain_order. If that trace's
post hasn't been sealed yet, prev_post_hash = None and verify_chain breaks.
Therefore every event in this wrapper is sealed as a one-shot pre+post pair
in the same dispatch — no traces are ever held open across events. Phase
context is preserved by writing self._current_phase / self._current_strategy
into each per-action trace's input_state, not by holding a phase trace open.

Defensive: every event handler is wrapped in try/except so a wrapper bug
cannot crash the underlying solver.
"""
from __future__ import annotations

import logging
from typing import Any

from hardseal_trace import SealedReasoningTracer, StructuredIntent

logger = logging.getLogger("hardseal_occam.tracer_callback")


class HardsealOccamCallback:
    """Stateful callback bridging Occam events → sealed reasoning traces.

    One tracer per game (re-initialised on game_start; closed game's chain
    snapshotted into the accumulator). All event seals are one-shot pre+post
    in the same dispatch — no held-open traces.
    """

    def __init__(self, hmac_key: bytes, run_uuid: str, agent_version: str = "0.4.0"):
        self._tracer: SealedReasoningTracer | None = None
        self._hmac_key = hmac_key
        self._run_uuid = run_uuid
        self._agent_version = agent_version
        self._all_chains: list[list] = []
        self._current_game: str | None = None
        self._current_strategy: str = ""
        self._current_phase: str | None = None
        self._stats = {
            "events_received": 0,
            "events_handled": 0,
            "traces_emitted": 0,
            "errors": 0,
        }

    def _get_tracer(self, game_id: str | None) -> SealedReasoningTracer:
        if self._tracer is None or (game_id and game_id != self._current_game):
            if self._tracer is not None:
                try:
                    self._all_chains.append(self._tracer.export_chain())
                except Exception:
                    logger.exception("Failed to export prior game chain")
            self._current_game = game_id
            self._tracer = SealedReasoningTracer(
                session_id=f"{self._run_uuid}::{game_id or 'session'}",
                game_id=game_id or "session",
                key_provider=self._hmac_key,
                agent_name="HardsealOccam",
                agent_version=self._agent_version,
            )
            self._current_strategy = ""
            self._current_phase = None
        return self._tracer

    def _seal_one_shot(
        self,
        claim_text: str,
        predicted: dict,
        observed: dict,
        candidate_action: dict,
        confidence_class: str = "exploratory",
        prediction_horizon: str = "next_frame",
        extra_input_state: dict | None = None,
    ) -> None:
        """Seal a complete pre+post trace pair for a single Occam event."""
        if self._tracer is None:
            return
        input_state = {
            "phase": self._current_phase,
            "strategy": self._current_strategy,
            "game_id": self._current_game,
        }
        if extra_input_state:
            input_state.update(extra_input_state)
        intent = StructuredIntent(
            claim_text=claim_text,
            predicted_outcome=predicted,
            confidence_class=confidence_class,
            prediction_horizon=prediction_horizon,
        )
        trace = self._tracer.seal_pre_action(
            input_state=input_state,
            structured_intent=intent,
            candidate_action=candidate_action,
        )
        self._tracer.seal_post_action(trace.trace_id, observed_outcome=observed)
        self._stats["traces_emitted"] += 1

    def __call__(self, payload: dict) -> None:
        """EventEmitter callback entry point — Occam invokes this per emit()."""
        self._stats["events_received"] += 1
        try:
            self._dispatch(payload)
            self._stats["events_handled"] += 1
        except Exception:
            self._stats["errors"] += 1
            logger.exception("Event handler raised; suppressing to protect solver")

    def _dispatch(self, payload: dict) -> None:
        et = payload.get("type")
        data = payload.get("data", {}) or {}

        if et == "benchmark_start":
            return  # session-level marker

        if et == "game_start":
            self._current_game = data.get("game_id")
            self._get_tracer(self._current_game)
            return

        if et == "phase_change":
            prev_phase = self._current_phase
            self._current_phase = data.get("phase")
            self._current_strategy = data.get("strategy", "")
            self._get_tracer(self._current_game)
            self._seal_one_shot(
                claim_text=(
                    f"Enter phase '{self._current_phase}' with strategy "
                    f"'{self._current_strategy}'"
                ),
                predicted={
                    "phase": self._current_phase,
                    "strategy": self._current_strategy,
                    "expected_yields_solution": self._current_phase == "execute",
                },
                observed={
                    "phase": self._current_phase,
                    "strategy": self._current_strategy,
                    "phase_prev": prev_phase,
                    "completion": "phase_entered",
                },
                candidate_action={"type": "phase_transition", "to": self._current_phase},
                confidence_class="high" if self._current_strategy else "exploratory",
                prediction_horizon="phase",
            )

        elif et == "probe":
            self._get_tracer(self._current_game)
            self._seal_one_shot(
                claim_text=f"Probe action {data.get('action')} for effectiveness",
                predicted={"action": data.get("action"), "expected_effective": True},
                observed={
                    "action": data.get("action"),
                    "effective": data.get("effective"),
                    "diff_pixels": data.get("diff_pixels", 0),
                },
                candidate_action={"id": data.get("action"), "type": "probe"},
                confidence_class="exploratory",
                prediction_horizon="next_frame",
            )

        elif et == "bfs_step":
            self._get_tracer(self._current_game)
            from_state = data.get("from_state", "") or ""
            self._seal_one_shot(
                claim_text=(
                    f"BFS expand from {from_state[:8]} via "
                    f"action {data.get('action')}"
                ),
                predicted={
                    "from_state": from_state,
                    "action": data.get("action"),
                    "expected_new_state": True,
                },
                observed={
                    "from_state": from_state,
                    "to_state": data.get("to_state"),
                    "is_new": data.get("is_new", False),
                },
                candidate_action={"id": data.get("action"), "type": "bfs_step"},
                confidence_class="exploratory",
                prediction_horizon="next_frame",
                extra_input_state={"from_state": from_state},
            )

        elif et == "reset":
            self._get_tracer(self._current_game)
            replay_prefix = data.get("replay_prefix", []) or []
            self._seal_one_shot(
                claim_text=f"Reset and replay prefix length {len(replay_prefix)}",
                predicted={
                    "expected_reset_success": True,
                    "replay_len": len(replay_prefix),
                },
                observed={
                    "reset_count": data.get("count", 0),
                    "replay_len": len(replay_prefix),
                    "completion": "reset",
                },
                candidate_action={
                    "type": "reset_replay",
                    "prefix_len": len(replay_prefix),
                },
                confidence_class="high",
                prediction_horizon="next_frame",
                extra_input_state={"reset_count": data.get("count", 0)},
            )

        elif et == "level_solved":
            self._get_tracer(self._current_game)
            self._seal_one_shot(
                claim_text=f"Level {data.get('level')} solved",
                predicted={
                    "level": data.get("level"),
                    "expected_completion": "solved",
                },
                observed={
                    "level": data.get("level"),
                    "actions": data.get("actions"),
                    "rhae": data.get("rhae"),
                    "completion": "solved",
                },
                candidate_action={
                    "type": "level_outcome",
                    "level": data.get("level"),
                },
                confidence_class="high",
                prediction_horizon="end_of_level",
                extra_input_state={"level": data.get("level")},
            )

        elif et == "level_failed":
            self._get_tracer(self._current_game)
            self._seal_one_shot(
                claim_text=(
                    f"Level {data.get('level')} failed: "
                    f"{data.get('reason', '')}"
                ),
                predicted={
                    "level": data.get("level"),
                    "expected_completion": "solved",
                },
                observed={
                    "level": data.get("level"),
                    "reason": data.get("reason", ""),
                    "completion": "failed",
                },
                candidate_action={
                    "type": "level_outcome",
                    "level": data.get("level"),
                },
                confidence_class="exploratory",
                prediction_horizon="end_of_level",
                extra_input_state={"level": data.get("level")},
            )

        elif et == "game_complete":
            # Snapshot the per-game chain into accumulator and detach so the
            # next game starts with a fresh tracer (chain_order resets there).
            if self._tracer is not None:
                try:
                    self._all_chains.append(self._tracer.export_chain())
                except Exception:
                    logger.exception("Failed to export game chain on game_complete")
                self._tracer = None
            self._current_game = None
            self._current_phase = None
            self._current_strategy = ""

        elif et == "benchmark_complete":
            # Final flush — if a game was still active (e.g., single-game runs
            # where game_complete didn't fire before benchmark_complete), snapshot.
            if self._tracer is not None:
                try:
                    self._all_chains.append(self._tracer.export_chain())
                except Exception:
                    logger.exception("Failed to export final game chain")
                self._tracer = None

        # state_discovered + frame_diff intentionally not sealed (per §4):
        # state_discovered = informational waypoint (no candidate action),
        # frame_diff = compression target (semantic content lives in the
        # enclosing bfs_step's observed_outcome).

    def export_chain(self) -> list:
        """Concatenation of all per-game chains.

        Each game has its own tracer + chain_order, so verify_chain must run
        per-game (each game is a self-consistent sub-chain). For convenience
        we still flatten across games for inspection / total-count purposes.
        """
        chains = list(self._all_chains)
        if self._tracer is not None:
            try:
                chains.append(self._tracer.export_chain())
            except Exception:
                logger.exception("Failed to export active tracer chain")
        return [t for chain in chains for t in chain]

    def export_per_game_chains(self) -> list[list]:
        """List of per-game chains. verify_chain runs cleanly on each entry."""
        chains = list(self._all_chains)
        if self._tracer is not None:
            try:
                chains.append(self._tracer.export_chain())
            except Exception:
                logger.exception("Failed to export active tracer chain")
        return chains

    def average_pods(self) -> float | None:
        """Mean PODS across all sealed traces. Returns None if no chain."""
        traces = self.export_chain()
        scores = [
            t.get("divergence_score") for t in traces
            if t.get("divergence_score") is not None
        ]
        if not scores:
            return None
        return sum(scores) / len(scores)

    def stats(self) -> dict:
        return dict(self._stats)
