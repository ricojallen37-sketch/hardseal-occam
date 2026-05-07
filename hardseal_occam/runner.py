# SPDX-License-Identifier: CC0-1.0 OR MIT
"""Hardseal-Occam runner — wires HardsealOccamCallback into Occam's
BenchmarkRunner and persists the sealed trace chain after the run.

Usage:
    python -m hardseal_occam.runner --game CD82
    python -m hardseal_occam.runner --quick
    python -m hardseal_occam.runner            # full 25-game bench
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path

from solver.benchmark import BenchmarkRunner

from hardseal_occam.tracer_callback import HardsealOccamCallback
from hardseal_trace import verify_chain


def main() -> int:
    parser = argparse.ArgumentParser(description="Hardseal-Occam runner")
    parser.add_argument("--game", type=str, help="Single game ID")
    parser.add_argument("--quick", action="store_true", help="Quick 5-game demo")
    parser.add_argument(
        "--max-actions", type=int, default=500000,
        help="Max actions per level (matches occam viewer/cli.py default)",
    )
    parser.add_argument(
        "--hmac-key", type=str,
        default="hardseal-occam-v0.4-hmac-key",
        help="HMAC key (string or env var name)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="hardseal_occam_runs",
        help="Output directory for sealed-trace JSONL + summary",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    log = logging.getLogger("hardseal_occam.runner")

    mode = "quick" if args.quick else ("single" if args.game else "full")
    run_uuid = str(uuid.uuid4())
    hmac_key = (
        args.hmac_key.encode("utf-8") if isinstance(args.hmac_key, str)
        else args.hmac_key
    )

    out_dir = Path(args.out_dir) / run_uuid
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Hardseal-Occam run_uuid=%s out_dir=%s mode=%s", run_uuid, out_dir, mode)

    callback = HardsealOccamCallback(hmac_key=hmac_key, run_uuid=run_uuid)

    runner = BenchmarkRunner(
        event_callback=callback,
        mode=mode,
        game_filter=args.game,
        max_actions=args.max_actions,
    )

    t0 = time.time()
    result = asyncio.run(runner.run())
    wall_s = time.time() - t0

    # Export and validate the sealed chain
    chain = callback.export_chain()
    avg_pods = callback.average_pods()
    log.info(
        "Hardseal-Occam complete: RHAE=%.4f%% games_solved=%d/%d trace_count=%d "
        "avg_PODS=%s wall=%.1fs",
        result["mean_rhae_pct"], result["games_solved"], result["n_games"],
        len(chain), f"{avg_pods:.4f}" if avg_pods is not None else "n/a", wall_s,
    )

    # Verify per-game chains (each game has its own self-consistent chain;
    # cross-game flat concatenation cannot be verified as one chain because
    # each per-game tracer starts a fresh chain_order).
    per_game = callback.export_per_game_chains()
    games_total = len(per_game)
    games_valid = 0
    chain_valid_results = []
    for i, gc in enumerate(per_game):
        if not gc:
            chain_valid_results.append((i, True, "empty-chain"))
            games_valid += 1
            continue
        try:
            ok, msg = verify_chain(gc, hmac_key)
        except Exception as e:
            ok, msg = False, f"exception: {e}"
            log.exception("verify_chain raised on game-chain %d", i)
        chain_valid_results.append((i, ok, msg))
        if ok:
            games_valid += 1
    chain_valid = (games_valid == games_total) and games_total > 0
    log.info(
        "verify_chain per-game: %d/%d valid", games_valid, games_total,
    )

    # Persist artifacts
    # export_chain returns asdict-serialized dicts already; just dump as JSONL
    chain_path = out_dir / "sealed_traces.jsonl"
    with chain_path.open("w") as f:
        for trace in chain:
            f.write(json.dumps(trace) + "\n")

    summary_path = out_dir / "summary.json"
    summary = {
        "run_uuid": run_uuid,
        "mode": mode,
        "game_filter": args.game,
        "wall_s": round(wall_s, 1),
        "rhae_pct": result["mean_rhae_pct"],
        "games_solved": result["games_solved"],
        "n_games": result["n_games"],
        "trace_count": len(chain),
        "avg_pods": avg_pods,
        "chain_valid": chain_valid,
        "per_game_chain_validation": [
            {"game_idx": i, "valid": ok, "msg": msg}
            for i, ok, msg in chain_valid_results
        ],
        "scorecard_id": result.get("scorecard_id"),
        "callback_stats": callback.stats(),
        "occam_results_pointer": "results/occam_results_*.json",
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    log.info("Wrote %s (%d traces) and %s", chain_path, len(chain), summary_path)
    print()
    print(f"Hardseal-Occam Complete")
    print(f"  run_uuid:        {run_uuid}")
    print(f"  RHAE:            {result['mean_rhae_pct']:.2f}%")
    print(f"  Games solved:    {result['games_solved']}/{result['n_games']}")
    print(f"  Trace count:     {len(chain)}")
    print(f"  Avg PODS:        {f'{avg_pods:.4f}' if avg_pods is not None else 'n/a'}")
    print(f"  Chain valid:     {chain_valid}")
    print(f"  Wall time:       {wall_s:.1f}s")
    print(f"  Artifacts:       {out_dir}/")
    return 0 if chain_valid or not chain else 1


if __name__ == "__main__":
    sys.exit(main())
