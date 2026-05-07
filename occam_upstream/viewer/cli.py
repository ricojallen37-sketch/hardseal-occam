# occam/viewer/cli.py
"""Occam CLI entry point."""
import argparse
import multiprocessing
import sys
import uvicorn

from viewer.runner import SolverProcess
from viewer.server import create_app


def main():
    parser = argparse.ArgumentParser(description="Occam ARC-AGI-3 Solver Viewer")
    parser.add_argument("command", choices=["run", "benchmark"], default="run", nargs="?")
    parser.add_argument("--game", type=str, help="Single game ID to run")
    parser.add_argument("--quick", action="store_true", help="Quick demo: 5 best games")
    parser.add_argument("--host", default="127.0.0.1", help="Server bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--max-actions", type=int, default=500000)
    args = parser.parse_args()

    mode = "quick" if args.quick else ("single" if args.game else "full")

    if args.command == "benchmark":
        import asyncio
        from solver.benchmark import BenchmarkRunner
        runner = BenchmarkRunner(mode=mode, game_filter=args.game, max_actions=args.max_actions)
        result = asyncio.run(runner.run())
        print(f"\nOccam Benchmark Complete")
        print(f"  RHAE: {result['mean_rhae_pct']:.2f}%")
        print(f"  Games solved: {result['games_solved']}/{result['n_games']}")
        print(f"  Time: {result['total_time_s']:.1f}s")
        sys.exit(0)

    # Viewer mode — server starts, solver is controlled from browser
    queue = multiprocessing.Queue(maxsize=10000)
    app = create_app(queue=queue, host=args.host, port=args.port)

    # Auto-start solver if --game or --quick specified
    if args.game or args.quick:
        solver = SolverProcess(queue=queue, mode=mode, game_filter=args.game, max_actions=args.max_actions)
        solver.start()

    print(f"\n  OCCAM Solver v0.1.0")
    print(f"  Open: http://{args.host}:{args.port}")
    if args.game or args.quick:
        print(f"  Mode: {mode} | Auto-started")
    else:
        print(f"  Mode: interactive | Use browser controls to start")
    if args.host == "127.0.0.1":
        print(f"  (localhost only — use --host 0.0.0.0 for remote access)")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
