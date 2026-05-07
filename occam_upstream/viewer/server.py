"""FastAPI WebSocket server for Occam viewer."""
from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse

logger = logging.getLogger("occam.server")

STATIC_DIR = Path(__file__).parent / "static"
RESULTS_DIR = Path("results")


def create_app(queue: multiprocessing.Queue, host: str = "127.0.0.1", port: int = 8080) -> FastAPI:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    events_file = RESULTS_DIR / f"events_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

    reconnect_buffer: deque = deque(maxlen=500)

    # Server-side frame state: accumulate frame_diffs so new clients get the full picture
    current_frame: list[int] = [0] * (64 * 64)

    state_snapshot: dict = {
        "games": {},
        "current_game": None,
        "mean_rhae": 0.0,
        "benchmark_running": False,
    }

    clients: set[WebSocket] = set()

    async_queue: asyncio.Queue = asyncio.Queue()

    def _drain_thread(loop: asyncio.AbstractEventLoop):
        """Dedicated thread: reads from multiprocessing.Queue, puts into asyncio.Queue.

        Must use loop.call_soon_threadsafe() because asyncio.Queue is not thread-safe.
        """
        while True:
            try:
                event = queue.get(timeout=0.5)
                loop.call_soon_threadsafe(async_queue.put_nowait, event)
            except Exception:
                continue

    async def queue_reader():
        """Background task: read events from async queue, persist + broadcast."""
        loop = asyncio.get_running_loop()
        t = threading.Thread(target=_drain_thread, args=(loop,), daemon=True)
        t.start()

        with open(events_file, "a") as f:
            while True:
                try:
                    event = await async_queue.get()

                    try:
                        f.write(json.dumps(event, default=str) + "\n")
                        f.flush()
                    except IOError:
                        pass

                    # Update state snapshot
                    etype = event.get("type")
                    edata = event.get("data", {})
                    if etype == "benchmark_start":
                        state_snapshot["benchmark_running"] = True
                    elif etype == "game_start":
                        gid = edata.get("game_id", "")
                        state_snapshot["current_game"] = gid
                        state_snapshot["games"][gid] = {
                            "status": "active", "score": 0,
                            "levels_completed": 0, "total_levels": edata.get("total_levels", 0)
                        }
                    elif etype == "game_complete":
                        gid = edata.get("game_id", "")
                        state_snapshot["games"][gid] = {
                            "status": "complete", "score": edata.get("score", 0),
                            "levels_completed": edata.get("levels_completed", 0),
                            "total_levels": edata.get("total_levels", 0)
                        }
                        state_snapshot["current_game"] = None
                    elif etype == "benchmark_complete":
                        state_snapshot["benchmark_running"] = False
                        state_snapshot["mean_rhae"] = edata.get("mean_rhae", 0)

                    # Track frame state server-side
                    if etype == "frame_diff":
                        for ch in edata.get("changes", []):
                            if len(ch) >= 3:
                                r, c, val = ch[0], ch[1], ch[2]
                                if 0 <= r < 64 and 0 <= c < 64:
                                    current_frame[r * 64 + c] = val
                    elif etype == "game_start":
                        # Reset frame for new game
                        for i in range(len(current_frame)):
                            current_frame[i] = 0

                    reconnect_buffer.append(event)

                    dead = set()
                    for ws in list(clients):
                        try:
                            await ws.send_json(event)
                        except Exception:
                            dead.add(ws)
                    clients.difference_update(dead)

                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("queue_reader error")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(queue_reader())
        yield
        task.cancel()

    app = FastAPI(title="Occam Solver Viewer", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
    )

    # Solver process management
    solver_state: dict = {"process": None, "mode": "full", "game_filter": None, "max_actions": 500000}

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = STATIC_DIR / "index.html"
        return HTMLResponse(content=html_path.read_text(), status_code=200)

    @app.get("/events")
    async def get_events_file():
        if events_file.exists():
            return FileResponse(events_file, media_type="application/x-ndjson")
        return {"error": "No events recorded yet"}

    @app.post("/api/start")
    async def start_benchmark(mode: str = "full", game: str | None = None):
        """Start a benchmark run via API."""
        from viewer.runner import SolverProcess
        if solver_state["process"] and solver_state["process"].is_alive():
            return {"error": "Benchmark already running"}
        # Reset all state
        state_snapshot["games"] = {}
        state_snapshot["current_game"] = None
        state_snapshot["mean_rhae"] = 0.0
        state_snapshot["benchmark_running"] = True
        reconnect_buffer.clear()
        for i in range(len(current_frame)):
            current_frame[i] = 0
        solver_state["process"] = SolverProcess(
            queue=queue, mode="single" if game else mode,
            game_filter=game, max_actions=solver_state["max_actions"],
        )
        solver_state["process"].start()
        return {"status": "started", "mode": mode, "game": game}

    @app.post("/api/stop")
    async def stop_benchmark():
        """Stop a running benchmark."""
        if solver_state["process"] and solver_state["process"].is_alive():
            solver_state["process"].terminate()
            state_snapshot["benchmark_running"] = False
            return {"status": "stopped"}
        return {"status": "not_running"}

    @app.get("/api/status")
    async def get_status():
        """Get current solver status."""
        running = solver_state["process"].is_alive() if solver_state["process"] else False
        return {
            "running": running,
            "state": state_snapshot,
        }

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        clients.add(ws)
        await ws.send_json({"type": "state_snapshot", "data": state_snapshot})
        # Send current frame + buffered events only if benchmark is active
        if state_snapshot.get("benchmark_running"):
            frame_changes = [
                [r, c, current_frame[r * 64 + c]]
                for r in range(64) for c in range(64)
                if current_frame[r * 64 + c] != 0
            ]
            if frame_changes:
                await ws.send_json({"type": "frame_diff", "data": {"changes": frame_changes}})
            for event in list(reconnect_buffer):
                if event.get("type") != "frame_diff":
                    await ws.send_json(event)
        try:
            while True:
                msg = await ws.receive_text()
                # Handle commands from browser
                try:
                    cmd = json.loads(msg)
                    if cmd.get("action") == "start":
                        await start_benchmark(
                            mode=cmd.get("mode", "full"),
                            game=cmd.get("game"),
                        )
                    elif cmd.get("action") == "stop":
                        await stop_benchmark()
                except (json.JSONDecodeError, Exception):
                    pass
        except WebSocketDisconnect:
            clients.discard(ws)

    return app
