# Hardseal-Occam — ARC Prize 2026 submission

**TL;DR:** Sean Donahoe's [Occam](https://github.com/g-baskin/occam) ARC-AGI-3 solver (MIT, 60.01% RHAE on the public 25-game eval) wrapped with [hardseal-trace](https://github.com/anthropics/hardseal-trace) v0.2.0, a cryptographic reasoning-trace primitive (Apache-2.0). Pure subscription pattern via Occam's `EventEmitter` — zero modifications to the solver. **Score parity vs stock: +0.0000% delta** on the 5-game quick-mode parity test (75.9657% wrapper vs 75.9657% stock to four decimals).

This repo is the Kaggle submission. License: `CC0-1.0 OR MIT` for wrapper code; MIT for vendored Occam; Apache-2.0 for vendored hardseal-trace. See `NOTICE.md` for the full license map.

## Repo layout

```
hardseal-occam-contest/
├── LICENSE                              # CC0-1.0 OR MIT (wrapper code)
├── NOTICE.md                            # provenance + full license map
├── README.md                            # this file
├── pyproject.toml                       # contest-side packaging
├── kaggle_submission.ipynb              # driver notebook for Kaggle eval
├── LICENSES/
│   ├── MIT-occam.txt                    # Sean Donahoe MIT (verbatim)
│   └── Apache-2.0-hardseal-trace.txt    # full Apache-2.0 (verbatim)
├── occam_upstream/                      # vendored verbatim from g-baskin/occam@e3be26a (MIT)
├── hardseal_trace/                      # vendored verbatim from hardseal-trace v0.2.0 (Apache-2.0)
├── hardseal_occam/                      # this repo's wrapper code (CC0-1.0 OR MIT)
│   ├── __init__.py
│   ├── tracer_callback.py               # EventEmitter → SealedReasoningTracer bridge
│   └── runner.py                        # standalone CLI runner
└── traces/                              # sealed reasoning trace artifacts (Apache-2.0)
    └── full_25_game_<run_uuid>.jsonl    # full chain from local 25-game wrapper run
```

## What the wrapper does

Occam already exposes an `event_callback: Callable[[dict], None]` constructor
parameter on `BenchmarkRunner` and `GameOrchestrator`. The
`HardsealOccamCallback` class subscribes to that callback and emits a
hash-chained, HMAC-signed sealed reasoning trace per relevant event:

| Occam event | Trace primitive |
|---|---|
| `phase_change` (game_start / discover / execute) | one-shot pre+post seal |
| `probe(action, effective, diff_pixels)` | one-shot pre+post seal |
| `bfs_step(from_state, action, to_state, is_new)` | one-shot pre+post seal |
| `reset(count, replay_prefix)` | one-shot pre+post seal |
| `level_solved` / `level_failed` | one-shot pre+post seal |
| `state_discovered` | informational, not sealed |
| `frame_diff` | viewer-only, not sealed |
| `game_complete` / `benchmark_complete` | tracer flush + chain export |

Each sealed trace records a `predicted_outcome` (the strategy's claim about
what the action will accomplish) and an `observed_outcome` (what actually
happened). The divergence between predicted and observed is a per-trace
**PODS** score in [0, 1]. Chain integrity is HMAC + SHA-256 hash chain
verifiable via `hardseal_trace.verify_chain(chain, hmac_key)`.

## Running locally

Requires Python ≥ 3.12 (we tested 3.13.12). Anonymous API key auto-fetches
from `three.arcprize.org`; no manual `ARC_API_KEY` setup required.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ./occam_upstream
pip install -e ./hardseal_trace        # if hardseal_trace ships pyproject; else add to PYTHONPATH
PYTHONPATH=. python -m hardseal_occam.runner --quick --hmac-key "your-hmac-key"
# Full 25-game run:
PYTHONPATH=. python -m hardseal_occam.runner --hmac-key "your-hmac-key"
# Single game:
PYTHONPATH=. python -m hardseal_occam.runner --game CD82 --hmac-key "your-hmac-key"
```

The runner writes `summary.json` + `sealed_traces.jsonl` to
`hardseal_occam_runs/<run_uuid>/` and also writes Occam's standard
`occam_results_*.json` to `results/`.

## Why not put this in the Hardseal repo?

Hardseal LLC's CMMC-readiness product and back-office IP are licensed
restrictively to protect commercial interests. The ARC Prize 2026 contest
requires submissions under permissive licenses (CC0 / MIT-0). To honor
both, this repo ships only the public-permissively-licensed wrapper +
vendored upstream + Apache-2.0 trace primitive. **Hardseal Core never
enters this repo.**

## License

See `LICENSE` for the wrapper-code license (CC0-1.0 OR MIT). See `NOTICE.md`
for the per-component license map. Vendored upstream code retains its
original license (Occam: MIT; hardseal-trace: Apache-2.0).

## Citations

If you use this submission's sealed-trace artifacts in research, cite:

- Occam: Sean Donahoe, *Occam: Algorithmic Solver for ARC-AGI-3* (2026), [github.com/g-baskin/occam](https://github.com/g-baskin/occam) | [Zenodo DOI 10.5281/zenodo.19448189](https://zenodo.org/records/19448189)
- ARC-AGI-3: ARC Prize Foundation, *ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence* (April 2026)
- Hardseal-trace: Hardseal LLC, *hardseal-trace: cryptographically verifiable sealed reasoning traces with PODS divergence metric* (2026)
