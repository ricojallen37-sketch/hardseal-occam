# NOTICE — license map and provenance

This repository is the **Hardseal-Occam** ARC-AGI-3 contest submission. It
combines three independently-licensed pieces. Every file in the tree falls
under exactly one of these three licenses; this NOTICE documents which is
which.

## License map

| Path | License | Origin |
|---|---|---|
| `occam_upstream/` | **MIT** | Vendored copy of `github.com/g-baskin/occam` at commit `e3be26a`. Copyright 2026 Sean Donahoe. Full text: `LICENSES/MIT-occam.txt`. Not modified. |
| `hardseal_trace/` | **Apache-2.0** | Vendored copy of `hardseal-trace` v0.2.0 from Hardseal LLC's internal codebase (cryptographic reasoning-trace primitive with PODS metric). Full text: `LICENSES/Apache-2.0-hardseal-trace.txt`. Not modified. |
| `hardseal_occam/` | **CC0-1.0 OR MIT** | This repo's wrapper code: `HardsealOccamCallback` + runner. Bridges Occam's `EventEmitter` to `SealedReasoningTracer`. Per ARC Prize 2026 contest rules; recipient's option. See `LICENSE`. |
| `kaggle_submission.ipynb` | **CC0-1.0 OR MIT** | Driver notebook for Kaggle eval. Same as wrapper code. |
| `LICENSE` | **CC0-1.0 OR MIT** | Top-level wrapper-code license file. |
| `LICENSES/MIT-occam.txt` | (text of MIT) | Full Occam upstream MIT license, preserved verbatim. |
| `LICENSES/Apache-2.0-hardseal-trace.txt` | (text of Apache-2.0) | Full hardseal-trace Apache-2.0 license, preserved verbatim. |
| `NOTICE.md` | **CC0-1.0 OR MIT** | This file. |
| `README.md` | **CC0-1.0 OR MIT** | This file. |
| `pyproject.toml` | **CC0-1.0 OR MIT** | Contest-side packaging metadata. |
| `traces/` | **Apache-2.0** | Sealed reasoning trace JSONL artifacts produced by the wrapper at run time. Same license as `hardseal_trace/` since the format and HMAC scheme are defined there. |

## Provenance summary

This contest submission is the wrapping work, not the underlying solver. The
solver (Occam) was authored entirely by **Sean Donahoe** and is included as
a **vendored verbatim copy** of his public MIT-licensed repository at commit
`e3be26a`. No solver changes. All ARC-AGI-3 score on this submission is
attributable to Occam's algorithmic work; the Hardseal contribution is the
cryptographic reasoning-trace layer that does not modify solver behavior
(verified: ±0.0000% RHAE delta on the 5-game quick-mode parity test;
75.9657% wrapper vs 75.9657% stock).

The hardseal-trace v0.2.0 primitive is owned by Hardseal LLC and released
under Apache-2.0. The contest-rules-compliant wrapper code in this repo is
released under CC0-1.0 OR MIT to satisfy ARC Prize 2026 submission terms
without entangling Hardseal Core IP.

## IP partition

Per Hardseal IP doctrine (Perplexity §9.2.7): **Hardseal Core never touches
the contest repo.** The contest submission lives under Rico Allen's personal
GitHub account (`ricojallen37-sketch`), imports `hardseal-trace` as a vendored
Apache-2.0 dependency, and forks/vendors Occam upstream. Hardseal's CMMC-
readiness product, the back-office, the SSP/POA&M generators, and any
Hardseal-org repos are **not** referenced or imported here.
