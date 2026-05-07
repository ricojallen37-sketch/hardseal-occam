# SPDX-License-Identifier: CC0-1.0 OR MIT
"""hardseal_occam — Hardseal-trace bridge for Occam's EventEmitter.

Pure subscription pattern. Zero modifications to Occam upstream.
"""
from hardseal_occam.tracer_callback import HardsealOccamCallback

__all__ = ["HardsealOccamCallback"]
__version__ = "0.4.0"
