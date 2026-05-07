"""Priority-based click targeting for ARC-AGI-3.

Frame segmentation and 5-tier action classification ported from
Just Explore (3rd place, 3.64% RHAE). Segments frame into connected
components, detects status bars, classifies clicks by visual salience.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
import numpy as np

# Color classification
SALIENT_COLORS = frozenset(range(6, 16))  # chromatic colors
NON_SALIENT_COLORS = frozenset(range(0, 6))  # grayscale
STATUS_BAR_EDGE_DIST = 3
STATUS_BAR_ASPECT_RATIO = 5
STATUS_BAR_MIN_TWINS = 3
MIN_WIDTH = 2
MAX_WIDTH = 32
NUM_TIERS = 5


@dataclass
class Segment:
    """A connected component in the frame."""
    color: int
    bbox: tuple[int, int, int, int]  # (y1, x1, y2, x2)
    area: int
    pixels: list[tuple[int, int]] = field(default_factory=list, repr=False)

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def center(self) -> tuple[int, int]:
        return ((self.bbox[0] + self.bbox[2]) // 2, (self.bbox[1] + self.bbox[3]) // 2)

    @property
    def is_salient(self) -> bool:
        return self.color in SALIENT_COLORS

    @property
    def is_medium(self) -> bool:
        return MIN_WIDTH <= self.width <= MAX_WIDTH and MIN_WIDTH <= self.height <= MAX_WIDTH

    @property
    def aspect_ratio(self) -> float:
        return max(self.width, self.height) / max(1, min(self.width, self.height))


def segment_frame(frame: np.ndarray) -> list[Segment]:
    """Flood-fill segmentation with 4-connectivity."""
    if frame.ndim == 3:
        frame = frame[:, :, 0] if frame.shape[2] > 1 else frame.squeeze(-1)
    h, w = frame.shape[:2]
    visited = np.zeros((h, w), dtype=bool)
    segments = []
    for r in range(h):
        for c in range(w):
            if visited[r, c]:
                continue
            color = int(frame[r, c])
            queue = deque([(r, c)])
            visited[r, c] = True
            pixels = []
            y_min, y_max, x_min, x_max = r, r, c, c
            while queue:
                cr, cc = queue.popleft()
                pixels.append((cr, cc))
                y_min, y_max = min(y_min, cr), max(y_max, cr)
                x_min, x_max = min(x_min, cc), max(x_max, cc)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and frame[nr, nc] == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            segments.append(Segment(color=color, bbox=(y_min, x_min, y_max, x_max), area=len(pixels), pixels=pixels))
    return segments


def identify_status_bars(segments: list[Segment], frame_size: int = 64) -> set[int]:
    """Detect status bar segment indices near frame edges."""
    twin_key = lambda s: (s.area, s.color)
    twin_counts: dict = {}
    for s in segments:
        k = twin_key(s)
        twin_counts[k] = twin_counts.get(k, 0) + 1

    status_ids = set()
    for i, s in enumerate(segments):
        near_edge = (
            s.bbox[0] < STATUS_BAR_EDGE_DIST
            or s.bbox[2] >= frame_size - STATUS_BAR_EDGE_DIST
            or s.bbox[1] < STATUS_BAR_EDGE_DIST
            or s.bbox[3] >= frame_size - STATUS_BAR_EDGE_DIST
        )
        if not near_edge:
            continue
        if s.aspect_ratio >= STATUS_BAR_ASPECT_RATIO:
            status_ids.add(i)
        elif twin_counts.get(twin_key(s), 0) >= STATUS_BAR_MIN_TWINS:
            status_ids.add(i)
    return status_ids


def classify_segments(segments: list[Segment], status_bar_ids: set[int] | None = None) -> dict[int, list[Segment]]:
    """Classify segments into 5 priority tiers."""
    if status_bar_ids is None:
        status_bar_ids = set()
    tiers: dict[int, list[Segment]] = {i: [] for i in range(NUM_TIERS)}
    for i, seg in enumerate(segments):
        if i in status_bar_ids:
            tiers[4].append(seg)
        elif seg.is_salient and seg.is_medium:
            tiers[0].append(seg)
        elif seg.is_medium:
            tiers[1].append(seg)
        elif seg.is_salient:
            tiers[2].append(seg)
        else:
            tiers[3].append(seg)
    return tiers


def get_priority_click_targets(frame: np.ndarray, max_targets: int = 32) -> list[tuple[int, int]]:
    """Get prioritized click target positions from a frame.

    Returns (y, x) positions sorted by priority tier (tier 0 first).
    """
    segments = segment_frame(frame)
    status_ids = identify_status_bars(segments)
    tiers = classify_segments(segments, status_ids)

    targets = []
    for tier in range(NUM_TIERS):
        for seg in tiers[tier]:
            if seg.area < 2:
                continue
            cy, cx = seg.center
            targets.append((cy, cx))
            if len(targets) >= max_targets:
                return targets
    return targets
