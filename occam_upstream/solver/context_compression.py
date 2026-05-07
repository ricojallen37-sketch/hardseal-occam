"""Context compression for spatial intelligence.

Converts raw 64x64 game frames into compact text descriptions for BFS reasoning.
The orchestrator should NEVER see raw grids -- only these compressed summaries.

Key design:
- detect_objects()   : find connected components, return structured dicts
- ObjectTracker      : persistent IDs across frames via IoU matching
- compress_l1()      : ~200-token structural summary with spatial relations
- compress_l2()      : ~50-token episodic summary of an action log
- compress_diff()    : object-level diff between two consecutive frames
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Color name palette (ARC-AGI uses 0-9; we support up to 15)
# ---------------------------------------------------------------------------
_COLOR_NAMES = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "magenta",
    7: "orange",
    8: "sky",
    9: "maroon",
    10: "color10",
    11: "color11",
    12: "color12",
    13: "color13",
    14: "color14",
    15: "color15",
}


def _color_name(c: int) -> str:
    return _COLOR_NAMES.get(int(c), f"color{c}")


# ---------------------------------------------------------------------------
# Object detection
# ---------------------------------------------------------------------------

def _extract_single_channel(frame: np.ndarray) -> np.ndarray:
    """Return a 2-D (H, W) integer frame from any input shape."""
    frame = np.asarray(frame)
    if frame.ndim == 3:
        # (N, H, W) -- take first channel
        frame = frame[0]
    if frame.ndim != 2:
        raise ValueError(f"Expected 2-D or 3-D frame, got shape {frame.shape}")
    return frame.astype(np.int32)


def _flood_fill(grid: np.ndarray, visited: np.ndarray, r: int, c: int, color: int) -> list[tuple[int, int]]:
    """BFS flood-fill. Returns list of (row, col) pixels in the component."""
    H, W = grid.shape
    pixels: list[tuple[int, int]] = []
    queue: deque[tuple[int, int]] = deque()
    queue.append((r, c))
    visited[r, c] = True
    while queue:
        cr, cc = queue.popleft()
        pixels.append((cr, cc))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and grid[nr, nc] == color:
                visited[nr, nc] = True
                queue.append((nr, nc))
    return pixels


def detect_objects(frame: np.ndarray) -> list[dict[str, Any]]:
    """Detect connected components of non-background colors.

    Args:
        frame: numpy array of shape (64, 64) or (N, 64, 64) with int values 0-15.

    Returns:
        List of object dicts, each with keys:
            color   : int
            size    : int (pixel count)
            bbox    : (min_r, min_c, max_r, max_c) -- inclusive
            center  : (row, col) floats
            shape   : "square" | "tall" | "wide" | "dot"
    """
    grid = _extract_single_channel(frame)
    H, W = grid.shape

    # Background = most frequent color
    unique, counts = np.unique(grid, return_counts=True)
    background = int(unique[np.argmax(counts)])

    visited = np.zeros((H, W), dtype=bool)
    objects: list[dict[str, Any]] = []

    for r in range(H):
        for c in range(W):
            color = int(grid[r, c])
            if color == background or visited[r, c]:
                continue
            pixels = _flood_fill(grid, visited, r, c, color)
            rows = [p[0] for p in pixels]
            cols = [p[1] for p in pixels]
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)
            height = max_r - min_r + 1
            width = max_c - min_c + 1
            aspect = height / max(width, 1)
            if height <= 1 and width <= 1:
                shape = "dot"
            elif aspect > 1.5:
                shape = "tall"
            elif aspect < 0.67:
                shape = "wide"
            else:
                shape = "square"
            objects.append({
                "color": color,
                "size": len(pixels),
                "bbox": (min_r, min_c, max_r, max_c),
                "center": (float(min_r + max_r) / 2.0, float(min_c + max_c) / 2.0),
                "shape": shape,
            })

    return objects


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection over union for two bboxes (min_r, min_c, max_r, max_c)."""
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    inter = max(0, r1 - r0 + 1) * max(0, c1 - c0 + 1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return inter / (area_a + area_b - inter)


# ---------------------------------------------------------------------------
# ObjectTracker
# ---------------------------------------------------------------------------

class ObjectTracker:
    """Maintains persistent object IDs across frames via IoU matching."""

    IOU_THRESHOLD = 0.1

    def __init__(self) -> None:
        self._next_id: int = 1
        self._active: dict[str, dict[str, Any]] = {}
        self.state_changes: list[str] = []

    def _new_id(self) -> str:
        oid = f"Obj_{self._next_id}"
        self._next_id += 1
        return oid

    def update(self, objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Match new objects to existing tracked objects; assign/persist IDs."""
        self.state_changes = []

        if not self._active:
            result = []
            for obj in objects:
                oid = self._new_id()
                tracked = {**obj, "id": oid}
                self._active[oid] = tracked
                result.append(tracked)
            return result

        unmatched_existing = set(self._active.keys())
        matched: dict[str, dict[str, Any]] = {}

        new_objects_copy = [dict(o) for o in objects]

        existing_ids = list(self._active.keys())
        scores: list[tuple[float, str, int]] = []
        for new_idx, new_obj in enumerate(new_objects_copy):
            for eid in existing_ids:
                iou = _bbox_iou(new_obj["bbox"], self._active[eid]["bbox"])
                scores.append((iou, eid, new_idx))

        scores.sort(key=lambda x: -x[0])

        assigned_new: set[int] = set()
        for iou, eid, new_idx in scores:
            if iou < self.IOU_THRESHOLD:
                break
            if eid not in unmatched_existing or new_idx in assigned_new:
                continue
            unmatched_existing.discard(eid)
            assigned_new.add(new_idx)
            tracked = {**new_objects_copy[new_idx], "id": eid}
            matched[eid] = tracked

            old_center = self._active[eid]["center"]
            new_center = tracked["center"]
            dr = new_center[0] - old_center[0]
            dc = new_center[1] - old_center[1]
            if abs(dr) > 0.4 or abs(dc) > 0.4:
                self.state_changes.append(
                    f"{eid} moved from ({old_center[0]:.1f},{old_center[1]:.1f}) "
                    f"to ({new_center[0]:.1f},{new_center[1]:.1f})"
                )

        result = list(matched.values())
        for new_idx, new_obj in enumerate(new_objects_copy):
            if new_idx not in assigned_new:
                oid = self._new_id()
                tracked = {**new_obj, "id": oid}
                result.append(tracked)
                matched[oid] = tracked
                self.state_changes.append(f"{oid} appeared at center {new_obj['center']}")

        self._active = matched
        return result


# ---------------------------------------------------------------------------
# Spatial relation helpers
# ---------------------------------------------------------------------------

def _spatial_relation(a: dict, b: dict) -> str:
    """Single-sentence spatial relation between two objects."""
    ar, ac = a["center"]
    br, bc = b["center"]
    dr = ar - br
    dc = ac - bc
    if abs(dr) >= abs(dc):
        return "below" if dr > 0 else "above"
    return "right-of" if dc > 0 else "left-of"


def _detect_symmetry(frame: np.ndarray) -> list[str]:
    """Return list of detected symmetry types."""
    grid = _extract_single_channel(frame)
    syms = []
    if np.array_equal(grid, np.fliplr(grid)):
        syms.append("horizontal")
    if np.array_equal(grid, np.flipud(grid)):
        syms.append("vertical")
    if np.array_equal(grid, np.rot90(grid, 2)):
        syms.append("rotational-180")
    return syms


# ---------------------------------------------------------------------------
# compress_l1
# ---------------------------------------------------------------------------

def compress_l1(frame: np.ndarray, tracker: ObjectTracker | None = None) -> str:
    """Structural description of a single frame (~200 tokens)."""
    raw_objects = detect_objects(frame)

    if tracker is not None:
        objects = tracker.update(raw_objects)
    else:
        objects = [{**o, "id": f"Obj_{i+1}"} for i, o in enumerate(raw_objects)]

    if not objects:
        return "Frame is empty (background only)."

    parts: list[str] = []

    for obj in objects:
        oid = obj["id"]
        cname = _color_name(obj["color"])
        r, c = obj["center"]
        parts.append(
            f"{oid}: {cname} {obj['shape']}, {obj['size']}px, "
            f"center=({r:.0f},{c:.0f}), bbox={obj['bbox']}"
        )

    objs_for_rel = objects[:4]
    relations: list[str] = []
    for i, a in enumerate(objs_for_rel):
        for b in objs_for_rel[i + 1:]:
            rel = _spatial_relation(a, b)
            relations.append(f"{a['id']} is {rel} {b['id']}")
    if relations:
        parts.append("Relations: " + "; ".join(relations))

    syms = _detect_symmetry(frame)
    if syms:
        parts.append("Symmetry: " + ", ".join(syms))

    summary = " | ".join(parts)
    return summary[:500]


# ---------------------------------------------------------------------------
# compress_l2
# ---------------------------------------------------------------------------

def compress_l2(action_log: list[dict]) -> str:
    """Episodic summary of an action log (~50 tokens)."""
    if not action_log:
        return "No actions recorded."

    n = len(action_log)
    total_reward = sum(e.get("reward", 0.0) for e in action_log)

    from collections import Counter
    action_counts = Counter(e.get("action") for e in action_log)
    most_common_action, most_common_count = action_counts.most_common(1)[0]

    reward_events = [e for e in action_log if e.get("reward", 0.0) > 0]

    step_start = action_log[0].get("step", 0)
    step_end = action_log[-1].get("step", n - 1)

    parts = [f"Steps {step_start}-{step_end} ({n} total)."]
    parts.append(f"Total reward: {total_reward:.1f}.")
    parts.append(f"Most frequent action: {most_common_action} ({most_common_count}x).")

    if reward_events:
        rewarding_actions = Counter(e.get("action") for e in reward_events)
        top_action, top_count = rewarding_actions.most_common(1)[0]
        top_reward = sum(e.get("reward", 0.0) for e in reward_events if e.get("action") == top_action)
        parts.append(f"Action {top_action} earned reward {top_count}x (total {top_reward:.1f}).")

    result = " ".join(parts)
    return result[:300]


# ---------------------------------------------------------------------------
# compress_diff
# ---------------------------------------------------------------------------

def compress_diff(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    tracker: ObjectTracker | None = None,
) -> str:
    """Object-level diff between two consecutive frames."""
    local_tracker = tracker if tracker is not None else ObjectTracker()

    prev_raw = detect_objects(prev_frame)
    prev_tracked = local_tracker.update(prev_raw)
    prev_ids = {o["id"] for o in prev_tracked}

    curr_raw = detect_objects(curr_frame)
    curr_tracked = local_tracker.update(curr_raw)
    curr_ids = {o["id"] for o in curr_tracked}

    parts: list[str] = []

    new_ids = curr_ids - prev_ids
    for obj in curr_tracked:
        if obj["id"] in new_ids:
            parts.append(f"{obj['id']} ({_color_name(obj['color'])}) appeared.")

    vanished_ids = prev_ids - curr_ids
    for obj in prev_tracked:
        if obj["id"] in vanished_ids:
            parts.append(f"{obj['id']} ({_color_name(obj['color'])}) vanished.")

    for change in local_tracker.state_changes:
        if "appeared" not in change:
            parts.append(change)

    unchanged = [o for o in curr_tracked if o["id"] not in new_ids]
    if not parts and unchanged:
        parts.append("No change detected. " + "; ".join(f"{o['id']} unchanged" for o in unchanged[:3]))
    elif unchanged and not any("unchanged" in p for p in parts):
        unchanged_ids = [o["id"] for o in unchanged if o["id"] not in {c for c in local_tracker.state_changes}]
        if unchanged_ids:
            moved_ids = {c.split()[0] for c in local_tracker.state_changes if "moved" in c}
            truly_unchanged = [oid for oid in [o["id"] for o in unchanged] if oid not in moved_ids]
            if truly_unchanged:
                parts.append("; ".join(f"{oid} unchanged" for oid in truly_unchanged[:3]))

    if not parts:
        return "No change detected."

    result = " ".join(parts)
    return result[:500]
