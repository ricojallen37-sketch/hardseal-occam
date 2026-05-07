"""StateGraph -- directed graph of (state, action) -> state for systematic exploration.

Implements the "Just Explore" approach from the 3rd-place ARC-AGI-3 solution:
track unique frames as nodes, actions as edges, prioritize untested
(state, action) pairs, and find shortest paths to frontier states.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


class StateGraph:
    """Directed graph where nodes are state hashes and edges are actions."""

    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions
        self._initial_state: str | None = None
        # node -> {action -> successor_node}
        self._edges: dict[str, dict[int, str]] = {}

    @property
    def nodes(self) -> set[str]:
        """All known states."""
        nodes = set(self._edges.keys())
        for successors in self._edges.values():
            nodes.update(successors.values())
        return nodes

    def set_initial_state(self, state: str) -> None:
        """Mark a state as the initial/reset state for the current level."""
        self._initial_state = state

    def is_reset_transition(self, state: str, action: int) -> bool:
        """True if taking action from state leads back to the initial state."""
        if self._initial_state is None:
            return False
        return self.get_successor(state, action) == self._initial_state

    def reset(self) -> None:
        self._edges.clear()
        self._initial_state = None

    def add_transition(self, state: str, action: int, next_state: str) -> None:
        """Record that taking `action` in `state` leads to `next_state`."""
        if state not in self._edges:
            self._edges[state] = {}
        self._edges[state][action] = next_state
        if next_state not in self._edges:
            self._edges[next_state] = {}

    def get_successor(self, state: str, action: int) -> str | None:
        """Get the known successor for (state, action), or None."""
        return self._edges.get(state, {}).get(action)

    def tried_actions(self, state: str) -> set[int]:
        """Actions that have been tried from this state."""
        return set(self._edges.get(state, {}).keys())

    def untried_actions(self, state: str) -> set[int]:
        """Actions NOT yet tried from this state."""
        return set(range(self.n_actions)) - self.tried_actions(state)

    def is_frontier(self, state: str) -> bool:
        """True if state has untried actions."""
        return len(self.tried_actions(state)) < self.n_actions

    def frontier_states(self) -> set[str]:
        """All states with at least one untried action."""
        return {s for s in self.nodes if self.is_frontier(s)}

    def path_to_frontier(self, current: str) -> list[int] | None:
        """BFS shortest path (as action sequence) from current to nearest frontier."""
        if self.is_frontier(current):
            return []

        visited: set[str] = {current}
        queue: deque[tuple[str, list[int]]] = deque()

        for action, successor in self._edges.get(current, {}).items():
            if successor not in visited:
                visited.add(successor)
                queue.append((successor, [action]))

        while queue:
            node, path = queue.popleft()
            if self.is_frontier(node):
                return path
            for action, successor in self._edges.get(node, {}).items():
                if successor not in visited:
                    visited.add(successor)
                    queue.append((successor, path + [action]))

        return None

    def is_noop(self, state: str, action: int) -> bool:
        """True if taking action from state leads back to same state."""
        successor = self.get_successor(state, action)
        return successor == state

    def action_effectiveness(self) -> dict[int, float]:
        """Score each action by how often it causes state changes."""
        action_total: dict[int, int] = {}
        action_moves: dict[int, int] = {}
        for state, edges in self._edges.items():
            for action, successor in edges.items():
                action_total[action] = action_total.get(action, 0) + 1
                if successor != state:
                    action_moves[action] = action_moves.get(action, 0) + 1

        result: dict[int, float] = {}
        for a in range(self.n_actions):
            total = action_total.get(a, 0)
            moves = action_moves.get(a, 0)
            result[a] = moves / total if total > 0 else 0.5
        return result

    def suggest_action(self, current: str) -> int | None:
        """Suggest the best next action from current state."""
        untried = self.untried_actions(current)
        if untried:
            effectiveness = self.action_effectiveness()
            scored = [(a, effectiveness.get(a, 0.5)) for a in untried]
            best_score = max(s for _, s in scored)
            best_actions = [a for a, s in scored if s >= best_score - 0.01]
            import random
            return random.choice(best_actions)

        path = self.path_to_frontier(current)
        if path:
            return path[0]

        return None

    def edges_from(self, state: str) -> list[tuple[int, str]]:
        """Return all (action, next_state) pairs from a state."""
        return list(self._edges.get(state, {}).items())

    def path_to(self, target: str) -> list[int] | None:
        """BFS from the initial state to target. Returns action sequence or None."""
        if self._initial_state is None:
            return None
        if self._initial_state == target:
            return []
        return self.shortest_path_between(self._initial_state, target)

    def shortest_path_between(self, source: str, target: str) -> list[int] | None:
        """BFS from source to target. Returns action sequence or None if unreachable."""
        if source == target:
            return []

        visited: set[str] = {source}
        queue: deque[tuple[str, list[int]]] = deque()

        for action, successor in self._edges.get(source, {}).items():
            if successor == target:
                return [action]
            if successor not in visited:
                visited.add(successor)
                queue.append((successor, [action]))

        while queue:
            node, path = queue.popleft()
            for action, successor in self._edges.get(node, {}).items():
                if successor == target:
                    return path + [action]
                if successor not in visited:
                    visited.add(successor)
                    queue.append((successor, path + [action]))

        return None

    def summary(self) -> str:
        """Return a human-readable summary of the state graph."""
        s = self.stats()
        effectiveness = self.action_effectiveness()
        eff_parts = ", ".join(
            f"action {a}: {v * 100:.0f}%"
            for a, v in sorted(effectiveness.items())
        )
        return (
            f"State graph: {s['nodes']} nodes, {s['edges']} edges, "
            f"{s['frontier']} frontier states ({s['exploration_pct']:.0f}% explored). "
            f"Effective actions: {eff_parts}."
        )

    def stats(self) -> dict:
        """Return exploration statistics."""
        all_nodes = self.nodes
        n_nodes = len(all_nodes)
        total_possible = n_nodes * self.n_actions
        total_tried = sum(len(self._edges.get(s, {})) for s in all_nodes)
        n_edges = sum(len(succs) for succs in self._edges.values())
        frontier = self.frontier_states()

        return {
            "nodes": n_nodes,
            "edges": n_edges,
            "frontier": len(frontier),
            "exploration_pct": (total_tried / total_possible * 100) if total_possible > 0 else 0.0,
        }
