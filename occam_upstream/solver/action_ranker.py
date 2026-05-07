"""Action effectiveness tracker for BFS exploration.

Tracks which actions produce state changes during probing,
then ranks actions so effective ones are tried first.
"""


class ActionRanker:
    def __init__(self):
        self._effective = set()
        self.experience_buffer = []

    def reset(self):
        self._effective = set()
        self.experience_buffer = []

    def record(self, frame, action, changed):
        self.experience_buffer.append((frame, action, changed))
        if changed:
            self._effective.add(action)

    def rank(self, frame, actions):
        """Return actions with effective ones first."""
        effective = [a for a in actions if a in self._effective]
        rest = [a for a in actions if a not in self._effective]
        return effective + rest
