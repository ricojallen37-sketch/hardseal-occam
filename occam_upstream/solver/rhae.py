def compute_rhae(ai_actions: int, human_baseline: int) -> float:
    if ai_actions == 0:
        return 0.0
    ratio = min(1.0, human_baseline / ai_actions)
    return ratio ** 2

def compute_game_score(level_scores: list[float]) -> float:
    if not level_scores:
        return 0.0
    total_weight = sum(range(1, len(level_scores) + 1))
    weighted_sum = sum(score * (i + 1) for i, score in enumerate(level_scores))
    return weighted_sum / total_weight

def weighted_game_score(level_scores: list[float], total_levels: int) -> float:
    """Weighted RHAE score matching ARC-AGI-3 formula.

    E(e) = sum(l * S(l,e)) / sum(1..n) where l is 1-indexed level number.
    """
    if not level_scores or total_levels == 0:
        return 0.0
    weight_sum = sum(range(1, total_levels + 1))
    score = sum((i + 1) * s for i, s in enumerate(level_scores))
    return score / weight_sum


def compute_giveup_budget(baseline: int, n_actions: int = 4) -> int:
    """Dynamic give-up budget. Short baselines get higher multipliers."""
    min_budget = n_actions + 30
    multiplier = max(5.0, (n_actions / max(baseline, 1)) + 3.0)
    return max(int(baseline * multiplier), min_budget)

def should_give_up(actions_taken: int, baseline: int, n_available_actions: int = 4) -> bool:
    budget = compute_giveup_budget(baseline, n_available_actions)
    return actions_taken >= budget
