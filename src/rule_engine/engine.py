"""
Rule engine entry point.

    from src.rule_engine.engine import score_hand
    from src.rule_engine.models import HandInput, WinContext, Meld, MeldType, GongKind

    result = score_hand(
        HandInput(
            melds=[],  # no pre-declared exposed melds/gongs in this example
            concealed_tiles=["1s","2s","3s", "4s","5s","6s", "7s","8s","9s",
                              "1p","2p","3p", "7p","8p","9p", "2m","2m"],
            flowers=["spring"],
            winning_tile="9s",
        ),
        WinContext(seat_wind="E", round_wind="E", self_draw=True),
    )
    print(result.total_points)
    print(result.breakdown())

See models.py for the full HandInput / WinContext field reference, and
rulesets/taiwanese.py's module docstring for the confirmed scoring policy
and known limitations (Four-in-N-ways, Nico Nico gong variants, etc.).
"""
from __future__ import annotations

from src.rule_engine.models import HandInput, ScoreResult, WinContext
from src.rule_engine.rulesets import get_ruleset


def score_hand(hand: HandInput, context: WinContext, ruleset: str = "taiwanese") -> ScoreResult:
    """Score a completed winning hand. Raises ValueError if the tiles don't form a valid hand."""
    return get_ruleset(ruleset).score(hand, context)
