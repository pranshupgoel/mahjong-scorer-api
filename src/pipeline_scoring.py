"""
Glue layer between "reviewed tile data" (what the app's review screen +
questionnaire produce, as plain JSON-friendly dicts) and the rule engine's
typed HandInput/WinContext/score_hand API.

This is the integration point flagged as missing in
claude/rule-engine-implementation.md's "Suggested next steps" -- pipeline.py
already does detector -> classifier -> TileGuess list; this module is what
turns a *user-confirmed* version of that (plus the WinContext questionnaire
answers) into something score_hand() can consume. It's deliberately
dict-in/dict-out so the FastAPI layer (api/main.py) can pass request JSON
straight through with minimal translation.

Expected input shape (see api/main.py's Pydantic models for the exact
schema the app sends):

    {
      "melds": [
        {"type": "pong", "tiles": ["5z","5z","5z"], "concealed": false, "gong_kind": "none"},
        {"type": "gong", "tiles": ["1z","1z","1z","1z"], "concealed": true, "gong_kind": "concealed"},
        ...
      ],
      "concealed_tiles": ["1s","2s","3s", "4s","5s","6s", "2m","2m"],
      "flowers": ["spring"],
      "winning_tile": "6s",
      "context": {
        "seat_wind": "E", "round_wind": "E", "self_draw": true,
        "win_from_flower_wall": false, "win_from_robbing_gong": false,
        "gong_gong_win": false, "tiles_on_table": null,
        "is_seabed": false, "is_earthly": false, "is_heavenly": false,
        "wait_type": "none", "closed_hand_declared": false,
        "triple_fan_dice": null, "dealer_win_streak": 0
      }
    }
"""
from __future__ import annotations

from typing import Any

from src.rule_engine.engine import score_hand
from src.rule_engine.models import (
    GongKind,
    HandInput,
    Meld,
    MeldType,
    ScoreResult,
    WaitType,
    WinContext,
)


def _meld_from_dict(d: dict[str, Any]) -> Meld:
    return Meld(
        type=MeldType(d["type"]),
        tiles=tuple(d["tiles"]),
        concealed=bool(d.get("concealed", False)),
        gong_kind=GongKind(d.get("gong_kind", "none")),
    )


def _context_from_dict(d: dict[str, Any]) -> WinContext:
    return WinContext(
        seat_wind=d["seat_wind"],
        round_wind=d["round_wind"],
        self_draw=bool(d["self_draw"]),
        win_from_flower_wall=bool(d.get("win_from_flower_wall", False)),
        win_from_robbing_gong=bool(d.get("win_from_robbing_gong", False)),
        gong_gong_win=bool(d.get("gong_gong_win", False)),
        tiles_on_table=d.get("tiles_on_table"),
        is_seabed=bool(d.get("is_seabed", False)),
        is_earthly=bool(d.get("is_earthly", False)),
        is_heavenly=bool(d.get("is_heavenly", False)),
        wait_type=WaitType(d.get("wait_type", "none")),
        closed_hand_declared=bool(d.get("closed_hand_declared", False)),
        triple_fan_dice=d.get("triple_fan_dice"),
        dealer_win_streak=int(d.get("dealer_win_streak", 0)),
    )


def hand_input_from_dict(d: dict[str, Any]) -> HandInput:
    return HandInput(
        melds=[_meld_from_dict(m) for m in d.get("melds", [])],
        concealed_tiles=list(d.get("concealed_tiles", [])),
        flowers=list(d.get("flowers", [])),
        winning_tile=d.get("winning_tile"),
    )


def score_result_to_dict(result: ScoreResult) -> dict[str, Any]:
    return {
        "total_points": result.total_points,
        "lines": [
            {"category": l.category, "label": l.label, "points": l.points}
            for l in result.lines
        ],
        "is_special_hand": result.is_special_hand,
        "is_chicken_hand": result.is_chicken_hand,
        "ruleset": result.ruleset,
        "notes": result.notes,
        "breakdown": result.breakdown(),
    }


def score_reviewed_hand(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Full glue: reviewed-tiles-and-questionnaire dict (see module docstring)
    -> score_hand() -> plain dict the API can return as JSON.

    Raises ValueError (propagated from score_hand) if the tiles don't form a
    valid winning hand -- the caller (api/main.py) turns this into a 422 with
    the error message, not a 500: it means the user's review-screen input
    doesn't represent a real hand, not a server bug.
    """
    hand = hand_input_from_dict(payload)
    context = _context_from_dict(payload["context"])
    ruleset = payload.get("ruleset", "taiwanese")
    result = score_hand(hand, context, ruleset=ruleset)
    return score_result_to_dict(result)
