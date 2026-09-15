"""
Smoke tests for the rule engine. Run with:
    python -m unittest src.rule_engine.test_engine -v

These are basic sanity checks (does it run, are totals plausible, are the
right categories triggered) rather than exhaustive verification against the
source guide's worked examples -- the guide's own sample hands (pages 38-40)
involve rule interactions (e.g. reusing suits across two "Mixed Dragon"
groupings) that go beyond what this v1 engine implements confidently; see
rulesets/taiwanese.py's module docstring for the flagged gaps. Treat this
file as a starting point to extend once real hands are being tested through
the photo pipeline.
"""
from __future__ import annotations

import unittest

from src.rule_engine.decomposition import decompose_concealed
from src.rule_engine.engine import score_hand
from src.rule_engine.models import GongKind, HandInput, Meld, MeldType, WinContext
from src.rule_engine.special_hands import detect_nico_nico, detect_thirteen_orphans


class TestDecomposition(unittest.TestCase):
    def test_simple_decomposition_count(self):
        # 5 sheungs + pair (17 tiles) should yield at least one valid decomposition.
        tiles = [
            "1s", "2s", "3s", "4s", "5s", "6s", "7p", "8p", "9p",
            "3m", "4m", "5m", "6p", "7p", "8p", "2p", "2p",
        ]
        decompositions = decompose_concealed(tiles)
        self.assertTrue(decompositions)
        for melds, pair in decompositions:
            self.assertEqual(len(melds), 5)
            self.assertEqual(pair[0], pair[1])

    def test_invalid_tile_count_yields_nothing(self):
        self.assertEqual(decompose_concealed(["1m", "2m", "3m"]), [])


class TestSpecialHands(unittest.TestCase):
    def test_nico_nico_shape(self):
        tiles = ["1m","1m","2m","2m","3m","3m","4m","4m","5m","5m","6m","6m","7m","7m","8m","8m","8m"]
        self.assertTrue(detect_nico_nico(tiles))

    def test_thirteen_orphans_shape(self):
        tiles = ["1m","9m","1p","9p","1s","9s","1z","2z","3z","4z","5z","6z","7z","2p","2p","2p","1m"]
        meld = detect_thirteen_orphans(tiles)
        self.assertIsNotNone(meld)


class TestEngine(unittest.TestCase):
    def test_all_sheung_no_honors_hand_scores(self):
        hand = HandInput(
            melds=[],
            concealed_tiles=[
                "1s", "2s", "3s", "4s", "5s", "6s", "7p", "8p", "9p",
                "3m", "4m", "5m", "6p", "7p", "8p", "2p", "2p",
            ],
            flowers=[],
            winning_tile="5m",
        )
        ctx = WinContext(seat_wind="E", round_wind="E", self_draw=True)
        result = score_hand(hand, ctx)
        labels = {l.label for l in result.lines}
        self.assertIn("All Sheung Hand (no flowers, no Winds & Dragons)", labels)
        self.assertIn("Base Point", labels)
        self.assertGreater(result.total_points, 5)

    def test_wind_pong_and_concealed_gong(self):
        hand = HandInput(
            melds=[
                Meld(MeldType.PONG, ("1z", "1z", "1z"), concealed=False),
                Meld(MeldType.GONG, ("5z", "5z", "5z", "5z"), concealed=True, gong_kind=GongKind.CONCEALED),
            ],
            concealed_tiles=["1m", "2m", "3m", "7s", "8s", "9s", "3p", "4p", "5p", "4p", "4p"],
            flowers=["spring", "summer"],
            winning_tile="9s",
        )
        ctx = WinContext(seat_wind="E", round_wind="E", self_draw=False)
        result = score_hand(hand, ctx)
        labels = {l.label for l in result.lines}
        self.assertIn("Pong of Winds - Player's seat", labels)
        self.assertIn("Pong of Winds - Wind of the Round", labels)
        self.assertIn("Concealed Gong", labels)

    def test_nico_nico_via_engine(self):
        hand = HandInput(
            melds=[],
            concealed_tiles=["1m","1m","2m","2m","3m","3m","4m","4m","5m","5m","6m","6m","7m","7m","8m","8m","8m"],
            flowers=[],
            winning_tile="8m",
        )
        ctx = WinContext(seat_wind="S", round_wind="E", self_draw=True)
        result = score_hand(hand, ctx)
        self.assertEqual(result.is_special_hand, "Nico Nico")
        self.assertGreaterEqual(result.total_points, 40 + 5)  # special hand fixed points + Base Point at minimum

    def test_invalid_hand_raises(self):
        hand = HandInput(melds=[], concealed_tiles=["1m", "2m", "3m"], flowers=[], winning_tile="3m")
        ctx = WinContext(seat_wind="S", round_wind="E", self_draw=True)
        with self.assertRaises(ValueError):
            score_hand(hand, ctx)


if __name__ == "__main__":
    unittest.main()
