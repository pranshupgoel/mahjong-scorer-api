"""
Tests for the table-context rules (dealer win streak, Closing Hand,
East-discard tile, server-side declare threshold). Run with:
    python -m unittest src.rule_engine.test_context_rules -v

Values come from the project's scoring reference (sections 0, 19 and 23):
streak bonus +3/+5/+7 and threshold 17/15/13 for 1/2/3 consecutive dealer wins,
Closing Hand +5 in West/North rounds, East Seat tile +1 when East wins or
discards the winning tile.
"""
from __future__ import annotations

import unittest

from src.pipeline_scoring import score_reviewed_hand
from src.rule_engine.engine import score_hand
from src.rule_engine.models import HandInput, Meld, MeldType, WinContext

SHEUNG_TILES = [
    "1s", "2s", "3s", "4s", "5s", "6s", "7p", "8p", "9p",
    "3m", "4m", "5m", "6p", "7p", "8p", "2p", "2p",
]
NICO_TILES = ["1m", "1m", "2m", "2m", "3m", "3m", "4m", "4m", "5m", "5m", "6m", "6m", "7m", "7m", "8m", "8m", "8m"]


def concealed_hand() -> HandInput:
    return HandInput(melds=[], concealed_tiles=list(SHEUNG_TILES), flowers=[], winning_tile="5m")


def exposed_hand() -> HandInput:
    melds = [
        Meld(MeldType.SHEUNG, ("1s", "2s", "3s"), concealed=False),
        Meld(MeldType.SHEUNG, ("4s", "5s", "6s"), concealed=False),
        Meld(MeldType.SHEUNG, ("7p", "8p", "9p"), concealed=False),
        Meld(MeldType.SHEUNG, ("3m", "4m", "5m"), concealed=False),
        Meld(MeldType.SHEUNG, ("6p", "7p", "8p"), concealed=False),
    ]
    return HandInput(melds=melds, concealed_tiles=["2p", "2p"], flowers=[], winning_tile="2p")


def labels(result):
    return [l.label for l in result.lines]


def total_with(hand, **ctx):
    return score_hand(hand, WinContext(**ctx))


class TestDealerWinStreak(unittest.TestCase):
    def test_bonus_and_threshold_by_streak(self):
        base = total_with(concealed_hand(), seat_wind="E", round_wind="E", self_draw=True)
        self.assertEqual(base.declare_threshold, 20)
        for streak, bonus, threshold in [(1, 3, 17), (2, 5, 15), (3, 7, 13)]:
            r = total_with(concealed_hand(), seat_wind="E", round_wind="E", self_draw=True, dealer_win_streak=streak)
            self.assertEqual(r.total_points - base.total_points, bonus, f"streak {streak}")
            self.assertEqual(r.declare_threshold, threshold, f"streak {streak}")
            self.assertIn(f"Dealer win streak ({streak} in a row)", labels(r))

    def test_streak_above_three_is_capped_with_a_note(self):
        r3 = total_with(concealed_hand(), seat_wind="E", round_wind="E", self_draw=True, dealer_win_streak=3)
        r5 = total_with(concealed_hand(), seat_wind="E", round_wind="E", self_draw=True, dealer_win_streak=5)
        self.assertEqual(r5.total_points, r3.total_points)
        self.assertEqual(r5.declare_threshold, 13)
        self.assertTrue(any("treated as 3" in n for n in r5.notes))

    def test_non_dealer_streak_is_ignored(self):
        plain = total_with(concealed_hand(), seat_wind="S", round_wind="E", self_draw=True)
        streak = total_with(concealed_hand(), seat_wind="S", round_wind="E", self_draw=True, dealer_win_streak=2)
        self.assertEqual(plain.total_points, streak.total_points)
        self.assertEqual(streak.declare_threshold, 20)

    def test_special_hand_gets_the_streak_bonus_too(self):
        hand = HandInput(melds=[], concealed_tiles=list(NICO_TILES), flowers=[], winning_tile="8m")
        base = total_with(hand, seat_wind="E", round_wind="E", self_draw=True)
        streak = total_with(hand, seat_wind="E", round_wind="E", self_draw=True, dealer_win_streak=2)
        self.assertEqual(streak.is_special_hand, "Nico Nico")
        self.assertEqual(streak.total_points - base.total_points, 5)


class TestClosingHand(unittest.TestCase):
    def test_adds_five_in_west_and_north_rounds(self):
        for rnd in ("W", "N"):
            plain = total_with(concealed_hand(), seat_wind="S", round_wind=rnd, self_draw=True)
            closed = total_with(concealed_hand(), seat_wind="S", round_wind=rnd, self_draw=True, closed_hand_declared=True)
            self.assertEqual(closed.total_points - plain.total_points, 5, rnd)
            self.assertIn("Closing Hand (West/North round)", labels(closed))

    def test_ignored_with_a_note_in_east_and_south_rounds(self):
        for rnd in ("E", "S"):
            plain = total_with(concealed_hand(), seat_wind="S", round_wind=rnd, self_draw=True)
            closed = total_with(concealed_hand(), seat_wind="S", round_wind=rnd, self_draw=True, closed_hand_declared=True)
            self.assertEqual(closed.total_points, plain.total_points, rnd)
            self.assertTrue(any("Closing Hand" in n for n in closed.notes), rnd)

    def test_closing_a_fully_exposed_hand_forfeits_the_exposed_points(self):
        plain = total_with(exposed_hand(), seat_wind="S", round_wind="W", self_draw=False)
        closed = total_with(exposed_hand(), seat_wind="S", round_wind="W", self_draw=False, closed_hand_declared=True)
        self.assertIn("Fully Exposed Hand (win from Discard)", labels(plain))
        self.assertNotIn("Fully Exposed Hand (win from Discard)", labels(closed))
        self.assertEqual(closed.total_points - plain.total_points, 5 - 15)


class TestEastDiscard(unittest.TestCase):
    def test_east_discard_adds_one_for_a_non_dealer_discard_win(self):
        plain = total_with(concealed_hand(), seat_wind="S", round_wind="E", self_draw=False)
        east = total_with(concealed_hand(), seat_wind="S", round_wind="E", self_draw=False, winning_tile_from_east_discard=True)
        self.assertEqual(east.total_points - plain.total_points, 1)
        self.assertIn("East Seat/Dealer tile (East discarded the winning tile)", labels(east))

    def test_ignored_for_self_draw_and_for_the_dealer(self):
        sd_plain = total_with(concealed_hand(), seat_wind="S", round_wind="E", self_draw=True)
        sd = total_with(concealed_hand(), seat_wind="S", round_wind="E", self_draw=True, winning_tile_from_east_discard=True)
        self.assertEqual(sd.total_points, sd_plain.total_points)
        self.assertTrue(any("East discarded" in n for n in sd.notes))
        d_plain = total_with(concealed_hand(), seat_wind="E", round_wind="E", self_draw=False)
        d = total_with(concealed_hand(), seat_wind="E", round_wind="E", self_draw=False, winning_tile_from_east_discard=True)
        self.assertEqual(d.total_points, d_plain.total_points)  # East tile already counted once


class TestApiGlue(unittest.TestCase):
    def payload(self, **ctx):
        context = {"seat_wind": "E", "round_wind": "E", "self_draw": True}
        context.update(ctx)
        return {"melds": [], "concealed_tiles": list(SHEUNG_TILES), "flowers": [], "winning_tile": "5m", "context": context}

    def test_response_carries_declare_threshold(self):
        self.assertEqual(score_reviewed_hand(self.payload())["declare_threshold"], 20)
        self.assertEqual(score_reviewed_hand(self.payload(dealer_win_streak=2))["declare_threshold"], 15)

    def test_new_context_field_is_optional_and_read(self):
        base = score_reviewed_hand(self.payload(seat_wind="S", self_draw=False))
        east = score_reviewed_hand(self.payload(seat_wind="S", self_draw=False, winning_tile_from_east_discard=True))
        self.assertEqual(east["total_points"] - base["total_points"], 1)


if __name__ == "__main__":
    unittest.main()
