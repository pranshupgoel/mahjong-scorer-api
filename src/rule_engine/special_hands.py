"""
Detectors for the three structurally-different Special Hands (Nico Nico,
13 Orphans, 16 Orphans). These do NOT follow the normal 5-melds+pair shape,
so they need their own recognizers rather than going through
decomposition.decompose_concealed.

All three are "fully concealed" per the source guide -- callers should pass
the whole hand (winning tile included) as HandInput.concealed_tiles with an
empty `melds` list (flowers/exposed gongs aside).

Chicken Hand is NOT here: it isn't a distinct tile shape, it's a post-hoc
scoring override (see rulesets/taiwanese.py) applied to an otherwise-normal
hand whose points happen to sum to only 1 before the Base Point.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Optional

from src.rule_engine.models import Meld, MeldType, Tile, tile_rank, tile_suit

ORPHAN_13_TYPES = {
    "1m", "9m", "1p", "9p", "1s", "9s",
    "1z", "2z", "3z", "4z", "5z", "6z", "7z",
}


def _single_meld_from(tiles: list[Tile]) -> Optional[Meld]:
    """If these exactly-3 tiles form a valid Pong or Sheung, return it."""
    if len(tiles) != 3:
        return None
    if len(set(tiles)) == 1:
        return Meld(MeldType.PONG, tuple(tiles), concealed=True)
    suits = {tile_suit(t) for t in tiles}
    if len(suits) == 1 and None not in suits:
        ranks = sorted(tile_rank(t) for t in tiles)
        if ranks == [ranks[0], ranks[0] + 1, ranks[0] + 2] and len(set(ranks)) == 3:
            ordered = sorted(tiles, key=lambda t: tile_rank(t))
            return Meld(MeldType.SHEUNG, tuple(ordered), concealed=True)
    return None


def detect_nico_nico(tiles: list[Tile]) -> bool:
    """
    7 pairs + 1 Pong, fully concealed, fully self-drawn except the winning
    tile. Canonical shape: 8 distinct tile-types, one with 3 copies (the
    Pong) and seven with 2 copies each (the pairs) -- 3 + 7*2 = 17 tiles.
    """
    if len(tiles) != 17:
        return False
    counter = Counter(tiles)
    counts = sorted(counter.values(), reverse=True)
    return counts == [3, 2, 2, 2, 2, 2, 2, 2]


def detect_thirteen_orphans(tiles: list[Tile]) -> Optional[Meld]:
    """
    1 of each of the 6 terminals + 1 of each of the 7 honors (13 distinct
    singles), + 1 more Pong/Sheung of ANY tiles, + 1 duplicate of one of the
    13 orphan types to form the Eyes. 13 + 3 + 1 = 17 tiles.
    Returns the extra Pong/Sheung Meld if matched, else None.
    """
    if len(tiles) != 17:
        return None
    counter = Counter(tiles)
    if not ORPHAN_13_TYPES.issubset(counter.keys()):
        return None
    orphan_counts = {t: counter[t] for t in ORPHAN_13_TYPES}
    pair_types = [t for t, c in orphan_counts.items() if c == 2]
    single_types = [t for t, c in orphan_counts.items() if c == 1]
    if not (len(pair_types) == 1 and len(single_types) == 12):
        return None

    remaining = counter.copy()
    for t in ORPHAN_13_TYPES:
        remaining[t] -= orphan_counts[t]
        if remaining[t] == 0:
            del remaining[t]
    remaining_tiles = list(remaining.elements())
    return _single_meld_from(remaining_tiles)


def detect_sixteen_orphans(tiles: list[Tile]) -> Optional[list[tuple[Tile, Tile, Tile]]]:
    """
    3 "spread" Sheungs (one per number suit -- 9 tiles), each made of 3
    tiles from that suit that are pairwise unrelated by at least 2 ranks
    (e.g. 1-4-7, no two ranks within 1 of each other), + 1 of each Wind &
    Dragon tile (7 singles), + 1 duplicate of any of those 16 orphan tile
    *types* (9 spread-tile ranks-per-suit + 7 honors) to form the Eyes.
    9 + 7 + 1 = 17 tiles.

    NOTE: the source guide's exact wording for the "spread Sheung" rule
    ("3 tiles unrelated by at least 2 numbers") is somewhat ambiguous about
    whether ranks must be pairwise >=2 apart (this implementation) or
    something stricter/looser. This is a rare, best-effort hand -- verify
    against a real example before trusting it in production.

    Returns the 3 spread-tile-sets (plain tuples, NOT Meld objects -- these
    are not real Sheungs and would fail Meld's consecutive-rank validation)
    if matched, else None.
    """
    if len(tiles) != 17:
        return None
    counter = Counter(tiles)
    honors = {"1z", "2z", "3z", "4z", "5z", "6z", "7z"}
    if not honors.issubset(counter.keys()):
        return None

    remaining = counter.copy()
    for t in honors:
        remaining[t] -= 1
        if remaining[t] == 0:
            del remaining[t]

    # Try every way to pick a "pair partner" duplicate among remaining tiles,
    # then check the rest splits into exactly 3 valid spread-sets, one per suit.
    for pair_tile, count in list(remaining.items()):
        if count < 2:
            continue
        after_pair = remaining.copy()
        after_pair[pair_tile] -= 2
        if after_pair[pair_tile] == 0:
            del after_pair[pair_tile]

        by_suit: dict[str, list[Tile]] = {"m": [], "p": [], "s": []}
        ok = True
        for t in after_pair.elements():
            suit = tile_suit(t)
            if suit is None:
                ok = False
                break
            by_suit[suit].append(t)
        if not ok:
            continue

        spread_sets = []
        valid = True
        for suit, suit_tiles in by_suit.items():
            if len(suit_tiles) != 3:
                valid = False
                break
            ranks = sorted(tile_rank(t) for t in suit_tiles)
            if len(set(ranks)) != 3 or any(ranks[i + 1] - ranks[i] < 2 for i in range(2)):
                valid = False
                break
            spread_sets.append(tuple(sorted(suit_tiles, key=tile_rank)))
        if valid and len(spread_sets) == 3:
            return spread_sets

    return None
