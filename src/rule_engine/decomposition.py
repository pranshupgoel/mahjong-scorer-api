"""
Combinatorial decomposition of a flat, ungrouped list of concealed tiles into
every valid (Sheung/Pong melds + pair) grouping.

Exposed melds and gongs are never ambiguous -- they're given explicitly by
the caller (see HandInput.melds docstring). What IS ambiguous is a fully- or
partially-concealed hand's remaining tiles: the same 13/16 tiles can often be
grouped into sets in more than one valid way, and different groupings can
score differently (e.g. one grouping might complete a Dragon Sequence, another
might not). The engine tries all of them and keeps whichever scores highest --
the standard "the player gets their best-scoring interpretation" convention.
"""
from __future__ import annotations

from collections import Counter

from src.rule_engine.models import Meld, MeldType, Tile, tile_rank, tile_suit, sort_key


def _decompose_sets(counter: Counter) -> list[list[Meld]]:
    """All ways to fully partition a multiset of tiles into Sheung/Pong melds."""
    if not counter:
        return [[]]

    tile = min(counter, key=sort_key)
    results: list[list[Meld]] = []

    # Option 1: a concealed Pong of this tile.
    if counter[tile] >= 3:
        remaining = counter.copy()
        remaining[tile] -= 3
        if remaining[tile] == 0:
            del remaining[tile]
        for rest in _decompose_sets(remaining):
            results.append([Meld(MeldType.PONG, (tile, tile, tile), concealed=True)] + rest)

    # Option 2: a concealed Sheung starting at this tile (number suits only).
    suit = tile_suit(tile)
    if suit is not None:
        rank = tile_rank(tile)
        if rank <= 7:
            t2, t3 = f"{rank + 1}{suit}", f"{rank + 2}{suit}"
            if counter.get(t2, 0) > 0 and counter.get(t3, 0) > 0:
                remaining = counter.copy()
                for t in (tile, t2, t3):
                    remaining[t] -= 1
                    if remaining[t] == 0:
                        del remaining[t]
                for rest in _decompose_sets(remaining):
                    results.append([Meld(MeldType.SHEUNG, (tile, t2, t3), concealed=True)] + rest)

    return results


def decompose_concealed(tiles: list[Tile]) -> list[tuple[list[Meld], tuple[Tile, Tile]]]:
    """
    Every valid (melds, pair) decomposition of `tiles` where melds are all
    Sheung/Pong and exactly one tile-type is set aside as the pair. Returns
    an empty list if no valid decomposition exists (e.g. wrong tile count,
    or tiles that just don't form valid sets -- a genuine misread from the
    classifier should surface this way rather than silently scoring wrong).
    """
    counter = Counter(tiles)
    decompositions: list[tuple[list[Meld], tuple[Tile, Tile]]] = []
    seen = set()

    for pair_tile, count in list(counter.items()):
        if count < 2:
            continue
        remaining = counter.copy()
        remaining[pair_tile] -= 2
        if remaining[pair_tile] == 0:
            del remaining[pair_tile]
        for melds in _decompose_sets(remaining):
            key = (pair_tile, tuple(sorted(tuple(sorted(m.tiles, key=sort_key)) for m in melds)))
            if key in seen:
                continue
            seen.add(key)
            decompositions.append((melds, (pair_tile, pair_tile)))

    return decompositions
