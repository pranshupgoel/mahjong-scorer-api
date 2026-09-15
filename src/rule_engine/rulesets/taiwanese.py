"""
Core Taiwanese Mahjong scoring ruleset, per the source guide extracted into
the project doc `taiwanese-mahjong-scoring-reference.md`.

Design decisions confirmed with Pranshu (2026-09-15):
  - Tiered categories (Win within 4/7/10 tiles, etc.): take the single best
    matching tier, never stack multiple tiers of the same family.
  - Bonus stacking across DIFFERENT categories (e.g. per-pong Wind points vs
    the full-hand Little/Big Winds bonus; Concealed Pongs count vs All Pong
    Hand; Terminal-set bonuses vs the full-hand Terminals categories): all
    additive, confirmed explicitly for winds/dragons/pongs/terminals and
    applied consistently to the remaining, structurally-analogous cases.
  - Anything not visually inferable from the tile photo (seat, round wind,
    self-draw vs discard, Triple Fan, tiles-on-table count, etc.) comes from
    WinContext, which the app should populate via a short user questionnaire
    -- not auto-tracked game state.

KNOWN LIMITATIONS (flagged rather than silently guessed -- see the project
doc's "Known ambiguous stacking cases" section and confirm before shipping):
  - "Four in 2/3/4 ways" is NOT implemented. Correctly detecting it requires
    enumerating every alternate valid meld-grouping of the same tiles, which
    the decomposition engine *can* surface (see engine.py) but scoring it
    precisely needs a worked real-hand example -- flagged as TODO.
  - Nico Nico's +10 (1 Gong) / +25 (2 Gongs) refinements are NOT implemented
    -- only the base 40-point Nico Nico shape is detected. The source page's
    caption ("2 pairs - not declared as a Concealed Gong") is ambiguous
    about the exact tile-count semantics; needs a worked example.
  - 16 Orphans' "3 tiles unrelated by at least 2 numbers" rule is
    implemented as "pairwise rank gap >= 2" -- confirm this reading.
  - Flower "Mixed bouquet" vs "Full bouquet" (pages 8/31): implemented as
    Full = all 4 of one flower group (season OR plant), Mixed = all 4 ranks
    1-4 covered using tiles from both groups. Confirm against the source's
    red/blue bouquet framing.
  - Multi-round game-state penalties (Chasing Tiles, False Mahjong, Dealer
    Die table payments, closing-hand mechanics) are out of scope for this
    module -- it scores one completed winning hand, per the agreed v1 scope.
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional

from src.rule_engine.decomposition import decompose_concealed
from src.rule_engine.models import (
    DRAGON_TILES,
    GREEN_DRAGON,
    RED_DRAGON,
    WHITE_DRAGON,
    HandInput,
    Meld,
    MeldType,
    ScoreLine,
    ScoreResult,
    Tile,
    WinContext,
    flower_seat_index,
    is_dragon,
    is_flower,
    is_good_eye_tile,
    is_honor,
    is_terminal,
    is_wind,
    tile_rank,
    tile_suit,
)
from src.rule_engine.special_hands import (
    detect_nico_nico,
    detect_sixteen_orphans,
    detect_thirteen_orphans,
)

_SEAT_INDEX = {"E": 1, "S": 2, "W": 3, "N": 4}
_TIER_POINTS_2_5 = {2: 5, 3: 10, 4: 15, 5: 80}  # Concealed Pongs count


def _all_tiles(melds: list[Meld], pair: tuple[Tile, Tile]) -> list[Tile]:
    tiles: list[Tile] = []
    for m in melds:
        tiles.extend(m.tiles)
    tiles.extend(pair)
    return tiles


def _pong_gong_melds(melds: list[Meld]) -> list[Meld]:
    return [m for m in melds if m.type in (MeldType.PONG, MeldType.GONG)]


# --------------------------------------------------------------------------
# Category scorers -- each takes the fully-assembled candidate hand and
# returns the ScoreLines it contributes. All additive per the confirmed
# stacking policy above.
# --------------------------------------------------------------------------


def _score_flowers(flowers: list[Tile], context: WinContext) -> list[ScoreLine]:
    lines = []
    if not flowers:
        lines.append(ScoreLine("flowers", "No Flower tiles", 1))
    seat_idx = _SEAT_INDEX[context.seat_wind]
    for f in flowers:
        lines.append(ScoreLine("flowers", f"Flower ({f})", 1))
        if flower_seat_index(f) == seat_idx:
            lines.append(ScoreLine("flowers", "Flower matches player's seat", 1))

    from src.rule_engine.models import PLANT_FLOWERS, SEASON_FLOWERS

    have_season = set(flowers) & set(SEASON_FLOWERS)
    have_plant = set(flowers) & set(PLANT_FLOWERS)
    if len(have_season) == 4:
        lines.append(ScoreLine("flowers", "Full bouquet (all 4 seasons)", 10))
    if len(have_plant) == 4:
        lines.append(ScoreLine("flowers", "Full bouquet (all 4 plants)", 10))
    if len(have_season) < 4 and len(have_plant) < 4:
        ranks_covered = {SEASON_FLOWERS.index(f) for f in have_season} | {
            PLANT_FLOWERS.index(f) for f in have_plant
        }
        if len(ranks_covered) == 4 and have_season and have_plant:
            lines.append(ScoreLine("flowers", "Mixed bouquet (ranks 1-4, both groups)", 5))
    return lines


def _score_honors_absence(melds: list[Meld], pair: tuple[Tile, Tile], flowers: list[Tile]) -> list[ScoreLine]:
    all_tiles = _all_tiles(melds, pair)
    no_honors = not any(is_honor(t) for t in all_tiles)
    no_flowers = not flowers
    lines = []
    if no_honors:
        lines.append(ScoreLine("flowers", "No Wind and Dragon tiles", 1))
    if no_honors and no_flowers:
        lines.append(ScoreLine("flowers", "No Flower, Wind and Dragon tiles", 5))
        if all(m.type == MeldType.SHEUNG for m in melds):
            lines.append(ScoreLine("flowers", "No Flower/Wind/Dragon in an ALL Sheung Hand", 15))
    return lines


def _score_winds(melds: list[Meld], pair: tuple[Tile, Tile], context: WinContext) -> list[ScoreLine]:
    lines = []
    wind_melds = [m for m in _pong_gong_melds(melds) if is_wind(m.tile)]
    from src.rule_engine.models import WIND_NAME

    for m in wind_melds:
        lines.append(ScoreLine("winds", "1 Pong of Winds", 1))
        if WIND_NAME[m.tile] == context.seat_wind:
            lines.append(ScoreLine("winds", "Pong of Winds - Player's seat", 1))
        if WIND_NAME[m.tile] == context.round_wind:
            lines.append(ScoreLine("winds", "Pong of Winds - Wind of the Round", 1))

    count = len(wind_melds)
    pair_is_wind = is_wind(pair[0])
    if count == 4:
        lines.append(ScoreLine("winds", "Big 4 Winds/Happiness", 60))
    elif count == 3 and pair_is_wind:
        lines.append(ScoreLine("winds", "Little 4 Winds/Happiness", 50))
    elif count == 3:
        lines.append(ScoreLine("winds", "Big 3 Winds/Happiness", 30))
    elif count == 2 and pair_is_wind:
        lines.append(ScoreLine("winds", "Little 3 Winds/Happiness", 15))
    return lines


def _score_dragons(melds: list[Meld], pair: tuple[Tile, Tile]) -> list[ScoreLine]:
    lines = []
    dragon_melds = [m for m in _pong_gong_melds(melds) if is_dragon(m.tile)]
    for _ in dragon_melds:
        lines.append(ScoreLine("dragons", "1 Pong of Dragons", 2))
    count = len(dragon_melds)
    pair_is_dragon = is_dragon(pair[0])
    if count == 3:
        lines.append(ScoreLine("dragons", "Big Dragons", 40))
    elif count == 2 and pair_is_dragon:
        lines.append(ScoreLine("dragons", "Little Dragons", 20))
    return lines


def _score_gongs(melds: list[Meld]) -> list[ScoreLine]:
    from src.rule_engine.models import GongKind

    lines = []
    for m in melds:
        if m.type != MeldType.GONG:
            continue
        if m.gong_kind == GongKind.OPEN:
            lines.append(ScoreLine("gongs", "Open Gong", 1))
        elif m.gong_kind == GongKind.CONCEALED:
            lines.append(ScoreLine("gongs", "Concealed Gong", 2))
    return lines


def _score_dragon_sequence(melds: list[Meld]) -> list[ScoreLine]:
    lines = []
    sheungs = [m for m in melds if m.type == MeldType.SHEUNG]
    for combo in combinations(sheungs, 3):
        starts = {m.start_rank for m in combo}
        if starts != {1, 4, 7}:
            continue
        suits = {m.suit for m in combo}
        concealed_all = all(m.concealed for m in combo)
        if len(suits) == 1:
            pts, tag = (20, "Concealed") if concealed_all else (10, "Exposed/Partially Concealed")
            lines.append(ScoreLine("dragon_sequence", f"Pure Dragon Sequence ({tag})", pts))
        elif len(suits) == 3:
            pts, tag = (10, "Concealed") if concealed_all else (5, "Exposed/Partially Concealed")
            lines.append(ScoreLine("dragon_sequence", f"Mixed Dragon Sequence ({tag})", pts))
    return lines


def _score_neighbours(melds: list[Meld]) -> list[ScoreLine]:
    """Uncle Pongs: same-suit sequential Pongs, e.g. 111,222,333."""
    lines = []
    by_suit: dict[str, set[int]] = {}
    for m in _pong_gong_melds(melds):
        if m.suit is None:
            continue
        by_suit.setdefault(m.suit, set()).add(tile_rank(m.tile))

    tier = {2: 5, 3: 15, 4: 30, 5: 60}
    for suit, ranks in by_suit.items():
        best_run = 0
        for r in sorted(ranks):
            if r - 1 not in ranks:
                run = 1
                while r + run in ranks:
                    run += 1
                best_run = max(best_run, run)
        if best_run >= 2:
            lines.append(ScoreLine("neighbours", f"Neighbours/Uncle Pongs x{best_run} (suit {suit})", tier[min(best_run, 5)]))
    return lines


def _score_brother_sheungs(melds: list[Meld]) -> list[ScoreLine]:
    lines = []
    counts: dict[tuple[str, int], int] = {}
    for m in melds:
        if m.type != MeldType.SHEUNG:
            continue
        counts[(m.suit, m.start_rank)] = counts.get((m.suit, m.start_rank), 0) + 1
    tier = {2: 5, 3: 15, 4: 30}
    for (suit, start), n in counts.items():
        if n >= 2:
            lines.append(ScoreLine("brother_sheungs", f"Brother Sheungs x{n} ({start}-{start+2}, suit {suit})", tier[min(n, 4)]))
    return lines


def _score_sister_sheungs(melds: list[Meld]) -> list[ScoreLine]:
    """
    Same run, different suits. Per guideline #12/page14: don't also take
    Sister points for start ranks 1 or 7 -- those are reserved for the
    Terminal Mixed/Pure Sheung Sets bonuses (see _score_terminals).
    """
    lines = []
    by_start: dict[int, set[str]] = {}
    for m in melds:
        if m.type != MeldType.SHEUNG or m.start_rank in (1, 7):
            continue
        by_start.setdefault(m.start_rank, set()).add(m.suit)
    tier = {2: 3, 3: 10}
    for start, suits in by_start.items():
        n = len(suits)
        if n >= 2:
            lines.append(ScoreLine("sister_sheungs", f"Sister Sheungs x{n} (start {start})", tier[min(n, 3)]))
    return lines


def _score_sister_pongs(melds: list[Meld]) -> list[ScoreLine]:
    lines = []
    by_rank: dict[int, set[str]] = {}
    for m in _pong_gong_melds(melds):
        if m.suit is None:
            continue
        by_rank.setdefault(tile_rank(m.tile), set()).add(m.suit)
    tier = {2: 3, 3: 10}
    for rank, suits in by_rank.items():
        n = len(suits)
        if n >= 2:
            lines.append(ScoreLine("sister_pongs", f"Sister Pongs x{n} (rank {rank})", tier[min(n, 3)]))
    return lines


def _score_concealed_pongs(melds: list[Meld]) -> list[ScoreLine]:
    count = sum(1 for m in melds if m.counts_as_concealed_pong)
    if count >= 2:
        return [ScoreLine("concealed_pongs", f"{count} Concealed Pongs", _TIER_POINTS_2_5[min(count, 5)])]
    return []


def _score_terminals(melds: list[Meld], pair: tuple[Tile, Tile]) -> list[ScoreLine]:
    lines = []
    all_tiles = _all_tiles(melds, pair)
    if not any(is_terminal(t) or is_honor(t) for t in all_tiles):
        lines.append(ScoreLine("terminals", "No Terminal tiles", 5))

    terminal_pongs = [m for m in _pong_gong_melds(melds) if m.is_terminal_set]
    if len(terminal_pongs) >= 2:
        lines.append(ScoreLine("terminals", "2 Terminal Pong Sets", 15))
    elif len(terminal_pongs) == 1:
        lines.append(ScoreLine("terminals", "1 Terminal Pong Set", 3))

    terminal_sheungs = [m for m in melds if m.type == MeldType.SHEUNG and m.is_terminal_set]
    by_suit: dict[str, set[int]] = {}
    for m in terminal_sheungs:
        by_suit.setdefault(m.suit, set()).add(m.start_rank)
    complete_suits = [s for s, starts in by_suit.items() if {1, 7} <= starts]
    if len(complete_suits) >= 2:
        pts = 20 if len(complete_suits) == 1 else 15  # all-same-suit (Pure) vs mixed-suit
        lines.append(ScoreLine("terminals", "2 Terminal Mixed Sheung Sets" if pts == 15 else "2 Terminal Pure Sheung Sets", pts))
    elif len(complete_suits) == 1:
        lines.append(ScoreLine("terminals", "1 Terminal Sheung Set", 3))
    return lines


def _score_step_up(melds: list[Meld]) -> list[ScoreLine]:
    sheungs = [m for m in melds if m.type == MeldType.SHEUNG]
    for combo in combinations(sheungs, 3):
        starts = sorted(m.start_rank for m in combo)
        if starts == [starts[0], starts[0] + 1, starts[0] + 2]:
            return [ScoreLine("step_up", "Step Up Sheungs (3 sets)", 5)]
    return []


def _hand_flags(melds: list[Meld]):
    is_concealed_hand = all(m.concealed or m.type == MeldType.GONG for m in melds)
    is_fully_exposed_hand = all(not m.concealed for m in melds)
    return is_concealed_hand, is_fully_exposed_hand


def _score_hand_composition(melds: list[Meld], pair: tuple[Tile, Tile], flowers: list[Tile]) -> list[ScoreLine]:
    lines = []
    suits_used = {m.suit for m in melds if m.suit}
    if tile_suit(pair[0]):
        suits_used.add(tile_suit(pair[0]))
    honors_used = any(m.is_honor_set for m in melds) or is_honor(pair[0])

    is_concealed_hand, is_fully_exposed_hand = _hand_flags(melds)
    if is_concealed_hand:
        pass  # scored with win-method context, see _score_winning_tile
    if is_fully_exposed_hand:
        pass

    if len(suits_used) == 1 and not honors_used:
        lines.append(ScoreLine("hand_shape", "Pure Suit Hand", 90))
    elif len(suits_used) == 1 and honors_used:
        lines.append(ScoreLine("hand_shape", "Semi Pure Hand", 30))
    elif len(suits_used) == 2 and not honors_used:
        lines.append(ScoreLine("hand_shape", "2 Suit Hand", 5))
        if not flowers:
            lines.append(ScoreLine("hand_shape", "2 Suit Hand, no flowers (extra)", 5))

    if len(suits_used) == 3 and honors_used:
        winds_present = any(is_wind(m.tile) for m in _pong_gong_melds(melds)) or is_wind(pair[0])
        dragons_present = any(is_dragon(m.tile) for m in _pong_gong_melds(melds)) or is_dragon(pair[0])
        if winds_present and dragons_present:
            lines.append(ScoreLine("hand_shape", "All 5 Suits Hand", 10))

    if all(m.type in (MeldType.PONG, MeldType.GONG) for m in melds):
        lines.append(ScoreLine("hand_shape", "All Pong Hand", 25))

    if all(m.type == MeldType.SHEUNG for m in melds):
        if flowers or honors_used:
            lines.append(ScoreLine("hand_shape", "All Sheung Hand (with flowers and/or Winds & Dragons)", 5))
        else:
            lines.append(ScoreLine("hand_shape", "All Sheung Hand (no flowers, no Winds & Dragons)", 15))
    return lines


def _score_named_special_categories(melds: list[Meld], pair: tuple[Tile, Tile]) -> list[ScoreLine]:
    lines = []

    # Neighbours/Uncles Hand (full 5-pong+pair sequential run, e.g. 111..555+66)
    pong_melds = _pong_gong_melds(melds)
    if len(pong_melds) == 5:
        suits = {m.suit for m in pong_melds}
        if len(suits) == 1 and None not in suits:
            ranks = sorted(tile_rank(m.tile) for m in pong_melds)
            if ranks == list(range(ranks[0], ranks[0] + 5)) and tile_suit(pair[0]) == m.suit and tile_rank(pair[0]) == ranks[-1] + 1:
                lines.append(ScoreLine("named", "Neighbours/Uncles Hand", 80))

    # Pure Honour Hand
    if all(m.is_honor_set for m in melds) and is_honor(pair[0]):
        lines.append(ScoreLine("named", "Pure Honour Hand", 140))

    # Terminals + Honours / Terminals Only
    if all(m.is_terminal_set or m.is_honor_set for m in melds) and (is_terminal(pair[0]) or is_honor(pair[0])):
        any_honor = any(m.is_honor_set for m in melds) or is_honor(pair[0])
        if any_honor:
            lines.append(ScoreLine("named", "Terminals + Honours", 20))
        elif is_terminal(pair[0]):
            lines.append(ScoreLine("named", "Terminals Only (No Honours)", 40))

    # ALL STEP UP Hand (full 5-sheung stepping run)
    if all(m.type == MeldType.SHEUNG for m in melds):
        ranks = sorted(m.start_rank for m in melds)
        if len(set(ranks)) == 5 and ranks == list(range(ranks[0], ranks[0] + 5)):
            lines.append(ScoreLine("named", "ALL STEP UP Hand", 20))

    # LRC Jewel hands: one suit + one specific dragon, nothing else.
    suits_used = {m.suit for m in melds if m.suit}
    if tile_suit(pair[0]):
        suits_used.add(tile_suit(pair[0]))
    honor_tiles_used = {m.tile for m in melds if m.is_honor_set}
    if is_honor(pair[0]):
        honor_tiles_used.add(pair[0])

    def _is_jewel(suit: str, dragon: str) -> bool:
        return suits_used <= {suit} and honor_tiles_used <= {dragon} and bool(honor_tiles_used)

    if _is_jewel("s", GREEN_DRAGON):
        lines.append(ScoreLine("named", "LRC Jade Hand (Green Dragon + Bamboo)", 20))
    if _is_jewel("m", RED_DRAGON):
        lines.append(ScoreLine("named", "LRC Ruby Hand (Red Dragon + Character)", 20))
    if _is_jewel("p", WHITE_DRAGON):
        lines.append(ScoreLine("named", "LRC Diamond Hand (White Dragon + Circle)", 20))
    if _is_jewel("m", WHITE_DRAGON):
        lines.append(ScoreLine("named", "LRC Sapphire Hand (White Dragon + Character)", 20))

    return lines


def _score_eyes(pair: tuple[Tile, Tile]) -> list[ScoreLine]:
    if is_good_eye_tile(pair[0]):
        return [ScoreLine("eyes", "Eyes (good eyes: 2, 5, or 8)", 2)]
    return []


def _score_waiting(context: WinContext) -> list[ScoreLine]:
    from src.rule_engine.models import WaitType

    mapping = {
        WaitType.PAIR_WAIT: "Calling by Pairs",
        WaitType.TRUE_SINGLE: "True Single Wait",
        WaitType.FALSE_SINGLE: "False Single Wait",
    }
    label = mapping.get(context.wait_type)
    return [ScoreLine("waiting", label, 2)] if label else []


def _score_winning_tile(melds: list[Meld], context: WinContext, force_concealed: Optional[bool] = None) -> list[ScoreLine]:
    lines = []
    if force_concealed is not None:
        # Special hands (Nico Nico / 13 Orphans / 16 Orphans) are always fully
        # concealed by definition -- _hand_flags' pong/sheung-based logic
        # doesn't apply to them (an empty meld list would vacuously satisfy
        # BOTH "all concealed" and "all exposed", which is wrong).
        is_concealed_hand, is_fully_exposed_hand = force_concealed, False
    else:
        is_concealed_hand, is_fully_exposed_hand = _hand_flags(melds)

    if context.is_dealer:
        lines.append(ScoreLine("winning_tile", "East Seat/Dealer tile", 1))

    if context.self_draw:
        lines.append(ScoreLine("winning_tile", "Self-Draw from the Wall", 1))
        if context.win_from_flower_wall:
            lines.append(ScoreLine("winning_tile", "Self-Draw from the Flower Wall", 5))
        if is_concealed_hand:
            lines.append(ScoreLine("winning_tile", "Self-Draw with a Concealed Hand", 10))
        if is_fully_exposed_hand:
            lines.append(ScoreLine("winning_tile", "Self-Draw in a Fully Exposed Hand", 10))
    else:
        if is_concealed_hand:
            lines.append(ScoreLine("winning_tile", "Concealed Hand (win from Discard)", 5))
        if is_fully_exposed_hand:
            lines.append(ScoreLine("winning_tile", "Fully Exposed Hand (win from Discard)", 15))

    if context.win_from_robbing_gong:
        lines.append(ScoreLine("winning_tile", "Robbing a Gong", 10))
    if context.gong_gong_win:
        lines.append(ScoreLine("winning_tile", "Gong-Gong Win", 30))

    if context.tiles_on_table is not None:
        t = context.tiles_on_table
        if t <= 4:
            lines.append(ScoreLine("winning_tile", "Human Hand - win within 4 tiles", 80))
        elif t <= 7:
            lines.append(ScoreLine("winning_tile", "Win within 7 tiles", 20))
        elif t <= 10:
            lines.append(ScoreLine("winning_tile", "Win within 10 tiles", 10))

    if context.is_seabed:
        lines.append(ScoreLine("winning_tile", "Seabed Hand", 10))
    if context.is_earthly:
        lines.append(ScoreLine("winning_tile", "Earthly Hand", 90))
    if context.is_heavenly:
        lines.append(ScoreLine("winning_tile", "Heavenly Hand", 100))

    return lines


_BASE_POINT = ScoreLine("base", "Base Point", 5)


def _score_normal_hand(melds: list[Meld], pair: tuple[Tile, Tile], flowers: list[Tile], context: WinContext) -> list[ScoreLine]:
    lines: list[ScoreLine] = []
    lines += _score_flowers(flowers, context)
    lines += _score_honors_absence(melds, pair, flowers)
    lines += _score_winds(melds, pair, context)
    lines += _score_dragons(melds, pair)
    lines += _score_gongs(melds)
    lines += _score_dragon_sequence(melds)
    lines += _score_neighbours(melds)
    lines += _score_brother_sheungs(melds)
    lines += _score_sister_sheungs(melds)
    lines += _score_sister_pongs(melds)
    lines += _score_concealed_pongs(melds)
    lines += _score_terminals(melds, pair)
    lines += _score_step_up(melds)
    lines += _score_hand_composition(melds, pair, flowers)
    lines += _score_named_special_categories(melds, pair)
    lines += _score_eyes(pair)
    lines += _score_waiting(context)
    lines += _score_winning_tile(melds, context)
    return lines


class TaiwaneseRuleset:
    name = "taiwanese"

    def score(self, hand: HandInput, context: WinContext) -> ScoreResult:
        flowers = hand.flowers

        # -- 7-Flowers / 8-Flowers instant self-draw win overrides everything --
        if len(flowers) >= 8:
            return ScoreResult(
                total_points=45,
                lines=[ScoreLine("flowers", "8 Flowers - Instant Self-Draw Win", 40), _BASE_POINT],
                is_special_hand="8 Flowers",
            )
        if len(flowers) == 7:
            return ScoreResult(
                total_points=25,
                lines=[ScoreLine("flowers", "7 Flowers - Instant Self-Draw Win", 20), _BASE_POINT],
                is_special_hand="7 Flowers",
            )

        # -- Special hands (structurally different from 5 melds + pair) --
        if not hand.melds:
            if detect_nico_nico(hand.concealed_tiles):
                lines = (
                    [ScoreLine("special", "Nico Nico", 40)]
                    + _score_flowers(flowers, context)
                    + _score_waiting(context)
                    + _score_winning_tile([], context, force_concealed=True)
                    + [_BASE_POINT]
                )
                return ScoreResult(total_points=sum(l.points for l in lines), lines=lines, is_special_hand="Nico Nico")

            orphan_meld = detect_thirteen_orphans(hand.concealed_tiles)
            if orphan_meld is not None:
                lines = (
                    [ScoreLine("special", "13 Orphans", 90)]
                    + _score_flowers(flowers, context)
                    + _score_waiting(context)
                    + _score_winning_tile([], context, force_concealed=True)
                    + [_BASE_POINT]
                )
                return ScoreResult(total_points=sum(l.points for l in lines), lines=lines, is_special_hand="13 Orphans")

            spread_sets = detect_sixteen_orphans(hand.concealed_tiles)
            if spread_sets is not None:
                lines = (
                    [ScoreLine("special", "16 Orphans", 50)]
                    + _score_flowers(flowers, context)
                    + _score_waiting(context)
                    + _score_winning_tile([], context, force_concealed=True)
                    + [_BASE_POINT]
                )
                return ScoreResult(total_points=sum(l.points for l in lines), lines=lines, is_special_hand="16 Orphans")

        # -- Standard 5-melds + pair hand: try every valid decomposition of
        #    the ungrouped concealed tiles, keep whichever scores highest. --
        decompositions = decompose_concealed(hand.concealed_tiles)
        if not decompositions:
            raise ValueError(
                "No valid decomposition found for the given concealed tiles -- "
                "check the tile list is a real, complete winning hand (5 melds "
                "+ pair combined with hand.melds) and that any Gongs are "
                "listed explicitly in hand.melds, not concealed_tiles."
            )

        best_lines: Optional[list[ScoreLine]] = None
        best_total = -1
        for decomposed_melds, pair in decompositions:
            full_melds = hand.melds + decomposed_melds
            if len(full_melds) != 5:
                continue  # wrong meld count for this decomposition -- not a valid 5-meld hand
            lines = _score_normal_hand(full_melds, pair, flowers, context)
            total = sum(l.points for l in lines)
            if total > best_total:
                best_total = total
                best_lines = lines

        if best_lines is None:
            raise ValueError(
                "concealed_tiles + hand.melds did not combine into any valid "
                "5-meld-plus-pair hand -- check tile counts."
            )

        # -- Chicken Hand override: a hand whose points sum to only 1 before
        #    the Base Point gets a flat 20 instead. --
        pre_base_total = sum(l.points for l in best_lines)
        if pre_base_total == 1:
            return ScoreResult(
                total_points=25,
                lines=[ScoreLine("special", "Chicken Hand", 20), _BASE_POINT],
                is_chicken_hand=True,
                notes=[
                    f"Raw pre-Base score was {pre_base_total}pt "
                    f"({[l.label for l in best_lines]}); overridden to the flat "
                    f"20pt Chicken Hand bonus per the source guide."
                ],
            )

        final_lines = best_lines + [_BASE_POINT]
        return ScoreResult(total_points=sum(l.points for l in final_lines), lines=final_lines)
