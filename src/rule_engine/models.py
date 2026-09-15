"""
Core data model for the Taiwanese Mahjong rule engine.

Tiles are canonical strings from src.common.labels.CANONICAL_CLASSES:
    1m..9m / 1p..9p / 1s..9s   -- the three number suits (character/dot/bamboo)
    1z..4z                     -- winds, in order East, South, West, North
    5z..7z                     -- dragons, in order Red, Green, White
    spring/summer/autumn/winter, plum/orchid/chrysanthemum/bamboo_flower -- flowers

IMPORTANT structural fact (easy to miss, confirmed from the source PDF):
Taiwanese Mahjong hands are **5 melds + 1 pair = 17 tiles**, not the 4-melds
+pair/14-tile shape used in Japanese/Chinese riichi-style mahjong. This is
explicit in the source doc: "All Pong Hand ... Five sets of three tiles to
make trios & a pair of Eyes" (and the same wording for All Sheung Hand), plus
the "17th tile" references throughout (East is dealt 16 tiles + draws a 17th
before the first discard). Get this wrong and every meld-count-based rule
breaks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

Tile = str

MELDS_IN_HAND = 5  # Taiwanese hands are 5 melds + 1 pair (17 tiles), not 4+1.

WIND_ORDER = ["1z", "2z", "3z", "4z"]  # E, S, W, N
WIND_NAME = {"1z": "E", "2z": "S", "3z": "W", "4z": "N"}
NAME_TO_WIND_TILE = {v: k for k, v in WIND_NAME.items()}

DRAGON_TILES = {"5z", "6z", "7z"}
RED_DRAGON, GREEN_DRAGON, WHITE_DRAGON = "5z", "6z", "7z"
HONOR_TILES = set(WIND_ORDER) | DRAGON_TILES

SUITS = ("m", "p", "s")
_SUIT_INDEX = {"m": 0, "p": 1, "s": 2}

# Seat index (1-based, E/S/W/N order) each flower matches -- page 6 of the
# source guide: seasons spring/summer/autumn/winter = seats 1-4 (E/S/W/N);
# plants plum/orchid/chrysanthemum/bamboo_flower = seats 1-4 the same way.
SEASON_FLOWERS = ["spring", "summer", "autumn", "winter"]
PLANT_FLOWERS = ["plum", "orchid", "chrysanthemum", "bamboo_flower"]
FLOWER_TILES = set(SEASON_FLOWERS) | set(PLANT_FLOWERS)


def flower_seat_index(tile: Tile) -> int:
    """1-based E/S/W/N index this flower tile corresponds to."""
    if tile in SEASON_FLOWERS:
        return SEASON_FLOWERS.index(tile) + 1
    if tile in PLANT_FLOWERS:
        return PLANT_FLOWERS.index(tile) + 1
    raise ValueError(f"{tile!r} is not a flower tile")


def tile_suit(tile: Tile) -> Optional[str]:
    return tile[-1] if tile[-1] in SUITS else None


def tile_rank(tile: Tile) -> Optional[int]:
    if tile[-1] in SUITS or tile[-1] == "z":
        return int(tile[:-1])
    return None


def sort_key(tile: Tile):
    suit = tile_suit(tile)
    if suit:
        return (0, _SUIT_INDEX[suit], tile_rank(tile))
    if tile[-1] == "z":
        return (1, tile_rank(tile))
    return (2, tile)  # flowers -- stable, arbitrary order


def is_number_tile(tile: Tile) -> bool:
    return tile_suit(tile) is not None


def is_honor(tile: Tile) -> bool:
    return tile in HONOR_TILES


def is_wind(tile: Tile) -> bool:
    return tile in WIND_ORDER


def is_dragon(tile: Tile) -> bool:
    return tile in DRAGON_TILES


def is_terminal(tile: Tile) -> bool:
    return is_number_tile(tile) and tile_rank(tile) in (1, 9)


def is_flower(tile: Tile) -> bool:
    return tile in FLOWER_TILES


def is_good_eye_tile(tile: Tile) -> bool:
    """'Good eyes' = a pair of 2s, 5s, or 8s of any suit."""
    return is_number_tile(tile) and tile_rank(tile) in (2, 5, 8)


class MeldType(str, Enum):
    SHEUNG = "sheung"
    PONG = "pong"
    GONG = "gong"


class GongKind(str, Enum):
    NONE = "none"
    OPEN = "open"
    CONCEALED = "concealed"


@dataclass(frozen=True)
class Meld:
    type: MeldType
    tiles: tuple[Tile, ...]
    concealed: bool
    gong_kind: GongKind = GongKind.NONE

    def __post_init__(self):
        if self.type == MeldType.SHEUNG and len(self.tiles) != 3:
            raise ValueError("a sheung must have exactly 3 tiles")
        if self.type == MeldType.PONG and len(self.tiles) != 3:
            raise ValueError("a pong must have exactly 3 tiles")
        if self.type == MeldType.GONG and len(self.tiles) != 4:
            raise ValueError("a gong must have exactly 4 tiles")
        if self.type in (MeldType.PONG, MeldType.GONG) and len(set(self.tiles)) != 1:
            raise ValueError("a pong/gong must be 3-4 copies of the same tile")
        if self.type == MeldType.SHEUNG:
            suits = {tile_suit(t) for t in self.tiles}
            if len(suits) != 1 or None in suits:
                raise ValueError("a sheung must be 3 tiles of one number suit")
            ranks = sorted(tile_rank(t) for t in self.tiles)
            if ranks != [ranks[0], ranks[0] + 1, ranks[0] + 2]:
                raise ValueError("a sheung must be 3 consecutive ranks")

    @property
    def tile(self) -> Tile:
        """The repeated tile for a pong/gong, or the lowest tile for a sheung."""
        return min(self.tiles, key=sort_key) if self.type == MeldType.SHEUNG else self.tiles[0]

    @property
    def suit(self) -> Optional[str]:
        return tile_suit(self.tile)

    @property
    def start_rank(self) -> Optional[int]:
        return tile_rank(self.tile)

    @property
    def is_terminal_set(self) -> bool:
        """Pong/gong of 1s or 9s, or a sheung of 123/789."""
        if self.type == MeldType.SHEUNG:
            ranks = sorted(tile_rank(t) for t in self.tiles)
            return ranks in ([1, 2, 3], [7, 8, 9])
        return is_terminal(self.tile)

    @property
    def is_honor_set(self) -> bool:
        return self.type != MeldType.SHEUNG and is_honor(self.tile)

    @property
    def counts_as_concealed_pong(self) -> bool:
        """
        1 open gong = 1 concealed pong, for combo-counting purposes (Concealed
        Pongs, Four-in-N-ways etc). Any tile drawn from the wall (including
        completing a gong) counts as concealed per the source guide.
        """
        if self.type == MeldType.PONG:
            return self.concealed
        if self.type == MeldType.GONG:
            return True
        return False


@dataclass
class HandInput:
    """
    melds: every pre-identified set -- exposed Pongs/Sheungs/Gongs AND any
        concealed Gongs. Gongs must always be given explicitly here: the
        engine can't infer "4 identical tiles = a gong" vs "a pong + 1 spare
        tile" from a flat tile list, and open-vs-concealed is a fact about
        game history, not tile appearance.
    concealed_tiles: the remaining, ungrouped concealed tiles (everything
        that is NOT one of the melds above and not a flower) -- includes the
        pair. The engine auto-decomposes this into Sheung/Pong sets + the
        pair, trying every valid grouping and keeping whichever scores
        highest (the standard "score your hand's best interpretation"
        convention).
    flowers: flower tiles held (bonus tiles, outside the 5-sets+pair shape).
    winning_tile: the tile that completed the hand. Must be one of the tiles
        in concealed_tiles (or the 4th tile of a meld in `melds`, for a
        self-drawn/robbed gong win) -- the caller is responsible for
        including it there; this field is only used to look up which tile it
        was for tie-breaking / wait-type-adjacent rules.
    """

    melds: list[Meld] = field(default_factory=list)
    concealed_tiles: list[Tile] = field(default_factory=list)
    flowers: list[Tile] = field(default_factory=list)
    winning_tile: Optional[Tile] = None


class WaitType(str, Enum):
    NONE = "none"
    PAIR_WAIT = "pair_wait"
    TRUE_SINGLE = "true_single"
    FALSE_SINGLE = "false_single"


@dataclass
class WinContext:
    """
    Everything the engine needs that ISN'T visually inferable from a photo of
    the tiles -- per Pranshu's call, the app should prompt the user directly
    for these on a short questionnaire rather than trying to track full
    game state automatically.
    """

    seat_wind: str                             # "E" / "S" / "W" / "N" -- this player's seat
    round_wind: str                             # "E" / "S" / "W" / "N" -- current round
    self_draw: bool
    win_from_flower_wall: bool = False          # self-draw was a flower-wall replacement tile
    win_from_robbing_gong: bool = False         # won by robbing someone's pong-to-gong upgrade
    gong_gong_win: bool = False                 # gong -> flower draw -> gong -> self-draw win chain
    tiles_on_table: Optional[int] = None        # total discarded+declared tiles when won (Human Hand brackets)
    is_seabed: bool = False                     # won on the very last wall tile (not a discard)
    is_earthly: bool = False                    # non-dealer wins on the dealer's first discard
    is_heavenly: bool = False                   # dealer dealt a complete hand before the first discard
    wait_type: WaitType = WaitType.NONE
    closed_hand_declared: bool = False          # "Calling/Closing Hand" (West/North round house rule)
    triple_fan_dice: Optional[int] = None       # 1-6 if dealer broke the wall with a triple, else None
    dealer_win_streak: int = 0                  # consecutive wins as dealer (Dealer Die), 0 if none/not dealer

    @property
    def is_dealer(self) -> bool:
        return self.seat_wind == "E"


@dataclass
class ScoreLine:
    category: str
    label: str
    points: int


@dataclass
class ScoreResult:
    total_points: int
    lines: list[ScoreLine]
    is_special_hand: Optional[str] = None
    is_chicken_hand: bool = False
    ruleset: str = "taiwanese"
    notes: list[str] = field(default_factory=list)

    def breakdown(self) -> str:
        rows = "\n".join(f"  {l.points:>4}  [{l.category}] {l.label}" for l in self.lines)
        extra = ("\n\nNotes:\n" + "\n".join(f"  - {n}" for n in self.notes)) if self.notes else ""
        return f"{rows}\n  ----\n  {self.total_points:>4}  TOTAL{extra}"
