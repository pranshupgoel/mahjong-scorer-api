"""
Canonical tile taxonomy for the pipeline, plus mappings from the two raw
dataset label schemes onto it.

Canonical scheme (riichi-style shorthand, 42 classes):
    1m..9m  -> characters / man     (dataset-master: characters-1..9)
    1p..9p  -> dots / pin           (dataset-master: dots-1..9)
    1s..9s  -> bamboo / sou         (dataset-master: bamboo-1..9)
    1z..7z  -> honors (E,S,W,N,red,green,white in that order)
    f1..f4  -> flower bonus tiles   (spring, summer, autumn, plum... see FLOWER_MAP)
    s1..s4  -> season/misc bonus    (dataset-master calls these bonus-* too;
                                      see note below on the split)

Two source datasets, two different naming conventions:
  - "yolo dataset" (Datasets/yolo dataset, Roboflow export): only used to
    train the detector, which is class-agnostic (single "tile" class -- see
    scripts/make_detector_class_agnostic.py). Its original per-tile-face
    labels (34 classes, no bonus tiles) don't matter for that reason: the
    detector never needs to tell tiles apart, only find them.
  - "mahjong-dataset-master" (Datasets/mahjong-dataset-master): uses
    long-form names (dots-1, bamboo-3, honors-east, bonus-winter, ...) with a
    numeric label 1-42 defined in its own README. Covers all 42 classes,
    including the 8 bonus tiles.

CANONICAL_CLASSES is the single ordered list the classifier trains against
/ outputs indices into. Use `master_label_to_canonical` to translate the
mahjong-dataset-master label scheme onto it. The detector doesn't need a
mapping at all -- it's class-agnostic (see YOLO_DATASET_ORIGINAL_CLASSES
below for why).
"""

from __future__ import annotations

# --- Canonical class list --------------------------------------------------

_SUITS = ["m", "p", "s"]  # man/characters, pin/dots, sou/bamboo
_HONORS = ["1z", "2z", "3z", "4z", "5z", "6z", "7z"]  # E, S, W, N, red, green, white
_BONUS = [
    "spring", "summer", "autumn", "winter",  # seasons
    "plum", "orchid", "chrysanthemum", "bamboo_flower",  # flowers
]

CANONICAL_CLASSES: list[str] = (
    [f"{n}{s}" for s in _SUITS for n in range(1, 10)]
    + _HONORS
    + _BONUS
)
assert len(CANONICAL_CLASSES) == 42, len(CANONICAL_CLASSES)

CANONICAL_TO_IDX = {name: i for i, name in enumerate(CANONICAL_CLASSES)}

# --- mahjong-dataset-master (long-form names, 1-indexed 1..42) -------------
# Copied from Datasets/mahjong-dataset-master/README.md "Classes" table.

_MASTER_TABLE_NAME_TO_CANONICAL = {
    "dots-1": "1p", "dots-2": "2p", "dots-3": "3p", "dots-4": "4p", "dots-5": "5p",
    "dots-6": "6p", "dots-7": "7p", "dots-8": "8p", "dots-9": "9p",
    "bamboo-1": "1s", "bamboo-2": "2s", "bamboo-3": "3s", "bamboo-4": "4s",
    "bamboo-5": "5s", "bamboo-6": "6s", "bamboo-7": "7s", "bamboo-8": "8s", "bamboo-9": "9s",
    "characters-1": "1m", "characters-2": "2m", "characters-3": "3m", "characters-4": "4m",
    "characters-5": "5m", "characters-6": "6m", "characters-7": "7m", "characters-8": "8m",
    "characters-9": "9m",
    "honors-east": "1z", "honors-south": "2z", "honors-west": "3z", "honors-north": "4z",
    "honors-red": "5z", "honors-green": "6z", "honors-white": "7z",
    "bonus-spring": "spring", "bonus-summer": "summer", "bonus-autumn": "autumn",
    "bonus-winter": "winter", "bonus-plum": "plum", "bonus-orchid": "orchid",
    "bonus-chrysanthemum": "chrysanthemum", "bonus-bamboo": "bamboo_flower",
}

# table-index (1-based, as used in the master dataset's data.csv "label" column)
# -> table-name, taken straight from its README.
_MASTER_INDEX_TO_TABLE_NAME = {
    1: "dots-1", 2: "dots-2", 3: "dots-3", 4: "dots-4", 5: "dots-5", 6: "dots-6",
    7: "dots-7", 8: "dots-8", 9: "dots-9",
    10: "bamboo-1", 11: "bamboo-2", 12: "bamboo-3", 13: "bamboo-4", 14: "bamboo-5",
    15: "bamboo-6", 16: "bamboo-7", 17: "bamboo-8", 18: "bamboo-9",
    19: "characters-1", 20: "characters-2", 21: "characters-3", 22: "characters-4",
    23: "characters-5", 24: "characters-6", 25: "characters-7", 26: "characters-8",
    27: "characters-9",
    28: "honors-east", 29: "honors-south", 30: "honors-west", 31: "honors-north",
    32: "honors-red", 33: "honors-green", 34: "honors-white",
    35: "bonus-spring", 36: "bonus-summer", 37: "bonus-autumn", 38: "bonus-winter",
    39: "bonus-plum", 40: "bonus-orchid", 41: "bonus-chrysanthemum", 42: "bonus-bamboo",
}


def master_label_to_canonical(label_index: int) -> str:
    """Map a mahjong-dataset-master `data.csv` `label` (1..42) to a canonical class name."""
    table_name = _MASTER_INDEX_TO_TABLE_NAME[label_index]
    return _MASTER_TABLE_NAME_TO_CANONICAL[table_name]


# --- yolo dataset (Roboflow export) -----------------------------------------
# The detector trained on this dataset is class-agnostic (single "tile"
# class), so there's no per-face mapping to maintain here. Kept as a
# reference only: these were the original 34 per-face class names before
# scripts/make_detector_class_agnostic.py collapsed them.

YOLO_DATASET_ORIGINAL_CLASSES = [
    "1m", "1p", "1s", "1z", "2m", "2p", "2s", "2z", "3m", "3p", "3s", "3z",
    "4m", "4p", "4s", "4z", "5m", "5p", "5s", "5z", "6m", "6p", "6s", "6z",
    "7m", "7p", "7s", "7z", "8m", "8p", "8s", "9m", "9p", "9s",
]
