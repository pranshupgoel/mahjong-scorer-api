"""
Ruleset registry. Core Taiwanese scoring is the only ruleset implemented for
now; this registry exists so house-rule variants from the source guide's
"Variation Games" section (Joker rules, holiday-themed hands, X-Stitch hands
-- pages 41-43) can be added later as additional Ruleset implementations
without touching engine.py or the core taiwanese.py module.
"""
from __future__ import annotations

from src.rule_engine.rulesets.base import Ruleset
from src.rule_engine.rulesets.taiwanese import TaiwaneseRuleset

_REGISTRY: dict[str, Ruleset] = {
    "taiwanese": TaiwaneseRuleset(),
}


def get_ruleset(name: str = "taiwanese") -> Ruleset:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown ruleset {name!r}. Available: {sorted(_REGISTRY)}") from None


def register_ruleset(name: str, ruleset: Ruleset) -> None:
    """For future variant rulesets (joker/holiday/X-Stitch) to plug themselves in."""
    _REGISTRY[name] = ruleset
