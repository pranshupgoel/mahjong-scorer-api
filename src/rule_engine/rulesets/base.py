"""Pluggable-ruleset scaffolding. See rulesets/__init__.py for the registry."""
from __future__ import annotations

from typing import Protocol

from src.rule_engine.models import HandInput, ScoreResult, WinContext


class Ruleset(Protocol):
    name: str

    def score(self, hand: HandInput, context: WinContext) -> ScoreResult:
        ...
