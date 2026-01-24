"""Persistence helpers for BotState."""
from __future__ import annotations

from pathlib import Path
from typing import Union

from diabot.models import BotState


def save_bot_state(state: BotState, path: Union[str, Path]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(state.to_json(), encoding="utf-8")


def load_bot_state(path: Union[str, Path]) -> BotState:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Bot state not found: {p}")
    return BotState.from_json(p.read_text(encoding="utf-8"))
