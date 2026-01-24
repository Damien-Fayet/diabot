"""Quest definitions per act (minimal skeleton)."""
from __future__ import annotations

QUESTS = {
    1: [
        {"id": "den", "name": "Den of Evil", "type": "mandatory"},
        {"id": "andariel", "name": "Andariel", "type": "mandatory"},
    ],
    2: [
        {"id": "radament", "name": "Radament's Lair", "type": "mandatory"},
        {"id": "duriel", "name": "Duriel", "type": "mandatory"},
    ],
    3: [
        {"id": "travincal", "name": "Travincal", "type": "mandatory"},
        {"id": "mephisto", "name": "Mephisto", "type": "mandatory"},
    ],
    4: [
        {"id": "izual", "name": "Izual", "type": "optional"},
        {"id": "diablo", "name": "Diablo", "type": "mandatory"},
    ],
    5: [
        {"id": "ancients", "name": "Ancients", "type": "mandatory"},
        {"id": "baal", "name": "Baal", "type": "mandatory"},
    ],
}
