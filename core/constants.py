# core/constants.py
from __future__ import annotations
from typing import Dict, List

ACTIONS: List[str] = ["box", "jump", "run", "walk"]
ACTION_TO_ID: Dict[str, int] = {a: i for i, a in enumerate(ACTIONS)}
ID_TO_ACTION: Dict[int, str] = {v: k for k, v in ACTION_TO_ID.items()}
