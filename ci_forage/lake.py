"""Toy lake. Default ask: Saturday with Sam, Jerry, Tom.

True Saturday: beach + jerry (surf / tacos, not Tom's chess).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .math.ops import EPS
from .model import BRANCHES, CUES, PEOPLE, FactorModel, normalize_A_gaze


@dataclass(frozen=True)
class LakeItem:
    id: str
    label: str


def _likelihood(table: dict[tuple[str, str], str], noise: float = 0.02) -> np.ndarray:
    n_o, n_b, n_t = len(CUES), len(BRANCHES), len(PEOPLE)
    A = np.zeros((n_o, n_b, n_t))
    for b_i, branch in enumerate(BRANCHES):
        for t_i, person in enumerate(PEOPLE):
            cue = table.get((branch, person), "none")
            A[:, b_i, t_i] = noise / n_o
            A[CUES.index(cue), b_i, t_i] += 1.0 - noise
            A[:, b_i, t_i] /= A[:, b_i, t_i].sum()
    return A


def _all(cue: str) -> dict[tuple[str, str], str]:
    return {(b, p): cue for b in BRANCHES for p in PEOPLE}


def sam_saturday_lake() -> tuple[FactorModel, np.ndarray, list[LakeItem], dict]:
    items_spec: list[tuple[str, str, dict[tuple[str, str], str]]] = [
        (
            "calendar_saturday",
            "calendar: Saturday beach trip",
            {("beach", p): "beach_mention" for p in PEOPLE}
            | {("office", p): "office_mention" for p in PEOPLE}
            | {("errands", p): "errands_mention" for p in PEOPLE},
        ),
        (
            "packed_shorts",
            "content: board shorts in the bag",
            {("beach", p): "beach_mention" for p in PEOPLE},
        ),
        (
            "jerry_text",
            "message: Jerry is free Saturday near the beach",
            {("beach", "jerry"): "jerry_mention", ("errands", "jerry"): "jerry_mention"},
        ),
        (
            "tom_chess",
            "event: Tom at chess club Saturday",
            {("beach", "tom"): "chess_mention", ("office", "tom"): "chess_mention"},
        ),
        (
            "surf_report",
            "content: good surf, Jerry always goes",
            {("beach", "jerry"): "surf_mention"},
        ),
        (
            "slack_weekend",
            "event: no standup Saturday",
            {("office", p): "office_mention" for p in PEOPLE},
        ),
        (
            "sam_friday",
            "event: Sam seen downtown Friday",
            _all("none"),
        ),
    ]

    items = [LakeItem(id=i, label=lab) for i, lab, _ in items_spec]
    process_A = normalize_A_gaze(
        np.stack([_likelihood(table) for _, _, table in items_spec], axis=-1)
    )
    model = FactorModel()
    model.bind_gaze(len(items))
    model.A = process_A.copy()
    model.a = model.A * 8.0
    model.C = np.ones(len(CUES)) / len(CUES)
    truth = {"branch": "beach", "texture": "jerry"}
    return model, process_A, items, truth


def observe(process_A: np.ndarray, truth: dict, gaze: int, rng: np.random.Generator) -> int:
    b = BRANCHES.index(truth["branch"])
    t = PEOPLE.index(truth["texture"])
    probs = np.clip(process_A[:, b, t, gaze], EPS, None)
    probs = probs / probs.sum()
    return int(rng.choice(process_A.shape[0], p=probs))
