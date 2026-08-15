"""Forage sidecar. Ticket CI-Active-Inference#19.

Empty POST /ask loads the Sam Saturday fixture. One current trial at a time;
pass trial_id to address another.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .lake import sam_saturday_lake
from .model import BRANCHES, PEOPLE
from .persist import write_trial
from .trial import ForageError, ForageTrial

app = FastAPI(
    title="CI Forage",
    description="Epistemic sampler over a memory lake. Discrete active inference, not a pixel eye.",
    version="0.2.0",
)


class AskBody(BaseModel):
    question: str = "where is Sam on Saturday"
    subject: str = "Sam"
    horizon: str = "saturday"
    seed: int = 0
    prefer_cue: str | None = None


class PickBody(BaseModel):
    branch: str | None = None


class ProbeBody(BaseModel):
    item_id: str


class ResolveBody(BaseModel):
    branch: str | None = None
    texture: str | None = None


class TrialStore:
    def __init__(self) -> None:
        self.trials: dict[str, ForageTrial] = {}
        self.current_id: str | None = None

    def start(self, body: AskBody) -> str:
        model, process_A, items, truth = sam_saturday_lake()
        if body.prefer_cue:
            model.set_preferences_for_outcome(body.prefer_cue)
        trial = ForageTrial(
            model=model,
            process_A=process_A,
            items=items,
            truth=truth,
            question=body.question,
            subject=body.subject,
            horizon=body.horizon,
            rng=np.random.default_rng(body.seed),
        )
        trial_id = str(uuid4())
        self.trials[trial_id] = trial
        self.current_id = trial_id
        self.persist(trial_id)
        return trial_id

    def get(self, trial_id: str | None = None) -> tuple[str, ForageTrial]:
        key = trial_id or self.current_id
        if key is None or key not in self.trials:
            raise HTTPException(status_code=404, detail="no trial; POST /ask first")
        return key, self.trials[key]

    def persist(self, trial_id: str) -> None:
        trial = self.trials[trial_id]
        write_trial(trial_id, {"trial_id": trial_id, **trial.snapshot()})


store = TrialStore()


def _run(trial_id: str | None, fn):
    key, trial = store.get(trial_id)
    try:
        result = fn(trial)
    except ForageError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    store.persist(key)
    return result


@app.get("/health")
def health() -> dict:
    return {"ok": True, "current_trial": store.current_id}


@app.post("/ask")
def ask(body: AskBody | None = None) -> dict:
    trial_id = store.start(body or AskBody())
    _, trial = store.get(trial_id)
    return {"trial_id": trial_id, **trial.snapshot()}


@app.get("/factors")
def factors(trial_id: str | None = None) -> dict:
    _, trial = store.get(trial_id)
    return {
        "question": trial.question,
        "horizon": trial.horizon,
        "factors": {"branch": list(BRANCHES), "texture": list(PEOPLE)},
        "note": "permutations are Q(texture | branch) from the joint",
    }


@app.get("/branches")
def branches(trial_id: str | None = None) -> dict:
    _, trial = store.get(trial_id)
    picked = trial.pick()
    return {
        "question": trial.question,
        "horizon": trial.horizon,
        "pin": trial.pin,
        "pick": picked["branch"],
        "ready": picked["ready"],
        "branches": [
            {
                "id": name,
                "mass": picked["branch_mass"][name],
                "ready": picked["branch"] == name,
                "past": trial.is_past(name),
                "pinned": trial.pin == name,
                "top_texture": max(trial.permutations(name), key=trial.permutations(name).get),
            }
            for name in BRANCHES
        ],
    }


@app.get("/branches/{name}")
def branch_story(name: str, trial_id: str | None = None) -> dict:
    _, trial = store.get(trial_id)
    try:
        perms = trial.permutations(name)
    except ForageError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return {
        "id": name,
        "mass": trial.pick()["branch_mass"][name],
        "past": trial.is_past(name),
        "pinned": trial.pin == name,
        "permutations": perms,
        "interests": trial.interests_for(name),
        "why": trial.why(name),
    }


@app.get("/branches/{name}/permutations")
def branch_permutations(name: str, trial_id: str | None = None) -> dict:
    _, trial = store.get(trial_id)
    try:
        perms = trial.permutations(name)
    except ForageError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return {
        "branch": name,
        "Q_texture_given_branch": perms,
    }


@app.post("/branches/{name}/pin")
def pin_branch(name: str, trial_id: str | None = None) -> dict:
    def go(trial: ForageTrial) -> dict:
        trial.set_pin(name)
        return {"pin": trial.pin, "branches": branches(trial_id)["branches"]}

    return _run(trial_id, go)


@app.delete("/branches/{name}/pin")
def unpin_branch(name: str, trial_id: str | None = None) -> dict:
    def go(trial: ForageTrial) -> dict:
        if trial.pin == name:
            trial.clear_pin()
        return {"pin": trial.pin}

    return _run(trial_id, go)


@app.get("/branches/{name}/expect")
def expect_branch(name: str, trial_id: str | None = None) -> dict:
    _, trial = store.get(trial_id)
    try:
        return trial.expect(name)
    except ForageError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.post("/pick")
def pick(body: PickBody | None = None, trial_id: str | None = None) -> dict:
    def go(trial: ForageTrial) -> dict:
        if body and body.branch:
            trial.set_pin(body.branch)
        return trial.pick()

    return _run(trial_id, go)


@app.get("/attention")
def attention(trial_id: str | None = None) -> dict:
    _, trial = store.get(trial_id)
    return {"items": trial.attention()}


@app.post("/sample")
def take_sample(trial_id: str | None = None) -> dict:
    def go(trial: ForageTrial) -> dict:
        rec = trial.sample_next()
        return {
            "read": {
                "item_id": rec.item_id,
                "cue": rec.cue,
                "G": rec.G,
                "epistemic": rec.epistemic,
            },
            "pick": trial.pick(),
        }

    return _run(trial_id, go)


@app.get("/trace")
def trace(trial_id: str | None = None) -> dict:
    _, trial = store.get(trial_id)
    return {
        "reads": [
            {
                "item_id": rec.item_id,
                "cue": rec.cue,
                "G": rec.G,
                "branch_delta": rec.branch_delta,
                "branch_mass_after": {
                    name: float(m)
                    for name, m in zip(
                        BRANCHES,
                        rec.joint_after.sum(axis=1) / rec.joint_after.sum(),
                    )
                },
            }
            for rec in trial.log
        ]
    }


@app.post("/probe")
def probe(body: ProbeBody, trial_id: str | None = None) -> dict:
    _, trial = store.get(trial_id)
    try:
        return trial.probe(body.item_id)
    except ForageError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.post("/resolve")
def resolve(body: ResolveBody | None = None, trial_id: str | None = None) -> dict:
    def go(trial: ForageTrial) -> dict:
        actual = None
        if body and body.branch and body.texture:
            actual = {"branch": body.branch, "texture": body.texture}
        return trial.resolve(actual)

    return _run(trial_id, go)


@app.get("/surprise")
def surprise(trial_id: str | None = None) -> dict:
    _, trial = store.get(trial_id)
    return trial.surprise()


@app.get("/postdict")
def postdict(trial_id: str | None = None) -> dict:
    _, trial = store.get(trial_id)
    return {"unread": trial.postdict()}


@app.post("/close")
def close_trial(trial_id: str | None = None) -> dict:
    def go(trial: ForageTrial) -> dict:
        trial.close_trial()
        return {"closed": True, "pick": trial.pick(), "surprise": trial.surprise()}

    return _run(trial_id, go)


@app.get("/assumptions")
def assumptions(trial_id: str | None = None) -> dict:
    _, trial = store.get(trial_id)
    return trial.assumptions()
