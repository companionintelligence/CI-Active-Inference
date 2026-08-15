"""Container contract for the forage sidecar."""

from fastapi.testclient import TestClient

from ci_forage.api import app, store


def setup_function() -> None:
    store.trials.clear()
    store.current_id = None


def test_ask_sample_pin_probe_resolve():
    client = TestClient(app)
    started = client.post("/ask", json={"seed": 0})
    assert started.status_code == 200
    assert started.json()["question"] == "where is Sam on Saturday"

    attention = client.get("/attention")
    assert attention.status_code == 200
    assert attention.json()["items"][0]["id"] == "calendar_saturday"

    first = client.post("/sample")
    assert first.status_code == 200
    assert first.json()["read"]["item_id"] == "calendar_saturday"
    assert first.json()["pick"]["branch"] == "beach"

    branches = client.get("/branches")
    ids = {row["id"] for row in branches.json()["branches"]}
    assert ids == {"beach", "office", "errands"}
    assert branches.json()["pick"] == "beach"

    pin = client.post("/branches/beach/pin")
    assert pin.status_code == 200
    assert pin.json()["pin"] == "beach"
    assert {row["id"] for row in pin.json()["branches"]} == ids

    perms = client.get("/branches/beach/permutations")
    assert perms.status_code == 200
    assert "jerry" in perms.json()["Q_texture_given_branch"]

    client.post("/sample")
    client.post("/sample")

    beach = client.get("/branches/beach")
    assert "surf" in beach.json()["interests"]["jerry"]
    assert "chess" in beach.json()["interests"]["tom"]

    before = client.get("/assumptions").json()
    probe = client.post("/probe", json={"item_id": "tom_chess"})
    assert probe.status_code == 200
    assert probe.json()["updated_q"] is False
    after = client.get("/assumptions").json()
    assert after["used"] == before["used"]

    resolved = client.post("/resolve")
    assert resolved.status_code == 200
    assert resolved.json()["truth"]["texture"] == "jerry"

    surpr = client.get("/surprise")
    assert surpr.json()["q_truth"] > 0.0

    expect_office = client.get("/branches/office/expect")
    assert expect_office.status_code == 409

    expect_beach = client.get("/branches/beach/expect")
    assert expect_beach.status_code == 200

    post = client.get("/postdict")
    assert "unread" in post.json()

    factors = client.get("/factors")
    assert factors.json()["factors"]["texture"] == ["jerry", "tom", "colleague", "solo"]

    trace = client.get("/trace")
    assert trace.json()["reads"][0]["item_id"] == "calendar_saturday"


def test_no_trial_is_404():
    client = TestClient(app)
    assert client.get("/attention").status_code == 404


def test_unknown_branch_is_404():
    client = TestClient(app)
    client.post("/ask")
    assert client.get("/branches/moon/permutations").status_code == 404
