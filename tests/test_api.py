import json, os, sys, time
from pathlib import Path
import numpy as np, pytest
from fastapi.testclient import TestClient
from app import main

DATA = Path(__file__).parent / "data"
MODEL_PATH = os.getenv("MODEL_PATH", "model/champion_model.pkl")

@pytest.fixture(scope="module")
def client():
    assert os.path.exists(MODEL_PATH), f"{MODEL_PATH} missing: export it from the notebook (deploy bundle cell) and commit it"
    with TestClient(main.app) as c:       # context manager runs the lifespan (model load)
        yield c

def sample(name):
    d = json.loads((DATA / f"sample_{name}.json").read_text())
    exp = {k: d.pop(k) for k in [k for k in d if k.startswith("_")]}
    return d, exp

def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready"); assert r.status_code == 200 and r.json()["model_version"]

def test_model_info(client):
    j = client.get("/model/info").json()
    assert 0 <= j["otp_threshold"] < j["decline_threshold"] <= 1 and "anomaly_score" in j["features"]

@pytest.mark.parametrize("name", ["genuine", "fraud"])
def test_training_serving_parity(client, name):
    """API must reproduce the notebook's score and decision for the same transaction (no training/serving skew)."""
    p, exp = sample(name)
    j = client.post("/score", json=p).json()
    assert abs(j["fraud_score"] - exp["_expected_score"]) < 1e-6
    assert j["decision"] == exp["_expected_decision"]

def test_behaviour_regression(client):
    pg, _ = sample("genuine"); pf, _ = sample("fraud")
    assert client.post("/score", json=pg).json()["decision"] == "APPROVE"
    jf = client.post("/score", json=pf).json()
    assert jf["decision"] in ("OTP", "DECLINE") and len(jf["reasons"]) >= 1

@pytest.mark.parametrize("mutate", [lambda p: p.pop("V1"), lambda p: p.update(hour=25), lambda p: p.update(amount=-5), lambda p: p.update(V3="abc")])
def test_bad_input_rejected(client, mutate):
    p, _ = sample("genuine"); mutate(p)
    assert client.post("/score", json=p).status_code == 422

def test_503_when_model_missing(client, monkeypatch):
    monkeypatch.setitem(main.STATE, "bundle", None)
    assert client.get("/ready").status_code == 503
    p, _ = sample("genuine"); assert client.post("/score", json=p).status_code == 503

def test_latency_under_sla(client):
    p, _ = sample("genuine"); client.post("/score", json=p)          # warm-up
    lat = []
    for _ in range(100):
        t = time.perf_counter(); assert client.post("/score", json=p).status_code == 200; lat.append((time.perf_counter() - t) * 1000)
    assert np.percentile(lat, 95) < 100, f"p95 {np.percentile(lat, 95):.1f} ms breaches the 100 ms SLA"

def test_stats_endpoint(client):
    j = client.get("/stats").json(); assert j["requests"] > 0 and "p99" in j["latency_ms"]

def test_library_versions_match_training():
    """pickles are version-sensitive: serving env must match the training env recorded in model_meta.json"""
    import sklearn, lightgbm
    meta = json.loads(Path(MODEL_PATH).with_name("model_meta.json").read_text())
    assert meta["sklearn"] == sklearn.__version__ and meta["lightgbm"] == lightgbm.__version__, \
        f"version skew: trained with sklearn {meta['sklearn']}/lightgbm {meta['lightgbm']}; update requirements-ml.txt"
