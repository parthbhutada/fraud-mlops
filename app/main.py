"""Real-time fraud scoring API (REST). Loads the champion model exported from the notebook."""
import json, logging, os, pickle, threading, time
from collections import Counter, deque
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

MODEL_PATH = os.getenv("MODEL_PATH", "model/champion_model.pkl")
V_COLS = [f"V{i}" for i in range(1, 29)]
IF_FEATS = V_COLS + ["log_amount"]          # same inputs the Isolation Forest was trained on
log = logging.getLogger("fraud-api")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ---- request schema (validated: no NaN/inf, hour 0-23, amount >= 0) ----
_f = lambda **kw: (float, Field(..., allow_inf_nan=False, **kw))
Transaction = create_model(
    "Transaction",
    txn_id=(Optional[str], None),
    amount=_f(ge=0),
    hour=(int, Field(..., ge=0, le=23)),
    tx_rate_5min=(int, Field(0, ge=0, description="payments seen platform-wide in previous 5 min (from feature store)")),
    **{v: _f() for v in V_COLS},
)

class Decision(BaseModel):
    txn_id: Optional[str]
    fraud_score: float
    decision: str                    # APPROVE | OTP | DECLINE
    model_version: str
    reasons: List[str]
    latency_ms: float

HUMAN = {"amount": "unusually large or small payment amount", "log_amount": "unusual payment amount",
         "hour": "unusual time of day", "tx_rate_5min": "burst of platform-wide payments in last 5 minutes",
         "anomaly_score": "overall pattern looks unusual vs normal behaviour"}

BUILD = os.getenv("GIT_SHA") or os.getenv("RENDER_GIT_COMMIT") or "unknown"     # which commit is running (used by the post-deploy check)
STATE = {"bundle": None, "error": None}
STATS = {"lock": threading.Lock(), "lat": deque(maxlen=2000), "scores": deque(maxlen=5000), "decisions": Counter(), "requests": 0, "errors": 0}

def load_model(path: str = MODEL_PATH):
    with open(path, "rb") as f:
        b = pickle.load(f)              # only ever load artifacts produced by our own training pipeline
    b["pipe"]["iso"].set_params(n_jobs=1); b["pipe"]["model"].set_params(n_jobs=1)
    return b

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        STATE["bundle"] = load_model(); STATE["error"] = None
        log.info(json.dumps({"event": "model_loaded", "version": STATE["bundle"]["version"], "path": MODEL_PATH}))
    except Exception as e:             # keep the process up so /ready reports the reason
        STATE["error"] = f"{type(e).__name__}: {e}"; log.error(json.dumps({"event": "model_load_failed", "error": STATE["error"]}))
    yield

app = FastAPI(title="Fraud Detection API", version="1.0", lifespan=lifespan,
              description="Approve / OTP / Decline decision for a payment in real time.")

def _bundle():
    if STATE["bundle"] is None:
        raise HTTPException(503, f"model not loaded: {STATE['error']}")
    return STATE["bundle"]

def build_features(rows: List[dict], b) -> pd.DataFrame:
    d = pd.DataFrame(rows)
    d["log_amount"] = np.log1p(d["amount"])
    X = d[b["features"]].copy()
    X["anomaly_score"] = -b["pipe"]["iso"].score_samples(d[IF_FEATS])
    return X

def decide(score: float, t1: float, t2: float) -> str:
    return "DECLINE" if score >= t2 else ("OTP" if score >= t1 else "APPROVE")

def reason_codes(X: pd.DataFrame, b, k: int = 3) -> List[str]:
    contrib = b["pipe"]["model"].booster_.predict(X, pred_contrib=True)[0][:-1]      # per-feature SHAP values (last col = bias)
    top = [i for i in np.argsort(-contrib)[:k] if contrib[i] > 0]
    return [HUMAN.get(X.columns[i], f"{X.columns[i]}: anonymised card-behaviour component") for i in top]

# ---- endpoints ----
@app.get("/health")
def health():                           # liveness: process is up
    return {"status": "ok"}

@app.get("/ready")
def ready():                            # readiness: model loaded
    b = _bundle()
    return {"status": "ready", "model_version": b["version"], "build": BUILD}

@app.get("/model/info")
def model_info():
    b = _bundle()
    return {"model_version": b["version"], "build": BUILD, "otp_threshold": b["t1"], "decline_threshold": b["t2"], "features": b["features"] + ["anomaly_score"]}

@app.post("/score", response_model=Decision)
def score(txn: Transaction):            # sync def: FastAPI runs it in a worker thread (CPU-bound model call)
    t0 = time.perf_counter()
    b = _bundle()
    try:
        X = build_features([txn.model_dump()], b)
        s = float(b["pipe"]["model"].predict_proba(X)[0, 1])
        dec = decide(s, b["t1"], b["t2"])
        reasons = reason_codes(X, b) if dec != "APPROVE" else []
    except Exception as e:
        with STATS["lock"]: STATS["errors"] += 1
        log.error(json.dumps({"event": "score_failed", "txn_id": txn.txn_id, "error": str(e)}))
        raise HTTPException(500, "scoring failed")
    ms = (time.perf_counter() - t0) * 1000
    with STATS["lock"]:
        STATS["requests"] += 1; STATS["lat"].append(ms); STATS["scores"].append(s); STATS["decisions"][dec] += 1
    log.info(json.dumps({"event": "scored", "txn_id": txn.txn_id, "score": round(s, 5), "decision": dec, "model": b["version"], "ms": round(ms, 2)}))
    return Decision(txn_id=txn.txn_id, fraud_score=s, decision=dec, model_version=b["version"], reasons=reasons, latency_ms=round(ms, 2))

@app.get("/stats")
def stats():                            # lightweight monitoring: latency percentiles, decision mix, score distribution (for drift checks)
    with STATS["lock"]:
        lat, sc = np.array(STATS["lat"]), np.array(STATS["scores"])
        out = {"requests": STATS["requests"], "errors": STATS["errors"], "decisions": dict(STATS["decisions"])}
    if len(lat):
        out["latency_ms"] = {f"p{q}": round(float(np.percentile(lat, q)), 2) for q in (50, 95, 99)}
        out["score_mean"], out["score_p99"] = round(float(sc.mean()), 5), round(float(np.percentile(sc, 99)), 5)
    return out
