"""LOCAL SMOKE-TEST ONLY. Trains a tiny stand-in model on synthetic data so you can run the API/tests
before you have the real champion_model.pkl. Never commit its output: the real model comes from the notebook export cell."""
import json, pickle, sys, os
import numpy as np, pandas as pd, sklearn, lightgbm as lgb
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(0)
V = [f"V{i}" for i in range(1, 29)]
n = 6000
d = pd.DataFrame(rng.normal(size=(n, 28)), columns=V)
d["amount"] = rng.gamma(2.0, 40.0, n); d["hour"] = rng.integers(0, 24, n); d["tx_rate_5min"] = rng.integers(0, 60, n)
d["is_fraud"] = (rng.random(n) < 0.03).astype(int)
d.loc[d.is_fraud == 1, ["V14", "V12", "V10"]] -= 4.0                     # make fraud separable
d["log_amount"] = np.log1p(d["amount"])
FEATS = V + ["amount", "log_amount", "hour", "tx_rate_5min"]; IF = V + ["log_amount"]
iso = IsolationForest(n_estimators=100, max_samples=2048, random_state=42, n_jobs=1).fit(d.loc[d.is_fraud == 0, IF])
X = d[FEATS].copy(); X["anomaly_score"] = -iso.score_samples(d[IF])
m = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.05, num_leaves=15, random_state=42, n_jobs=1, verbose=-1).fit(X, d.is_fraud)
s = m.predict_proba(X)[:, 1]
t1, t2 = float(np.quantile(s, 0.95)), float(np.quantile(s, 0.99))
os.makedirs("model", exist_ok=True); os.makedirs("tests/data", exist_ok=True)
pickle.dump({"pipe": {"iso": iso, "model": m}, "version": "dummy", "t1": t1, "t2": t2, "features": FEATS}, open("model/champion_model.pkl", "wb"))
json.dump({"sklearn": sklearn.__version__, "lightgbm": lgb.__version__, "numpy": np.__version__, "pandas": pd.__version__,
           "python": sys.version.split()[0], "note": "DUMMY MODEL"}, open("model/model_meta.json", "w"), indent=1)

def payload(i):
    r = d.iloc[i]; p = {c: float(r[c]) for c in V + ["amount"]}; p.update(hour=int(r["hour"]), tx_rate_5min=int(r["tx_rate_5min"]), txn_id=str(i))
    sc = float(s[i]); p["_expected_score"] = sc; p["_expected_decision"] = "DECLINE" if sc >= t2 else ("OTP" if sc >= t1 else "APPROVE"); return p
fi = int(np.argmax(np.where(d.is_fraud == 1, s, -1))); g = np.where(d.is_fraud == 0)[0]; gi = int(g[np.argsort(s[g])[len(g) // 2]])
json.dump(payload(fi), open("tests/data/sample_fraud.json", "w"), indent=1); json.dump(payload(gi), open("tests/data/sample_genuine.json", "w"), indent=1)
print("dummy model + samples written")
