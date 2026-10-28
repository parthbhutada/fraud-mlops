# ---- Export deployment bundle: model + pinned versions + REAL sample payloads for the CI tests ----
import sys, shutil, sklearn, lightgbm
B = "deploy_bundle"; shutil.rmtree(B, ignore_errors=True)
os.makedirs(f"{B}/model"); os.makedirs(f"{B}/tests/data")

# 1) champion model, in the exact structure the API loads
pickle.dump({"pipe": champ["pipe"], "version": champ["version"], "t1": float(champ["t1"]), "t2": float(champ["t2"]), "features": BASE_FEATS},
            open(f"{B}/model/champion_model.pkl", "wb"))

# 2) library versions: a pickle only loads safely on the same versions, so the API/CI pin them
meta = {"model_version": champ["version"], "otp_threshold": float(champ["t1"]), "decline_threshold": float(champ["t2"]), "data_hash": data_hash,
        "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__, "numpy": np.__version__, "pandas": pd.__version__, "python": sys.version.split()[0]}
json.dump(meta, open(f"{B}/model/model_meta.json", "w"), indent=1)
open(f"{B}/requirements-ml.txt", "w").write(f"numpy=={np.__version__}\npandas=={pd.__version__}\nscikit-learn=={sklearn.__version__}\nlightgbm=={lightgbm.__version__}\n")

# 3) two real transactions + the notebook's own score for them (CI checks the API reproduces it = no training/serving skew)
sc = score(champ["pipe"], te0); y = te0.is_fraud.values
def payload(i):
    r = te0.iloc[i]
    p = {c: float(r[c]) for c in V_COLS + ["amount"]}
    p.update(hour=int(r["hour"]), tx_rate_5min=int(r["tx_rate_5min"]), txn_id=str(int(r["txn_id"])))
    s = float(sc[i]); p["_expected_score"] = s
    p["_expected_decision"] = "DECLINE" if s >= champ["t2"] else ("OTP" if s >= champ["t1"] else "APPROVE")
    return p
fi = int(np.argmax(np.where(y == 1, sc, -1)))                            # highest-scored known fraud
g = np.where(y == 0)[0]; gi = int(g[np.argsort(sc[g])[len(g) // 2]])     # typical genuine payment (median score)
json.dump(payload(fi), open(f"{B}/tests/data/sample_fraud.json", "w"), indent=1)
json.dump(payload(gi), open(f"{B}/tests/data/sample_genuine.json", "w"), indent=1)

shutil.make_archive("deploy_bundle", "zip", B)
print("model", champ["version"], "| thresholds", round(champ["t1"], 4), round(champ["t2"], 4), "| sklearn", sklearn.__version__, "lightgbm", lightgbm.__version__, "python", sys.version.split()[0])
print("fraud sample ->", payload(fi)["_expected_decision"], "| genuine sample ->", payload(gi)["_expected_decision"])
try:
    from google.colab import files; files.download("deploy_bundle.zip")
except Exception:
    pass
