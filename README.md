# Fraud Detection API: CI/CD + REST deployment

REST service that returns **APPROVE / OTP / DECLINE** for a payment, with reason codes, plus the CI/CD pipeline that tests, containerises and deploys it.

## 1. Add your trained model (from the notebook)
1. In Colab, run the **export cell** (copy in `scripts/colab_export_cell.py`) after the last notebook cell. It downloads `deploy_bundle.zip`.
2. Unzip it into this folder (merges `model/`, `tests/data/`, `requirements-ml.txt`). Do not skip: CI fails without the real model, by design.
3. Check `model/model_meta.json` -> `python`. If it is not 3.13, change `PYTHON_VERSION` in `Dockerfile` and `python-version` in `.github/workflows/mlops_pipeline.yml`.

## 2. Run and test locally
```bash
pip install -r requirements-ml.txt -r requirements.txt -r requirements-dev.txt
pytest -q                                   # 13 checks incl. score parity with the notebook
uvicorn app.main:app --port 8000            # then open http://localhost:8000/docs
curl -X POST localhost:8000/score -H 'content-type: application/json' -d @tests/data/sample_fraud.json
```
Endpoints: `POST /score`, `GET /health` (liveness), `GET /ready` (readiness), `GET /model/info`, `GET /stats` (latency p50/p95/p99, decision mix, score stats).

## 3. CI/CD (GitHub Actions)
```bash
git init && git add . && git commit -m "fraud api" && git branch -M main
git remote add origin https://github.com/<you>/<repo>.git && git push -u origin main
```
Pipeline on every push: **test -> build image -> container smoke test -> push to GHCR -> deploy -> verify live -> auto-rollback**.
Pull requests run test + build only.

## 4. Deploy over REST (Render, free tier)
1. Render -> New -> Web Service -> connect the repo -> Runtime **Docker** -> **Auto-Deploy: Off**.
2. Service Settings -> copy the **Deploy Hook URL**.
3. GitHub repo -> Settings -> Secrets and variables -> Actions: secret `RENDER_DEPLOY_HOOK_URL` = hook URL; variable `SERVICE_URL` = `https://<service>.onrender.com`.
4. Push to `main`. The deploy job calls the hook with `&ref=<commit sha>`, waits until `/model/info` reports that commit, scores the known-fraud sample, and re-deploys the previous commit if any check fails.

Free instances sleep when idle; the first call after a sleep is slow. The 100 ms SLA applies to a warm instance.

## 5. Kubernetes (scaling, canary)
`k8s/` has the Deployment (2 replicas, rolling update, probes, resource limits, non-root), Service, HPA (2-10 pods at 65% CPU) and a canary Deployment.
Run **Actions -> k8s-kind-demo -> Run workflow** to apply them to a throwaway kind cluster and call the API through the Service.
