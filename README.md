# Signature backend (Windows, no Docker)

## 1. Setup (once)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```
Edit `.env`: paste your Upstash `REDIS_URL`, and put your detector weights at `weights\best.pt`.

## 2. Run (two terminals, venv active in both)
Terminal 1, the API:
```
uvicorn app.main:app --reload
```
Terminal 2, the worker (`--pool=solo` is required on Windows):
```
celery -A app.celery_app worker --pool=solo --loglevel=info
```
Wait for the worker to print that both models are loaded.

## 3. Test
Open http://127.0.0.1:8000/docs, use `POST /api/v1/jobs` with a document image and a
reference signature, copy the `job_id`, then call `GET /api/v1/jobs/{job_id}` until
`status` is `completed`.

Or from the command line:
```
curl.exe -F "document=@doc.jpg" -F "reference=@ref.png" http://127.0.0.1:8000/api/v1/jobs
curl.exe http://127.0.0.1:8000/api/v1/jobs/<job_id>
```

## Job status values
queued -> processing -> completed (or failed, with an `error` message)
