# Connecting this frontend to the FastAPI backend

Static site: no build step. Pages: `index.html`, `detect.html`, `similarity.html`.

## 1. Point it at your API
Edit `js/config.js`:
```js
window.SIG_API_BASE = "https://your-api.onrender.com";   // no trailing slash
```

## 2. Allow the site in CORS (backend `.env`)
```
ALLOWED_ORIGINS=http://localhost:5500,https://your-site.vercel.app
```
Restart the API. Remove `*` and list only your real frontend URL in production.

## 3. What the frontend calls
| Page | Request |
|---|---|
| detect.html | `POST /api/v1/jobs` with `document` |
| similarity.html | `POST /api/v1/jobs` with `document` + `reference` |
| both | `GET /api/v1/jobs/{id}` every 2 s until `completed` or `failed` |

Crops are shown from `SIG_API_BASE + crop_url`. Boxes are drawn from `result.detections[].box`
in the original image's pixels. Verdict colours come from `ACCEPT / REVIEW / REJECT`.

## 4. One backend change needed: make `reference` optional
The detection page sends only the document, but the current API requires `reference`.
In `app/main.py`, change the parameter to `reference: UploadFile | None = File(None)`,
save it only if present, and store `reference_key=None` (make that column nullable in `db.py`).
In `worker/tasks.py`, skip the compare step when `job.reference_key` is empty:
```python
if job.reference_key:
    score, verdict = pipeline.compare(...)
    det["similarity_score"], det["verdict"] = score, verdict
```
The frontend already handles results with and without scores.

## 5. Run locally
`python -m http.server 5500` in this folder, then open http://localhost:5500.
Start the API (`uvicorn app.main:app --reload`) and worker as in the backend README.

## 6. Deploy
Vercel or Cloudflare Pages: connect the repo, framework "Other", no build command, output directory `/`.
Then set `SIG_API_BASE` to the deployed API URL.

## 7. Before going live
- Crops are served by the API's local `/files` folder. When storage moves to Supabase/R2, return full URLs from the API and drop the `SIG_API_BASE +` prefix in `js/tool.js`.
- Replace the "About us" copy with your real team details.
