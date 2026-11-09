// Talks to the FastAPI backend. Returns the job result once the worker finishes.
async function runJob(form, onStatus) {
  const base = (window.SIG_API_BASE || "").replace(/\/+$/, "");
  let r;
  try {
    r = await fetch(base + "/api/v1/jobs", { method: "POST", body: form });
  } catch (err) {
    throw new Error("Cannot reach backend at " + base + " (" + err.message + "). Check CORS or backend status.");
  }
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    throw new Error(e.detail || "Upload failed (" + r.status + ")");
  }
  const { job_id } = await r.json();
  for (let i = 0; i < 90; i++) {
    await new Promise((s) => setTimeout(s, 2000));
    try {
      const res = await fetch(`${base}/api/v1/jobs/${job_id}`);
      if (!res.ok) continue;
      const j = await res.json();
      onStatus(j.status);
      if (j.status === "completed") return j.result;
      if (j.status === "failed") throw new Error(j.error || "Job failed");
    } catch (pollErr) {
      if (pollErr.message && pollErr.message.includes("Job failed")) throw pollErr;
      console.warn("Poll attempt hiccup, retrying...", pollErr);
    }
  }
  throw new Error("Timed out waiting for the worker");
}
