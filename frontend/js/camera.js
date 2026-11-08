/**
 * camera.js — Live camera capture for SignIQ
 *
 * Adds a 📷 button under every .drop zone.
 * On click → opens fullscreen modal → live preview → countdown → snap
 * → shows preview → "Use This" injects blob as a File into the existing
 *   files{} store and drop UI, exactly as if the user had picked a file.
 *
 * Zero backend changes needed — the captured frame becomes a JPEG File
 * and flows through the same FormData → /jobs → Celery pipeline.
 */

(function () {
  /* ── Wait for DOM ──────────────────────────────────────────────── */
  document.addEventListener("DOMContentLoaded", () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;

    /* ── Inject modal HTML once ───────────────────────────────────── */
    const tpl = document.createElement("div");
    tpl.innerHTML = `
<div id="cam-modal" role="dialog" aria-modal="true" aria-label="Camera capture">
  <div id="cam-backdrop"></div>
  <div id="cam-shell">

    <div id="cam-header">
      <span id="cam-label" class="cam-mono">📷 LIVE CAMERA</span>
      <button id="cam-close" aria-label="Close camera" title="Close">✕</button>
    </div>

    <div id="cam-body">

      <!-- Live feed view -->
      <div id="cam-live-view">
        <div id="cam-viewfinder">
          <video id="cam-video" autoplay playsinline muted></video>
          <div id="cam-corners">
            <span class="cc tl"></span><span class="cc tr"></span>
            <span class="cc bl"></span><span class="cc br"></span>
          </div>
          <div id="cam-flash"></div>
          <div id="cam-countdown"></div>
        </div>

        <div id="cam-hint" class="cam-mono">Position document flat · Keep all edges in frame</div>

        <div id="cam-controls">
          <button id="cam-flip" title="Flip camera" aria-label="Switch camera">⇄</button>
          <button id="cam-snap" aria-label="Capture">
            <span id="cam-shutter-ring"></span>
          </button>
          <div id="cam-spacer"></div><!-- balances flip btn -->
        </div>
      </div>

      <!-- Preview view -->
      <div id="cam-preview-view" hidden>
        <img id="cam-preview-img" alt="Captured image">
        <div id="cam-preview-label" class="cam-mono">Review your capture</div>
        <div id="cam-preview-actions">
          <button id="cam-retake" class="cam-btn cam-btn-sec">↩ Retake</button>
          <button id="cam-use"    class="cam-btn cam-btn-pri">✓ Use This</button>
        </div>
      </div>

    </div><!-- /cam-body -->
  </div><!-- /cam-shell -->
</div><!-- /cam-modal -->`;
    document.body.appendChild(tpl.firstElementChild);

    /* ── State ────────────────────────────────────────────────────── */
    const modal      = document.getElementById("cam-modal");
    const video      = document.getElementById("cam-video");
    const flash      = document.getElementById("cam-flash");
    const countdown  = document.getElementById("cam-countdown");
    const liveView   = document.getElementById("cam-live-view");
    const previewView= document.getElementById("cam-preview-view");
    const previewImg = document.getElementById("cam-preview-img");
    const snapBtn    = document.getElementById("cam-snap");
    const flipBtn    = document.getElementById("cam-flip");
    const retakeBtn  = document.getElementById("cam-retake");
    const useBtn     = document.getElementById("cam-use");
    const closeBtn   = document.getElementById("cam-close");
    const backdrop   = document.getElementById("cam-backdrop");
    const camLabel   = document.getElementById("cam-label");

    let stream       = null;
    let capturedBlob = null;
    let targetDrop   = null;   // which .drop zone triggered open
    let facingMode   = "environment"; // prefer rear camera
    let isShooting   = false;

    /* ── Add 📷 trigger buttons under each .drop ─────────────────── */
    document.querySelectorAll(".drop").forEach(drop => {
      const inp = drop.querySelector("input[type=file]");
      if (!inp) return;

      const wrapper = document.createElement("div");
      wrapper.className = "cam-trigger-wrap";

      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "cam-trigger";
      btn.innerHTML = `<span class="cam-trigger-icon">📷</span><span>Use Camera</span>`;
      btn.title = "Capture with camera";

      btn.addEventListener("click", e => {
        e.preventDefault();
        targetDrop = drop;
        openCamera();
      });

      wrapper.appendChild(btn);
      // Insert below the drop label
      drop.insertAdjacentElement("afterend", wrapper);
    });

    /* ── Open / close ─────────────────────────────────────────────── */
    async function openCamera() {
      showLive();
      modal.classList.add("open");
      document.body.style.overflow = "hidden";
      await startStream();
    }

    function closeCamera() {
      stopStream();
      modal.classList.remove("open");
      document.body.style.overflow = "";
      capturedBlob = null;
      isShooting   = false;
      showLive();
    }

    closeBtn.addEventListener("click", closeCamera);
    backdrop.addEventListener("click", closeCamera);
    document.addEventListener("keydown", e => {
      if (e.key === "Escape" && modal.classList.contains("open")) closeCamera();
    });

    /* ── Stream ───────────────────────────────────────────────────── */
    async function startStream() {
      stopStream();
      const constraints = {
        video: { facingMode, width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false
      };
      try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        await video.play();
        // Hide flip btn if only one camera
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoCams = devices.filter(d => d.kind === "videoinput");
        flipBtn.style.display = videoCams.length > 1 ? "" : "hidden";
      } catch (err) {
        countdown.textContent = "⚠ " + (
          err.name === "NotAllowedError"  ? "Camera permission denied" :
          err.name === "NotFoundError"    ? "No camera found"          :
                                           "Cannot open camera"
        );
        countdown.style.display = "flex";
      }
    }

    function stopStream() {
      if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
      video.srcObject = null;
    }

    /* ── Flip camera ─────────────────────────────────────────────── */
    flipBtn.addEventListener("click", () => {
      facingMode = facingMode === "environment" ? "user" : "environment";
      startStream();
    });

    /* ── Capture with countdown ───────────────────────────────────── */
    snapBtn.addEventListener("click", async () => {
      if (isShooting) return;
      isShooting = true;
      snapBtn.disabled = true;

      await runCountdown(3);

      /* Shutter flash */
      flash.classList.add("flash-go");
      await delay(350);
      flash.classList.remove("flash-go");

      /* Draw frame to canvas */
      const canvas = document.createElement("canvas");
      canvas.width  = video.videoWidth  || 1280;
      canvas.height = video.videoHeight || 720;
      canvas.getContext("2d").drawImage(video, 0, 0);

      canvas.toBlob(blob => {
        capturedBlob = blob;
        previewImg.src = URL.createObjectURL(blob);
        showPreview();
        snapBtn.disabled = false;
        isShooting = false;
      }, "image/jpeg", 0.92);
    });

    async function runCountdown(from) {
      countdown.style.display = "flex";
      for (let n = from; n >= 1; n--) {
        countdown.textContent = n;
        countdown.classList.add("pop");
        await delay(700);
        countdown.classList.remove("pop");
        await delay(300);
      }
      countdown.style.display = "none";
    }

    /* ── Preview actions ─────────────────────────────────────────── */
    retakeBtn.addEventListener("click", () => {
      capturedBlob = null;
      previewImg.src = "";
      showLive();
    });

    useBtn.addEventListener("click", () => {
      if (!capturedBlob || !targetDrop) { closeCamera(); return; }

      /* Turn the blob into a File and inject it into the drop zone */
      const ts   = new Date().toISOString().slice(0,19).replace(/[:T]/g,"-");
      const inp  = targetDrop.querySelector("input[type=file]");
      const name = `camera-${inp ? inp.name : "capture"}-${ts}.jpg`;
      const file = new File([capturedBlob], name, { type: "image/jpeg" });

      /* Inject into the existing files{} map via a CustomEvent that
         tool.js listens for — OR trigger the same logic manually.      */
      const ev = new CustomEvent("camera:captured", {
        bubbles: true,
        detail: { file, dropEl: targetDrop }
      });
      targetDrop.dispatchEvent(ev);

      closeCamera();
    });

    /* ── View helpers ─────────────────────────────────────────────── */
    function showLive() {
      liveView.hidden    = false;
      previewView.hidden = true;
      camLabel.textContent = "📷 LIVE CAMERA";
    }
    function showPreview() {
      liveView.hidden    = true;
      previewView.hidden = false;
      camLabel.textContent = "🖼 REVIEW CAPTURE";
    }

    /* ── Wire camera:captured into tool.js's files store ─────────── */
    document.addEventListener("camera:captured", e => {
      const { file, dropEl } = e.detail;
      const inp     = dropEl.querySelector("input[type=file]");
      const span    = dropEl.querySelector("span");
      const prevImg = dropEl.querySelector(".drop-preview");
      if (!inp) return;

      /* Update the files{} object exposed by tool.js */
      if (window._sigFiles) window._sigFiles[inp.name] = file;

      dropEl.classList.add("has");
      if (span) span.textContent = file.name;
      if (prevImg) {
        prevImg.src = URL.createObjectURL(file);
      }

      /* Keep docURL / refURL in sync */
      const url = URL.createObjectURL(file);
      if (inp.name === "document") window._sigDocURL = url;
      if (inp.name === "reference") window._sigRefURL = url;
    });

    function delay(ms) { return new Promise(r => setTimeout(r, ms)); }
  });
})();
