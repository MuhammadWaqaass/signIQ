/**
 * camera.js — Live camera capture for SignIQ
 * Snap → flash → preview immediately (no countdown).
 * "Use This" injects JPEG blob into the existing drop zone / files{} pipeline.
 */
(function () {
  document.addEventListener("DOMContentLoaded", () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;

    /* ── Build modal once ─────────────────────────────────────────── */
    document.body.insertAdjacentHTML("beforeend", `
<div id="cam-modal" role="dialog" aria-modal="true" aria-label="Camera capture">
  <div id="cam-bd"></div>
  <div id="cam-shell">

    <!-- ① LIVE VIEW (shown first) -->
    <div id="cam-live">
      <video id="cam-vid" autoplay playsinline muted></video>

      <!-- corner guide brackets -->
      <div id="cam-corners">
        <i class="cc tl"></i><i class="cc tr"></i>
        <i class="cc bl"></i><i class="cc br"></i>
      </div>

      <!-- white flash overlay -->
      <div id="cam-flash"></div>

      <!-- top bar: label + close -->
      <div id="cam-top">
        <span id="cam-lbl" class="cmono">📷 Camera</span>
        <button id="cam-x" aria-label="Close">✕</button>
      </div>

      <!-- bottom bar: flip + shutter + spacer -->
      <div id="cam-bar">
        <button id="cam-flip" aria-label="Flip camera" title="Switch camera">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 4v6h6"/><path d="M23 20v-6h-6"/><path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4-4.64 4.36A9 9 0 0 1 3.51 15"/></svg>
        </button>
        <button id="cam-snap" aria-label="Take photo">
          <span id="cam-ring"></span>
        </button>
        <div style="width:48px"></div>
      </div>

      <!-- hint -->
      <div id="cam-hint" class="cmono">Keep document flat &amp; in frame</div>
    </div>

    <!-- ② PREVIEW VIEW (hidden until snap) -->
    <div id="cam-prev" hidden>
      <img id="cam-img" alt="Captured photo">
      <div id="cam-prev-actions">
        <button id="cam-retake">↩ Retake</button>
        <button id="cam-use">✓ Use This Photo</button>
      </div>
    </div>

  </div>
</div>`);

    /* ── Refs ─────────────────────────────────────────────────────── */
    const modal    = document.getElementById("cam-modal");
    const backdrop = document.getElementById("cam-bd");
    const liveView = document.getElementById("cam-live");
    const prevView = document.getElementById("cam-prev");
    const video    = document.getElementById("cam-vid");
    const flash    = document.getElementById("cam-flash");
    const snapBtn  = document.getElementById("cam-snap");
    const flipBtn  = document.getElementById("cam-flip");
    const closeBtn = document.getElementById("cam-x");
    const prevImg  = document.getElementById("cam-img");
    const retakeBtn= document.getElementById("cam-retake");
    const useBtn   = document.getElementById("cam-use");
    const lbl      = document.getElementById("cam-lbl");

    let stream = null, blob = null, targetDrop = null;
    let facing = "environment", busy = false;

    /* ── Trigger buttons — injected INTO each drop zone ──────────── */
    document.querySelectorAll(".drop").forEach(drop => {
      const inp = drop.querySelector("input[type=file]");
      if (!inp) return;

      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "cam-open-btn";
      btn.setAttribute("aria-label", "Use camera");
      btn.innerHTML = `<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>Use Camera`;
      btn.onclick = e => { e.preventDefault(); targetDrop = drop; open(); };

      /* Place inside the drop-content div so it sits under the label text */
      const content = drop.querySelector(".drop-content");
      if (content) content.appendChild(btn);
      else drop.appendChild(btn);
    });

    /* ── Open / close ─────────────────────────────────────────────── */
    async function open() {
      showLive();
      modal.classList.add("open");
      document.body.style.overflow = "hidden";
      await startStream();
    }

    function close() {
      stopStream();
      modal.classList.remove("open");
      document.body.style.overflow = "";
      blob = null; busy = false;
    }

    closeBtn.onclick = close;
    backdrop.onclick = close;
    document.addEventListener("keydown", e => {
      if (e.key === "Escape" && modal.classList.contains("open")) close();
    });

    /* ── Stream ───────────────────────────────────────────────────── */
    async function startStream() {
      stopStream();
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: facing, width: { ideal: 1920 }, height: { ideal: 1080 } },
          audio: false
        });
        video.srcObject = stream;
        await video.play();
      } catch (err) {
        const msg = err.name === "NotAllowedError" ? "Camera permission denied"
                  : err.name === "NotFoundError"   ? "No camera detected"
                  : "Cannot access camera";
        document.getElementById("cam-hint").textContent = "⚠ " + msg;
      }
    }

    function stopStream() {
      if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
      video.srcObject = null;
    }

    /* ── Flip ─────────────────────────────────────────────────────── */
    flipBtn.onclick = () => { facing = facing === "environment" ? "user" : "environment"; startStream(); };

    /* ── Snap — instant, no countdown ────────────────────────────── */
    snapBtn.onclick = async () => {
      if (busy) return;
      busy = true;
      snapBtn.disabled = true;

      /* Flash */
      flash.classList.add("on");
      await tick(80);
      flash.classList.remove("on");

      /* Capture frame */
      const c = document.createElement("canvas");
      c.width  = video.videoWidth  || 1280;
      c.height = video.videoHeight || 720;
      c.getContext("2d").drawImage(video, 0, 0);

      c.toBlob(b => {
        blob = b;
        prevImg.src = URL.createObjectURL(b);
        stopStream();          // kill camera IMMEDIATELY
        showPreview();
        snapBtn.disabled = false;
        busy = false;
      }, "image/jpeg", 0.93);
    };

    /* ── Preview actions ─────────────────────────────────────────── */
    retakeBtn.onclick = async () => {
      blob = null; prevImg.src = "";
      showLive();
      await startStream();
    };

    useBtn.onclick = () => {
      if (!blob || !targetDrop) { close(); return; }

      const inp  = targetDrop.querySelector("input[type=file]");
      const ts   = new Date().toISOString().slice(0,19).replace(/[:T]/g,"-");
      const name = `cam-${inp ? inp.name : "photo"}-${ts}.jpg`;
      const file = new File([blob], name, { type: "image/jpeg" });

      /* Fire custom event — tool.js listener will inject into drop */
      targetDrop.dispatchEvent(new CustomEvent("camera:captured", {
        bubbles: true, detail: { file, dropEl: targetDrop }
      }));
      close();
    };

    /* ── View switchers ───────────────────────────────────────────── */
    function showLive() {
      liveView.hidden = false;
      prevView.hidden = true;
      lbl.textContent = "📷 Camera";
    }
    function showPreview() {
      liveView.hidden = true;
      prevView.hidden = false;
      lbl.textContent = "📸 Review photo";
    }

    /* ── Event: inject captured file into drop zone ──────────────── */
    document.addEventListener("camera:captured", e => {
      const { file, dropEl } = e.detail;
      const inp     = dropEl.querySelector("input[type=file]");
      const span    = dropEl.querySelector("span");
      const preview = dropEl.querySelector(".drop-preview");
      if (!inp) return;

      if (window._sigFiles) window._sigFiles[inp.name] = file;
      dropEl.classList.add("has");
      if (span) span.textContent = file.name;
      if (preview) preview.src = URL.createObjectURL(file);

      const url = URL.createObjectURL(file);
      if (inp.name === "document") window._sigDocURL = url;
      if (inp.name === "reference") window._sigRefURL = url;
    });

    function tick(ms) { return new Promise(r => setTimeout(r, ms)); }
  });
})();
