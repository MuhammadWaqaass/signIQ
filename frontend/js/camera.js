/**
 * camera.js — Compact, theme-matching live camera capture for SignIQ
 * 
 * Features:
 * - Stylish camera button integrated right into the upload drop box
 * - Compact modal dialog matching SignIQ's design system
 * - Instant capture without lag
 * - On capture: live camera vanishes completely, only photo + 2 prominent theme buttons appear
 * - Retake restarts camera; Use Photo injects file directly into SignIQ pipeline
 */
(function () {
  document.addEventListener("DOMContentLoaded", () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;

    /* ── Inject Modal HTML ────────────────────────────────────────── */
    document.body.insertAdjacentHTML("beforeend", `
<div id="cam-modal" class="cam-modal" aria-hidden="true" role="dialog">
  <div id="cam-bd" class="cam-backdrop"></div>
  <div class="cam-card">
    
    <!-- Top Header -->
    <div class="cam-header">
      <div class="cam-title">
        <span class="cam-title-icon">📷</span>
        <span id="cam-heading">Camera Capture</span>
      </div>
      <button type="button" id="cam-close" class="cam-btn-close" aria-label="Close">✕</button>
    </div>

    <!-- Live Viewfinder -->
    <div id="cam-live-box" class="cam-stage">
      <div class="cam-viewfinder">
        <video id="cam-vid" autoplay playsinline muted></video>
        <div class="cam-brackets">
          <span class="b-tl"></span><span class="b-tr"></span>
          <span class="b-bl"></span><span class="b-br"></span>
        </div>
        <div id="cam-flash" class="cam-flash"></div>
      </div>
      <p class="cam-tip">Center signature or document inside the frame</p>
      <div class="cam-controls">
        <button type="button" id="cam-flip" class="cam-btn-flip" title="Switch camera" aria-label="Switch camera">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M20 10c0-4.418-3.582-8-8-8s-8 3.582-8 8v2"/>
            <path d="m7 9-3 3-3-3"/>
            <path d="M4 14c0 4.418 3.582 8 8 8s8-3.582 8-8v-2"/>
            <path d="m17 15 3-3 3 3"/>
          </svg>
        </button>
        <button type="button" id="cam-snap" class="cam-btn-shutter" aria-label="Take Photo">
          <span class="shutter-core"></span>
        </button>
        <div class="cam-control-spacer"></div>
      </div>
    </div>

    <!-- Preview View (Displaces camera on snap) -->
    <div id="cam-preview-box" class="cam-stage" style="display:none;">
      <div class="cam-preview-wrap">
        <img id="cam-preview-img" alt="Captured photo" />
      </div>
      <div class="cam-actions-row">
        <button type="button" id="cam-retake" class="cam-btn-action cam-btn-retake">
          <span>↩</span> Retake
        </button>
        <button type="button" id="cam-use" class="cam-btn-action cam-btn-use">
          <span>✓</span> Use This Photo
        </button>
      </div>
    </div>

  </div>
</div>`);

    /* ── Elements ─────────────────────────────────────────────────── */
    const modal       = document.getElementById("cam-modal");
    const backdrop    = document.getElementById("cam-bd");
    const closeBtn    = document.getElementById("cam-close");
    const heading     = document.getElementById("cam-heading");
    const liveBox     = document.getElementById("cam-live-box");
    const previewBox  = document.getElementById("cam-preview-box");
    const video       = document.getElementById("cam-vid");
    const flash       = document.getElementById("cam-flash");
    const flipBtn     = document.getElementById("cam-flip");
    const snapBtn     = document.getElementById("cam-snap");
    const previewImg  = document.getElementById("cam-preview-img");
    const retakeBtn   = document.getElementById("cam-retake");
    const useBtn      = document.getElementById("cam-use");

    let stream = null;
    let currentBlob = null;
    let targetDrop = null;
    let facingMode = "environment";
    let isProcessing = false;

    /* ── Inject "Take Photo" button into every .drop box ─────────── */
    document.querySelectorAll(".drop").forEach(drop => {
      const inp = drop.querySelector("input[type=file]");
      if (!inp) return;

      const triggerBtn = document.createElement("button");
      triggerBtn.type = "button";
      triggerBtn.className = "cam-drop-trigger";
      triggerBtn.innerHTML = `
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
          <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
          <circle cx="12" cy="13" r="4"/>
        </svg>
        <span>Open Camera</span>
      `;

      triggerBtn.addEventListener("click", e => {
        e.preventDefault();
        e.stopPropagation(); // prevent opening file browser
        targetDrop = drop;
        openModal();
      });

      const content = drop.querySelector(".drop-content") || drop;
      content.appendChild(triggerBtn);
    });

    /* ── Modal Show / Hide ────────────────────────────────────────── */
    async function openModal() {
      currentBlob = null;
      isProcessing = false;
      snapBtn.disabled = false;

      // Ensure live camera is shown, preview is hidden
      liveBox.style.display = "flex";
      previewBox.style.display = "none";
      heading.textContent = "Take Photo";

      modal.classList.add("active");
      document.body.style.overflow = "hidden";
      await startCamera();
    }

    function closeModal() {
      stopCamera();
      modal.classList.remove("active");
      document.body.style.overflow = "";
      currentBlob = null;
      isProcessing = false;
    }

    closeBtn.addEventListener("click", closeModal);
    backdrop.addEventListener("click", closeModal);
    document.addEventListener("keydown", e => {
      if (e.key === "Escape" && modal.classList.contains("active")) {
        closeModal();
      }
    });

    /* ── Camera Stream ────────────────────────────────────────────── */
    async function startCamera() {
      stopCamera();
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: facingMode,
            width: { ideal: 1920 },
            height: { ideal: 1080 }
          },
          audio: false
        });
        video.srcObject = stream;
        await video.play();

        // Check if multiple camera inputs exist to show flip button
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoInputs = devices.filter(d => d.kind === "videoinput");
        flipBtn.style.visibility = videoInputs.length > 1 ? "visible" : "hidden";
      } catch (err) {
        console.warn("Camera access failed:", err);
        const vf = document.querySelector(".cam-viewfinder");
        const msg = err.name === "NotAllowedError" ? "Camera permission blocked" : "Unable to access camera";
        vf.insertAdjacentHTML("beforeend", `<div class="cam-error-overlay">⚠️ ${msg}</div>`);
      }
    }

    function stopCamera() {
      if (stream) {
        stream.getTracks().forEach(t => t.stop());
        stream = null;
      }
      video.srcObject = null;
      const errOverlay = document.querySelector(".cam-error-overlay");
      if (errOverlay) errOverlay.remove();
    }

    /* ── Flip Camera ──────────────────────────────────────────────── */
    flipBtn.addEventListener("click", () => {
      facingMode = facingMode === "environment" ? "user" : "environment";
      startCamera();
    });

    /* ── Instant Snap ─────────────────────────────────────────────── */
    snapBtn.addEventListener("click", async () => {
      if (isProcessing) return;
      isProcessing = true;
      snapBtn.disabled = true;

      // 1. Shutter flash effect
      flash.classList.add("fired");
      await new Promise(r => setTimeout(r, 70));

      // 2. Render frame to offscreen canvas
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth || 1280;
      canvas.height = video.videoHeight || 720;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0);

      canvas.toBlob(blob => {
        currentBlob = blob;
        previewImg.src = URL.createObjectURL(blob);

        // 3. Immediately shut down camera stream & disappear live view
        stopCamera();
        flash.classList.remove("fired");
        liveBox.style.display = "none";

        // 4. Show preview with 2 prominent theme buttons
        heading.textContent = "Review Photo";
        previewBox.style.display = "flex";

        isProcessing = false;
        snapBtn.disabled = false;
      }, "image/jpeg", 0.94);
    });

    /* ── Preview Buttons ─────────────────────────────────────────── */
    retakeBtn.addEventListener("click", async () => {
      currentBlob = null;
      previewImg.src = "";
      previewBox.style.display = "none";
      liveBox.style.display = "flex";
      heading.textContent = "Take Photo";
      await startCamera();
    });

    useBtn.addEventListener("click", () => {
      if (!currentBlob || !targetDrop) {
        closeModal();
        return;
      }

      const inp = targetDrop.querySelector("input[type=file]");
      const ts = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
      const fileName = `snap-${inp ? inp.name : "photo"}-${ts}.jpg`;
      const file = new File([currentBlob], fileName, { type: "image/jpeg" });

      targetDrop.dispatchEvent(new CustomEvent("camera:captured", {
        bubbles: true,
        detail: { file, dropEl: targetDrop }
      }));

      closeModal();
    });

    /* ── Update files in tool.js store ────────────────────────────── */
    document.addEventListener("camera:captured", e => {
      const { file, dropEl } = e.detail;
      const inp = dropEl.querySelector("input[type=file]");
      const span = dropEl.querySelector("span");
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

  });
})();
