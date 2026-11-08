const mode=document.body.dataset.mode,$=s=>document.querySelector(s),files={};
let docURL=null, refURL=null;
// Expose internals so camera.js can inject captured frames
window._sigFiles = files;
Object.defineProperty(window, '_sigDocURL', { get:()=>docURL, set:v=>{ docURL=v; } });
Object.defineProperty(window, '_sigRefURL', { get:()=>refURL, set:v=>{ refURL=v; } });
document.querySelectorAll(".drop").forEach(d=>{
  const span=d.querySelector("span");
  if(span) d.dataset.defaultText = span.textContent;
  const inp=d.querySelector("input");
  const remBtn=d.querySelector(".drop-remove");
  const prevImg=d.querySelector(".drop-preview");

  const set=f=>{
    if(!f)return;
    files[inp.name]=f;
    d.classList.add("has");
    span.textContent=f.name;
    const url = URL.createObjectURL(f);
    if(prevImg) prevImg.src = url;
    if(inp.name==="document") docURL=url;
    if(inp.name==="reference") refURL=url;
  };

  inp.onchange=()=>set(inp.files[0]);
  d.ondragover=e=>{e.preventDefault();d.classList.add("over")};d.ondragleave=()=>d.classList.remove("over");
  d.ondrop=e=>{e.preventDefault();d.classList.remove("over");set(e.dataTransfer.files[0])};
  if(remBtn){
    remBtn.onclick=e=>{
      e.preventDefault();e.stopPropagation();
      inp.value="";delete files[inp.name];
      d.classList.remove("has");
      if(prevImg) prevImg.src="";
      if(d.dataset.defaultText) span.textContent=d.dataset.defaultText;
      if(inp.name==="document") docURL=null;
      if(inp.name==="reference") refURL=null;
    };
  }
});

const msg={
  queued: "Waiting in the queue for an available AI worker...",
  processing: "Scanning document & running LibreYOLO + Siamese models..."
};

$("#run").onclick=async()=>{
  const st=$("#status");
  if(!files.document||(mode==="similarity"&&!files.reference)){
    st.className="error";
    st.innerHTML="<span>⚠️ Please upload "+(mode==="similarity"?"both document and reference images":"a document image")+" first.</span>";
    return;
  }
  const fd=new FormData();fd.append("document",files.document);if(files.reference)fd.append("reference",files.reference);
  $(".drops").style.display="none";
  $("#run").style.display="none";
  $("#out").style.display="none";
  try{
    st.className="active";
    st.innerHTML='<span class="status-spinner"></span><span>Uploading images to server...</span>';
    const res=await runJob(fd,s=>{
      st.className="active";
      st.innerHTML=`<span class="status-spinner"></span><span>${msg[s]||s}</span>`;
    });
    st.className="";
    st.style.display="none";
    st.innerHTML="";
    show(res);
  }catch(e){
    st.className="error";
    st.innerHTML=`<span>❌ Something went wrong: ${e.message}</span>`;
    $(".drops").style.display="";
    $("#run").style.display="";
  }
};
function resetAll(){
  const out=$("#out");if(out)out.style.display="none";
  const c=$("#cv");if(c){const ctx=c.getContext("2d");ctx.clearRect(0,0,c.width,c.height);}
  const r=$("#res");if(r)r.replaceChildren();
  document.querySelectorAll(".drop").forEach(d=>{
    const inp=d.querySelector("input");if(inp)inp.value="";
    d.classList.remove("has");
    const prev=d.querySelector(".drop-preview");if(prev)prev.src="";
    const span=d.querySelector("span");if(span&&d.dataset.defaultText)span.textContent=d.dataset.defaultText;
  });
  for(const k in files)delete files[k];
  docURL=null;
  refURL=null;
  $(".drops").style.display="";
  $("#run").style.display="";
  const st=$("#status");
  if(st){st.className="";st.innerHTML="";st.style.display="";}
  $(".tool").scrollIntoView({behavior:"smooth"});
}
document.addEventListener("DOMContentLoaded",()=>{
  const bn=$("#btn-next");if(bn)bn.onclick=resetAll;
  const br=$("#btn-reset");if(br)br.onclick=resetAll;
});
const bn=$("#btn-next");if(bn)bn.onclick=resetAll;
const br=$("#btn-reset");if(br)br.onclick=resetAll;

function getVerdict(d) {
  if (d.similarity_score == null) return null;
  const score = d.similarity_score;
  const pct = Math.round(score * 100);
  const isSimilar = score >= 0.7785;
  if (isSimilar) {
    return {
      key: "SIMILAR",
      chip: "SIMILAR",
      title: "✔ SIMILAR",
      desc: `The signatures match (${pct}% similarity). They are written by the same person.`,
      col: "#1f7a6d"
    };
  } else {
    return {
      key: "NOT_SIMILAR",
      chip: "NOT SIMILAR",
      title: "✖ NOT SIMILAR",
      desc: `The signatures do not match (${pct}% similarity). They are not written by the same person.`,
      col: "#d63031"
    };
  }
}

function show(res){
  const c=$("#cv");
  if(c && docURL){
    const img=new Image();
    img.onload=()=>{
      const x=c.getContext("2d");c.width=img.width;c.height=img.height;x.drawImage(img,0,0);
      const lw=Math.max(3,img.width/250);
      res.detections.forEach((d,idx)=>{
        const [a,b,e,f]=d.box;
        const v = getVerdict(d);
        const k = v ? v.col : "#1f7a6d";
        x.lineWidth=lw;x.strokeStyle=k;x.strokeRect(a,b,e-a,f-b);
        x.font=`700 ${Math.max(13,lw*4)}px monospace`;
        const t=d.similarity_score!=null?`#${idx+1}: ${Math.round(d.similarity_score*100)}% (${v ? v.chip : 'MATCH'})`:`Signature #${idx+1} (${Math.round(d.confidence*100)}%)`;
        const tw=x.measureText(t).width;
        const th=Math.max(18,lw*5);
        const ly=Math.max(0,b-th);
        x.fillStyle=k;x.fillRect(a,ly,tw+lw*4,th);
        x.fillStyle="#ffffff";
        x.fillText(t,a+lw*2,ly+th-4);
      });
    };
    img.src=docURL;
  }

  const out=$("#out");
  if(out){
    out.style.display=mode==="similarity"?"block":"grid";
    out.scrollIntoView({behavior:"smooth"});
  }

  // Update document badge if present (detect mode)
  const badge=$("#doc-badge");
  if(badge){
    badge.textContent=res.signatures_found?`${res.signatures_found} FOUND`:"0 FOUND";
    badge.className=`chip ${res.signatures_found ? 'ACCEPT' : 'REJECT'}`;
  }

  // Update summary card (detection mode only)
  const sb=$("#summary-box");
  if(sb && mode === "detect"){
    if(res.signatures_found > 0){
      sb.className = "summary-box found";
      sb.innerHTML = `<div class="summary-title">✔ ${res.signatures_found} Signature${res.signatures_found > 1 ? "s" : ""} Located</div><div class="summary-desc">Located and bounded on the scanned document preview on the left.</div>`;
    } else {
      sb.className = "summary-box empty";
      sb.innerHTML = `<div class="summary-title">✖ No Signatures Located</div><div class="summary-desc">${res.error_message || "No signatures were detected with confidence ≥ 38.5% on this page. Please ensure the document is clear, correctly oriented, and well-lit."}</div>`;
    }
  }

  // Handle invalid state in similarity mode
  if(mode === "similarity"){
    if(res.reference_is_signature === false){
      $("#res").innerHTML = `
        <div class="verdict-banner NOT_SIMILAR" style="margin:0;padding:18px">
          <div class="verdict-title">✖ Reference Is Not A Signature</div>
          <div class="verdict-desc">${res.error_message || "The reference image was not recognized as a valid signature. Please upload a clear signature image."}</div>
        </div>
      `;
      return;
    }
    if(res.signatures_found === 0){
      $("#res").innerHTML = `
        <div class="verdict-banner NOT_SIMILAR" style="margin:0;padding:18px">
          <div class="verdict-title">✖ No Signature Detected on Document</div>
          <div class="verdict-desc">${res.error_message || "No signature was detected on the document image. Please upload a document with a clear signature."}</div>
        </div>
      `;
      return;
    }
  } else {
    if(res.signatures_found === 0){
      $("#res").replaceChildren();
      return;
    }
  }

  // Populate individual signature cards
  $("#res").replaceChildren(...res.detections.map((d,i)=>{
    const el=document.createElement("div");
    const s=d.similarity_score!=null;
    const v=getVerdict(d);

    if (s && v && refURL) {
      el.className = "card similarity-compare-card";
      el.innerHTML = `
        <div class="compare-pair">
          <div class="compare-box">
            <span class="compare-tag doc-tag">📄 Document Signature</span>
            <div class="compare-img-wrap">
              <img src="${window.SIG_API_BASE + d.crop_url}" alt="Detected #${i+1}">
            </div>
            <span class="compare-sub">Detected #${i+1} · ${Math.round(d.confidence*100)}% Conf</span>
          </div>

          <div class="compare-vs">VS</div>

          <div class="compare-box">
            <span class="compare-tag ref-tag">🖋 Reference Signature</span>
            <div class="compare-img-wrap">
              <img src="${refURL}" alt="Reference">
            </div>
            <span class="compare-sub">Target Reference</span>
          </div>
        </div>

        <div class="compare-results">
          <div class="verdict-metric">
            <div>
              <span class="mono" style="font-size:11px;display:block;margin-bottom:2px">SIMILARITY MATCH</span>
              <b style="font:800 32px 'Bricolage Grotesque'">${Math.round(d.similarity_score*100)}% Match</b>
            </div>
            <span class="chip ${v.key}">${v.chip}</span>
          </div>

          <div class="bar ${v.key}"><i style="width:0"></i></div>

          <div class="verdict-banner ${v.key}">
            <div class="verdict-title">${v.title}</div>
            <div class="verdict-desc">${v.desc}</div>
          </div>
        </div>
      `;
      setTimeout(()=>el.querySelector(".bar i").style.width=d.similarity_score*100+"%",100);
    } else {
      el.className = "card";
      const pct = Math.round(d.confidence * 100);
      el.innerHTML = `
        <div class="card-thumb">
          <img src="${window.SIG_API_BASE+d.crop_url}" alt="Signature ${i+1}">
          <span class="mono">SIGNATURE #${i+1}</span>
        </div>
        <div class="card-details">
          <p class="mono" style="font-size:11px;letter-spacing:.05em;text-transform:uppercase">DETECTION CONFIDENCE · <b>${pct}%</b></p>
          <div class="bar ACCEPT" style="margin:6px 0 10px"><i style="width:0"></i></div>
          <div class="verdict-banner ACCEPT" style="margin-top:4px">
            <div class="verdict-title">✔ SIGNATURE LOCATED</div>
            <div class="verdict-desc">Bounding box successfully outlined on document.</div>
          </div>
        </div>
      `;
      setTimeout(() => {
        const b = el.querySelector(".bar i");
        if (b) b.style.width = pct + "%";
      }, 100);
    }
    return el;
  }));
}
