const io=new IntersectionObserver(e=>e.forEach(x=>x.isIntersecting&&(x.target.classList.add("in"),io.unobserve(x.target))),{threshold:.15});
document.querySelectorAll(".rv").forEach((el,i)=>{el.style.transitionDelay=(i%3)*.12+"s";io.observe(el)});
const bg=document.querySelector(".burger"),nv=document.querySelector("nav");
if(bg){const t=o=>{nv.classList.toggle("open",o);bg.setAttribute("aria-expanded",o)};bg.onclick=()=>t(!nv.classList.contains("open"));nv.onclick=e=>e.target.tagName==="A"&&t(false)}
