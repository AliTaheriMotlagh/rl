// Shared helpers: navigation, charts, sliders, colors.

const LESSONS = [
  { href: "index.html", n: "", title: "خانه" },
  { href: "1-env.html", n: 1, title: "محیط و پاداش" },
  { href: "2-qlearning.html", n: 2, title: "Q-Learning" },
  { href: "3-snake-q.html", n: 3, title: "مار با Q-Table" },
  { href: "4-neural.html", n: 4, title: "شبکهٔ عصبی" },
  { href: "5-dqn.html", n: 5, title: "Deep Q-Network" },
  { href: "6-tokens.html", n: 6, title: "توکن و Bigram" },
  { href: "7-gpt.html", n: 7, title: "ساخت GPT" },
  { href: "8-rlhf.html", n: 8, title: "RLHF" },
];

const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => Array.from(root.querySelectorAll(s));
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : String(v));
const faNum = (n) => String(n).replace(/\d/g, (d) => "۰۱۲۳۴۵۶۷۸۹"[d]);

function buildNav() {
  const here = location.pathname.split("/").pop() || "index.html";
  const nav = document.createElement("nav");
  nav.className = "topnav";
  nav.innerHTML = `<div class="inner">${LESSONS.map(
    (l) => `<a href="${l.href}" class="${l.href === here ? "active" : ""}">${l.n ? `<span class="num">${faNum(l.n)}</span> ` : "🏠 "}${l.title}</a>`
  ).join("")}</div>`;
  document.body.prepend(nav);
  const idx = LESSONS.findIndex((l) => l.href === here);
  const pager = document.getElementById("pager");
  if (pager && idx >= 0) {
    const prev = LESSONS[idx - 1], next = LESSONS[idx + 1];
    pager.className = "pager";
    pager.innerHTML =
      (prev ? `<a href="${prev.href}"><span>→ قبلی</span>${prev.title}</a>` : "<div></div>") +
      (next ? `<a class="next" href="${next.href}"><span>بعدی ←</span>درس ${faNum(next.n)}: ${next.title}</a>` : "<div></div>");
  }
  const active = nav.querySelector(".active");
  if (active) active.scrollIntoView({ inline: "center", block: "nearest" });
}
document.addEventListener("DOMContentLoaded", buildNav);

// Slider helper: <div class="slider" data-key> with label + range input.
// Returns an object whose .value is always current.
function slider(container, { label, min, max, step, value, digits = 2, onChange }) {
  const wrap = document.createElement("div");
  wrap.className = "slider";
  wrap.innerHTML = `<label><span>${label}</span><b></b></label><input type="range" min="${min}" max="${max}" step="${step}" value="${value}">`;
  const input = wrap.querySelector("input"), out = wrap.querySelector("b");
  const s = { value: Number(value), input };
  const upd = () => {
    s.value = Number(input.value);
    out.textContent = s.value.toFixed(digits);
    onChange && onChange(s.value);
  };
  input.addEventListener("input", upd);
  out.textContent = s.value.toFixed(digits);
  s.set = (v) => { input.value = v; upd(); };
  (typeof container === "string" ? $(container) : container).appendChild(wrap);
  return s;
}

// Fit a canvas to its CSS size (sharp on retina). Returns {ctx, w, h} in CSS pixels.
function fitCanvas(canvas, height) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 300, h = height || canvas.clientHeight || 200;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.direction = "ltr"; // numbers like -0.5 must not be flipped by the RTL page
  return { ctx, w, h };
}

// Simple line chart: raw values faint, moving average bold.
class LineChart {
  constructor(canvas, { series, window = 20, logY = false, yMin = null }) {
    this.canvas = canvas;
    this.series = series.map((s) => ({ ...s, data: [] }));
    this.window = window;
    this.logY = logY;
    this.yMin = yMin;
  }
  push(i, v) { this.series[i].data.push(v); }
  clear() { this.series.forEach((s) => (s.data = [])); this.draw(); }
  smooth(d) {
    const out = [], w = this.window;
    let s = 0;
    for (let i = 0; i < d.length; i++) {
      s += d[i];
      if (i >= w) s -= d[i - w];
      out.push(s / Math.min(i + 1, w));
    }
    return out;
  }
  draw() {
    const { ctx, w, h } = fitCanvas(this.canvas);
    ctx.clearRect(0, 0, w, h);
    const pad = { l: 44, r: 10, t: 22, b: 20 };
    const tr = (v) => (this.logY ? Math.log10(Math.max(v, 1e-4)) : v);
    let n = 0, lo = Infinity, hi = -Infinity;
    for (const s of this.series) {
      n = Math.max(n, s.data.length);
      for (const v of s.data) { const t = tr(v); if (t < lo) lo = t; if (t > hi) hi = t; }
    }
    if (this.yMin !== null) lo = Math.min(lo, tr(this.yMin));
    ctx.font = "11px Vazirmatn, Tahoma, sans-serif";
    ctx.fillStyle = cssVar("--muted");
    ctx.direction = "rtl"; // Persian labels
    // legend
    let lx = w - pad.r;
    ctx.textAlign = "right";
    for (const s of this.series) {
      const last = s.data.length ? this.smooth(s.data).at(-1) : null;
      const label = s.label + (last !== null ? `: ${fmt(last, s.digits ?? 2)}` : "");
      ctx.fillStyle = s.color;
      ctx.fillRect(lx - 10, 7, 10, 10);
      ctx.fillStyle = cssVar("--muted");
      ctx.fillText(label, lx - 14, 16);
      lx -= ctx.measureText(label).width + 32;
    }
    if (n < 2) {
      ctx.textAlign = "center";
      ctx.fillText("هنوز داده‌ای نیست — آموزش را شروع کن", w / 2, h / 2);
      return;
    }
    if (hi - lo < 1e-9) { hi += 1; lo -= 1; }
    const X = (i) => pad.l + (i / (n - 1)) * (w - pad.l - pad.r);
    const Y = (v) => pad.t + (1 - (tr(v) - lo) / (hi - lo)) * (h - pad.t - pad.b);
    // grid + y labels
    ctx.strokeStyle = cssVar("--grid");
    ctx.textAlign = "left";
    ctx.direction = "ltr";
    for (let k = 0; k <= 4; k++) {
      const t = lo + ((hi - lo) * k) / 4, y = pad.t + (1 - k / 4) * (h - pad.t - pad.b);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      const v = this.logY ? Math.pow(10, t) : t;
      ctx.fillText(Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(Math.abs(v) < 1 ? 3 : 1), 4, y + 4);
    }
    ctx.fillText(`${n}`, w - pad.r - 30, h - 4);
    ctx.fillText("0", pad.l, h - 4);
    ctx.direction = "inherit";
    for (const s of this.series) {
      if (s.data.length < 2) continue;
      const step = Math.max(1, Math.floor(s.data.length / 600));
      ctx.strokeStyle = s.color;
      ctx.globalAlpha = 0.25;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < s.data.length; i += step) (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(i), Y(s.data[i]));
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.lineWidth = 2.2;
      const sm = this.smooth(s.data);
      ctx.beginPath();
      for (let i = 0; i < sm.length; i += step) (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(i), Y(sm[i]));
      ctx.stroke();
    }
  }
}

// Diverging color: negative → red, 0 → neutral, positive → green. Returns rgba string.
function divColor(v, maxAbs = 1, alpha = 1) {
  const t = Math.max(-1, Math.min(1, v / (maxAbs || 1)));
  const neutral = [128, 128, 140];
  const pos = [46, 190, 120], neg = [235, 87, 87];
  const target = t >= 0 ? pos : neg, a = Math.abs(t);
  const c = neutral.map((n, i) => Math.round(n + (target[i] - n) * a));
  return `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
}

// Render a set of horizontal bars into a container (Q-values, probabilities...)
function renderBars(container, items, { digits = 2, highlight = -1, max = null } = {}) {
  const vals = items.map((i) => i.value);
  const mx = max ?? Math.max(...vals.map(Math.abs), 1e-6);
  container.innerHTML = items
    .map((it, k) => {
      const pct = Math.min(100, (Math.abs(it.value) / mx) * 100);
      const neg = it.value < 0 ? `style="width:${pct}%;background:var(--bad)"` : `style="width:${pct}%"`;
      return `<div class="bar ${k === highlight ? "best" : ""}"><span>${it.label}</span><div class="track"><div class="fill" ${neg}></div></div><span class="v">${fmt(it.value, digits)}</span></div>`;
    })
    .join("");
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
