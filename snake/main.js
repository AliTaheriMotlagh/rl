// =====================================================================
// main.js — the browser side: drawing, keyboard, game loop.
// It only TALKS to the environment through env.reset() and env.step().
// =====================================================================

const env = new SnakeEnv(20, 20);
const canvas = document.getElementById("board");
const ctx = canvas.getContext("2d");
const CELL = canvas.width / env.cols;

const el = {
  score: document.getElementById("score"),
  best: document.getElementById("best"),
  reward: document.getElementById("reward"),
  total: document.getElementById("total"),
  steps: document.getElementById("steps"),
  message: document.getElementById("message"),
};

const TICK_MS = 110; // game speed: one step every 110ms
let timer = null;
let started = false;
let queue = []; // buffered key presses, so fast turns aren't lost

let best = 0;
try { best = Number(localStorage.getItem("snake-best")) || 0; } catch (_) {}

const KEYS = {
  ArrowUp: 0, w: 0, W: 0,
  ArrowRight: 1, d: 1, D: 1,
  ArrowDown: 2, s: 2, S: 2,
  ArrowLeft: 3, a: 3, A: 3,
};

// ---------- input ----------
function pushAction(a) {
  // compare with the last action we'll be heading in
  const last = queue.length ? queue[queue.length - 1] : env.action;
  if (a === last || (a + 2) % 4 === last) return; // same or reverse: useless
  if (queue.length < 3) queue.push(a);
  if (!started) start();
}

document.addEventListener("keydown", (e) => {
  if (e.key in KEYS) {
    e.preventDefault();
    pushAction(KEYS[e.key]);
  } else if (e.key === " ") {
    e.preventDefault();
    if (env.done) restart();
    else if (!started) start();
    else togglePause();
  }
});

document.querySelectorAll("[data-action]").forEach((btn) =>
  btn.addEventListener("click", () => {
    if (env.done) restart();
    pushAction(Number(btn.dataset.action));
  })
);

// ---------- game loop ----------
function start() {
  started = true;
  el.message.textContent = "";
  timer = setInterval(tick, TICK_MS);
}

function togglePause() {
  if (timer) {
    clearInterval(timer);
    timer = null;
    el.message.textContent = "⏸ مکث — Space برای ادامه";
  } else start();
}

function restart() {
  clearInterval(timer);
  timer = null;
  started = false;
  queue = [];
  env.reset();
  el.reward.textContent = "0";
  el.message.textContent = "یک جهت‌نما بزن تا شروع شود";
  render();
}

function tick() {
  const action = queue.length ? queue.shift() : env.action;
  const { reward, done } = env.step(action); // <-- the RL loop, driven by a human

  el.reward.textContent = reward > 0 ? "+" + reward : String(reward);
  el.reward.className = reward > 0 ? "pos" : reward < 0 ? "neg" : "";
  render();

  if (done) {
    clearInterval(timer);
    timer = null;
    if (env.score > best) {
      best = env.score;
      try { localStorage.setItem("snake-best", String(best)); } catch (_) {}
    }
    el.message.textContent = env.food === null ? "🏆 بردی!" : "💀 باختی — Space برای بازی دوباره";
    render();
  }
}

// ---------- drawing ----------
function render() {
  const css = getComputedStyle(document.documentElement);
  ctx.fillStyle = css.getPropertyValue("--board");
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // grid
  ctx.strokeStyle = css.getPropertyValue("--grid");
  ctx.lineWidth = 1;
  for (let i = 1; i < env.cols; i++) {
    ctx.beginPath();
    ctx.moveTo(i * CELL + 0.5, 0);
    ctx.lineTo(i * CELL + 0.5, canvas.height);
    ctx.moveTo(0, i * CELL + 0.5);
    ctx.lineTo(canvas.width, i * CELL + 0.5);
    ctx.stroke();
  }

  // food
  if (env.food) {
    ctx.fillStyle = css.getPropertyValue("--food");
    ctx.beginPath();
    ctx.arc((env.food.x + 0.5) * CELL, (env.food.y + 0.5) * CELL, CELL * 0.38, 0, Math.PI * 2);
    ctx.fill();
  }

  // snake
  env.snake.forEach((p, i) => {
    ctx.fillStyle = css.getPropertyValue(i === 0 ? "--head" : "--body");
    ctx.beginPath();
    ctx.roundRect(p.x * CELL + 1, p.y * CELL + 1, CELL - 2, CELL - 2, 4);
    ctx.fill();
  });

  el.score.textContent = env.score;
  el.best.textContent = Math.max(best, env.score);
  el.total.textContent = env.totalReward;
  el.steps.textContent = env.steps;
}

restart();
