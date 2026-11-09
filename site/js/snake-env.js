// =====================================================================
// SnakeEnv — the environment. Pure game rules, no screen, no keyboard.
// Humans, hand-written policies, Q-learning and DQN all use the same
// reset() / step(action) interface.
// =====================================================================

// Absolute actions: 0 = up, 1 = right, 2 = down, 3 = left
const ACTIONS = [
  { x: 0, y: -1 },
  { x: 1, y: 0 },
  { x: 0, y: 1 },
  { x: -1, y: 0 },
];
const ACTION_NAMES = ["بالا ↑", "راست →", "پایین ↓", "چپ ←"];
// Relative actions (what RL agents use): 0 = straight, 1 = turn right, 2 = turn left
const REL_NAMES = ["مستقیم", "گردش به راست", "گردش به چپ"];
const relToAbs = (dir, rel) => (rel === 0 ? dir : rel === 1 ? (dir + 1) % 4 : (dir + 3) % 4);

class SnakeEnv {
  constructor(cols = 20, rows = 20, opts = {}) {
    this.cols = cols;
    this.rows = rows;
    this.rewards = { food: 10, death: -10, step: 0, ...(opts.rewards || {}) };
    // End an episode if the snake wanders this many steps × length without eating
    this.maxIdle = opts.maxIdle ?? Infinity;
    this.reset();
  }

  reset() {
    const cx = Math.floor(this.cols / 2), cy = Math.floor(this.rows / 2);
    this.snake = [{ x: cx, y: cy }, { x: cx - 1, y: cy }, { x: cx - 2, y: cy }];
    this.action = 1;
    this.score = 0;
    this.steps = 0;
    this.idle = 0;
    this.totalReward = 0;
    this.done = false;
    this.deathReason = null;
    this.placeFood();
    return this.getState();
  }

  placeFood() {
    const free = [];
    for (let y = 0; y < this.rows; y++)
      for (let x = 0; x < this.cols; x++)
        if (!this.snake.some((s) => s.x === x && s.y === y)) free.push({ x, y });
    this.food = free.length ? free[Math.floor(Math.random() * free.length)] : null;
  }

  // Would the head die at point p on the next move?
  isCollision(p) {
    if (p.x < 0 || p.x >= this.cols || p.y < 0 || p.y >= this.rows) return true;
    for (let i = 0; i < this.snake.length - 1; i++) // tail will have moved away
      if (this.snake[i].x === p.x && this.snake[i].y === p.y) return true;
    return false;
  }

  step(action) {
    if (this.done) throw new Error("Episode is over — call reset() first.");
    if ((action + 2) % 4 !== this.action) this.action = action; // no 180° turns

    const d = ACTIONS[this.action], head = this.snake[0];
    const nh = { x: head.x + d.x, y: head.y + d.y };
    this.steps++;
    this.idle++;
    const ate = this.food && nh.x === this.food.x && nh.y === this.food.y;

    const hitWall = nh.x < 0 || nh.x >= this.cols || nh.y < 0 || nh.y >= this.rows;
    const body = ate ? this.snake : this.snake.slice(0, -1);
    const hitSelf = !hitWall && body.some((s) => s.x === nh.x && s.y === nh.y);
    if (hitWall || hitSelf) {
      this.done = true;
      this.deathReason = hitWall ? "دیوار" : "بدن خودش";
      return this._result(this.rewards.death);
    }

    this.snake.unshift(nh);
    if (ate) {
      this.score++;
      this.idle = 0;
      this.placeFood();
      if (!this.food) { this.done = true; this.deathReason = "برد! صفحه پر شد"; }
      return this._result(this.rewards.food);
    }
    this.snake.pop();
    if (this.idle >= this.maxIdle * this.snake.length) {
      this.done = true;
      this.deathReason = "دور خودش چرخید (وقت تمام شد)";
      return this._result(this.rewards.death);
    }
    return this._result(this.rewards.step);
  }

  _result(reward) {
    this.totalReward += reward;
    return { state: this.getState(), reward, done: this.done };
  }

  getState() {
    return {
      snake: this.snake.map((p) => ({ ...p })),
      food: this.food ? { ...this.food } : null,
      direction: this.action,
      score: this.score,
    };
  }
}

// ---------------------------------------------------------------------
// The 11 features an agent "sees" — a compact description of the state.
// ---------------------------------------------------------------------
const FEATURE_NAMES = [
  "خطر روبرو", "خطر سمت راست", "خطر سمت چپ",
  "جهت: چپ", "جهت: راست", "جهت: بالا", "جهت: پایین",
  "غذا: چپ‌تر", "غذا: راست‌تر", "غذا: بالاتر", "غذا: پایین‌تر",
];

function snakeFeatures(env) {
  const h = env.snake[0], d = env.action;
  const look = (dir) => (env.isCollision({ x: h.x + ACTIONS[dir].x, y: h.y + ACTIONS[dir].y }) ? 1 : 0);
  const f = env.food || h;
  return [
    look(d), look((d + 1) % 4), look((d + 3) % 4),
    d === 3 ? 1 : 0, d === 1 ? 1 : 0, d === 0 ? 1 : 0, d === 2 ? 1 : 0,
    f.x < h.x ? 1 : 0, f.x > h.x ? 1 : 0, f.y < h.y ? 1 : 0, f.y > h.y ? 1 : 0,
  ];
}
const featuresToIndex = (f) => f.reduce((s, b, i) => s | (b << i), 0);

// ---------------------------------------------------------------------
// Drawing (browser only)
// ---------------------------------------------------------------------
function drawSnake(canvas, env, opts = {}) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height, cw = W / env.cols, ch = H / env.rows;
  const c = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  ctx.fillStyle = c("--board");
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = c("--grid");
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 1; i < env.cols; i++) { ctx.moveTo(Math.round(i * cw) + 0.5, 0); ctx.lineTo(Math.round(i * cw) + 0.5, H); }
  for (let i = 1; i < env.rows; i++) { ctx.moveTo(0, Math.round(i * ch) + 0.5); ctx.lineTo(W, Math.round(i * ch) + 0.5); }
  ctx.stroke();

  if (env.food) {
    ctx.fillStyle = c("--food");
    ctx.beginPath();
    ctx.arc((env.food.x + 0.5) * cw, (env.food.y + 0.5) * ch, Math.min(cw, ch) * 0.36, 0, Math.PI * 2);
    ctx.fill();
  }
  const n = env.snake.length;
  env.snake.forEach((p, i) => {
    ctx.fillStyle = i === 0 ? c("--head") : c("--body");
    ctx.globalAlpha = i === 0 ? 1 : 1 - (0.45 * i) / Math.max(n, 1);
    const r = Math.min(cw, ch) * 0.2;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(p.x * cw + 1, p.y * ch + 1, cw - 2, ch - 2, r);
    else ctx.rect(p.x * cw + 1, p.y * ch + 1, cw - 2, ch - 2);
    ctx.fill();
  });
  ctx.globalAlpha = 1;

  // optional: highlight the 3 cells the agent checks for danger
  if (opts.showSensors && !env.done) {
    const h = env.snake[0], d = env.action;
    [d, (d + 1) % 4, (d + 3) % 4].forEach((dir) => {
      const p = { x: h.x + ACTIONS[dir].x, y: h.y + ACTIONS[dir].y };
      const bad = env.isCollision(p);
      ctx.strokeStyle = bad ? c("--bad") : c("--accent");
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(p.x * cw + 2, p.y * ch + 2, cw - 4, ch - 4);
      ctx.setLineDash([]);
    });
  }
  if (env.done && opts.overlay !== false) {
    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.fillRect(0, 0, W, H);
  }
}

if (typeof module !== "undefined")
  module.exports = { SnakeEnv, ACTIONS, relToAbs, snakeFeatures, featuresToIndex, FEATURE_NAMES };
