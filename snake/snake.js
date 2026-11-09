// =====================================================================
// SnakeEnv — the game logic ("the environment" in RL terms)
//
// This file knows NOTHING about the screen, keyboard or browser.
// It only knows the rules of the game. That separation is on purpose:
// later, an RL agent will call exactly the same reset() / step() API
// that the human player uses through the keyboard.
// =====================================================================

// The 4 possible actions. The index is the action number.
//   0 = up, 1 = right, 2 = down, 3 = left
const ACTIONS = [
  { x: 0, y: -1 },
  { x: 1, y: 0 },
  { x: 0, y: 1 },
  { x: -1, y: 0 },
];

// Rewards: the "score signal" an RL agent will learn from.
const REWARD_FOOD = 10;
const REWARD_DEATH = -10;
const REWARD_STEP = 0;

class SnakeEnv {
  constructor(cols = 20, rows = 20) {
    this.cols = cols;
    this.rows = rows;
    this.reset();
  }

  // Start a new episode (a new game) and return the first state.
  reset() {
    const cx = Math.floor(this.cols / 2);
    const cy = Math.floor(this.rows / 2);
    // snake[0] is the head
    this.snake = [
      { x: cx, y: cy },
      { x: cx - 1, y: cy },
      { x: cx - 2, y: cy },
    ];
    this.action = 1; // moving right
    this.score = 0;
    this.steps = 0;
    this.totalReward = 0;
    this.done = false;
    this.placeFood();
    return this.getState();
  }

  // Put food on a random empty cell.
  placeFood() {
    const free = [];
    for (let y = 0; y < this.rows; y++) {
      for (let x = 0; x < this.cols; x++) {
        if (!this.snake.some((s) => s.x === x && s.y === y)) free.push({ x, y });
      }
    }
    this.food = free.length ? free[Math.floor(Math.random() * free.length)] : null;
  }

  // Advance the game by ONE tick.
  // Input: an action (0..3). Output: { state, reward, done }.
  // This (state, action) -> (next state, reward, done) loop is the heart of RL.
  step(action) {
    if (this.done) throw new Error("Episode is over — call reset() first.");

    // A snake can't turn 180° into itself; ignore reversing actions.
    if ((action + 2) % 4 !== this.action) this.action = action;

    const dir = ACTIONS[this.action];
    const head = this.snake[0];
    const newHead = { x: head.x + dir.x, y: head.y + dir.y };
    this.steps++;

    const ate = this.food !== null && newHead.x === this.food.x && newHead.y === this.food.y;

    // If we don't eat, the tail moves out of the way this tick,
    // so moving into the current tail cell is allowed.
    const body = ate ? this.snake : this.snake.slice(0, -1);
    const hitWall =
      newHead.x < 0 || newHead.x >= this.cols || newHead.y < 0 || newHead.y >= this.rows;
    const hitSelf = body.some((s) => s.x === newHead.x && s.y === newHead.y);

    if (hitWall || hitSelf) {
      this.done = true;
      return this._result(REWARD_DEATH);
    }

    this.snake.unshift(newHead); // grow at the head
    if (ate) {
      this.score++;
      this.placeFood();
      if (this.food === null) this.done = true; // board full — you won!
      return this._result(REWARD_FOOD);
    }
    this.snake.pop(); // didn't eat: remove tail so length stays the same
    return this._result(REWARD_STEP);
  }

  _result(reward) {
    this.totalReward += reward;
    return { state: this.getState(), reward, done: this.done };
  }

  // A copy of everything needed to describe the current situation.
  getState() {
    return {
      snake: this.snake.map((p) => ({ ...p })),
      food: this.food ? { ...this.food } : null,
      direction: this.action,
      score: this.score,
    };
  }
}

// Lets us also use this file from Node.js (for tests / training later).
if (typeof module !== "undefined") module.exports = { SnakeEnv, ACTIONS };
