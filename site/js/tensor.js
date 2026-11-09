// =====================================================================
// tensor.js — a tiny automatic-differentiation (autograd) engine.
//
// Everything in this website (neural nets, DQN, the mini-GPT) is built
// on these ~400 lines. No TensorFlow, no PyTorch: just arrays + calculus.
//
// A Tensor is a 2-D matrix (rows × cols) stored in a Float32Array.
// Every operation remembers its inputs ("parents") and a function that
// knows how to push gradients back to them (the chain rule).
// Calling loss.backward() walks this graph in reverse.
// =====================================================================

class Tensor {
  constructor(data, rows, cols, parents, backFn) {
    this.data = data;
    this.rows = rows;
    this.cols = cols;
    this.grad = null;
    this._parents = parents || [];
    this._back = backFn || null;
    this.needsGrad = false;
  }

  static zeros(rows, cols) {
    return new Tensor(new Float32Array(rows * cols), rows, cols);
  }
  static fill(rows, cols, v) {
    return new Tensor(new Float32Array(rows * cols).fill(v), rows, cols);
  }
  // Gaussian random numbers (Box–Muller)
  static randn(rows, cols, std = 1) {
    const d = new Float32Array(rows * cols);
    for (let i = 0; i < d.length; i++) {
      const u = 1 - Math.random(), v = Math.random();
      d[i] = std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
    return new Tensor(d, rows, cols);
  }
  static from2D(arr) {
    const rows = arr.length, cols = arr[0].length;
    const d = new Float32Array(rows * cols);
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) d[i * cols + j] = arr[i][j];
    return new Tensor(d, rows, cols);
  }
  static scalar(v) {
    return new Tensor(new Float32Array([v]), 1, 1);
  }

  // Mark as a learnable parameter.
  param() { this.needsGrad = true; return this; }
  get(i, j) { return this.data[i * this.cols + j]; }
  item() { return this.data[0]; }
  row(i) { return Array.from(this.data.subarray(i * this.cols, (i + 1) * this.cols)); }
  ensureGrad() { if (!this.grad) this.grad = new Float32Array(this.data.length); return this.grad; }
  zeroGrad() { if (this.grad) this.grad.fill(0); }
  clone() { const t = new Tensor(new Float32Array(this.data), this.rows, this.cols); t.needsGrad = this.needsGrad; return t; }

  // Reverse-mode autodiff: topological sort, then call each node's backFn.
  backward(seed = 1) {
    const order = [], seen = new Set();
    const visit = (t) => {
      if (seen.has(t) || !t.needsGrad) return;
      seen.add(t);
      for (const p of t._parents) visit(p);
      order.push(t);
    };
    visit(this);
    this.ensureGrad().fill(seed);
    for (let i = order.length - 1; i >= 0; i--) {
      const t = order[i];
      if (t._back) t._back();
    }
  }
}

// When G.noGrad is true no graph is recorded (fast inference).
const G = { noGrad: false };

function out(data, rows, cols, parents, backFn) {
  const t = new Tensor(data, rows, cols);
  if (!G.noGrad && parents.some((p) => p.needsGrad)) {
    t.needsGrad = true;
    t._parents = parents;
    t._back = () => backFn(t);
  }
  return t;
}

// ---------------------------------------------------------------------
// Operations. Each one: (1) compute forward value, (2) define backward.
// ---------------------------------------------------------------------

// C = A·B
G.matmul = (a, b) => {
  if (a.cols !== b.rows) throw new Error(`matmul shape ${a.rows}x${a.cols} · ${b.rows}x${b.cols}`);
  const n = a.rows, k = a.cols, m = b.cols, A = a.data, B = b.data;
  const C = new Float32Array(n * m);
  for (let i = 0; i < n; i++) {
    const ci = i * m;
    for (let p = 0; p < k; p++) {
      const av = A[i * k + p];
      if (av === 0) continue;
      const bp = p * m;
      for (let j = 0; j < m; j++) C[ci + j] += av * B[bp + j];
    }
  }
  return out(C, n, m, [a, b], (o) => {
    const dC = o.grad;
    if (a.needsGrad) { // dA = dC · Bᵀ
      const dA = a.ensureGrad();
      for (let i = 0; i < n; i++)
        for (let p = 0; p < k; p++) {
          let s = 0;
          for (let j = 0; j < m; j++) s += dC[i * m + j] * B[p * m + j];
          dA[i * k + p] += s;
        }
    }
    if (b.needsGrad) { // dB = Aᵀ · dC
      const dB = b.ensureGrad();
      for (let i = 0; i < n; i++)
        for (let p = 0; p < k; p++) {
          const av = A[i * k + p];
          if (av === 0) continue;
          for (let j = 0; j < m; j++) dB[p * m + j] += av * dC[i * m + j];
        }
    }
  });
};

// C = A·Bᵀ   (used for attention scores Q·Kᵀ)
G.matmulT = (a, b) => {
  if (a.cols !== b.cols) throw new Error("matmulT shape");
  const n = a.rows, k = a.cols, m = b.rows, A = a.data, B = b.data;
  const C = new Float32Array(n * m);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let p = 0; p < k; p++) s += A[i * k + p] * B[j * k + p];
      C[i * m + j] = s;
    }
  return out(C, n, m, [a, b], (o) => {
    const dC = o.grad;
    if (a.needsGrad) { // dA = dC · B
      const dA = a.ensureGrad();
      for (let i = 0; i < n; i++)
        for (let j = 0; j < m; j++) {
          const g = dC[i * m + j];
          if (g === 0) continue;
          for (let p = 0; p < k; p++) dA[i * k + p] += g * B[j * k + p];
        }
    }
    if (b.needsGrad) { // dB = dCᵀ · A
      const dB = b.ensureGrad();
      for (let i = 0; i < n; i++)
        for (let j = 0; j < m; j++) {
          const g = dC[i * m + j];
          if (g === 0) continue;
          for (let p = 0; p < k; p++) dB[j * k + p] += g * A[i * k + p];
        }
    }
  });
};

// Elementwise A + B (same shape) — or A + b where b is a 1×cols row (broadcast)
G.add = (a, b) => {
  const bcast = b.rows === 1 && a.rows !== 1;
  if (!bcast && (a.rows !== b.rows || a.cols !== b.cols)) throw new Error("add shape");
  const C = new Float32Array(a.data.length), c = a.cols;
  for (let i = 0; i < C.length; i++) C[i] = a.data[i] + b.data[bcast ? i % c : i];
  return out(C, a.rows, a.cols, [a, b], (o) => {
    if (a.needsGrad) { const g = a.ensureGrad(); for (let i = 0; i < C.length; i++) g[i] += o.grad[i]; }
    if (b.needsGrad) { const g = b.ensureGrad(); for (let i = 0; i < C.length; i++) g[bcast ? i % c : i] += o.grad[i]; }
  });
};

G.sub = (a, b) => G.add(a, G.scale(b, -1));

// Elementwise A * B (same shape)
G.mul = (a, b) => {
  const C = new Float32Array(a.data.length);
  for (let i = 0; i < C.length; i++) C[i] = a.data[i] * b.data[i];
  return out(C, a.rows, a.cols, [a, b], (o) => {
    if (a.needsGrad) { const g = a.ensureGrad(); for (let i = 0; i < C.length; i++) g[i] += o.grad[i] * b.data[i]; }
    if (b.needsGrad) { const g = b.ensureGrad(); for (let i = 0; i < C.length; i++) g[i] += o.grad[i] * a.data[i]; }
  });
};

G.scale = (a, s) => {
  const C = new Float32Array(a.data.length);
  for (let i = 0; i < C.length; i++) C[i] = a.data[i] * s;
  return out(C, a.rows, a.cols, [a], (o) => {
    const g = a.ensureGrad();
    for (let i = 0; i < C.length; i++) g[i] += o.grad[i] * s;
  });
};

G.square = (a) => G.mul(a, a);

// Activation functions
G.relu = (a) => {
  const C = new Float32Array(a.data.length);
  for (let i = 0; i < C.length; i++) C[i] = a.data[i] > 0 ? a.data[i] : 0;
  return out(C, a.rows, a.cols, [a], (o) => {
    const g = a.ensureGrad();
    for (let i = 0; i < C.length; i++) if (a.data[i] > 0) g[i] += o.grad[i];
  });
};
G.tanh = (a) => {
  const C = new Float32Array(a.data.length);
  for (let i = 0; i < C.length; i++) C[i] = Math.tanh(a.data[i]);
  return out(C, a.rows, a.cols, [a], (o) => {
    const g = a.ensureGrad();
    for (let i = 0; i < C.length; i++) g[i] += o.grad[i] * (1 - C[i] * C[i]);
  });
};
G.sigmoid = (a) => {
  const C = new Float32Array(a.data.length);
  for (let i = 0; i < C.length; i++) C[i] = 1 / (1 + Math.exp(-a.data[i]));
  return out(C, a.rows, a.cols, [a], (o) => {
    const g = a.ensureGrad();
    for (let i = 0; i < C.length; i++) g[i] += o.grad[i] * C[i] * (1 - C[i]);
  });
};
G.act = (name, a) => G[name](a);

// Pick rows of a table by index: embeddings lookup.
G.gather = (table, ids) => {
  const c = table.cols, C = new Float32Array(ids.length * c);
  for (let i = 0; i < ids.length; i++) C.set(table.data.subarray(ids[i] * c, ids[i] * c + c), i * c);
  return out(C, ids.length, c, [table], (o) => {
    const g = table.ensureGrad();
    for (let i = 0; i < ids.length; i++)
      for (let j = 0; j < c; j++) g[ids[i] * c + j] += o.grad[i * c + j];
  });
};

// Glue matrices side by side (used to join attention heads).
G.concatCols = (list) => {
  const rows = list[0].rows, cols = list.reduce((s, t) => s + t.cols, 0);
  const C = new Float32Array(rows * cols);
  let off = 0;
  for (const t of list) {
    for (let i = 0; i < rows; i++)
      for (let j = 0; j < t.cols; j++) C[i * cols + off + j] = t.data[i * t.cols + j];
    off += t.cols;
  }
  return out(C, rows, cols, list, (o) => {
    let off2 = 0;
    for (const t of list) {
      if (t.needsGrad) {
        const g = t.ensureGrad();
        for (let i = 0; i < rows; i++)
          for (let j = 0; j < t.cols; j++) g[i * t.cols + j] += o.grad[i * cols + off2 + j];
      }
      off2 += t.cols;
    }
  });
};

// Row-wise softmax. causal=true hides the future (j > i) — the GPT mask.
G.softmaxRows = (a, causal = false) => {
  const n = a.rows, m = a.cols, C = new Float32Array(n * m);
  for (let i = 0; i < n; i++) {
    const lim = causal ? Math.min(i + 1, m) : m;
    let mx = -Infinity;
    for (let j = 0; j < lim; j++) mx = Math.max(mx, a.data[i * m + j]);
    let s = 0;
    for (let j = 0; j < lim; j++) { const e = Math.exp(a.data[i * m + j] - mx); C[i * m + j] = e; s += e; }
    for (let j = 0; j < lim; j++) C[i * m + j] /= s;
  }
  return out(C, n, m, [a], (o) => {
    const g = a.ensureGrad();
    for (let i = 0; i < n; i++) {
      let dot = 0;
      for (let j = 0; j < m; j++) dot += o.grad[i * m + j] * C[i * m + j];
      for (let j = 0; j < m; j++) g[i * m + j] += C[i * m + j] * (o.grad[i * m + j] - dot);
    }
  });
};

// Layer normalization (per row) with learnable gain & bias (1×cols).
G.layerNorm = (a, gain, bias, eps = 1e-5) => {
  const n = a.rows, m = a.cols, C = new Float32Array(n * m);
  const xhat = new Float32Array(n * m), inv = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let mu = 0;
    for (let j = 0; j < m; j++) mu += a.data[i * m + j];
    mu /= m;
    let v = 0;
    for (let j = 0; j < m; j++) { const d = a.data[i * m + j] - mu; v += d * d; }
    inv[i] = 1 / Math.sqrt(v / m + eps);
    for (let j = 0; j < m; j++) {
      const xh = (a.data[i * m + j] - mu) * inv[i];
      xhat[i * m + j] = xh;
      C[i * m + j] = xh * gain.data[j] + bias.data[j];
    }
  }
  return out(C, n, m, [a, gain, bias], (o) => {
    const dy = o.grad;
    if (gain.needsGrad || bias.needsGrad) {
      const gg = gain.ensureGrad(), gb = bias.ensureGrad();
      for (let i = 0; i < n * m; i++) { gg[i % m] += dy[i] * xhat[i]; gb[i % m] += dy[i]; }
    }
    if (a.needsGrad) {
      const ga = a.ensureGrad();
      for (let i = 0; i < n; i++) {
        let s1 = 0, s2 = 0;
        for (let j = 0; j < m; j++) {
          const dxh = dy[i * m + j] * gain.data[j];
          s1 += dxh; s2 += dxh * xhat[i * m + j];
        }
        for (let j = 0; j < m; j++) {
          const dxh = dy[i * m + j] * gain.data[j];
          ga[i * m + j] += (inv[i] / m) * (m * dxh - s1 - xhat[i * m + j] * s2);
        }
      }
    }
  });
};

// Softmax + cross-entropy against integer targets.
// Without weights: mean over rows of −log p(target).
// With weights:    Σ weights[i] · (−log p(target_i))   (used by REINFORCE)
G.crossEntropy = (logits, targets, weights = null) => {
  const n = logits.rows, m = logits.cols, P = new Float32Array(n * m);
  let loss = 0;
  for (let i = 0; i < n; i++) {
    let mx = -Infinity;
    for (let j = 0; j < m; j++) mx = Math.max(mx, logits.data[i * m + j]);
    let s = 0;
    for (let j = 0; j < m; j++) { const e = Math.exp(logits.data[i * m + j] - mx); P[i * m + j] = e; s += e; }
    for (let j = 0; j < m; j++) P[i * m + j] /= s;
    const nll = -Math.log(P[i * m + targets[i]] + 1e-12);
    loss += weights ? weights[i] * nll : nll / n;
  }
  const res = out(new Float32Array([loss]), 1, 1, [logits], (o) => {
    const g = logits.ensureGrad(), up = o.grad[0];
    for (let i = 0; i < n; i++) {
      const w = (weights ? weights[i] : 1 / n) * up;
      if (w === 0) continue;
      for (let j = 0; j < m; j++) g[i * m + j] += w * (P[i * m + j] - (j === targets[i] ? 1 : 0));
    }
  });
  res.probs = P;
  return res;
};

// DQN loss: mean over rows of Huber( Q[i, action_i] − target_i ).
G.huberSelect = (pred, actions, targets, delta = 1) => {
  const n = pred.rows, m = pred.cols;
  let loss = 0;
  const diffs = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const d = pred.data[i * m + actions[i]] - targets[i];
    diffs[i] = d;
    loss += (Math.abs(d) <= delta ? 0.5 * d * d : delta * (Math.abs(d) - 0.5 * delta)) / n;
  }
  return out(new Float32Array([loss]), 1, 1, [pred], (o) => {
    const g = pred.ensureGrad();
    for (let i = 0; i < n; i++) {
      const d = diffs[i];
      g[i * m + actions[i]] += (Math.abs(d) <= delta ? d : delta * Math.sign(d)) / n * o.grad[0];
    }
  });
};

// ---------------------------------------------------------------------
// Building blocks
// ---------------------------------------------------------------------

class Linear {
  constructor(nIn, nOut, std) {
    this.W = Tensor.randn(nIn, nOut, std ?? Math.sqrt(2 / nIn)).param();
    this.b = Tensor.zeros(1, nOut).param();
  }
  forward(x) { return G.add(G.matmul(x, this.W), this.b); }
  params() { return [this.W, this.b]; }
}

// Multi-layer perceptron: sizes = [in, hidden..., out]
class MLP {
  constructor(sizes, act = "relu") {
    this.sizes = sizes;
    this.act = act;
    this.layers = [];
    for (let i = 0; i < sizes.length - 1; i++) {
      const std = act === "tanh" ? Math.sqrt(1 / sizes[i]) : Math.sqrt(2 / sizes[i]);
      this.layers.push(new Linear(sizes[i], sizes[i + 1], std));
    }
  }
  // If `record` is an array, every layer's activations are pushed into it (for visualization).
  forward(x, record) {
    if (record) record.push(x);
    for (let i = 0; i < this.layers.length; i++) {
      x = this.layers[i].forward(x);
      if (i < this.layers.length - 1) x = G.act(this.act, x);
      if (record) record.push(x);
    }
    return x;
  }
  params() { return this.layers.flatMap((l) => l.params()); }
}

// Adam optimizer: gradient descent with momentum + per-parameter step sizes.
class Adam {
  constructor(params, lr = 1e-3, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
    Object.assign(this, { params, lr, b1, b2, eps, t: 0 });
    this.m = params.map((p) => new Float32Array(p.data.length));
    this.v = params.map((p) => new Float32Array(p.data.length));
  }
  zeroGrad() { for (const p of this.params) p.zeroGrad(); }
  gradNorm() {
    let s = 0;
    for (const p of this.params) if (p.grad) for (const g of p.grad) s += g * g;
    return Math.sqrt(s);
  }
  step(clip = 0) {
    this.t++;
    let k = 1;
    if (clip > 0) { const n = this.gradNorm(); if (n > clip) k = clip / n; }
    const { b1, b2, eps, lr, t } = this;
    const c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t);
    this.params.forEach((p, idx) => {
      if (!p.grad) return;
      const m = this.m[idx], v = this.v[idx];
      for (let i = 0; i < p.data.length; i++) {
        const g = p.grad[i] * k;
        m[i] = b1 * m[i] + (1 - b1) * g;
        v[i] = b2 * v[i] + (1 - b2) * g * g;
        p.data[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps);
      }
    });
  }
}

// Copy parameter values (e.g. into a DQN target network).
function copyParams(from, to) {
  from.forEach((p, i) => to[i].data.set(p.data));
}
function countParams(params) {
  return params.reduce((s, p) => s + p.data.length, 0);
}

if (typeof module !== "undefined")
  module.exports = { Tensor, G, Linear, MLP, Adam, copyParams, countParams };
