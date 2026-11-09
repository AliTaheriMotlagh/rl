// =====================================================================
// gpt.js — a real (tiny) GPT, built on tensor.js.
// Same architecture as GPT-2, just thousands of times smaller:
//   token embedding + position embedding
//   → N × [ LayerNorm → multi-head causal self-attention → residual
//            LayerNorm → MLP (d → 4d → d)                → residual ]
//   → LayerNorm → linear → logits over the vocabulary
// =====================================================================

class CharTokenizer {
  constructor(text) {
    this.chars = Array.from(new Set(Array.from(text))).sort();
    this.stoi = new Map(this.chars.map((c, i) => [c, i]));
  }
  get size() { return this.chars.length; }
  encode(s) { return Array.from(s).filter((c) => this.stoi.has(c)).map((c) => this.stoi.get(c)); }
  decode(ids) { return ids.map((i) => this.chars[i]).join(""); }
}

class TinyGPT {
  constructor({ vocab, block = 32, d = 48, heads = 2, layers = 2 }) {
    Object.assign(this, { vocab, block, d, heads, nLayers: layers });
    const s = 0.08;
    this.tokEmb = Tensor.randn(vocab, d, s).param();
    this.posEmb = Tensor.randn(block, d, s).param();
    const dh = d / heads;
    this.layers = [];
    for (let l = 0; l < layers; l++) {
      const proj = Math.sqrt(1 / d) / Math.sqrt(2 * layers);
      this.layers.push({
        ln1g: Tensor.fill(1, d, 1).param(), ln1b: Tensor.zeros(1, d).param(),
        heads: Array.from({ length: heads }, () => ({
          Wq: Tensor.randn(d, dh, Math.sqrt(1 / d)).param(),
          Wk: Tensor.randn(d, dh, Math.sqrt(1 / d)).param(),
          Wv: Tensor.randn(d, dh, Math.sqrt(1 / d)).param(),
        })),
        Wo: Tensor.randn(d, d, proj).param(), bo: Tensor.zeros(1, d).param(),
        ln2g: Tensor.fill(1, d, 1).param(), ln2b: Tensor.zeros(1, d).param(),
        W1: Tensor.randn(d, 4 * d, Math.sqrt(2 / d)).param(), b1: Tensor.zeros(1, 4 * d).param(),
        W2: Tensor.randn(4 * d, d, proj).param(), b2: Tensor.zeros(1, d).param(),
      });
    }
    this.lnfg = Tensor.fill(1, d, 1).param();
    this.lnfb = Tensor.zeros(1, d).param();
    this.Wout = Tensor.randn(d, vocab, 0.02).param();
    this.bout = Tensor.zeros(1, vocab).param();
  }

  params() {
    const ps = [this.tokEmb, this.posEmb];
    for (const L of this.layers) {
      ps.push(L.ln1g, L.ln1b);
      for (const h of L.heads) ps.push(h.Wq, h.Wk, h.Wv);
      ps.push(L.Wo, L.bo, L.ln2g, L.ln2b, L.W1, L.b1, L.W2, L.b2);
    }
    ps.push(this.lnfg, this.lnfb, this.Wout, this.bout);
    return ps;
  }

  clone() {
    const m = new TinyGPT({ vocab: this.vocab, block: this.block, d: this.d, heads: this.heads, layers: this.nLayers });
    copyParams(this.params(), m.params());
    return m;
  }

  // ids: array of token ids (length ≤ block). Returns logits (T × vocab).
  // If `trace` is an object, attention matrices are stored in trace.att[layer][head].
  forward(ids, trace) {
    const T = ids.length;
    const pos = Array.from({ length: T }, (_, i) => i);
    let x = G.add(G.gather(this.tokEmb, ids), G.gather(this.posEmb, pos));
    const dh = this.d / this.heads;
    if (trace) trace.att = [];
    for (const L of this.layers) {
      const h = G.layerNorm(x, L.ln1g, L.ln1b);
      const layerAtt = [];
      const headOuts = L.heads.map((H) => {
        const q = G.matmul(h, H.Wq), k = G.matmul(h, H.Wk), v = G.matmul(h, H.Wv);
        // attention(Q,K,V) = softmax(Q·Kᵀ / √d_head, causal mask) · V
        const att = G.softmaxRows(G.scale(G.matmulT(q, k), 1 / Math.sqrt(dh)), true);
        layerAtt.push(att);
        return G.matmul(att, v);
      });
      if (trace) trace.att.push(layerAtt);
      x = G.add(x, G.add(G.matmul(G.concatCols(headOuts), L.Wo), L.bo));
      const h2 = G.layerNorm(x, L.ln2g, L.ln2b);
      x = G.add(x, G.add(G.matmul(G.relu(G.add(G.matmul(h2, L.W1), L.b1)), L.W2), L.b2));
    }
    x = G.layerNorm(x, this.lnfg, this.lnfb);
    return G.add(G.matmul(x, this.Wout), this.bout);
  }

  // Probability distribution over the next token, given a context.
  nextProbs(ids, temperature = 1) {
    const ctx = ids.slice(-this.block);
    G.noGrad = true;
    const logits = this.forward(ctx.length ? ctx : [0]);
    G.noGrad = false;
    const row = logits.row(logits.rows - 1);
    const t = Math.max(temperature, 1e-3);
    const mx = Math.max(...row);
    const e = row.map((v) => Math.exp((v - mx) / t));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }

  generate(ids, n, temperature = 1) {
    const outIds = ids.slice();
    for (let i = 0; i < n; i++) outIds.push(sampleFrom(this.nextProbs(outIds, temperature)));
    return outIds;
  }

  // Sum of log-probabilities of tokens seq[from..] given what came before.
  logProb(seq, from) {
    G.noGrad = true;
    const logits = this.forward(seq.slice(0, -1));
    G.noGrad = false;
    let lp = 0;
    for (let i = from - 1; i < seq.length - 1; i++) {
      const row = logits.row(i), mx = Math.max(...row);
      const lse = mx + Math.log(row.reduce((s, v) => s + Math.exp(v - mx), 0));
      lp += row[seq[i + 1]] - lse;
    }
    return lp;
  }
}

function sampleFrom(probs) {
  let r = Math.random();
  for (let i = 0; i < probs.length; i++) { r -= probs[i]; if (r <= 0) return i; }
  return probs.length - 1;
}

// ---- Save / load (so lesson 8 can fine-tune the model trained in lesson 7) ----
function gptToJSON(model, tok, meta = {}) {
  const ps = model.params(), total = countParams(ps), flat = new Float32Array(total);
  let off = 0;
  for (const p of ps) { flat.set(p.data, off); off += p.data.length; }
  const bytes = new Uint8Array(flat.buffer);
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return JSON.stringify({
    cfg: { vocab: model.vocab, block: model.block, d: model.d, heads: model.heads, layers: model.nLayers },
    chars: tok.chars, meta, weights: btoa(bin),
  });
}
function gptFromJSON(json) {
  const o = JSON.parse(json);
  const model = new TinyGPT(o.cfg);
  const bin = atob(o.weights), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const flat = new Float32Array(bytes.buffer);
  let off = 0;
  for (const p of model.params()) { p.data.set(flat.subarray(off, off + p.data.length)); off += p.data.length; }
  const tok = new CharTokenizer("");
  tok.chars = o.chars;
  tok.stoi = new Map(o.chars.map((c, i) => [c, i]));
  return { model, tok, meta: o.meta || {} };
}

if (typeof module !== "undefined") module.exports = { TinyGPT, CharTokenizer, sampleFrom, gptToJSON, gptFromJSON };
