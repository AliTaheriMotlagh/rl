// =====================================================================
// rlhf.js — Reinforcement Learning on a language model.
//
// Policy gradient (REINFORCE) with a KL penalty to a frozen reference
// model. This is the core idea behind RLHF (PPO adds clipping + a value
// network, but the heart is the same):
//
//   1. the model writes K samples
//   2. each sample gets a reward R (from a rule, or from a human 👍/👎)
//   3. penalty: β · (log π(y) − log π_ref(y))  — don't drift too far
//   4. advantage A = score − average score of samples with the same prompt
//   5. raise log-probability of above-average samples, lower the rest:
//        loss = − Σ A · log π(tokens)
// =====================================================================

const REWARDS = {
  // Verifiable reward: WE ask "a+b=", the model writes only the answer.
  // (Like RL on math/code in modern reasoning models.)
  math: {
    name: "درستی جواب جمع (پاداش قابل‌بررسی)",
    len: 3,
    prompt: () => {
      const a = Math.floor(Math.random() * 10), b = Math.floor(Math.random() * 10);
      return { text: `\n${a}+${b}=`, a, b };
    },
    fn: (text, _target, q) => {
      const m = text.match(/^(\d+)\n/);
      return m && Number(m[1]) === q.a + q.b ? 1 : -1;
    },
  },
  // Reward for each occurrence of a target word
  word: {
    name: "تکرار یک کلمهٔ هدف",
    len: 30,
    fn: (text, target) => (target ? text.split(target).length - 1 : 0),
  },
  // Penalty for each occurrence of a forbidden character
  avoid: {
    name: "پرهیز از یک حرف",
    len: 30,
    fn: (text, target) => -(target ? Array.from(text).filter((c) => c === target).length : 0) / 4,
  },
};

// Score and update from a set of already-generated samples.
// samples: [{ ids, from, R }]  — ids includes the prompt, `from` = prompt length
function reinforceUpdate(policy, ref, opt, samples, beta) {
  for (const s of samples) {
    s.lp = policy.logProb(s.ids, s.from);
    s.lpRef = ref.logProb(s.ids, s.from);
    s.kl = s.lp - s.lpRef;               // how far the policy drifted on this sample
    s.score = s.R - beta * s.kl;
  }
  const K = samples.length;
  // Group baseline (GRPO-style): compare each sample with others that got the SAME prompt.
  const groups = new Map();
  for (const s of samples) { const g = s.group ?? 0; if (!groups.has(g)) groups.set(g, []); groups.get(g).push(s); }
  for (const list of groups.values()) {
    const mean = list.reduce((a, s) => a + s.score, 0) / list.length;
    const std = Math.sqrt(list.reduce((a, s) => a + (s.score - mean) ** 2, 0) / list.length);
    for (const s of list) s.adv = std > 1e-6 ? (s.score - mean) / std : 0;
  }
  opt.zeroGrad();
  for (const s of samples) {
    // s.adv: advantage — was this sample better or worse than its group's average?
    const n = s.ids.length - 1, w = new Float32Array(n);
    for (let i = s.from - 1; i < n; i++) w[i] = s.adv / K / (n - s.from + 1);
    // Σ w·(−log p)  → its gradient pushes log p UP when adv > 0, DOWN when adv < 0
    const loss = G.crossEntropy(policy.forward(s.ids.slice(0, -1)), s.ids.slice(1), w);
    loss.backward();
  }
  opt.step(1);
  return {
    meanR: samples.reduce((a, s) => a + s.R, 0) / K,
    meanKL: samples.reduce((a, s) => a + s.kl, 0) / K,
  };
}

// One full RL iteration with an automatic reward.
// mode: an entry of REWARDS. startText: prompt to use when the mode has no prompt().
function rlStep(policy, ref, opt, { K = 8, groupSize = 8, beta = 0.05, mode, target, tok, startText = "\n", temperature = 1 }) {
  const samples = [];
  const nGroups = mode.prompt ? Math.max(1, Math.round(K / groupSize)) : 1;
  const perGroup = mode.prompt ? groupSize : K;
  for (let g = 0; g < nGroups; g++) {
    const q = mode.prompt ? mode.prompt() : { text: startText };
    let prompt = tok.encode(q.text);
    if (!prompt.length) prompt = [0];
    for (let k = 0; k < perGroup; k++) {
      const ids = policy.generate(prompt, Math.min(mode.len, policy.block - prompt.length), temperature);
      const text = tok.decode(ids.slice(prompt.length));
      samples.push({ ids, group: g, from: prompt.length, prompt: q.text, text, R: mode.fn(text, target, q) });
    }
  }
  return { ...reinforceUpdate(policy, ref, opt, samples, beta), samples };
}

if (typeof module !== "undefined") module.exports = { REWARDS, reinforceUpdate, rlStep };
