// Synthetic camera recordings with known ground truth, and a helper that runs the app's analysis loop.
import { analyzeWindow, BpmSmoother, BeatTracker } from '../js/dsp.js';

// Deterministic PRNG (mulberry32) so test results are reproducible.
export function rng(seed) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  next.gauss = () => { let s = 0; for (let i = 0; i < 6; i++) s += next(); return (s - 3) / Math.SQRT1_2; };   // ~N(0,1)
  return next;
}

// Skin ROI mean colour over time.
//   bpm      mean heart rate
//   rsa      respiratory sinus arrhythmia: IBI modulation amplitude (s), at 6 breaths/min
//   amp      relative pulse amplitude (fraction of DC); spread over RGB like real skin PPG [0.33, 0.77, 0.53]
//   noise    per-frame additive noise std, in 8-bit pixel units
//   drift    slow lighting change (fraction), motion: intensity flicker at 0.9 Hz (fraction)
export function makeRecording({
  bpm = 70, rsa = 0.06, dur = 120, fps = 30, amp = 0.002, noise = 0.05,
  drift = 0.02, motion = 0, ampMod = 0.1, seed = 1,
} = {}) {
  const rand = rng(seed);
  const truth = [];
  for (let tb = 0.3; tb < dur + 2; ) {
    truth.push(tb);
    tb += 60 / bpm + rsa * Math.sin(2 * Math.PI * tb / 10) + 0.005 * rand.gauss();
  }
  const pulse = t => {
    let v = 0;
    for (const b of truth) {
      const d = t - b;
      if (d < -0.4 || d > 0.8) continue;
      const am = 1 + ampMod * Math.sin(2 * Math.PI * b / 10 + 1);   // breathing also modulates pulse height
      v += am * (Math.exp(-0.5 * (d / 0.09) ** 2) + 0.35 * Math.exp(-0.5 * ((d - 0.3) / 0.12) ** 2));
    }
    return v;
  };
  const samples = [];
  for (let t = 0; t < dur; ) {
    t += 1 / fps + (rand() - 0.5) * 0.008;        // jittery camera timestamps
    const p = pulse(t);
    const light = 1 + drift * Math.sin(2 * Math.PI * 0.15 * t) + motion * Math.sin(2 * Math.PI * 0.9 * t);
    samples.push({
      t,
      r: light * 180 * (1 - amp * 0.33 * p) + noise * rand.gauss(),   // more blood → darker, mostly green
      g: light * 120 * (1 - amp * 0.77 * p) + noise * rand.gauss(),
      b: light * 100 * (1 - amp * 0.53 * p) + noise * rand.gauss(),
    });
  }
  return { samples, truth };
}

// Reference RMSSD (ms) of the ground-truth beats in [t0, t1].
export function trueRmssd(truth, t0, t1) {
  const b = truth.filter(x => x >= t0 && x <= t1);
  let s = 0, n = 0;
  for (let i = 2; i < b.length; i++) { const d = ((b[i] - b[i - 1]) - (b[i - 1] - b[i - 2])) * 1000; s += d * d; n++; }
  return Math.sqrt(s / n);
}

// Same loop as the app: analysis every 0.25 s on the last 10 s, 15 s of history kept.
export function runPipeline(samples, { method = 'pos' } = {}) {
  const smoother = new BpmSmoother(), tracker = new BeatTracker();
  const buf = [], estimates = [];
  let last = -Infinity;
  for (const s of samples) {
    buf.push(s);
    while (buf[0].t < s.t - 15) buf.shift();
    if (s.t - last < 0.25) continue;
    last = s.t;
    const a = analyzeWindow(buf, { method });
    if (!a.ready) continue;
    const bpm = smoother.push(a.bpm, a.snr);
    estimates.push({ t: s.t, instant: a.bpm, bpm, snr: a.snr });
    tracker.detect(a.raw, a.t0, bpm);
  }
  return { smoother, tracker, estimates };
}
