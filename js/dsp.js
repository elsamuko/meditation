// Signal processing for camera-based pulse measurement (rPPG). Pure functions/classes, no DOM — testable.

export const FS = 30;                 // resampling rate for windowed analysis (Hz)
export const LO_BPM = 42, HI_BPM = 200;

// ================= basics =================
export const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
export const std = a => { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length) || 1e-9; };

// RBJ biquad (Butterworth Q). Returns normalised [b0, b1, b2, a1, a2].
export function biquad(type, f0, fs = FS) {
  const w0 = 2 * Math.PI * f0 / fs, c = Math.cos(w0), al = Math.sin(w0) / (2 * Math.SQRT1_2);
  let b0, b1, b2;
  if (type === 'lp') { b0 = (1 - c) / 2; b1 = 1 - c; b2 = (1 - c) / 2; }
  else { b0 = (1 + c) / 2; b1 = -(1 + c); b2 = (1 + c) / 2; }
  const a0 = 1 + al;
  return [b0 / a0, b1 / a0, b2 / a0, -2 * c / a0, (1 - al) / a0];
}
export const bqState = () => [0, 0, 0, 0];
export function bqStep(c, s, x) {
  const y = c[0] * x + c[1] * s[0] + c[2] * s[1] - c[3] * s[2] - c[4] * s[3];
  s[1] = s[0]; s[0] = x; s[3] = s[2]; s[2] = y; return y;
}
export function filt(x, c) {
  const y = new Float64Array(x.length), s = bqState();
  for (let i = 0; i < x.length; i++) y[i] = bqStep(c, s, x[i]);
  return y;
}
// Mean removal, then the biquad chain forward and backward (zero phase).
export function zeroPhase(x, chain) {
  const m = mean(x);
  let y = Float64Array.from(x, v => v - m);
  for (const c of chain) y = filt(y, c);
  y.reverse();
  for (const c of chain) y = filt(y, c);
  return y.reverse();
}
const HP = biquad('hp', LO_BPM / 60 * 0.85), LP = biquad('lp', 3.5);
export const bandpass = x => zeroPhase(x, [HP, LP]);

// Linear interpolation of irregular camera timestamps onto a uniform grid. samples: [{t, r, g, b}]
export function resample(samples, key, t0, N, fs = FS) {
  const out = new Float64Array(N);
  let j = 0;
  for (let i = 0; i < N; i++) {
    const ti = t0 + i / fs;
    while (j < samples.length - 2 && samples[j + 1].t < ti) j++;
    const a = samples[j], b = samples[j + 1];
    const f = Math.min(Math.max((ti - a.t) / (b.t - a.t || 1), 0), 1);
    out[i] = a[key] + f * (b[key] - a[key]);
  }
  return out;
}

// ================= pulse extraction =================
// POS — Wang, den Brinker, Stuijk, de Haan, "Algorithmic Principles of Remote PPG", IEEE TBME 2017.
// Projects temporally normalised RGB onto a plane orthogonal to skin tone so intensity/specular changes
// cancel and the pulse adds up. Output sign: up = more blood.
export function pos(R, G, B, fs = FS) {
  const N = R.length, L = Math.round(1.6 * fs), H = new Float64Array(N);
  const s1 = new Float64Array(L), s2 = new Float64Array(L), h = new Float64Array(L);
  for (let n = 0; n + L <= N; n++) {
    let mr = 0, mg = 0, mb = 0;
    for (let k = 0; k < L; k++) { mr += R[n + k]; mg += G[n + k]; mb += B[n + k]; }
    mr /= L; mg /= L; mb /= L;
    for (let k = 0; k < L; k++) {
      const r = R[n + k] / mr, g = G[n + k] / mg, b = B[n + k] / mb;
      s1[k] = g - b; s2[k] = g + b - 2 * r;
    }
    const alpha = std(s1) / std(s2);
    let hm = 0;
    for (let k = 0; k < L; k++) { h[k] = s1[k] + alpha * s2[k]; hm += h[k]; }
    hm /= L;
    for (let k = 0; k < L; k++) H[n + k] -= h[k] - hm;   // overlap-add
  }
  return H;
}
// Classic green-channel method: more blood → less green, so invert.
export function greenPulse(G) { const m = mean(G); return G.map(v => -(v / m - 1)); }

// Power at 40–200 bpm in 0.5 bpm steps (direct DFT, Hann window).
export function spectrum(x, fs = FS) {
  const N = x.length, out = [];
  const w = x.map((v, i) => v * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N - 1))));
  for (let bpm = 40; bpm <= 200; bpm += 0.5) {
    const om = 2 * Math.PI * bpm / 60 / fs;
    let re = 0, im = 0;
    for (let i = 0; i < N; i++) { re += w[i] * Math.cos(om * i); im -= w[i] * Math.sin(om * i); }
    out.push({ bpm, p: re * re + im * im });
  }
  return out;
}

// Spectral peak + SNR (power within ±6 bpm of the peak and its 1st harmonic vs. rest of band; de Haan 2013).
export function peakAndSnr(spec) {
  let best = spec[0];
  for (const s of spec) if (s.bpm >= LO_BPM && s.bpm <= HI_BPM && s.p > best.p) best = s;
  let sigP = 0, noiseP = 0;
  for (const s of spec) {
    if (s.bpm < LO_BPM || s.bpm > HI_BPM) continue;
    if (Math.abs(s.bpm - best.bpm) <= 6 || Math.abs(s.bpm - 2 * best.bpm) <= 6) sigP += s.p; else noiseP += s.p;
  }
  return { bpm: best.bpm, snr: 10 * Math.log10(sigP / (noiseP || 1e-12)) };
}

// Full windowed analysis of the last `winSec` seconds of samples.
// Returns { ready: false, sec } while collecting, else { ready: true, t0, raw, sig, spec, bpm, snr }.
export function analyzeWindow(samples, { method = 'pos', winSec = 10, minSec = 6, fs = FS } = {}) {
  if (samples.length < 20) return { ready: false, sec: 0 };
  const sec = Math.min(samples[samples.length - 1].t - samples[0].t, winSec);
  if (sec < minSec) return { ready: false, sec };
  const N = Math.floor(sec * fs);
  const t0 = samples[samples.length - 1].t - (N - 1) / fs;
  const G = resample(samples, 'g', t0, N, fs);
  const raw = method === 'pos'
    ? pos(resample(samples, 'r', t0, N, fs), G, resample(samples, 'b', t0, N, fs), fs)
    : greenPulse(G);
  const sig = bandpass(raw);
  const spec = spectrum(sig, fs);
  return { ready: true, t0, raw, sig, spec, ...peakAndSnr(spec) };
}

// Median of recent confident estimates.
export class BpmSmoother {
  constructor(n = 12, minSnr = -3) { this.n = n; this.minSnr = minSnr; this.reset(); }
  reset() { this.hist = []; this.value = null; }
  push(bpm, snr) {
    if (snr > this.minSnr) this.hist.push(bpm);
    if (this.hist.length > this.n) this.hist.shift();
    const s = [...this.hist].sort((a, b) => a - b);
    this.value = s.length ? s[s.length >> 1] : null;
    return this.value;
  }
}

// ================= beats & HRV =================
// Beat timing for HRV. RMSSD is very sensitive to timing jitter (measured² ≈ true² + 6·jitter²), and at
// 30 fps picking peak tops is too noisy. Instead: a narrow zero-phase band-pass around the current heart
// rate (f0 ± 0.5 Hz — wide enough to keep breathing-related IBI modulation, which sits at f0 ± 0.1–0.4 Hz),
// then each beat is timed at the rising zero crossing, the steepest (most noise-robust) point, with linear
// interpolation between frames. Beats are accepted only once 1.5 s old, away from the filter's edge effects.
// Call detect() repeatedly on overlapping windows; already-seen beats are skipped.
export class BeatTracker {
  constructor(fs = FS) { this.fs = fs; this.reset(); }
  reset() { this.beats = []; this.ibis = []; this.lastBeatT = 0; }   // ibis: {t, ibi (s), ok, amp, omega}

  detect(raw, t0, bpm) {
    if (!bpm) return;
    const fs = this.fs, f0 = bpm / 60, ibiExp = 1 / f0;
    const hp = biquad('hp', Math.max(0.5, f0 - 0.5), fs), lp = biquad('lp', f0 + 0.5, fs);
    const sig = zeroPhase(raw, [hp, hp, lp, lp]);
    const N = sig.length, tEnd = t0 + (N - 1) / fs;
    for (let i = 1; i < N; i++) {
      if (!(sig[i - 1] < 0 && sig[i] >= 0)) continue;
      const tb = t0 + (i - 1 + sig[i - 1] / (sig[i - 1] - sig[i])) / fs;
      if (tb <= this.lastBeatT + 0.6 * ibiExp || tb > tEnd - 1.5) continue;
      // height of the following crest — its beat-to-beat scatter measures the in-band noise (see rmssd)
      let amp = 0;
      for (let k = i; k < Math.min(N, i + Math.round(0.5 * ibiExp * fs)); k++) amp = Math.max(amp, sig[k]);
      this._add(tb, ibiExp, amp, 2 * Math.PI * f0);
    }
  }

  _add(tb, ibiExp, amp, omega) {
    if (this.lastBeatT) {
      const ibi = tb - this.lastBeatT;
      // reject artifacts: physiologically implausible, or >25% off the recent median interval
      const recent = this.ibis.slice(-8).map(b => b.ibi).sort((x, y) => x - y);
      const ref = recent.length >= 3 ? recent[recent.length >> 1] : ibiExp;
      const ok = ibi > 0.33 && ibi < 1.5 && Math.abs(ibi - ref) < 0.25 * ref;
      this.ibis.push({ t: tb, ibi, ok, amp, omega });
      while (this.ibis.length && this.ibis[0].t < tb - 120) this.ibis.shift();
    }
    this.lastBeatT = tb;
    this.beats.push(tb);
    while (this.beats.length && this.beats[0] < tb - 15) this.beats.shift();
  }

  get okCount() { return this.ibis.filter(b => b.ok).length; }

  // RMSSD: root mean square of successive IBI differences — the standard short-term HRV measure, mostly
  // reflecting parasympathetic (rest-and-digest) activity. Only pairs of consecutive clean beats count.
  // Returns null until 10 pairs are available.
  rmssd(sec = 60) {
    const tMin = this.lastBeatT - sec, win = this.ibis.filter(b => b.t >= tMin);
    let sum = 0, pairs = 0;
    for (let i = 1; i < win.length; i++) {
      if (!win[i].ok || !win[i - 1].ok) continue;
      const d = (win[i].ibi - win[i - 1].ibi) * 1000;
      sum += d * d; pairs++;
    }
    if (pairs < 10) return null;
    const rejected = Math.round(100 * win.filter(b => !b.ok).length / win.length);
    // Timing jitter: for a near-sinusoid A·sin(ωt) + noise σ, the zero-crossing error is σ/(Aω) and the crest
    // height scatters by σ. Breathing also modulates crest height, but slowly (~every 10 beats), so σ is taken
    // from second differences of consecutive crests (var = 6σ² for white noise; slow modulation mostly cancels).
    // Jitter inflates RMSSD² by ≈ 6·jitter² (each successive difference uses three beat times: 1, −2, 1).
    // Subtracting that proved too fragile in simulation (×6 amplifies estimation error), so jitter is used
    // as a reliability gate instead: below ~30 % of RMSSD (and ≤ 15 ms absolute) the inflation is small (≲ 25 %).
    let d2 = 0, nd2 = 0, ampSum = 0, omSum = 0, nA = 0;
    for (let i = 0; i < win.length; i++) {
      if (!win[i].ok) continue;
      ampSum += win[i].amp; omSum += win[i].omega; nA++;
      if (i >= 2 && win[i - 1].ok && win[i - 2].ok) {
        const d = win[i].amp - 2 * win[i - 1].amp + win[i - 2].amp; d2 += d * d; nd2++;
      }
    }
    const jitter = nd2 ? Math.sqrt(d2 / nd2 / 6) / ((ampSum / nA) * (omSum / nA)) : 0;
    const value = Math.sqrt(sum / pairs), jitMs = jitter * 1000;
    return { value, jitter: jitMs, pairs, rejected, reliable: jitMs < Math.min(0.3 * value, 15) && rejected < 20 };
  }
}

// ================= real-time pulse =================
// Sample-by-sample POS: running-mean normalisation, causal band-pass, running alpha.
// step() returns the pulse normalised to ~unit std (up = more blood). Used for the tint and heart flash.
export class CausalPulse {
  constructor() { this.reset(); }
  reset() { this.init = false; }
  step(r, g, b, dt, method = 'pos') {
    const fsNow = Math.min(Math.max(1 / dt, 10), 60);
    const HPc = biquad('hp', 0.7, fsNow), LPc = biquad('lp', 3.5, fsNow);
    const bp = (st, x) => bqStep(LPc, st[1], bqStep(HPc, st[0], x));
    if (!this.init) Object.assign(this, {
      init: true, mr: r, mg: g, mb: b, v1: 1e-10, v2: 1e-10, vp: 1e-10,
      f1: [bqState(), bqState()], f2: [bqState(), bqState()], fg: [bqState(), bqState()],
    });
    const km = 1 - Math.exp(-dt / 1.6), kv = 1 - Math.exp(-dt / 2.5);
    this.mr += km * (r - this.mr); this.mg += km * (g - this.mg); this.mb += km * (b - this.mb);
    const rn = r / this.mr - 1, gn = g / this.mg - 1, bn = b / this.mb - 1;
    const s1 = bp(this.f1, gn - bn), s2 = bp(this.f2, gn + bn - 2 * rn), sg = bp(this.fg, gn);
    this.v1 += kv * (s1 * s1 - this.v1); this.v2 += kv * (s2 * s2 - this.v2);
    const out = method === 'pos' ? -(s1 + Math.sqrt(this.v1 / this.v2) * s2) : -sg;
    this.vp += kv * (out * out - this.vp);
    return out / Math.sqrt(this.vp);
  }
}

// ================= Eulerian video magnification (Wu et al., SIGGRAPH 2012) =================
// Temporal part only: per-bin band-pass around the heart rate (difference of two first-order IIR low-passes),
// amplified, split into positive / negative RGBA images (the caller adds them back on the GPU).
// Bins are filled by the caller each frame via clear() + add(); faceSkin weights fade out non-face pixels.
export class EvmFilter {
  constructor(n) {
    this.n = n; this.sum = new Float32Array(n * 3); this.cnt = new Uint16Array(n); this.skin = new Uint16Array(n);
    this.lo1 = new Float32Array(n * 3); this.lo2 = new Float32Array(n * 3); this.init = false;
  }
  clear() { this.sum.fill(0); this.cnt.fill(0); this.skin.fill(0); }
  add(j, R, G, B, faceSkin) {
    const k = j * 3; this.sum[k] += R; this.sum[k + 1] += G; this.sum[k + 2] += B; this.cnt[j]++;
    if (faceSkin) this.skin[j]++;
  }
  // P, Q: RGBA byte arrays (n*4) receiving the positive / negative amplified change.
  update(dt, bpm, gain, P, Q) {
    const f0 = bpm ? bpm / 60 : 1.2;                         // band centred on the current heart rate
    const aHi = 1 - Math.exp(-2 * Math.PI * f0 * 1.6 * dt);
    const aLo = 1 - Math.exp(-2 * Math.PI * f0 * 0.6 * dt);
    const { n, sum, cnt, skin, lo1, lo2 } = this;
    for (let j = 0; j < n; j++) {
      const c = cnt[j] || 1, w = skin[j] / c, w2 = w * w;     // fade out at face edges (motion artifacts)
      for (let ch = 0; ch < 3; ch++) {
        const k = j * 3 + ch, x = sum[k] / c;
        if (!this.init) { lo1[k] = lo2[k] = x; }
        lo1[k] += aHi * (x - lo1[k]);
        lo2[k] += aLo * (x - lo2[k]);
        const d = (lo1[k] - lo2[k]) * gain * w2;
        if (P) { P[j * 4 + ch] = d > 0 ? Math.min(d, 255) : 0; Q[j * 4 + ch] = d < 0 ? Math.min(-d, 255) : 0; }
      }
      if (P) P[j * 4 + 3] = Q[j * 4 + 3] = 255;
    }
    this.init = true;
  }
}
