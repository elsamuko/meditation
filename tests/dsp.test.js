// DSP tests on synthetic recordings. Open tests/index.html in a browser, or run tests/run.sh.
import { biquad, filt, resample, analyzeWindow, BpmSmoother, CausalPulse, EvmFilter, mean, std } from '../js/dsp.js';
import { makeRecording, trueRmssd, runPipeline } from './synth.js';

const results = [];
function test(name, fn) {
  const t0 = performance.now();
  try { const info = fn(); results.push({ name, ok: true, info, ms: performance.now() - t0 }); }
  catch (e) { results.push({ name, ok: false, info: e.message, ms: performance.now() - t0 }); }
}
function assert(cond, msg) { if (!cond) throw new Error(msg); }
const near = (a, b, tol, what) => assert(Math.abs(a - b) <= tol, `${what}: expected ${b} ± ${tol}, got ${a.toFixed(2)}`);

// ---------- basics ----------
test('biquad: low-pass passes DC, high-pass blocks it', () => {
  const ones = new Float64Array(600).fill(1);
  near(filt(ones, biquad('lp', 3.5)).at(-1), 1, 1e-6, 'LP DC gain');
  near(filt(ones, biquad('hp', 0.7)).at(-1), 0, 1e-6, 'HP DC gain');
});

test('resample: exact on a linear signal with jittery timestamps', () => {
  const { samples } = makeRecording({ dur: 5, noise: 0, amp: 0, drift: 0 });
  const lin = samples.map(s => ({ ...s, g: 3 * s.t + 7 }));
  const out = resample(lin, 'g', 1, 60);
  out.forEach((v, i) => near(v, 3 * (1 + i / 30) + 7, 1e-9, `sample ${i}`));
});

// ---------- heart rate ----------
for (const bpm of [50, 72, 95, 130, 170]) {
  // A single 10 s window resolves ~±3 bpm under noise; the displayed value is the median of recent windows.
  test(`heart rate: POS finds ${bpm} bpm under lighting drift and flicker`, () => {
    const { samples } = makeRecording({ bpm, rsa: 0, dur: 30, amp: 0.002, noise: 0.05, motion: 0.015, seed: bpm });
    const { smoother, estimates } = runPipeline(samples, { method: 'pos' });
    near(smoother.value, bpm, 2, 'displayed bpm');
    return `${smoother.value} bpm, SNR ${estimates.at(-1).snr.toFixed(1)} dB`;
  });
}

test('heart rate: POS cancels intensity flicker that fools the green-only method', () => {
  const { samples } = makeRecording({ bpm: 80, rsa: 0, dur: 20, amp: 0.001, noise: 0.05, motion: 0.015, seed: 3 });
  const w = samples.filter(s => s.t > 8);
  const p = analyzeWindow(w, { method: 'pos' }), g = analyzeWindow(w, { method: 'green' });
  near(p.bpm, 80, 1.5, 'POS bpm');
  assert(Math.abs(g.bpm - 80) > 5, `green unexpectedly right (${g.bpm})`);
  return `POS ${p.bpm}, green ${g.bpm} (flicker at 54 bpm)`;
});

test('BpmSmoother: ignores low-SNR estimates, returns median', () => {
  const s = new BpmSmoother();
  [70, 71, 72].forEach(b => s.push(b, 5));
  s.push(150, -10);
  near(s.value, 71, 0, 'median');
});

test('analyzeWindow: not ready before 6 s of data', () => {
  const { samples } = makeRecording({ dur: 5 });
  assert(!analyzeWindow(samples).ready, 'should not be ready');
});

// ---------- HRV ----------
test('HRV: clean signal → RMSSD within 15 % of truth, flagged reliable', () => {
  const { samples, truth } = makeRecording({ bpm: 70, amp: 0.004, noise: 0.01, seed: 11 });
  const { tracker } = runPipeline(samples, { method: 'green' });
  const h = tracker.rmssd(60), ref = trueRmssd(truth, tracker.lastBeatT - 60, tracker.lastBeatT);
  assert(h, 'no RMSSD');
  near(h.value, ref, 0.15 * ref, 'RMSSD');
  assert(h.reliable, `not flagged reliable (jitter ${h.jitter.toFixed(1)} ms)`);
  return `true ${ref.toFixed(1)} ms, est ${h.value.toFixed(1)} ms, jitter ${h.jitter.toFixed(1)} ms`;
});

test('HRV: beat count matches ground truth (no doubles, few misses)', () => {
  const { samples, truth } = makeRecording({ bpm: 70, amp: 0.004, noise: 0.01, seed: 12 });
  const { tracker } = runPipeline(samples, { method: 'green' });
  const last = tracker.ibis.at(-1).t, first = tracker.ibis[0].t;
  const expected = truth.filter(x => x >= first - 0.3 && x <= last + 0.3).length - 1;
  near(tracker.ibis.length, expected, 2, 'intervals');
  assert(tracker.ibis.filter(b => !b.ok).length <= 2, 'too many rejected');
});

test('HRV: never flags a reading reliable when it is > 30 % off (noise sweep)', () => {
  const rows = [];
  for (const method of ['pos', 'green'])
    for (const amp of [0.002, 0.004])
      for (const noise of [0, 0.05, 0.15, 0.3, 0.6]) {
        const { samples, truth } = makeRecording({ bpm: 72, amp, noise, seed: 100 + noise * 100 + amp * 1e4 });
        const { tracker } = runPipeline(samples, { method });
        const h = tracker.rmssd(60);
        if (!h) continue;
        const ref = trueRmssd(truth, tracker.lastBeatT - 60, tracker.lastBeatT);
        const err = Math.abs(h.value - ref) / ref;
        rows.push(`${method}/${amp}/${noise}: ${h.value.toFixed(0)} vs ${ref.toFixed(0)} ms ${h.reliable ? '✓' : '✗'}`);
        assert(!(h.reliable && err > 0.3), `${method} amp ${amp} noise ${noise}: flagged reliable but ${(err * 100).toFixed(0)} % off`);
      }
  return rows.join(' | ');
});

// ---------- real-time pulse & magnification ----------
test('CausalPulse: tracks the true pulse (correlation > 0.6)', () => {
  const { samples } = makeRecording({ bpm: 72, rsa: 0, dur: 30, amp: 0.004, noise: 0.02, seed: 21 });
  const clean = makeRecording({ bpm: 72, rsa: 0, dur: 30, amp: 0.004, noise: 0, drift: 0, seed: 21 }).samples;
  const cp = new CausalPulse(), out = [], ref = [];
  let prevT = 0;
  samples.forEach((s, i) => {
    const v = cp.step(s.r, s.g, s.b, prevT ? s.t - prevT : 1 / 30, 'pos'); prevT = s.t;
    if (s.t > 10) { out.push(v); ref.push(-clean[i].g); }
  });
  // causal filters add delay: take the best correlation within 0–300 ms lag
  let best = -1;
  for (let lag = 0; lag <= 9; lag++) {
    const a = out.slice(lag), b = ref.slice(0, ref.length - lag), ma = mean(a), mb = mean(b);
    const c = a.reduce((s, v, i) => s + (v - ma) * (b[i] - mb), 0) / (a.length * std(a) * std(b));
    best = Math.max(best, c);
  }
  assert(best > 0.6, `correlation ${best.toFixed(2)}`);
  return `correlation ${best.toFixed(2)}`;
});

test('EvmFilter: static scene → no change; pulsing bin → amplified change', () => {
  const evm = new EvmFilter(2), P = new Uint8ClampedArray(8), Q = new Uint8ClampedArray(8);
  let maxStatic = 0, maxPulse = 0;
  for (let i = 0; i < 300; i++) {
    const t = i / 30;
    evm.clear();
    evm.add(0, 150, 100, 90, true);                                         // static
    evm.add(1, 150, 100 - 0.2 * Math.sin(2 * Math.PI * 1.2 * t), 90, true);  // 0.2 px green pulse at 72 bpm
    evm.update(1 / 30, 72, 80, P, Q);
    if (i > 150) { maxStatic = Math.max(maxStatic, P[1], Q[1]); maxPulse = Math.max(maxPulse, P[5], Q[5]); }
  }
  assert(maxStatic === 0, `static bin changed by ${maxStatic}`);
  assert(maxPulse >= 5, `pulse bin only ${maxPulse}`);
  return `pulse amplified to ±${maxPulse} px`;
});

// ---------- report ----------
const failed = results.filter(r => !r.ok).length;
const summary = `${failed ? 'FAIL' : 'PASS'}: ${results.length - failed}/${results.length} tests passed`;
const lines = results.map(r => `${r.ok ? '✓' : '✗'} ${r.name}  (${r.ms.toFixed(0)} ms)${r.info ? '\n    ' + r.info : ''}`);
document.getElementById('out').textContent = lines.join('\n') + '\n\n' + summary;
document.title = summary;
console.log(lines.join('\n') + '\n' + summary);
