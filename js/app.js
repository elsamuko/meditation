// Camera, face tracking, visualisation and UI wiring. Signal processing lives in dsp.js.
import { FS, analyzeWindow, BpmSmoother, BeatTracker, CausalPulse, EvmFilter } from './dsp.js';
import { drawLine, drawSpec, drawTrend } from './plots.js';

// ================= config =================
const WIN = 10;           // analysis window (s)
const MIN_SEC = 6;        // data needed before the first estimate (s)
const KEEP = 15;          // history kept (s)
const TREND_SEC = 300;    // trend charts window (s)
const WORK_W = 320;       // analysis resolution (width); height follows aspect ratio
const BIN = 4;            // Eulerian magnification works on WORK/BIN (spatial low-pass)

// Vendored locally (fully offline): @mediapipe/tasks-vision 0.10.14 (Apache-2.0) + face_landmarker float16 v1
const MP_URL = new URL('../vendor/mediapipe', import.meta.url).href;
const MODEL = new URL('../vendor/models/face_landmarker.task', import.meta.url).href;

// MediaPipe face-mesh landmark indices (subject's left/right are mirrored in the image)
const FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];
const FOREHEAD  = [67,109,10,338,297,333,299,337,151,108,69,104];
const CHEEK_A   = [116,117,118,101,36,205,187,123];
const CHEEK_B   = [345,346,347,330,266,425,411,352];
const ROIS = [FOREHEAD, CHEEK_A, CHEEK_B];

// ================= DOM / state =================
const $ = id => document.getElementById(id);
const video = $('video'), view = $('view');
const vctx = view.getContext('2d');
const mkCanvas = () => document.createElement('canvas');
const work = mkCanvas(), wctx = work.getContext('2d', { willReadFrequently: true });
const mask = mkCanvas(), mctx = mask.getContext('2d', { willReadFrequently: true });
const posC = mkCanvas(), pctx = posC.getContext('2d');
const negC = mkCanvas(), nctx = negC.getContext('2d');

let VW = 640, VH = 480, WW = 320, WH = 240, SW = 80, SH = 60;
let landmarker = null;
let manualRoi = { cx: 0.5, cy: 0.28 };
let samples = [];                  // {t, r, g, b}
let lastAnalysis = 0, fpsCount = 0, fpsT0 = 0, fps = 30, lastT = 0, lastMpT = 0;
let lastFlash = 0, prevPn = 0;
let evm = null, pImg = null, nImg = null;
let trend = [], lastTrendT = 0;    // {t, hr, hrv}; kept across resets

const smoother = new BpmSmoother();
const tracker = new BeatTracker();
const causal = new CausalPulse();
const method = () => $('method').value;

const reset = () => { samples = []; smoother.reset(); tracker.reset(); causal.reset(); };
$('start').onclick = start;
$('method').onchange = reset;
$('roimode').onchange = reset;
view.onclick = e => {
  if ($('roimode').value !== 'manual') return;
  const r = view.getBoundingClientRect();
  manualRoi.cx = 1 - (e.clientX - r.left) / r.width;     // canvas is mirrored on screen
  manualRoi.cy = (e.clientY - r.top) / r.height;
  reset();
};

async function loadLandmarker() {
  $('track').textContent = 'loading face model…';
  try {
    const { FaceLandmarker, FilesetResolver } = await import(`${MP_URL}/vision_bundle.mjs`);
    const fileset = await FilesetResolver.forVisionTasks(`${MP_URL}/wasm`);
    const make = delegate => FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL, delegate }, runningMode: 'VIDEO', numFaces: 1,
    });
    try { landmarker = await make('GPU'); } catch (_) { landmarker = await make('CPU'); }
    $('track').textContent = 'face model ready';
  } catch (err) {
    console.error(err);
    $('track').textContent = 'face model failed to load → manual box';
    $('roimode').value = 'manual';
  }
}

// (Re)size all buffers to the camera frame. Called again when a phone rotates.
function setupSizes() {
  VW = video.videoWidth; VH = video.videoHeight;
  view.width = VW; view.height = VH;
  WW = WORK_W; WH = Math.round(WORK_W * VH / VW);
  SW = Math.floor(WW / BIN); SH = Math.floor(WH / BIN);
  work.width = mask.width = WW; work.height = mask.height = WH;
  posC.width = negC.width = SW; posC.height = negC.height = SH;
  evm = new EvmFilter(SW * SH);
  pImg = new ImageData(SW, SH); nImg = new ImageData(SW, SH);
}

// Keep the phone screen from dimming/locking during a measurement.
async function keepAwake() {
  try { await navigator.wakeLock?.request('screen'); } catch (_) {}
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && video.srcObject) keepAwake();
});

async function start() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    $('stats').textContent = 'Camera needs a secure page: open via https:// (or http://localhost on this computer).';
    return;
  }
  $('start').disabled = true;
  const lmPromise = loadLandmarker();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    // Auto-exposure / white balance drift swamps the tiny signal: lock them where supported.
    const track = stream.getVideoTracks()[0];
    const caps = track.getCapabilities ? track.getCapabilities() : {};
    const adv = {};
    if (caps.exposureMode?.includes('manual')) adv.exposureMode = 'manual';
    if (caps.whiteBalanceMode?.includes('manual')) adv.whiteBalanceMode = 'manual';
    if (Object.keys(adv).length) { try { await track.applyConstraints({ advanced: [adv] }); } catch (_) {} }

    setupSizes();
    keepAwake();

    $('start').textContent = 'Running';
    fpsT0 = performance.now();
    await lmPromise;
    const next = video.requestVideoFrameCallback
      ? cb => video.requestVideoFrameCallback(cb) : cb => requestAnimationFrame(cb);
    const step = () => { onFrame(performance.now()); next(step); };
    next(step);
  } catch (err) {
    $('stats').textContent = 'Camera error: ' + err.message;
    $('start').disabled = false;
  }
}

// ================= per frame =================
const isSkin = (R, G, B) => {
  const Y = 0.299 * R + 0.587 * G + 0.114 * B;
  return Y > 35 && Y < 245 && R > G && R > B;
};

function manualRect() {
  const w = WW * 0.18, h = WH * 0.11;
  return {
    x: Math.min(Math.max(manualRoi.cx * WW - w / 2, 0), WW - w),
    y: Math.min(Math.max(manualRoi.cy * WH - h / 2, 0), WH - h), w, h,
  };
}

function polyPath(ctx, lm, idx, sx, sy) {
  ctx.beginPath();
  idx.forEach((k, i) => { const p = lm[k]; i ? ctx.lineTo(p.x * sx, p.y * sy) : ctx.moveTo(p.x * sx, p.y * sy); });
  ctx.closePath();
}

function onFrame(t) {
  if (video.videoWidth && (video.videoWidth !== VW || video.videoHeight !== VH)) { setupSizes(); reset(); }
  const dt = lastT ? Math.min((t - lastT) / 1000, 0.2) : 1 / 30;
  lastT = t;
  fpsCount++;
  if (t - fpsT0 > 1000) { fps = fpsCount * 1000 / (t - fpsT0); fpsCount = 0; fpsT0 = t; }

  // 1) face landmarks
  let lm = null;
  const useFace = $('roimode').value === 'face' && landmarker;
  if (landmarker) {
    const ts = Math.max(t, lastMpT + 1); lastMpT = ts;
    lm = landmarker.detectForVideo(video, ts).faceLandmarks?.[0] ?? null;
  }

  // 2) frame at analysis resolution
  wctx.drawImage(video, 0, 0, WW, WH);
  const fr = wctx.getImageData(0, 0, WW, WH).data;

  // 3) masks: red channel = measurement ROI, green channel = whole face
  mctx.globalCompositeOperation = 'source-over';
  mctx.fillStyle = '#000'; mctx.fillRect(0, 0, WW, WH);
  if (lm) { mctx.fillStyle = '#0f0'; polyPath(mctx, lm, FACE_OVAL, WW, WH); mctx.fill(); }
  mctx.globalCompositeOperation = 'lighter';
  mctx.fillStyle = '#f00';
  if (useFace) { if (lm) for (const r of ROIS) { polyPath(mctx, lm, r, WW, WH); mctx.fill(); } }
  else { const m = manualRect(); mctx.fillRect(m.x, m.y, m.w, m.h); }
  const mk = mctx.getImageData(0, 0, WW, WH).data;

  // 4) one pass: ROI skin mean + spatially binned frame for magnification
  let r = 0, g = 0, b = 0, n = 0, nRoi = 0;
  evm.clear();
  for (let y = 0; y < SH * BIN; y++) {
    const rowBin = (y / BIN | 0) * SW;
    for (let x = 0; x < SW * BIN; x++) {
      const i = (y * WW + x) * 4, R = fr[i], G = fr[i + 1], B = fr[i + 2];
      const skin = isSkin(R, G, B);
      evm.add(rowBin + (x / BIN | 0), R, G, B, skin && mk[i + 1] > 127);
      if (mk[i] > 127) { nRoi++; if (skin) { r += R; g += G; b += B; n++; } }
    }
  }
  const tracking = useFace ? !!lm : true;
  let pn = 0;
  if (tracking && n > 30) {
    r /= n; g /= n; b /= n;
    samples.push({ t: t / 1000, r, g, b });
    pn = causal.step(r, g, b, dt, method());
  }
  while (samples.length && samples[0].t < t / 1000 - KEEP) samples.shift();

  // 5) draw
  vctx.globalCompositeOperation = 'source-over';
  vctx.drawImage(video, 0, 0, VW, VH);
  const vis = $('vis').value, gain = +$('gain').value;
  const showEvm = vis === 'evm' && lm && gain > 0;
  evm.update(dt, smoother.value, gain, showEvm ? pImg.data : null, showEvm ? nImg.data : null);   // keeps filters warm
  if (showEvm) {
    // signed add on the GPU: 'lighter' adds the positive part, 'difference' subtracts the negative
    pctx.putImageData(pImg, 0, 0); nctx.putImageData(nImg, 0, 0);
    vctx.imageSmoothingEnabled = true;
    vctx.globalCompositeOperation = 'lighter';    vctx.drawImage(posC, 0, 0, VW, VH);
    vctx.globalCompositeOperation = 'difference'; vctx.drawImage(negC, 0, 0, VW, VH);
    vctx.globalCompositeOperation = 'source-over';
  }
  if (vis === 'tint' && lm && tracking) {
    const k = Math.min(Math.max(pn, 0) * gain * 0.5, 140);
    vctx.globalCompositeOperation = 'multiply';           // redden + darken = more blood
    vctx.fillStyle = `rgb(255,${255 - k | 0},${255 - k * 0.8 | 0})`;
    polyPath(vctx, lm, FACE_OVAL, VW, VH); vctx.fill();
    vctx.globalCompositeOperation = 'source-over';
  }
  if ($('showroi').checked) drawOutlines(lm, useFace, n / Math.max(nRoi, 1));

  // heart flash on each causal pulse peak
  if (prevPn < 0.8 && pn >= 0.8 && t - lastFlash > 300) {
    lastFlash = t; $('heart').style.transform = 'scale(1.35)';
    setTimeout(() => $('heart').style.transform = '', 120);
  }
  prevPn = pn;

  if (landmarker) $('track').textContent = useFace ? (lm ? 'face tracked' : 'no face found') : 'manual box';
  if (t - lastAnalysis > 250) { lastAnalysis = t; analyze(n / Math.max(nRoi, 1), tracking); }
}

function drawOutlines(lm, useFace, skinFrac) {
  vctx.lineWidth = 2;
  vctx.strokeStyle = skinFrac > 0.6 ? 'rgba(61,220,132,.9)' : 'rgba(255,176,32,.9)';
  if (useFace && lm) for (const r of ROIS) { polyPath(vctx, lm, r, VW, VH); vctx.stroke(); }
  if (!useFace) { const m = manualRect(), s = VW / WW; vctx.strokeRect(m.x * s, m.y * s, m.w * s, m.h * s); }
}

// ================= analysis (4× per second) =================
function analyze(skinFrac, tracking) {
  if (!tracking) { $('stats').textContent = 'no face — look at the camera'; return; }
  const a = analyzeWindow(samples, { method: method(), winSec: WIN, minSec: MIN_SEC });
  if (!a.ready) {
    $('stats').textContent = `collecting… ${a.sec.toFixed(1)} / ${MIN_SEC} s   |   ${fps.toFixed(0)} fps`;
    drawRaw(); return;
  }
  const bpm = smoother.push(a.bpm, a.snr);
  $('bpmval').textContent = bpm ? Math.round(bpm) : '--';
  const q = a.snr > 3 ? 'good' : a.snr > 0 ? 'ok' : 'poor — hold still / more light';
  $('stats').textContent =
    `instant ${a.bpm.toFixed(1)} bpm   |   SNR ${a.snr.toFixed(1)} dB (${q})   |   ${fps.toFixed(0)} fps   |   skin ${(skinFrac * 100).toFixed(0)}%`;

  // beats → inter-beat intervals → HRV
  tracker.detect(a.raw, a.t0, bpm);
  const hrv = tracker.rmssd(60);
  $('hrvval').textContent = hrv ? Math.round(hrv.value) : '--';
  $('hrvval').style.opacity = hrv && hrv.reliable ? 1 : 0.45;
  $('hrvinfo').textContent = hrv
    ? `${hrv.reliable ? '✓ reliable' : '⚠ unreliable — too noisy, hold still / more light'}  ·  ` +
      `timing jitter ±${hrv.jitter.toFixed(0)} ms  ·  ${hrv.pairs} beat pairs, ${hrv.rejected}% rejected`
    : `collecting beats… (${tracker.okCount} so far, need ~12)`;

  const now = samples[samples.length - 1].t;
  if (bpm && now - lastTrendT >= 2) {
    lastTrendT = now;
    trend.push({ t: now, hr: bpm, hrv: hrv && hrv.reliable ? hrv.value : null });
    while (trend.length && trend[0].t < now - TREND_SEC) trend.shift();
  }
  drawTrends();

  const marks = tracker.beats.filter(tb => tb >= a.t0).map(tb => (tb - a.t0) * FS);
  drawLine($('sig'), [{ data: a.sig, color: '#ff5d73', marks }]);
  drawSpec($('spec'), a.spec, a.bpm);
  drawRaw();
}

// ================= plots =================
const hover = new Map();   // trend canvas → pointer x (0..1)
for (const id of ['trendHr', 'trendHrv']) {
  const cv = $(id);
  cv.addEventListener('pointermove', e => {
    const r = cv.getBoundingClientRect(); hover.set(cv, (e.clientX - r.left) / r.width); drawTrends();
  });
  cv.addEventListener('pointerleave', () => { hover.delete(cv); drawTrends(); });
}

function drawTrends() {
  const now = samples.length ? samples[samples.length - 1].t : (trend.at(-1)?.t ?? 0);
  const common = { points: trend, now, windowSec: TREND_SEC };
  drawTrend($('trendHr'), { ...common, key: 'hr', color: '#ff5d73', unit: 'bpm', minPad: 3, hoverX: hover.get($('trendHr')) });
  drawTrend($('trendHrv'), { ...common, key: 'hrv', color: '#4dd0c8', unit: 'ms', minPad: 5, hoverX: hover.get($('trendHrv')),
    empty: 'waiting for reliable HRV…' });
}
drawTrends();   // empty state until data arrives

function drawRaw() {
  if (samples.length < 2) return;
  const take = samples.slice(-WIN * 30);
  drawLine($('raw'), [
    { data: take.map(s => s.r), color: '#ff6b6b' },
    { data: take.map(s => s.g), color: '#51cf66' },
    { data: take.map(s => s.b), color: '#4dabf7' },
  ]);
}
