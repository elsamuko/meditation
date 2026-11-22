// Canvas plotting. Canvases are drawn at 2× their CSS size for sharpness.

const GRID = '#2a2f36', MUTED = '#9aa0a6', INK = '#e8eaed', PLOT_BG = '#0d1013';

// Line plot, each series auto-scaled. series: [{ data, color, marks? }] — marks are fractional sample indices.
export function drawLine(cv, series) {
  const c = cv.getContext('2d'), W = cv.width, H = cv.height;
  c.clearRect(0, 0, W, H);
  c.strokeStyle = GRID; c.lineWidth = 1;
  c.beginPath(); c.moveTo(0, H / 2); c.lineTo(W, H / 2); c.stroke();
  for (const { data, color, marks } of series) {
    let mn = Infinity, mx = -Infinity;
    for (const v of data) { if (v < mn) mn = v; if (v > mx) mx = v; }
    const rng = (mx - mn) || 1;
    const X = i => i / (data.length - 1) * W, Y = v => H - 10 - (v - mn) / rng * (H - 20);
    c.strokeStyle = color; c.lineWidth = 3; c.beginPath();
    data.forEach((v, i) => { i ? c.lineTo(X(i), Y(v)) : c.moveTo(X(i), Y(v)); });
    c.stroke();
    for (const m of marks || []) {
      const i = Math.floor(m), f = m - i, v = data[i] + f * ((data[i + 1] ?? data[i]) - data[i]);
      c.fillStyle = INK; c.strokeStyle = PLOT_BG; c.lineWidth = 3;
      c.beginPath(); c.arc(X(m), Y(v), 7, 0, 2 * Math.PI); c.fill(); c.stroke();
    }
  }
}

// Spectrum 40–200 bpm with the peak marked.
export function drawSpec(cv, spec, peak) {
  const c = cv.getContext('2d'), W = cv.width, H = cv.height;
  c.clearRect(0, 0, W, H);
  const mx = Math.max(...spec.map(s => s.p)) || 1;
  const X = bpm => (bpm - 40) / 160 * W;
  c.fillStyle = MUTED; c.font = '22px system-ui'; c.lineWidth = 1;
  for (let b = 40; b <= 200; b += 20) {
    c.strokeStyle = GRID; c.beginPath(); c.moveTo(X(b), 0); c.lineTo(X(b), H); c.stroke();
    c.fillText(b, X(b) + 4, H - 6);
  }
  c.strokeStyle = '#ff5d73'; c.lineWidth = 3; c.beginPath();
  spec.forEach((s, i) => { const y = H - 30 - s.p / mx * (H - 40); i ? c.lineTo(X(s.bpm), y) : c.moveTo(X(s.bpm), y); });
  c.stroke();
  c.strokeStyle = '#fff'; c.setLineDash([6, 6]); c.beginPath(); c.moveTo(X(peak), 0); c.lineTo(X(peak), H); c.stroke(); c.setLineDash([]);
}

function niceStep(x) {
  const p = 10 ** Math.floor(Math.log10(x)), m = x / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
}

// Single-series time trend with y-axis, minute ticks, gaps, latest-value dot and optional hover readout.
// opts: { points: [{t, [key]}], key, now, windowSec, color, unit, fmt, minPad, hoverX (0..1 | undefined), empty }
export function drawTrend(cv, { points, key, now, windowSec = 300, color, unit, fmt = v => v.toFixed(0), minPad = 3, hoverX, empty = 'waiting for data…' }) {
  const c = cv.getContext('2d'), W = cv.width, H = cv.height;
  const L = 70, Rp = 20, T = 20, B = 44;
  c.clearRect(0, 0, W, H);
  const pts = points.filter(p => p[key] != null);
  const X = t => L + (t - (now - windowSec)) / windowSec * (W - L - Rp);

  let mn = 0, mx = 1;
  if (pts.length) { mn = Math.min(...pts.map(p => p[key])); mx = Math.max(...pts.map(p => p[key])); }
  const pad = Math.max((mx - mn) * 0.15, minPad);
  mn = Math.max(0, mn - pad); mx += pad;
  const step = niceStep((mx - mn) / 4);
  mn = Math.floor(mn / step) * step; mx = Math.ceil(mx / step) * step;
  const Y = v => T + (1 - (v - mn) / (mx - mn)) * (H - T - B);

  // recessive grid + axis labels
  c.font = '22px system-ui'; c.fillStyle = MUTED; c.strokeStyle = GRID; c.lineWidth = 1;
  c.textAlign = 'right'; c.textBaseline = 'middle';
  for (let v = mn; v <= mx + step / 2 && pts.length; v += step) {   // no y-axis until there is data
    c.beginPath(); c.moveTo(L, Y(v)); c.lineTo(W - Rp, Y(v)); c.stroke(); c.fillText(fmt(v), L - 10, Y(v));
  }
  c.textAlign = 'center'; c.textBaseline = 'alphabetic';
  const mins = Math.round(windowSec / 60);
  for (let m = mins; m >= 0; m--) c.fillText(m ? `−${m} min` : 'now', Math.min(X(now - m * 60), W - Rp - 20), H - 10);

  if (!pts.length) { c.fillText(empty, W / 2, H / 2); return; }

  // the line (broken across gaps, e.g. while no face was visible)
  c.strokeStyle = color; c.lineWidth = 4; c.lineJoin = c.lineCap = 'round'; c.beginPath();
  let prev = null;
  for (const p of pts) {
    const x = X(p.t), y = Y(p[key]);
    if (!prev || p.t - prev.t > 15) c.moveTo(x, y); else c.lineTo(x, y);
    prev = p;
  }
  c.stroke();

  const dot = p => {
    c.fillStyle = color; c.strokeStyle = PLOT_BG; c.lineWidth = 4;
    c.beginPath(); c.arc(X(p.t), Y(p[key]), 8, 0, 2 * Math.PI); c.fill(); c.stroke();
  };
  dot(pts.at(-1));   // latest value

  // hover: crosshair + tooltip on the nearest sample
  if (hoverX === undefined) return;
  const tH = now - windowSec + (hoverX * W - L) / (W - L - Rp) * windowSec;
  let near = pts[0];
  for (const p of pts) if (Math.abs(p.t - tH) < Math.abs(near.t - tH)) near = p;
  const x = X(near.t);
  c.strokeStyle = MUTED; c.lineWidth = 2; c.setLineDash([6, 6]);
  c.beginPath(); c.moveTo(x, T); c.lineTo(x, H - B); c.stroke(); c.setLineDash([]);
  dot(near);
  const ago = Math.round(now - near.t);
  const txt = `${fmt(near[key])} ${unit}  ·  ${ago < 60 ? ago + ' s' : Math.floor(ago / 60) + ' min ' + (ago % 60) + ' s'} ago`;
  c.font = '24px system-ui';
  const tw = c.measureText(txt).width + 24, bx = Math.min(Math.max(x - tw / 2, L), W - Rp - tw);
  c.fillStyle = 'rgba(32,37,43,.96)'; c.fillRect(bx, T, tw, 40);
  c.fillStyle = INK; c.textAlign = 'left'; c.textBaseline = 'middle'; c.fillText(txt, bx + 12, T + 20);
}
