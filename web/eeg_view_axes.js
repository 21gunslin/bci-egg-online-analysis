/* Pure viewing geometry. Never modifies source samples, PSD, or FIF metadata. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.EEGViewAxes = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  function niceCeil(value) {
    if (!Number.isFinite(value) || value <= 0) return 1;
    const order = 10 ** Math.floor(Math.log10(value));
    return [1, 2, 5, 10].find((n) => n * order >= value * (1 - 1e-12)) * order;
  }
  function stats(row) {
    if (row.stats) return row.stats;
    const values = row.points.map((p) => p[1]),
      sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2),
      median = sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
    return {
      min: sorted[0],
      max: sorted.at(-1),
      median,
      ptp: sorted.at(-1) - sorted[0],
      n: values.length,
    };
  }
  function axes(rows, { mode = 'channel', gain = 1, fixedStep = 100 } = {}) {
    if (
      !['channel', 'shared', 'fixed'].includes(mode) ||
      !Number.isFinite(gain) ||
      gain < 0.001 ||
      gain > 100 ||
      !Number.isFinite(fixedStep) ||
      fixedStep <= 0
    )
      throw new Error('纵轴参数无效。');
    const prepared = rows.map((row) => {
      const s = stats(row),
        center = s.median,
        extent = Math.max(Math.abs(s.max - center), Math.abs(s.min - center));
      return { row, s, center, extent };
    });
    const shared = new Map();
    for (const p of prepared)
      shared.set(p.row.unit, Math.max(shared.get(p.row.unit) || 0, p.extent));
    return prepared.map((p) => {
      const extent = mode === 'shared' ? shared.get(p.row.unit) : p.extent;
      const floor = p.row.unit === 'µV' ? 1 : Math.max(Math.abs(p.center) * 1e-12, 1e-6);
      const baseStep =
        mode === 'fixed' && p.row.unit === 'µV'
          ? fixedStep
          : niceCeil(Math.max(extent / 1.25, floor));
      const step = baseStep / gain,
        halfRange = step * 1.4;
      return {
        name: p.row.name,
        unit: p.row.unit,
        center: p.center,
        step,
        halfRange,
        yMin: p.center - halfRange,
        yMax: p.center + halfRange,
        flat: p.s.ptp === 0,
        min: p.s.min,
        max: p.s.max,
        ptp: p.s.ptp,
        stats: p.s,
        clipped: p.s.min < p.center - halfRange || p.s.max > p.center + halfRange,
      };
    });
  }
  function zoomTime({ start, seconds, total, fraction = 0.5, factor }) {
    if (
      ![start, seconds, total, fraction, factor].every(Number.isFinite) ||
      total <= 0 ||
      seconds <= 0 ||
      factor <= 0
    )
      throw new Error('横轴参数无效。');
    const minimum = Math.min(0.01, total),
      next = clamp(seconds / factor, minimum, total),
      anchor = start + clamp(fraction, 0, 1) * seconds;
    return { start: clamp(anchor - clamp(fraction, 0, 1) * next, 0, total - next), seconds: next };
  }
  function panTime({ start, seconds, total, fraction }) {
    return {
      start: clamp(start + fraction * seconds, 0, Math.max(0, total - seconds)),
      seconds: Math.min(seconds, total),
    };
  }
  return { niceCeil, stats, axes, zoomTime, panTime };
});
