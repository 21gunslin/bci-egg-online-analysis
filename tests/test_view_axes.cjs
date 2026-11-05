const test = require('node:test');
const assert = require('node:assert/strict');
const { axes, zoomTime, panTime } = require('../web/eeg_view_axes.js');
function row(name, min, max, median, unit = 'µV') {
  return {
    name,
    unit,
    points: [
      [0, min],
      [1, max],
    ],
    stats: { min, max, median, ptp: max - min, n: 1000 },
  };
}
test('per-channel fit covers all actual extrema and does not normalize samples', () => {
  const rows = [row('O1', -718000, -712000, -715000), row('F3', -5000, 2000, -2000)];
  const before = JSON.stringify(rows),
    a = axes(rows);
  for (let i = 0; i < rows.length; i++) {
    assert.ok(a[i].yMin <= rows[i].stats.min);
    assert.ok(a[i].yMax >= rows[i].stats.max);
    assert.equal(a[i].clipped, false);
  }
  assert.equal(JSON.stringify(rows), before);
});
test('shared axes use same physical step for same units despite different DC levels', () => {
  const a = axes(
    [
      row('F3', -5000, 2000, -2000),
      row('O1', -716000, -714000, -715000),
      row('count', 0, 10, 5, 'pcs'),
    ],
    { mode: 'shared' },
  );
  assert.equal(a[0].step, a[1].step);
  assert.notEqual(a[0].step, a[2].step);
});
test('flat input stays flat with explicit original value', () => {
  const a = axes([row('T2', 750000, 750000, 750000)])[0];
  assert.equal(a.flat, true);
  assert.equal(a.ptp, 0);
  assert.equal(a.center, 750000);
  assert.equal(a.clipped, false);
});
test('manual 100uV is retained and reports clipping instead of changing data', () => {
  const a = axes([row('F3', -5000, 2000, -2000)], { mode: 'fixed', fixedStep: 100 })[0];
  assert.equal(a.step, 100);
  assert.equal(a.clipped, true);
});
test('time zoom preserves cursor anchor and allows the entire recording', () => {
  const a = zoomTime({ start: 10, seconds: 10, total: 96, fraction: 0.25, factor: 2 });
  assert.equal(a.seconds, 5);
  assert.equal(a.start, 11.25);
  const b = zoomTime({ start: 10, seconds: 10, total: 96, fraction: 0.5, factor: 0.001 });
  assert.deepEqual(b, { start: 0, seconds: 96 });
});
test('time pan clamps to native boundaries', () => {
  assert.deepEqual(panTime({ start: 0, seconds: 1, total: 96, fraction: -10 }), {
    start: 0,
    seconds: 1,
  });
  assert.deepEqual(panTime({ start: 95, seconds: 1, total: 96, fraction: 10 }), {
    start: 95,
    seconds: 1,
  });
});
