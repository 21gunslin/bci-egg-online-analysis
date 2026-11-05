/* Manual review coverage is measured in exact channel-seconds, not viewed pages. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.EEGPatrolMath = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  function merge(intervals) {
    const sorted = intervals.map((r) => [r[0], r[1]]).sort((a, b) => a[0] - b[0]),
      out = [];
    for (const [a, b] of sorted) {
      if (b <= a) continue;
      if (out.length && a <= out.at(-1)[1]) out.at(-1)[1] = Math.max(out.at(-1)[1], b);
      else out.push([a, b]);
    }
    return out;
  }
  function intervals(entries, name) {
    return merge(entries.filter((r) => r.channels.includes(name)).map((r) => [r.start, r.end]));
  }
  function coverage(entries, names, duration) {
    let covered = 0;
    for (const n of names)
      for (const [a, b] of intervals(entries, n))
        covered += Math.max(0, Math.min(duration, b) - Math.max(0, a));
    const total = names.length * duration;
    return { covered, total, percent: total ? (100 * covered) / total : 0 };
  }
  function cellFraction(entries, name, start, end) {
    let seen = 0;
    for (const [a, b] of intervals(entries, name))
      seen += Math.max(0, Math.min(end, b) - Math.max(start, a));
    return Math.min(1, seen / (end - start));
  }
  function pages(names, size, index) {
    const count = Math.max(1, Math.ceil(names.length / size)),
      page = Math.max(0, Math.min(count - 1, index));
    return { page, count, names: names.slice(page * size, (page + 1) * size) };
  }
  function buildDraft(scan, badsText = '', segmentsText = '') {
    if (
      !scan ||
      !Array.isArray(scan.candidates) ||
      !Number.isFinite(scan.duration) ||
      scan.duration <= 0
    )
      throw new Error('请先扫描当前记录。');
    const names = new Set(
        badsText
          .split(/[,，\n]/)
          .map((n) => n.trim())
          .filter(Boolean),
      ),
      addedNames = [],
      groups = new Map(),
      allowed = new Set(scan.channels.map((c) => c.name));
    for (const c of scan.candidates) {
      if (!allowed.has(c.channel) || /[,，\r\n]/.test(c.channel))
        throw new Error('候选通道名不能快速填入，请核对通道名。');
      if (c.persistent) {
        if (!names.has(c.channel)) {
          names.add(c.channel);
          addedNames.push(c.channel);
        }
        continue;
      }
      if (
        !Number.isFinite(c.start) ||
        !Number.isFinite(c.end) ||
        c.start < 0 ||
        c.end > scan.duration ||
        c.start >= c.end
      )
        throw new Error('候选时间无效，请重新扫描。');
      const key = c.start + '|' + c.end;
      if (!groups.has(key)) groups.set(key, { start: c.start, end: c.end, labels: new Set() });
      groups.get(key).labels.add(
        c.channel +
          ':' +
          String(c.reason)
            .replace(/[,，\r\n]/g, '；')
            .slice(0, 120),
      );
    }
    const existing = segmentsText
      .split('\n')
      .map((line) => line.split(/[,，]/).map((s) => s.trim()))
      .filter(
        (p) =>
          p.length === 3 &&
          p[0] !== '' &&
          p[1] !== '' &&
          Number.isFinite(Number(p[0])) &&
          Number.isFinite(Number(p[1])) &&
          Number(p[1]) > Number(p[0]),
      )
      .map((p) => [Number(p[0]), Number(p[1])]);
    const rows = [];
    let covered = merge(existing);
    for (const g of [...groups.values()].sort((a, b) => a.start - b.start || a.end - b.end)) {
      if (covered.some(([a, b]) => a <= g.start && b >= g.end)) continue;
      // Use the exact candidate bounds, not its surrounding display context or a persistent envelope.
      rows.push(g.start + ',' + g.end + ',待核候选_' + [...g.labels].join('；'));
      covered = merge([...covered, [g.start, g.end]]);
    }
    const append = (original, values, separator) =>
      !values.length
        ? original
        : original + (original.trim() ? separator : '') + values.join(separator);
    const nextSegments = append(segmentsText, rows, '\n');
    if (nextSegments.split('\n').filter((s) => s.trim()).length > 1000)
      throw new Error('坏段草稿超过1000条，请调整扫描参数后重试；原草稿未更改。');
    return {
      badsText: append(badsText, addedNames, ', '),
      segmentsText: nextSegments,
      addedBads: addedNames.length,
      addedSegments: rows.length,
      totalBads: names.size,
      allEegBad:
        scan.channels.some((c) => c.eeg) &&
        scan.channels.filter((c) => c.eeg).every((c) => names.has(c.name)),
    };
  }
  return { merge, coverage, cellFraction, pages, buildDraft };
});
