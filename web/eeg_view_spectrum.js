'use strict';
(() => {
  let record = null,
    last = null,
    forceFollow = false;
  function setRecord(r) {
    record = r;
    last = null;
    forceFollow = false;
    $('rawPsdCharts').replaceChildren();
    $('rawPsdTable').replaceChildren();
    $('rawPsdDownloads').replaceChildren();
    $('rawPsdInfo').textContent = '尚未读取本记录频域；可点击黄色按钮自动定位并显示对应PSD。';
    $('rawPsdChannels').replaceChildren();
    for (const [value, label] of [
      ['visible', '当前显示的通道页/单通道'],
      ['eeg', '全部EEG候选通道'],
      ['all', '全部信号（分单位）'],
      ...(r?.channels || []).map((c) => ['ch:' + c.name, c.name]),
    ]) {
      const o = el('option', label);
      o.value = value;
      $('rawPsdChannels').append(o);
    }
  }
  function prepareFollow() {
    $('rawPsdPanel').open = true;
    $('rawPsdFollow').checked = true;
    $('rawPsdScope').value = 'window';
    $('rawPsdChannels').value = 'visible';
    forceFollow = true;
  }
  function isFollowing() {
    return Boolean(
      record &&
      (last || forceFollow) &&
      $('rawPsdFollow').checked &&
      $('rawPsdScope').value === 'window',
    );
  }
  function options() {
    const wave = EEGRawViewer.getWave();
    if (!record || !wave) throw new Error('请先载入并读取原始波形。');
    const selection = $('rawPsdChannels').value;
    const channels =
      selection === 'visible'
        ? EEGRawViewer.getVisibleRows().map((r) => r.name)
        : selection === 'all'
          ? record.channels.map((r) => r.name)
          : selection === 'eeg'
            ? record.channels.filter((r) => r.eeg).map((r) => r.name)
            : [selection.slice(3)];
    return {
      run_id: record.run_id,
      channels,
      start: wave.start,
      seconds: wave.stop - wave.start,
      scope: $('rawPsdScope').value,
      welch_seconds: Number($('rawPsdWindow').value),
      fmin: Number($('rawPsdMin').value),
      fmax: Number($('rawPsdMax').value),
    };
  }
  async function loadCurrent(exportValues = false) {
    const opts = options();
    $('rawPsdInfo').textContent = '正在估计原始信号Welch PSD；不滤波、不删除坏道/坏段…';
    const result = await post('/api/raw-qc/psd', { ...opts, export: exportValues });
    if (record?.run_id !== opts.run_id) return;
    last = result;
    forceFollow = false;
    $('rawPsdPanel').open = true;
    redraw();
    $('rawPsdDownloads').replaceChildren();
    if (result.files) {
      for (const f of result.files) {
        const a = el('a', f.name);
        a.href = f.url;
        a.download = f.name;
        $('rawPsdDownloads').append(a);
      }
    }
    return '原始频域已更新，仅估计PSD，未改动BDF/FIF。';
  }
  function redraw() {
    if (!last) return;
    const root = $('rawPsdCharts');
    root.replaceChildren();
    const groups = new Map();
    for (const c of last.curves) {
      if (!c.frequency.length) continue;
      if (!groups.has(c.unit)) groups.set(c.unit, []);
      groups.get(c.unit).push(c);
    }
    const db = $('rawPsdY').value === 'db',
      colors = ['#286a93', '#338874', '#886da7', '#bd8241', '#477d91', '#805f62'];
    for (const [unit, curves] of groups) {
      const holder = el('div', undefined, 'psd-chart'),
        canvas = el('canvas');
      canvas.setAttribute('role', 'img');
      canvas.setAttribute('aria-label', `原始Welch PSD，${unit}平方每Hz`);
      holder.append(canvas);
      root.append(holder);
      const width = Math.max(460, holder.clientWidth - 2),
        height = 360,
        ratio = window.devicePixelRatio || 1;
      canvas.width = width * ratio;
      canvas.height = height * ratio;
      canvas.style.width = width + 'px';
      canvas.style.height = height + 'px';
      const ctx = canvas.getContext('2d');
      ctx.scale(ratio, ratio);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, width, height);
      const left = 76,
        right = 25,
        top = 42,
        bottom = 58,
        xmin = last.fmin,
        xmax = Math.min(last.fmax, Math.max(...curves.map((c) => c.nyquist))),
        pw = width - left - right,
        ph = height - top - bottom;
      const values = [];
      for (const c of curves.filter((c) => !c.flat)) {
        for (const p of c.power)
          if (Number.isFinite(p) && p > 0) values.push(db ? 10 * Math.log10(p) : p);
      }
      if (!values.length)
        for (const c of curves)
          for (const p of c.power) if (p > 0) values.push(db ? 10 * Math.log10(p) : p);
      const upper = values.reduce((m, v) => Math.max(m, v), db ? -300 : 0),
        lower = values.reduce((m, v) => Math.min(m, v), upper);
      let ymin = db ? Math.max(lower - 5, upper - 120) : 0,
        ymax = db ? upper + 5 : Math.max(upper * 1.1, 1e-12);
      if (ymax - ymin < 1e-12) ymax = ymin + 1;
      const x = (f) => left + ((f - xmin) / Math.max(xmax - xmin, 1e-12)) * pw,
        y = (p) => top + ph - ((p - ymin) / (ymax - ymin)) * ph;
      ctx.fillStyle = '#284b65';
      ctx.font = '12px sans-serif';
      ctx.fillText(
        `${last.start.toFixed(3)}–${last.stop.toFixed(3)}秒 · Welch PSD · ${db ? 'dB re 1 ' : ''}${unit}²/Hz`,
        8,
        20,
      );
      ctx.font = '10px sans-serif';
      for (let i = 0; i <= 5; i++) {
        const f = xmin + ((xmax - xmin) * i) / 5,
          xx = x(f);
        ctx.strokeStyle = '#e0e7ed';
        ctx.beginPath();
        ctx.moveTo(xx, top);
        ctx.lineTo(xx, top + ph);
        ctx.stroke();
        ctx.fillStyle = '#496579';
        ctx.fillText(fmt(f), xx - 9, height - 35);
        const value = ymin + ((ymax - ymin) * i) / 5,
          yy = y(value);
        ctx.beginPath();
        ctx.moveTo(left, yy);
        ctx.lineTo(left + pw, yy);
        ctx.stroke();
        ctx.fillText(fmt(value), 3, yy + 3);
      }
      for (const hz of [50, 60])
        if (hz >= xmin && hz <= xmax) {
          ctx.setLineDash([4, 3]);
          ctx.strokeStyle = '#b99451';
          ctx.beginPath();
          ctx.moveTo(x(hz), top);
          ctx.lineTo(x(hz), top + ph);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = '#8a6427';
          ctx.fillText(hz + ' Hz', x(hz) + 3, top + 12);
        }
      curves.forEach((c, index) => {
        const hint = window.EEGPatrol?.overlay(c.name, last.start, last.stop),
          suspect = Boolean(hint?.spans.length),
          confirmed = (record.options?.bads || []).includes(c.name),
          color = suspect || confirmed ? '#c33432' : colors[index % colors.length];
        ctx.save();
        ctx.beginPath();
        ctx.rect(left, top, pw, ph);
        ctx.clip();
        ctx.strokeStyle = color;
        ctx.lineWidth = suspect || confirmed ? 1.5 : 1;
        ctx.beginPath();
        c.frequency.forEach((f, i) => {
          const value = db ? (c.power[i] > 0 ? 10 * Math.log10(c.power[i]) : ymin) : c.power[i],
            yy = y(Math.max(ymin, Math.min(ymax, value)));
          if (i) ctx.lineTo(x(f), yy);
          else ctx.moveTo(x(f), yy);
        });
        ctx.stroke();
        ctx.restore();
        const legend = el(
          'span',
          `${c.name}${confirmed ? ' [已保存BAD]' : suspect ? ' [候选待核]' : ''}${c.flat ? ' [原始平线]' : ''} · Δf=${fmt(c.df)}Hz`,
          'small',
        );
        legend.style.color = color;
        legend.style.display = 'inline-block';
        legend.style.margin = '6px';
        holder.append(legend);
      });
      ctx.fillStyle = '#496579';
      ctx.fillText('频率 (Hz)', left + pw / 2 - 20, height - 12);
      if (curves.every((c) => c.flat)) {
        ctx.fillStyle = '#a63f33';
        ctx.fillText(
          '原始零方差/平线：非DC数值可能仅为浮点误差，不作节律解释',
          left + 5,
          top + ph / 2,
        );
      }
    }
    if (!groups.size) root.append(el('p', '当前频率范围或时间窗内没有可估计的谱点。', 'notice'));
    $('rawPsdInfo').textContent =
      `${last.scope === 'full' ? '整份记录' : '当前窗口'} ${fmt(last.start)}–${fmt(last.stop)}秒；${last.curves.length}通道。${last.method} dB图仅显示最多120dB动态范围，极低/零功率落在图底；CSV保留原始PSD数值。${last.warnings.join(' ')}`;
    $('rawPsdTable').replaceChildren();
    table($('rawPsdTable'), last.summary, '当前PSD参数与工频突出度');
  }
  async function show() {
    if (window.EEGPatrol) EEGPatrol.pause();
    await loadCurrent();
    $('rawPsdPanel').scrollIntoView({ block: 'start', behavior: 'smooth' });
    return '频域已显示，可调整通道、时间范围与频率范围后更新。';
  }
  $('rawShowPsd').onclick = () => run('rawQcStatus', show);
  $('rawPsdLoad').onclick = () => run('rawQcStatus', show);
  $('rawPsdExport').onclick = () => run('rawQcStatus', () => loadCurrent(true));
  $('rawPsdY').onchange = redraw;
  window.addEventListener('resize', redraw);
  window.EEGSpectrum = { setRecord, prepareFollow, isFollowing, loadCurrent, redraw };
})();
