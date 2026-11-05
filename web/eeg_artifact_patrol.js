'use strict';
(() => {
  let record = null,
    scan = null,
    entries = [],
    page = 0,
    current = null,
    candidateIndex = -1,
    playing = false,
    timer = null,
    mapGeometry = null,
    highlighting = false,
    activeCandidate = null;
  const M = EEGPatrolMath;
  function stop() {
    playing = false;
    clearTimeout(timer);
    $('patrolPlay').textContent = '开始自动翻页';
  }
  function size() {
    return Number($('patrolPageSize').value);
  }
  function visibleRows(rows) {
    if (rows.length <= 1) return rows;
    const p = M.pages(rows, size(), page);
    page = p.page;
    return p.names;
  }
  function scopedNames() {
    return scan
      ? scan.channels.map((r) => r.name)
      : (record?.channels || [])
          .filter((r) => $('patrolScope').value === 'all' || r.eeg)
          .map((r) => r.name);
  }
  function coverage() {
    if (!record) return;
    const names = scopedNames(),
      c = M.coverage(entries, names, record.duration),
      uncertain = entries.filter((r) => r.status === 'uncertain').length;
    $('patrolCoverage').textContent =
      `人工浏览覆盖 ${c.percent.toFixed(1)}%（${fmt(c.covered)} / ${fmt(c.total)} 通道·秒；${names.length}通道范围） · 需再看登记${uncertain}条。只统计明确点击登记的范围，不代表全部无伪迹。`;
  }
  function afterDraw(wave, rows) {
    current = { start: wave.start, end: wave.stop, channels: rows.map((r) => r.name) };
    const single = $('rawWaveFocus').value,
      p = M.pages(wave.rows, size(), page);
    $('patrolPageInfo').textContent = single
      ? `单通道 ${single} · ${fmt(wave.start)}–${fmt(wave.stop)}秒`
      : `通道第${p.page + 1}/${p.count}页 · 当前${rows.map((r) => r.name).join('、')} · ${fmt(wave.start)}–${fmt(wave.stop)}秒`;
    if ($('rawYMode').value === 'shared')
      $('rawScaleNotice').textContent += ' 当前统一标尺按本页计算；跨页比较请使用相同手动标尺。';
    coverage();
  }
  function setRecord(value) {
    stop();
    highlighting = false;
    activeCandidate = null;
    record = value;
    scan = null;
    entries = [];
    page = 0;
    current = null;
    candidateIndex = -1;
    $('patrolFillDraft').dataset.ready = 'false';
    $('patrolFillDraft').disabled = true;
    $('rawQcAutoDraftNotice').classList.add('hide');
    $('patrolResults').classList.add('hide');
    $('patrolCandidates').replaceChildren();
    $('patrolDownloads').replaceChildren();
    $('patrolStatus').textContent = '';
    if (value) $('patrolBin').value = String(Math.max(2, Math.ceil(value.duration / 1000)));
    coverage();
  }
  async function restoreReview() {
    if (!record) return;
    const id = record.run_id,
      r = await post('/api/raw-qc/review', { run_id: id });
    if (record?.run_id === id) {
      entries = r.entries;
      coverage();
    }
  }
  function drawMap() {
    if (!scan) return;
    const canvas = $('patrolMap'),
      ratio = window.devicePixelRatio || 1,
      left = 125,
      top = 30,
      cw = Math.max(7, Math.floor(700 / scan.bins)),
      rh = 21,
      width = left + scan.bins * cw + 15,
      height = top + scan.channels.length * rh + 35;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(ratio, ratio);
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, width, height);
    ctx.font = '11px sans-serif';
    const ranges = new Map(
      scan.channels.map((c) => [
        c.name,
        M.merge(entries.filter((e) => e.channels.includes(c.name)).map((e) => [e.start, e.end])),
      ]),
    );
    scan.channels.forEach((channel, i) => {
      ctx.fillStyle = '#36506b';
      ctx.fillText(channel.name + (channel.eeg ? '' : ' [辅助]'), 4, top + i * rh + 14);
      for (let j = 0; j < scan.bins; j++) {
        const cell = scan.cells[i][j],
          x = left + j * cw,
          y = top + i * rh,
          abrupt = cell.reasons.some((r) => /跳变|离群|基线突变/.test(r));
        ctx.fillStyle = cell.reasons.length
          ? abrupt || cell.score >= 8
            ? '#c95642'
            : '#e4a164'
          : '#e8edf2';
        ctx.fillRect(x, y, cw - 1, rh - 2);
        let covered = 0;
        for (const [a, b] of ranges.get(channel.name)) {
          if (b <= cell.start) continue;
          if (a >= cell.end) break;
          covered += Math.max(0, Math.min(cell.end, b) - Math.max(cell.start, a));
        }
        const fraction = covered / (cell.end - cell.start);
        if (fraction > 0) {
          ctx.strokeStyle = fraction >= 0.999 ? '#236faf' : '#79c4dd';
          ctx.lineWidth = 1.4;
          ctx.strokeRect(x + 0.7, y + 0.7, cw - 2, rh - 3);
        }
      }
    });
    ctx.fillStyle = '#536c83';
    for (let j = 0; j <= 5; j++) {
      const t = (scan.duration * j) / 5;
      ctx.fillText(fmt(t) + 's', left + (scan.bins * cw * j) / 5 - 10, height - 9);
    }
    ctx.fillText('点格子定位原始片段', left, 17);
    mapGeometry = { left, top, cw, rh };
  }
  function candidates() {
    return (scan?.candidates || []).filter(
      (c) =>
        $('patrolFilter').value === 'all' ||
        ($('patrolFilter').value === 'persistent') === c.persistent,
    );
  }
  function renderCandidates() {
    const list = candidates(),
      holder = $('patrolCandidates');
    holder.replaceChildren();
    holder.append(
      el('p', `${list.length}项候选；持续性问题已合并。候选不等于已确认坏段。`, 'small muted'),
    );
    const begin = candidateIndex >= 0 ? Math.floor(candidateIndex / 20) * 20 : 0;
    list.slice(begin, begin + 20).forEach((c, i) => {
      const row = el('div', undefined, 'patrol-candidate' + (highlighting ? ' candidate-red' : ''));
      const text = el(
        'div',
        `${begin + i + 1}. ${c.channel} · ${c.reason}${c.persistent ? '（持续性通道提示）' : ''}`,
      );
      text.append(
        el(
          'p',
          `${fmt(c.start)}–${fmt(c.end)}秒；触发${(c.fraction * 100).toFixed(0)}%窗口${c.persistent ? '，包络内仍可能有间隙' : ''}`,
          'muted small',
        ),
      );
      const button = el('button', '跳到原始波形', 'secondary run');
      button.type = 'button';
      button.onclick = () =>
        run('patrolStatus', async () => {
          candidateIndex = begin + i;
          return jumpCandidate(c);
        });
      row.append(text, button);
      holder.append(row);
    });
    if (list.length > 20)
      holder.append(
        el(
          'p',
          `当前显示${begin + 1}–${Math.min(begin + 20, list.length)}；用上/下一个候选继续。`,
          'small muted',
        ),
      );
  }
  async function jumpCandidate(c) {
    stop();
    activeCandidate = c;
    page = 0;
    const seconds = Math.min(record.duration, 4),
      start = Math.max(0, Math.min(record.duration - seconds, c.focus_time - 1));
    await EEGRawViewer.jump({ start, seconds, channel: c.channel });
    renderCandidates();
    $('rawWaveCanvas').scrollIntoView({ block: 'center', behavior: 'smooth' });
    return `${c.channel}：${c.reason}。仅定位，尚未计入人工复核。`;
  }
  async function scanAll() {
    if (!record) throw new Error('请先载入或恢复一份BDF。');
    stop();
    const id = record.run_id;
    status('patrolStatus', '正在扫描全记录原始数据，生成候选定位；不滤波、不删数据…');
    const r = await post('/api/raw-qc/patrol', {
      run_id: id,
      seconds: Number($('patrolBin').value),
      sensitivity: Number($('patrolZ').value),
      scope: $('patrolScope').value,
    });
    if (record?.run_id !== id) return;
    scan = r;
    $('patrolFillDraft').dataset.ready = 'true';
    activeCandidate = null;
    entries = r.review.entries;
    candidateIndex = -1;
    $('patrolResults').classList.remove('hide');
    $('patrolRules').replaceChildren(
      ...r.rules.map((t) => el('p', t, 'small muted')),
      el('p', r.notice, 'notice small'),
    );
    $('patrolDownloads').replaceChildren();
    for (const f of r.files) {
      const a = el('a', f.name);
      a.href = f.url;
      a.download = f.name;
      $('patrolDownloads').append(a);
    }
    coverage();
    drawMap();
    renderCandidates();
    paintChannels();
    if (highlighting) EEGRawViewer.redraw();
    return `扫描完成：${r.channels.length}通道 × ${r.bins}窗，${r.candidates.length}项合并候选。优先检查候选；其余仍待人工复核。`;
  }
  $('patrolScan').onclick = () => run('patrolStatus', scanAll);
  $('patrolFillDraft').onclick = () =>
    run('patrolStatus', () => {
      stop();
      if (!record || !scan) throw new Error('请先点击黄色按钮，扫描当前记录。');
      if (
        scan.seconds !== Number($('patrolBin').value) ||
        scan.sensitivity !== Number($('patrolZ').value) ||
        scan.scope !== $('patrolScope').value
      )
        throw new Error('扫描参数已更改，请重新点击黄色按钮后再填入。');
      const draft = M.buildDraft(scan, $('rawQcBads').value, $('rawQcBadSegments').value);
      if (new TextEncoder().encode(JSON.stringify(draft)).length > 50000)
        throw new Error('候选草稿过多，请提高扫描阈值或缩小扫描范围后重试；现有内容未更改。');
      const message = `已追加${draft.addedBads}个坏道候选、${draft.addedSegments}条坏段候选，原手工内容保留；可直接删除或补充。${draft.allEegBad ? '注意：当前全部EEG均列为坏道，请先删改核对，否则无法分析。' : ''}坏段导出会作用于全部通道；尚未导出FIF。`;
      EEGRawViewer.applyDraft(draft, message);
      return message;
    });

  $('patrolMap').onclick = (event) => {
    if (!scan || !mapGeometry) return;
    const rect = $('patrolMap').getBoundingClientRect(),
      g = mapGeometry,
      i = Math.floor((event.clientY - rect.top - g.top) / g.rh),
      j = Math.floor((event.clientX - rect.left - g.left) / g.cw);
    if (i < 0 || i >= scan.channels.length || j < 0 || j >= scan.bins) return;
    const cell = scan.cells[i][j];
    run('patrolStatus', () =>
      jumpCandidate({
        channel: scan.channels[i].name,
        focus_time: cell.focus_time ?? cell.start,
        start: cell.start,
        end: cell.end,
        reason: cell.reasons.join('；') || '未触发程序规则的待核查片段',
      }),
    );
  };
  $('patrolFilter').onchange = () => {
    candidateIndex = -1;
    renderCandidates();
  };
  for (const [id, step] of [
    ['patrolNextCandidate', 1],
    ['patrolPrevCandidate', -1],
  ])
    $(id).onclick = () =>
      run('patrolStatus', async () => {
        const list = candidates();
        if (!list.length) return '当前无候选，不等于无伪迹。';
        candidateIndex = Math.max(0, Math.min(list.length - 1, candidateIndex + step));
        return jumpCandidate(list[candidateIndex]);
      });
  for (const [id, step] of [
    ['patrolPagePrev', -1],
    ['patrolPageNext', 1],
  ])
    $(id).onclick = () => {
      stop();
      $('rawWaveFocus').value = '';
      page += step;
      EEGRawViewer.redraw();
    };
  $('patrolPageSize').onchange = () => {
    stop();
    page = 0;
    EEGRawViewer.redraw();
  };
  async function advance() {
    const wave = EEGRawViewer.getWave();
    if (!wave || !record) return false;
    const p = M.pages(wave.rows, size(), page);
    if (!$('rawWaveFocus').value && p.page + 1 < p.count) {
      page++;
      EEGRawViewer.redraw();
      return true;
    }
    if (wave.stop >= record.duration - 1e-9) return false;
    page = 0;
    const seconds = Math.min(Number($('patrolWindow').value), record.duration - wave.stop);
    await EEGRawViewer.jump({ start: wave.stop, seconds });
    return true;
  }
  $('patrolAdvance').onclick = () =>
    run('patrolStatus', async () => {
      stop();
      return (await advance())
        ? '已切换下一巡查页；未自动登记复核。'
        : '已到末尾；请按人工复核覆盖进度检查遗漏。';
    });
  async function tick() {
    if (!playing) return;
    if (busy) {
      timer = setTimeout(tick, 400);
      return;
    }
    let more = false;
    await run('patrolStatus', async () => {
      more = await advance();
      return more ? '自动翻页中；不会自动计为已复核。' : '已到末尾，自动浏览不等于完成复核。';
    });
    if (playing && more) timer = setTimeout(tick, Number($('patrolDelay').value) * 1000);
    else stop();
  }
  $('patrolPlay').onclick = async () => {
    if (!record || busy) return;
    const seconds = Number($('patrolWindow').value),
      delay = Number($('patrolDelay').value);
    if (
      !Number.isFinite(seconds) ||
      seconds < 0.1 ||
      seconds > 60 ||
      !Number.isFinite(delay) ||
      delay < 2 ||
      delay > 30
    ) {
      status('patrolStatus', '巡查窗长0.1–60秒，翻页间隔2–30秒。', true);
      return;
    }
    stop();
    page = 0;
    $('rawWaveFocus').value = '';
    await run('patrolStatus', () =>
      EEGRawViewer.jump({
        start: EEGRawViewer.getWave()?.start || 0,
        seconds: Math.min(seconds, record.duration),
        channel: '',
      }),
    );
    playing = true;
    $('patrolPlay').textContent = '自动翻页中（不计复核）';
    timer = setTimeout(tick, delay * 1000);
  };
  $('patrolPause').onclick = () => {
    stop();
    status('patrolStatus', '自动翻页已暂停；未自动登记人工复核。');
  };
  async function mark(statusValue, next = false) {
    stop();
    if (!record || !current) throw new Error('请先读取波形。');
    const id = record.run_id,
      r = await post('/api/raw-qc/review', {
        run_id: id,
        action: 'append',
        ...current,
        status: statusValue,
      });
    if (record?.run_id !== id) return;
    entries = r.entries;
    coverage();
    drawMap();
    if (next) await advance();
    return '仅登记实际显示的通道与时间范围；BDF、FIF及坏道/BAD标记均未改变。';
  }
  $('patrolReviewed').onclick = () => run('patrolStatus', () => mark('reviewed'));
  $('patrolUncertain').onclick = () => run('patrolStatus', () => mark('uncertain'));
  $('patrolReviewNext').onclick = () => run('patrolStatus', () => mark('reviewed', true));
  $('patrolUndo').onclick = () =>
    run('patrolStatus', async () => {
      stop();
      if (!record) return;
      const r = await post('/api/raw-qc/review', { run_id: record.run_id, action: 'undo' });
      entries = r.entries;
      coverage();
      drawMap();
      return '已撤销上次人工复核登记，不影响原数据或BAD标记。';
    });
  $('patrolReviewExport').onclick = () =>
    run('patrolStatus', async () => {
      if (!record) return;
      const r = await post('/api/raw-qc/review-export', { run_id: record.run_id });
      for (const f of r.files) {
        const a = el('a', f.name);
        a.href = f.url;
        a.download = f.name;
        $('patrolDownloads').append(a);
      }
      $('patrolResults').classList.remove('hide');
      return '人工复核进度已导出，不是无伪迹证明，也未转换为BAD标记。';
    });
  function paintChannels() {
    const names = new Set(highlighting ? (scan?.candidates || []).map((c) => c.channel) : []);
    for (const row of document.querySelectorAll('.raw-channel')) {
      const name = row.querySelector('[data-raw-channel]')?.value;
      row.classList.toggle('candidate-red', names.has(name));
      row.title = names.has(name) ? '程序候选，待人工核对；尚未写成坏道' : '';
    }
  }
  function overlay(name, start, end) {
    if (!highlighting || !scan) return { spans: [], reasons: [], active: null };
    const index = scan.channels.findIndex((c) => c.name === name);
    if (index < 0) return { spans: [], reasons: [], active: null };
    const spans = [],
      reasons = new Set();
    for (const cell of scan.cells[index]) {
      if (!cell.reasons.length || cell.end <= start || cell.start >= end) continue;
      cell.reasons.forEach((r) => reasons.add(r));
      const a = Math.max(start, cell.start),
        b = Math.min(end, cell.end);
      if (spans.length && a <= spans.at(-1).end) spans.at(-1).end = Math.max(b, spans.at(-1).end);
      else spans.push({ start: a, end: b });
    }
    const active =
      activeCandidate?.channel === name &&
      Number.isFinite(activeCandidate.start) &&
      activeCandidate.start < end &&
      activeCandidate.end > start
        ? { start: Math.max(start, activeCandidate.start), end: Math.min(end, activeCandidate.end) }
        : null;
    return { spans, reasons: [...reasons], active };
  }
  $('patrolAutoRed').onclick = () =>
    run('patrolStatus', async () => {
      await scanAll();
      if (!record || !scan) return '记录已变更，请重新载入。';
      highlighting = true;
      $('patrolFilter').value = 'all';
      candidateIndex = 0;
      paintChannels();
      renderCandidates();
      if (!scan.candidates.length) {
        EEGRawViewer.redraw();
        return '未触发当前规则，没有自动写坏道；仍需人工核查。';
      }
      if (window.EEGSpectrum) EEGSpectrum.prepareFollow();
      const message = await jumpCandidate(scan.candidates[0]);
      return '已红标候选通道/时间窗并跳转；对应PSD已显示在波形下方。' + message;
    });
  $('patrolClearRed').onclick = () => {
    highlighting = false;
    activeCandidate = null;
    paintChannels();
    renderCandidates();
    EEGRawViewer.redraw();
    if (window.EEGSpectrum) EEGSpectrum.redraw();
    status('patrolStatus', '已清除程序候选红标，人工BAD标记和原数据不变。');
  };
  window.EEGPatrol = { setRecord, restoreReview, visibleRows, afterDraw, pause: stop, overlay };
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stop();
  });
  document.querySelectorAll('[data-tab]').forEach((button) =>
    button.addEventListener('click', () => {
      if (button.dataset.tab !== 'rawqc') stop();
    }),
  );
})();
