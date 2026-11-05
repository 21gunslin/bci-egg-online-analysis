'use strict';
// Separate unfiltered workflow. Reuses upload/result/image zoom helpers only.
(() => {
  const tab = el('button', '原始BDF → 质控 / FIF');
  tab.type = 'button';
  tab.setAttribute('role', 'tab');
  tab.dataset.tab = 'rawqc';
  tab.setAttribute('aria-selected', 'false');
  document.querySelector('.tabs').append(tab);
  tab.onclick = () => {
    document
      .querySelectorAll('[data-tab]')
      .forEach((t) => t.setAttribute('aria-selected', String(t === tab)));
    document
      .querySelectorAll('.tab-panel')
      .forEach((p) => p.classList.toggle('hide', p.id !== 'rawqc'));
  };
  let localToken = null,
    active = null,
    lastWave = null,
    draftVersion = 0,
    savedVersion = -1,
    zoomTimer = null;
  const fields = [
    'rawQcBads',
    'rawQcBadSegments',
    'rawQcEyes',
    'rawQcMapping',
    'rawQcWelch',
    'rawQcSegment',
  ];
  function dirty() {
    draftVersion++;
    if (active) status('rawQcStatus', '标记/参数草稿有更改，尚未写入FIF；应用后生成新版本。');
  }
  fields.forEach((id) => $(id).addEventListener('input', dirty));
  function badNames() {
    return $('rawQcBads')
      .value.split(/[,，\n]/)
      .map((x) => x.trim())
      .filter(Boolean);
  }
  function syncDraftControls() {
    const bads = new Set(badNames());
    for (const row of document.querySelectorAll('.raw-channel')) {
      const name = row.querySelector('[data-raw-channel]')?.value,
        button = row.querySelector('button');
      if (button) {
        button.textContent = bads.has(name) ? '草稿BAD：取消' : '标为坏道';
        button.setAttribute('aria-pressed', String(bads.has(name)));
      }
    }
    if (lastWave) draw(lastWave);
  }
  $('rawQcBads').addEventListener('input', syncDraftControls);
  $('rawQcBadSegments').addEventListener('input', () => {
    if (lastWave) draw(lastWave);
  });
  function applyDraft(draft, message) {
    if (!active) throw new Error('请先载入原始记录。');
    const changed =
      $('rawQcBads').value !== draft.badsText || $('rawQcBadSegments').value !== draft.segmentsText;
    $('rawQcBads').value = draft.badsText;
    $('rawQcBadSegments').value = draft.segmentsText;
    if (changed) dirty();
    syncDraftControls();
    const notice = $('rawQcAutoDraftNotice');
    notice.textContent = message;
    notice.classList.remove('hide');
    notice.classList.toggle('draft-all-bad', draft.allEegBad);
    $('rawQcBads').closest('details').open = true;
    notice.scrollIntoView({ block: 'center', behavior: 'smooth' });
    status('rawQcStatus', message);
  }
  function lines(id, columns) {
    return $(id)
      .value.split('\n')
      .map((s) => s.trim())
      .filter(Boolean)
      .map((s, i) => {
        const parts = s.split(/[,，]/).map((x) => x.trim());
        if (parts.length !== columns)
          throw new Error(`${id} 第${i + 1}行需要${columns}列，请使用逗号分隔。`);
        return parts;
      });
  }
  function options() {
    const intervals = (id) =>
      lines(id, 3).map(([a, b, label]) => {
        if (!a || !b || !Number.isFinite(Number(a)) || !Number.isFinite(Number(b)))
          throw new Error('区间起止必须填写有效秒数。');
        return { start: Number(a), end: Number(b), label };
      });
    return {
      bads: badNames(),
      bad_segments: intervals('rawQcBadSegments'),
      eyes: intervals('rawQcEyes'),
      overrides: lines('rawQcMapping', 3).map(([name, type, standard]) => ({
        name,
        type,
        standard,
      })),
      welch_seconds: Number($('rawQcWelch').value),
      segment_seconds: Number($('rawQcSegment').value),
    };
  }
  function channelControls(result, previous) {
    const holder = $('rawWaveChannels');
    holder.replaceChildren();
    const bads = new Set(badNames());
    result.channels.forEach((r, i) => {
      const line = el('div', undefined, 'raw-channel');
      const label = el('label');
      const check = el('input');
      check.type = 'checkbox';
      check.value = r.name;
      check.checked = previous
        ? previous.has(r.name)
        : result.channels.some((c) => c.eeg)
          ? r.eeg && i < 64
          : i < 16;
      check.dataset.rawChannel = 'true';
      label.append(
        check,
        document.createTextNode(`${r.name} · ${r.region || ''} · ${fmt(r.fs)} Hz`),
      );
      const mark = el('button', bads.has(r.name) ? '已标BAD：取消' : '标为坏道', 'secondary');
      mark.type = 'button';
      mark.setAttribute('aria-pressed', String(bads.has(r.name)));
      mark.onclick = () => {
        const names = new Set(badNames());
        if (names.has(r.name)) names.delete(r.name);
        else names.add(r.name);
        $('rawQcBads').value = [...names].join(', ');
        mark.textContent = names.has(r.name) ? '已标BAD：取消' : '标为坏道';
        mark.setAttribute('aria-pressed', String(names.has(r.name)));
        dirty();
        if (lastWave) draw(lastWave);
      };
      line.append(label, mark);
      holder.append(line);
    });
  }
  async function inspect({ follow = true } = {}) {
    if (!active) throw new Error('请先读取BDF。');
    const channels = [...document.querySelectorAll('[data-raw-channel]:checked')].map(
      (x) => x.value,
    );
    lastWave = await post('/api/raw-qc/waveform', {
      run_id: active.run_id,
      channels,
      start: Number($('rawWaveStart').value),
      seconds: Number($('rawWaveSeconds').value),
    });
    const previous = $('rawWaveFocus').value;
    $('rawWaveFocus').replaceChildren();
    const all = el('option', '所选多通道堆叠');
    all.value = '';
    $('rawWaveFocus').append(all);
    for (const row of lastWave.rows) {
      const option = el('option', row.name + ' 单通道放大');
      option.value = row.name;
      $('rawWaveFocus').append(option);
    }
    if (lastWave.rows.some((r) => r.name === previous)) $('rawWaveFocus').value = previous;
    draw(lastWave);
    if (follow && window.EEGSpectrum?.isFollowing()) await EEGSpectrum.loadCurrent();
    return '原始时间窗已读取，仅调整查看方式，未重新导出FIF。';
  }
  function viewRows() {
    const focus = $('rawWaveFocus').value,
      rows = lastWave ? lastWave.rows.filter((r) => !focus || r.name === focus) : [];
    return window.EEGPatrol ? EEGPatrol.visibleRows(rows) : rows;
  }
  function viewOptions() {
    return {
      mode: $('rawYMode').value,
      gain: Number($('rawWaveGain').value),
      fixedStep: Number($('rawWaveStep').value),
    };
  }
  function draw(wave) {
    const selected = $('rawWaveFocus').value,
      baseRows = wave.rows.filter((r) => !selected || r.name === selected),
      rows = window.EEGPatrol ? EEGPatrol.visibleRows(baseRows) : baseRows;
    if (!rows.length) return;
    const settings = EEGViewAxes.axes(rows, viewOptions()),
      single = rows.length === 1;
    const canvas = $('rawWaveCanvas'),
      ratio = window.devicePixelRatio || 1,
      width = Math.max(440, canvas.parentElement.clientWidth - 2),
      rowHeight = single ? 420 : 128,
      top = 44,
      height = rows.length * rowHeight + 90;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(ratio, ratio);
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, width, height);
    const left = 200,
      right = 32,
      span = wave.stop - wave.start,
      mode = $('rawYMode').value;
    const titles = {
      channel: '逐通道自适应：各行刻度不同',
      shared: '同单位统一自适应：可比较振幅',
      fixed: '手动电压标尺',
    };
    ctx.fillStyle = '#36536e';
    ctx.font = '12px sans-serif';
    ctx.fillText(titles[mode] + ' | 只调坐标，原始值不变', 8, 21);
    const bads = new Set(badNames());
    let badSpans = (active?.annotations || [])
      .filter((a) => a['来源'] === 'BDF' && a['标签'].toLowerCase().startsWith('bad'))
      .map((a) => ({ start: a['起点_s'], end: a['起点_s'] + a['持续_s'] }));
    try {
      badSpans = badSpans.concat(
        lines('rawQcBadSegments', 3)
          .map(([a, b]) => ({ start: Number(a), end: Number(b) }))
          .filter((r) => Number.isFinite(r.start) && Number.isFinite(r.end) && r.start < r.end),
      );
    } catch {}
    const parameterRows = [];
    rows.forEach((row, i) => {
      const axis = settings[i],
        center = top + i * rowHeight + rowHeight / 2,
        halfPixels = rowHeight / 2 - 14,
        scale = halfPixels / axis.halfRange;
      const hint = window.EEGPatrol
        ? EEGPatrol.overlay(row.name, wave.start, wave.stop)
        : { spans: [], reasons: [], active: null };
      ctx.fillStyle = bads.has(row.name) || hint.spans.length ? '#bf302f' : '#1b3955';
      ctx.font = '13px sans-serif';
      ctx.fillText(
        row.name +
          (bads.has(row.name)
            ? (active?.options?.bads || []).includes(row.name)
              ? ' [已保存BAD]'
              : ' [BAD草稿]'
            : hint.spans.length
              ? ' [候选待核]'
              : ''),
        8,
        center - 28,
      );
      ctx.font = '11px sans-serif';
      ctx.fillText(`${fmt(axis.step)} ${row.unit}/格`, 8, center - 10);
      ctx.fillText(`中心 ${fmt(axis.center)}`, 8, center + 7);
      ctx.font = '10px sans-serif';
      ctx.fillText(`原始最小 ${fmt(axis.min)}`, 8, center + 24);
      ctx.fillText(`原始最大 ${fmt(axis.max)}`, 8, center + 39);
      ctx.save();
      ctx.beginPath();
      ctx.rect(left, top + i * rowHeight, width - left - right, rowHeight);
      ctx.clip();
      ctx.fillStyle = i % 2 ? '#fff' : '#f6f9fc';
      ctx.fillRect(left, top + i * rowHeight, width - left - right, rowHeight);
      for (const interval of badSpans) {
        const a = Math.max(wave.start, interval.start),
          b = Math.min(wave.stop, interval.end);
        if (b > a) {
          ctx.fillStyle = '#fce4df';
          ctx.fillRect(
            left + ((a - wave.start) / span) * (width - left - right),
            top + i * rowHeight,
            ((b - a) / span) * (width - left - right),
            rowHeight,
          );
        }
      }
      for (const interval of hint.spans) {
        ctx.fillStyle = '#fde8e580';
        ctx.fillRect(
          left + ((interval.start - wave.start) / span) * (width - left - right),
          top + i * rowHeight,
          ((interval.end - interval.start) / span) * (width - left - right),
          rowHeight,
        );
      }
      if (hint.active) {
        ctx.fillStyle = '#ef9b8a66';
        ctx.fillRect(
          left + ((hint.active.start - wave.start) / span) * (width - left - right),
          top + i * rowHeight,
          ((hint.active.end - hint.active.start) / span) * (width - left - right),
          rowHeight,
        );
      }
      for (let j = 0; j <= 10; j++) {
        const x = left + ((width - left - right) * j) / 10;
        ctx.strokeStyle = '#e0e7ef';
        ctx.beginPath();
        ctx.moveTo(x, top + i * rowHeight);
        ctx.lineTo(x, top + (i + 1) * rowHeight);
        ctx.stroke();
      }
      for (const tick of [-1, 0, 1]) {
        const y = center - tick * axis.step * scale;
        ctx.strokeStyle = tick ? '#e3e9ef' : '#c2d1df';
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(width - right, y);
        ctx.stroke();
      }
      ctx.lineWidth = 0.8;
      let previous = null,
        color = null;
      ctx.beginPath();
      row.points.forEach(([t, value]) => {
        const x = left + ((t - wave.start) / span) * (width - left - right),
          y = center - Math.max(-halfPixels, Math.min(halfPixels, (value - axis.center) * scale)),
          next =
            bads.has(row.name) || hint.spans.some((s) => t >= s.start && t < s.end)
              ? '#c33432'
              : '#245e83';
        if (next !== color) {
          if (color) {
            ctx.strokeStyle = color;
            ctx.stroke();
          }
          ctx.beginPath();
          if (previous) ctx.moveTo(previous.x, previous.y);
          else ctx.moveTo(x, y);
          color = next;
        }
        ctx.lineTo(x, y);
        previous = { x, y };
      });
      if (color) {
        ctx.strokeStyle = color;
        ctx.stroke();
      }
      ctx.restore();
      if (hint.reasons.length) {
        ctx.fillStyle = '#a6302b';
        ctx.font = '9px sans-serif';
        ctx.fillText('候选：' + hint.reasons.join('；').slice(0, 19), 8, center + 54);
      }
      if (single) {
        ctx.font = '11px sans-serif';
        ctx.fillStyle = '#486175';
        for (const tick of [-1, 0, 1])
          ctx.fillText(
            `${fmt(axis.center + tick * axis.step)} ${row.unit}`,
            left + 7,
            center - tick * axis.step * scale - 5,
          );
      }
      if (axis.flat || axis.clipped) {
        ctx.fillStyle = '#af4736';
        ctx.font = '10px sans-serif';
        ctx.fillText(
          axis.flat ? '原始平线：保持原值' : '超出显示范围（仅绘图）；可点击适配',
          left + 6,
          top + i * rowHeight + 12,
        );
      }
      parameterRows.push({
        通道: row.name,
        单位: row.unit,
        采样点数: axis.stats.n,
        原始最小: axis.min,
        原始最大: axis.max,
        原始均值: axis.stats.mean,
        原始标准差: axis.stats.std,
        峰峰值: axis.ptp,
        显示中心: axis.center,
        每格幅度: axis.step,
        纵轴最小: axis.yMin,
        纵轴最大: axis.yMax,
        原始平线: axis.flat,
        超出显示范围: axis.clipped,
      });
    });
    ctx.fillStyle = '#29465c';
    ctx.font = '12px sans-serif';
    for (let i = 0; i <= 5; i++)
      ctx.fillText(
        fmt(wave.start + (span * i) / 5) + ' s',
        left + ((width - left - right) * i) / 5 - 10,
        height - 22,
      );
    $('rawScaleNotice').textContent =
      mode === 'channel'
        ? '逐通道使用不同刻度以看清形态，图上高度不能直接比较；可看数值标尺或切换统一尺度。所有极值用于自动适配，原始平线仍是平线。'
        : mode === 'shared'
          ? '相同单位采用共同标尺，可比较振幅；极大通道可能使其他波形显小。可切换逐通道或单通道放大。'
          : '电压使用手动标尺，辅助信号使用自己的单位；超界仅裁切绘图，原始值保留。';
    $('rawWaveInfo').textContent =
      `原始时间 ${fmt(wave.start)}–${fmt(wave.stop)} 秒 / 总长${fmt(active.duration)}秒；${rows.length}通道；${titles[mode]}。滚轮：纵轴；Shift＋滚轮：时间；左右拖动：平移。长窗用极值包络，缩短窗口可看细节。`;
    $('rawWaveStats').replaceChildren();
    table($('rawWaveStats'), parameterRows, '当前窗口原始数值及显示范围');
    if (window.EEGPatrol) EEGPatrol.afterDraw(wave, rows);
  }
  async function analyze(revise) {
    const config = options();
    let payload;
    status(
      'rawQcStatus',
      revise
        ? '正在应用人工标记并生成新版本，原导出保留…'
        : '正在上传原始BDF并解码、绘制全频谱、核验FIF回读…',
    );
    if (revise) {
      if (!active) throw new Error('请先读取BDF。');
      payload = { source_run: active.run_id };
    } else {
      payload = await sourcePayload();
    }
    if ($('rawQcRecord').files[0])
      payload.recordinfo_upload = await upload($('rawQcRecord').files[0], 'recordinfo');
    const requestVersion = draftVersion;
    const result = await post('/api/raw-qc', {
      ...payload,
      ...config,
      source_path: $('rawQcSourcePath').value.trim(),
    });
    await mountResult(result, requestVersion, false);
    return result.local_fif
      ? '已保存：' +
          result.local_fif.path +
          (result.local_fif.backup ? '；旧版备份：' + result.local_fif.backup : '')
      : '分析完成。';
  }
  async function mountResult(result, requestVersion, restored) {
    const previous = active
      ? new Set([...document.querySelectorAll('[data-raw-channel]:checked')].map((n) => n.value))
      : null;
    active = result;
    savedVersion = requestVersion;
    if (result.original_source) $('rawQcSourcePath').value = result.original_source;
    if (window.EEGPatrol) EEGPatrol.setRecord(result);
    if (window.EEGSpectrum) EEGSpectrum.setRecord(result);
    $('rawViewDownloads').replaceChildren();
    try {
      localStorage.setItem('eeg_raw_last_run', result.run_id);
    } catch {}
    if (restored) {
      const o = result.options || {};
      $('rawQcBads').value = (o.bads || []).join(', ');
      $('rawQcBadSegments').value = (o.bad_segments || [])
        .map((r) => `${r.start},${r.end},${r.label}`)
        .join('\n');
      $('rawQcEyes').value = (o.eyes || []).map((r) => `${r.start},${r.end},${r.label}`).join('\n');
      $('rawQcMapping').value = (o.overrides || [])
        .map((r) => `${r.name},${r.type},${r.standard || ''}`)
        .join('\n');
      $('rawQcWelch').value = o.welch_seconds || 4;
      $('rawQcSegment').value = o.segment_seconds || 30;
    }
    const metrics = result.view_only_import
      ? [
          ['模式', '只读查看'],
          ['原始通道', result.channels.length],
          ['时长（秒）', result.duration],
          ['FIF', '未生成'],
        ]
      : [
          ['原始通道', result.channels.length],
          ['时长（秒）', result.duration],
          ['FIF采样率组', result.checks.length],
          [
            '眼状态α峰（检出/评估）',
            `${result.summary[0]['alpha检出区间通道数']}/${result.summary[0]['alpha评估区间通道数']}`,
          ],
        ];
    const card = render('rawQcResult', result, metrics);
    table(card, result.summary, result.view_only_import ? '本次载入信息' : '数据质控汇总');
    if (!result.view_only_import) {
      table(card, result.checks, '双精度FIF回读核验');
      table(card, result.quality, '逐通道PSD质控（候选提示需人工确认）');
    }
    if (restored && !result.view_only_import)
      card.prepend(
        el(
          'p',
          '已恢复历史结果；原有图片沿用当时设置。上方自适应检视和“只导出当前视图”无需重新分析或写入FIF。',
          'notice',
        ),
      );
    if (result.occipital_mean) {
      card.append(el('p', result.occipital_mean.method, 'notice'));
      table(card, result.occipital_mean.rows, '枕区平均功率谱与平均谱α峰');
      for (const warning of result.occipital_mean.warnings || [])
        card.append(el('p', warning, 'muted'));
    }
    const alphaLabels = {
      alpha_peak_hz: 'α峰频率_Hz',
      peak_db_uv2_hz: '峰高_dB相对1µV²/Hz',
      prominence_db: '突出度_dB',
      alpha_power_uv2: 'α功率_µV²',
      detected: '检出标记_1是0否',
    };
    if (!result.view_only_import) {
      if (result.alpha.length)
        table(
          card,
          result.alpha.map((row) =>
            Object.fromEntries(Object.entries(row).map(([k, v]) => [alphaLabels[k] || k, v])),
          ),
          '睁闭眼区间α峰数值',
        );
      else
        card.append(
          el('h3', '睁闭眼区间α峰数值'),
          el(
            'p',
            '当前没有可评估的眼状态区间×枕区通道组合；未评估不等于未检出α峰。逐段枕区指标另见完整CSV。',
          ),
        );
      const details = el('details');
      details.append(
        el(
          'summary',
          `逐段质控（网页前${result.segments.length}行 / 共${result.segment_total}行，完整表请下载）`,
        ),
      );
      table(details, result.segments, '逐段参数');
      card.append(details);
    } else
      card.append(
        el(
          'p',
          '本次只读查看，未运行正式质控或α分析。可用“只导出当前视图”保存图片与坐标参数。',
          'notice',
        ),
      );
    if (result.annotations.length)
      table(card, result.annotations, '源文件事件与已有坏段标记（相对首样本秒数）');
    channelControls(result, previous);
    $('rawQcInspector').classList.remove('hide');
    $('rawQcRevise').dataset.ready = 'true';
    $('rawWaveStart').max = String(Math.max(0, result.duration - 0.001));
    $('rawWaveSeconds').max = String(result.duration);
    if (restored || !lastWave) {
      $('rawWaveStart').value = '0';
      $('rawWaveSeconds').value = String(
        result.view_defaults?.seconds || Math.min(10, result.duration),
      );
      $('rawWaveGain').value = '1';
      $('rawYMode').value = 'channel';
      $('rawWaveStep').disabled = true;
      $('rawTimePreset').value = 'auto';
    } else
      $('rawWaveSeconds').value = String(
        Math.min(Number($('rawWaveSeconds').value), result.duration),
      );
    if (Number($('rawWaveStart').value) >= result.duration) $('rawWaveStart').value = '0';
    try {
      if (window.EEGPatrol) await EEGPatrol.restoreReview();
      await inspect();
    } catch (e) {
      $('rawWaveInfo').textContent = e.message;
    }
    if (draftVersion !== savedVersion)
      card.prepend(
        el('p', '分析过程中草稿有更改，当前下载对应提交时的标记。请再次应用。', 'notice'),
      );
  }
  $('rawQcForm').onsubmit = (e) => {
    e.preventDefault();
    run('rawQcStatus', () => analyze(false));
  };
  $('rawQcOpen').onclick = () =>
    run('rawQcStatus', async () => {
      status('rawQcStatus', '正在载入原始BDF，仅准备查看坐标…');
      const payload = await sourcePayload();
      if ($('rawQcRecord').files[0])
        payload.recordinfo_upload = await upload($('rawQcRecord').files[0], 'recordinfo');
      const result = await post('/api/raw-qc/open', payload);
      await mountResult(result, draftVersion, true);
      return 'BDF已只读载入；未运行特征分析或导出FIF，可直接调整横纵轴。';
    });
  $('rawQcRevise').setAttribute('data-requires-result', 'true');
  $('rawQcFile').onchange = () => {
    localToken = null;
    $('rawQcSourcePath').value = '';
    clearTimeout(zoomTimer);
    if (window.EEGPatrol) EEGPatrol.setRecord(null);
    if (active) {
      active = null;
      lastWave = null;
      $('rawQcInspector').classList.add('hide');
      $('rawQcRevise').disabled = true;
      delete $('rawQcRevise').dataset.ready;
      status('rawQcStatus', '已选择新BDF，请复核人工标记草稿后重新读取；下方旧结果尚未更新。');
    }
  };
  async function sourcePayload() {
    if (localToken) return { local_token: localToken };
    const file = $('rawQcFile').files[0];
    if (file) {
      checkFormat(file, 'bdf');
      return { upload_id: await upload(file, 'eeg') };
    }
    const path = $('rawQcSourcePath').value.trim();
    if (path) {
      const r = await post('/api/raw-qc/local-source', { path });
      localToken = r.local_token;
      return { local_token: localToken };
    }
    throw new Error('请从本机选择BDF，或上传BDF并填写完整路径。');
  }
  $('rawQcSourcePath').addEventListener('input', () => {
    localToken = null;
  });
  $('rawQcLocalPick').onclick = () =>
    run('rawQcStatus', async () => {
      const r = await post('/api/raw-qc/local-source', {});
      if (r.cancelled) return '已取消选择。';
      $('rawQcFile').value = '';
      $('rawQcFile').onchange();
      localToken = r.local_token;
      $('rawQcSourcePath').value = r.source_path;
      return '已选择源BDF；导出位置：' + r.target_path;
    });
  $('rawQcRevise').onclick = () => run('rawQcStatus', () => analyze(true));
  $('rawWaveLoad').onclick = () => run('rawQcStatus', inspect);
  $('rawWavePrev').onclick = () =>
    run('rawQcStatus', async () => {
      if (!active) return;
      $('rawWaveStart').value = String(
        Math.max(0, Number($('rawWaveStart').value) - Number($('rawWaveSeconds').value)),
      );
      await inspect();
    });
  $('rawWaveNext').onclick = () =>
    run('rawQcStatus', async () => {
      if (!active) return;
      $('rawWaveStart').value = String(
        Math.min(
          Math.max(0, active.duration - Number($('rawWaveSeconds').value)),
          Number($('rawWaveStart').value) + Number($('rawWaveSeconds').value),
        ),
      );
      await inspect();
    });
  $('rawWaveGain').oninput = () => {
    if (lastWave) {
      try {
        draw(lastWave);
      } catch (e) {
        status('rawQcStatus', e.message, true);
      }
    }
  };
  function queueInspect() {
    clearTimeout(zoomTimer);
    const load = () => {
      if (!active) return;
      if (busy) {
        zoomTimer = setTimeout(load, 180);
        return;
      }
      run('rawQcStatus', inspect);
    };
    zoomTimer = setTimeout(load, 180);
  }
  function fitView() {
    if (!lastWave) return;
    $('rawWaveGain').value = '1';
    if ($('rawYMode').value === 'fixed') $('rawYMode').value = 'channel';
    $('rawWaveStep').disabled = $('rawYMode').value !== 'fixed';
    draw(lastWave);
  }
  $('rawYMode').onchange = () => {
    $('rawWaveStep').disabled = $('rawYMode').value !== 'fixed';
    $('rawWaveGain').value = '1';
    if (lastWave) draw(lastWave);
  };
  $('rawWaveStep').onchange = () => {
    if (lastWave) {
      try {
        draw(lastWave);
      } catch (e) {
        status('rawQcStatus', e.message, true);
      }
    }
  };
  $('rawWaveFocus').onchange = () => {
    if (lastWave) draw(lastWave);
  };
  $('rawWaveFit').onclick = fitView;
  $('rawTimePreset').onchange = () => {
    if (!active) return;
    const preset = $('rawTimePreset').value;
    const seconds =
      preset === 'all'
        ? active.duration
        : preset === 'auto'
          ? active.view_defaults?.seconds || Math.min(10, active.duration)
          : Math.min(active.duration, Number(preset));
    $('rawWaveSeconds').value = String(seconds);
    $('rawWaveStart').value = String(
      preset === 'all'
        ? 0
        : Math.max(0, Math.min(Number($('rawWaveStart').value), active.duration - seconds)),
    );
    queueInspect();
  };
  $('rawQcRestore').onclick = () =>
    run('rawQcStatus', async () => {
      status('rawQcStatus', '正在恢复本机已有结果，仅准备查看，不重新导出FIF…');
      let runId;
      try {
        runId = localStorage.getItem('eeg_raw_last_run');
      } catch {}
      let result;
      try {
        result = await post('/api/raw-qc/restore', runId ? { run_id: runId } : {});
      } catch (e) {
        if (!runId) throw e;
        result = await post('/api/raw-qc/restore', {});
      }
      await mountResult(result, draftVersion, true);
      return '已有结果已恢复，仅更新显示，未重新分析或写入FIF。';
    });
  $('rawWaveExport').onclick = () =>
    run('rawQcStatus', async () => {
      if (!active || !lastWave) throw new Error('请先读取或恢复一份记录。');
      const o = viewOptions(),
        result = await post('/api/raw-qc/view', {
          run_id: active.run_id,
          channels: viewRows().map((r) => r.name),
          start: lastWave.start,
          seconds: lastWave.stop - lastWave.start,
          mode: o.mode,
          gain: o.gain,
          fixed_step: o.fixedStep,
        });
      const holder = $('rawViewDownloads');
      holder.replaceChildren();
      holder.append(el('p', result.notice));
      for (const f of result.files) {
        const a = el('a', f.name);
        a.href = f.url;
        a.download = f.name;
        holder.append(a);
      }
      return '当前视图与坐标参数已导出，BDF/FIF未改动。';
    });
  $('rawWaveCanvas').addEventListener(
    'wheel',
    (event) => {
      if (!lastWave || !active) return;
      event.preventDefault();
      const factor = Math.exp(-Math.sign(event.deltaY) * 0.16);
      if (event.shiftKey) {
        const rect = $('rawWaveCanvas').getBoundingClientRect(),
          fraction = (event.clientX - rect.left - 200) / (rect.width - 232),
          next = EEGViewAxes.zoomTime({
            start: Number($('rawWaveStart').value),
            seconds: Number($('rawWaveSeconds').value),
            total: active.duration,
            fraction,
            factor,
          });
        $('rawWaveSeconds').value = String(next.seconds);
        $('rawWaveStart').value = String(next.start);
        queueInspect();
      } else {
        $('rawWaveGain').value = String(
          Number(
            Math.max(0.001, Math.min(100, Number($('rawWaveGain').value) * factor)).toPrecision(5),
          ),
        );
        draw(lastWave);
      }
    },
    { passive: false },
  );
  let dragging = null;
  $('rawWaveCanvas').onpointerdown = (event) => {
    const rect = $('rawWaveCanvas').getBoundingClientRect();
    if (!lastWave || event.button !== 0 || event.clientX - rect.left < 200) return;
    event.preventDefault();
    dragging = {
      x: event.clientX,
      start: lastWave.start,
      seconds: lastWave.stop - lastWave.start,
      width: rect.width - 232,
      moved: false,
    };
    $('rawWaveCanvas').setPointerCapture(event.pointerId);
  };
  $('rawWaveCanvas').onpointermove = (event) => {
    if (!dragging) return;
    const dx = event.clientX - dragging.x;
    if (Math.abs(dx) < 4) return;
    dragging.moved = true;
    const next = EEGViewAxes.panTime({
      start: dragging.start,
      seconds: dragging.seconds,
      total: active.duration,
      fraction: -dx / dragging.width,
    });
    $('rawWaveStart').value = String(next.start);
    $('rawWaveInfo').textContent =
      `松开后读取原始时间 ${fmt(next.start)}–${fmt(next.start + next.seconds)} 秒；当前波形尚未重读。`;
  };
  $('rawWaveCanvas').onpointerup = () => {
    const moved = dragging?.moved;
    dragging = null;
    if (moved) queueInspect();
  };
  $('rawWaveCanvas').onpointercancel = () => {
    dragging = null;
  };
  $('rawWaveCanvas').ondblclick = fitView;
  window.addEventListener('resize', () => {
    if (lastWave && !$('rawQcInspector').classList.contains('hide')) draw(lastWave);
  });
  $('rawWaveMark').onclick = () => {
    if (!lastWave) return;
    const box = $('rawQcBadSegments');
    box.value +=
      (box.value.trim() ? '\n' : '') + `${lastWave.start},${lastWave.stop},manual_review`;
    dirty();
  };
  window.EEGRawViewer = {
    getRecord: () => active,
    getWave: () => lastWave,
    getVisibleRows: viewRows,
    applyDraft,
    redraw: () => {
      if (lastWave) draw(lastWave);
    },
    jump: async ({ start, seconds, channel }) => {
      if (!active) throw new Error('请先载入或恢复BDF。');
      seconds = Math.max(Math.min(0.01, active.duration), Math.min(seconds || 4, active.duration));
      $('rawWaveStart').value = String(
        Math.max(0, Math.min(start || 0, active.duration - seconds)),
      );
      $('rawWaveSeconds').value = String(seconds);
      if (channel) {
        for (const check of document.querySelectorAll('[data-raw-channel]'))
          if (check.value === channel) check.checked = true;
      }
      await inspect({ follow: false });
      if (channel !== undefined) {
        $('rawWaveFocus').value = channel;
        draw(lastWave);
      }
      if (window.EEGSpectrum?.isFollowing()) await EEGSpectrum.loadCurrent();
      return '已定位到原始波形，未自动标坏或计入人工复核。';
    },
  };
})();
