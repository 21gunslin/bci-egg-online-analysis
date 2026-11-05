// 多份文件：整理队列、匹配JSON和evt，再按时间提交。
// 阅读顺序见 docs/CODE_READING_GUIDE.md；使用原生JavaScript，无需前端框架。
let batchQueue = [],
  queueReading = false,
  folderQueueMode = false;
function drawBatchQueue() {
  if (folderQueueMode) $('batchCount').value = Math.max(2, Math.min(30, batchQueue.length));
  const root = $('batchQueue');
  root.replaceChildren();
  $('batchQueueInfo').textContent =
    `已添加 ${batchQueue.length} / ${$('batchCount').value} 份 · 顺序会发送给服务器复核文件头时间`;
  batchQueue.forEach((item, i) => {
    const row = el('div', undefined, 'queue-item'),
      name = el('div', undefined, 'queue-name');
    name.append(
      el('strong', `第 ${i + 1} 段 · ${item.file.name}`),
      el('p', `${item.clock} · ${(item.file.size / 1024 ** 2).toFixed(2)} MB`, 'muted small'),
    );
    if (item.path && item.path !== item.file.name) name.append(el('p', item.path, 'muted small'));
    if (batchMeta.datainfo) name.append(el('p', queueMetadataMatch(item), 'muted small'));
    const actions = el('div', undefined, 'queue-actions');
    for (const [text, step] of [
      ['上移', -1],
      ['下移', 1],
      ['移除', 0],
    ]) {
      const b = el('button', text, 'secondary');
      b.type = 'button';
      b.setAttribute('aria-label', `第${i + 1}段${text}`);
      b.disabled = (step === -1 && i === 0) || (step === 1 && i === batchQueue.length - 1);
      b.onclick = () => {
        if (busy || queueReading) return;
        if (step === 0) batchQueue.splice(i, 1);
        else [batchQueue[i], batchQueue[i + step]] = [batchQueue[i + step], batchQueue[i]];
        drawBatchQueue();
      };
      actions.append(b);
    }
    row.append(name, actions);
    root.append(row);
  });
}
$('batchCount').oninput = () => {
  folderQueueMode = false;
  drawBatchQueue();
};
$('batchFiles').onchange = async () => {
  if (busy || queueReading) return;
  queueReading = true;
  lock(true);
  try {
    const files = [...$('batchFiles').files],
      limit = folderQueueMode ? 30 : Number($('batchCount').value);
    if (!Number.isInteger(limit) || limit < 2 || limit > 30)
      throw new Error('先填写2–30之间的份数。');
    if (batchQueue.length + files.length > limit)
      throw new Error(`本次最多${limit}份，请先移除不需要的片段或调整份数。`);
    const additions = [];
    for (const file of files) additions.push(await queueEeg(file));
    batchQueue.push(...additions);
    status('batchStatus', '已加入文件，可继续逐份添加或调整顺序。');
  } catch (e) {
    status('batchStatus', e.message, true);
  } finally {
    $('batchFiles').value = '';
    queueReading = false;
    lock(false);
    drawBatchQueue();
  }
};
$('batchSort').onclick = () => {
  if (busy || queueReading) return;
  if (batchQueue.some((i) => i.timestamp === null)) {
    status('batchStatus', '有文件未识别出开始时间，请手动核对。', true);
    return;
  }
  batchQueue.sort((a, b) => a.timestamp - b.timestamp);
  drawBatchQueue();
};
$('batchForm').onsubmit = (e) => {
  e.preventDefault();
  run('batchStatus', async () => {
    if (queueReading) throw new Error('文件头正在读取，请稍后提交。');
    const queued = batchQueue.slice(),
      count = Number($('batchCount').value);
    for (const item of queued) checkFormat(item.file, inputFormat('batch'));
    if (queued.length !== count)
      throw new Error(`设定 ${count} 份，但已添加 ${queued.length} 份，请补齐或调整份数。`);
    if (queued.reduce((n, i) => n + i.file.size, 0) > 2 * 1024 ** 3)
      throw new Error('本组 data 总大小超过2 GB。');
    const evt = $('batchEvt').files[0];
    check(evt);
    if (evt.size > 20 * 1024 ** 2) throw new Error('evt 文件不能超过20 MB。');
    const opts = settings('batch'),
      offset = Number($('batchEvtOffset').value),
      items = [],
      metadataUploads = {};
    const metadataFiles = {
      datainfo: $('batchDataInfo').files[0],
      recordinfo: $('batchRecordInfo').files[0],
    };
    for (const [kind, file] of Object.entries(metadataFiles)) {
      if (file) {
        status('batchStatus', '验证 JSON 元数据……');
        metadataUploads[kind + '_upload'] = (await jsonMetadata(file, kind, false)).upload_id;
      }
    }
    for (let i = 0; i < queued.length; i++) {
      status('batchStatus', `上传第 ${i + 1}/${count} 份 data，请等待……`);
      items.push({ upload_id: await upload(queued[i].file, 'eeg') });
    }
    status('batchStatus', '上传本组唯一的 evt……');
    const evt_upload = await upload(evt, 'evt');
    status('batchStatus', '正在逐段质控、核对共同通道与事件时间，再汇总四域特征；请保持页面打开……');
    const r = await post('/api/session', {
        input_format: inputFormat('batch'),
        items,
        evt_upload,
        evt_offset: offset,
        settings: opts,
        skip_empty: $('batchSkipEmpty').checked,
        ...metadataUploads,
      }),
      m = r.metrics;
    const card = render('batchResult', r, [
      ['参与分析份数', m.files],
      ['仅展示/未纳入份数', m.skipped_files || 0],
      ['共同通道', m.common_channels],
      ['实际采样率 Hz', m.sampling_rate],
      ['已选原始总时长 s', m.recorded_seconds],
      ['首尾跨度 s（含间隔）', m.span_seconds],
      ['有效时长 s', m.effective_seconds],
      ['独立连续段数', m.continuous_runs],
    ]);
    if (r.skipped_files?.length) table(card, r.skipped_files, '质控失败：仅展示，不参与汇总');
    table(
      card,
      r.review_segments || r.segments,
      r.review_segments ? '全部分段：时间位置、质控状态与是否参与汇总' : '时间顺序与分段映射',
    );
    if (r.review_only) {
      card.append(
        el(
          'p',
          '当前为仅复核模式：请查看失败段原始波形和事件位置；没有合格数据，因此不计算四域特征。',
          'notice',
        ),
      );
      return;
    }
    card.append(el('p', '共同通道：' + r.common_labels.join(', '), 'muted'));
    const domain = el('select');
    domain.setAttribute('aria-label', '多份特征域筛选');
    ['全部', '时域', '频域', '时频域', '网络', '汇总'].forEach((v) => {
      const o = el('option', v);
      o.value = v;
      domain.append(o);
    });
    const search = el('input');
    search.placeholder = '搜索多份汇总特征';
    const filters = el('div', undefined, 'row');
    filters.append(domain, search);
    card.append(filters);
    const holder = el('div');
    card.append(holder);
    function update() {
      holder.replaceChildren();
      table(
        holder,
        r.rows.filter(
          (row) =>
            (domain.value === '全部' || row['特征域'] === domain.value) &&
            row['特征'].toLowerCase().includes(search.value.toLowerCase()),
        ),
        r.feature_definition
          ? '论文方案：独立段窗口汇集特征（287项）'
          : '逐连续段按有效时长加权的特征均值',
      );
    }
    domain.onchange = update;
    search.oninput = update;
    update();
    if (r.table_truncated) card.append(el('p', '页面仅展示部分特征，完整结果请下载 CSV。'));
  });
};

let batchMeta = { datainfo: null, recordinfo: null };
async function jsonMetadata(file, kind, preview) {
  if (!file || !/\.json$/i.test(file.name) || file.size > 2 * 1024 ** 2 || !file.size)
    throw new Error('请选择不超过2 MB的JSON文件。');
  return request(
    '/api/upload?kind=' +
      kind +
      '&preview=' +
      (preview ? '1' : '0') +
      '&filename=' +
      encodeURIComponent(file.name),
    { method: 'POST', body: file },
  );
}
function drawBatchMetadata() {
  const root = $('batchMetadataPreview');
  root.replaceChildren();
  if (batchMeta.datainfo) {
    const d = batchMeta.datainfo;
    root.append(
      el(
        'p',
        `整次 ${fmt(d.duration_seconds)} 秒 · ${d.entry_count} 个时间条目 · ${d.waveform_count} 个有波形条目 · 波形合计 ${fmt(d.recorded_seconds)} 秒 · 其余 ${fmt(d.gap_seconds)} 秒`,
      ),
    );
    const details = el('details');
    details.open = true;
    details.append(el('summary', '自动分段时间表（JSON，UTC+8）'));
    table(details, d.segments, 'JSON 分段计划');
    root.append(details);
  }
  if (batchMeta.recordinfo) {
    const d = batchMeta.recordinfo;
    root.append(
      el(
        'p',
        '配置通道 ' +
          d.channel_count +
          ' 个：' +
          Object.entries(d.type_counts)
            .map(([k, v]) => k + ' ' + v)
            .join('，'),
      ),
    );
    const details = el('details');
    details.append(el('summary', '查看安全通道配置（不含摄像头或身份字段）'));
    table(details, d.channels, 'RecordInfo 通道类型');
    root.append(details);
    if (d.montage_info) renderMontage(root, d.montage_info, '当前RecordInfo的Montage映射');
  }
}
function queueMetadataMatch(item) {
  const d = batchMeta.datainfo;
  if (!d || item.timestamp === null || item.duration === null)
    return '元数据预匹配：等待服务器核对';
  const start = item.timestamp / 1000 - 8 * 3600 - d.start_timestamp;
  const candidates = d.segments
    .filter(
      (r) =>
        r['持续_s'] > 0 &&
        Math.abs(r['起点_s'] - start) <= 1.05 &&
        Math.abs(r['持续_s'] - item.duration) <= 0.001,
    )
    .sort((a, b) => Math.abs(a['起点_s'] - start) - Math.abs(b['起点_s'] - start));
  if (!candidates.length) return '元数据预匹配：未找到对应条目，请核对文件';
  if (
    candidates.length > 1 &&
    Math.abs(
      Math.abs(candidates[0]['起点_s'] - start) - Math.abs(candidates[1]['起点_s'] - start),
    ) < 1e-9
  )
    return '元数据预匹配：编号不唯一，须核查';
  const r = candidates[0];
  return `预匹配 JSON data${r['JSON编号']} · 起点差 BDF−JSON ${fmt(start - r['起点_s'])} 秒（以服务器核对为准）`;
}
for (const [kind, id] of [
  ['datainfo', 'batchDataInfo'],
  ['recordinfo', 'batchRecordInfo'],
]) {
  $(id).onchange = async () => {
    if (busy) {
      status('batchMetadataStatus', '请等待当前上传或分析完成后再读取JSON。', true);
      return;
    }
    const file = $(id).files[0];
    if (!file) {
      batchMeta[kind] = null;
      drawBatchMetadata();
      drawBatchQueue();
      return;
    }
    lock(true);
    try {
      status('batchMetadataStatus', '正在读取元数据安全预览……');
      const result = await jsonMetadata(file, kind, true);
      batchMeta[kind] = result.metadata;
      let note = '元数据预览已生成。';
      if (kind === 'datainfo') {
        const n = result.metadata.waveform_count;
        if (folderQueueMode) {
          note = `JSON包含${n}份非零波形条目；当前按队列中的${batchQueue.length}份处理，可手动移除或补加。`;
        } else if (n >= 2 && n <= 30) {
          $('batchCount').value = n;
          note = `已自动设置 ${n} 份有波形记录；零时长节点不计入。可手动改为本次选择的子集份数。`;
        } else note = '已生成全部时间表；本入口每次仍需选择2–30份，可分析其子集。';
      }
      drawBatchMetadata();
      drawBatchQueue();
      status('batchMetadataStatus', note);
    } catch (e) {
      batchMeta[kind] = null;
      $(id).value = '';
      drawBatchMetadata();
      drawBatchQueue();
      status('batchMetadataStatus', e.message, true);
    } finally {
      lock(false);
    }
  };
}
$('batchClearMetadata').onclick = () => {
  if (busy) return;
  batchMeta = { datainfo: null, recordinfo: null };
  $('batchDataInfo').value = '';
  $('batchRecordInfo').value = '';
  drawBatchMetadata();
  drawBatchQueue();
  status('batchMetadataStatus', 'JSON辅助已清除，已添加的data保留；份数可自行调整。');
};
function renderMetadataResult(parent, info) {
  const section = el('section');
  section.append(el('h3', 'JSON 元数据核对'));
  if (info.time_checks.length)
    table(section, info.time_checks, 'JSON 与 BDF 时间核对（不覆盖BDF）');
  if (info.channel_checks.length) {
    const details = el('details');
    details.append(el('summary', '通道名称与类型核对明细'));
    table(details, info.channel_checks, 'RecordInfo 映射及分析选择');
    section.append(details);
  }
  if (info.datainfo)
    section.append(
      el(
        'p',
        `JSON共有 ${info.datainfo.waveform_count} 份有波形记录，本次匹配 ${info.time_checks.length} 份。`,
      ),
    );
  if (info.recordinfo)
    section.append(
      el(
        'p',
        'RecordInfo声明：' +
          Object.entries(info.recordinfo.type_counts)
            .map(([k, v]) => k + ' ' + v)
            .join('，') +
          '；这不等于最终质控保留数。',
      ),
    );
  parent.append(section);
}

let batchFolderRecords = [];
function drawFolderChoices() {
  const choice = $('batchFolderRecord'),
    selected = choice.value,
    query = $('batchFolderSearch').value;
  choice.replaceChildren();
  const candidates = batchFolderRecords
    .map((record, index) => ({ record, index }))
    .filter(({ record }) => EEGFolderImport.matchesRecord(record, query));
  const placeholder = el(
    'option',
    `匹配${candidates.length}/${batchFolderRecords.length}次记录，请选择一次`,
  );
  placeholder.value = '';
  choice.append(placeholder);
  for (const { record, index } of candidates) {
    const option = el('option', `${record.title} · ${record.data.length}份data`);
    option.value = String(index);
    choice.append(option);
  }
  choice.value = candidates.some((item) => String(item.index) === selected) ? selected : '';
}
function importedFile(id, file) {
  const input = $(id);
  if (!file) {
    input.value = '';
    return;
  }
  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
}
async function populateFolderRecord(record) {
  const items = [];
  for (const entry of record.data) items.push(await queueEeg(entry.file, entry.path));
  const sorted = EEGFolderImport.sortQueue(items),
    messages = [],
    safeMeta = { datainfo: null, recordinfo: null },
    chosen = { evt: null, datainfo: null, recordinfo: null };
  for (const kind of ['evt', 'datainfo', 'recordinfo']) {
    const files = record.aux[kind];
    if (files.length === 1) chosen[kind] = files[0];
    else
      messages.push(
        files.length
          ? `${kind}有多个候选，未擅自选择，请手动指定。`
          : `未找到${kind === 'evt' ? '配套evt.bdf（须手动添加）' : kind + '.json（可选）'}。`,
      );
  }
  for (const kind of ['datainfo', 'recordinfo']) {
    if (chosen[kind]) {
      try {
        safeMeta[kind] = (await jsonMetadata(chosen[kind], kind, true)).metadata;
      } catch (error) {
        messages.push(`${kind}.json未自动采用：${error.message}`);
        chosen[kind] = null;
      }
    }
  }
  // Commit together, so an unreadable BDF does not leave a half-replaced queue.
  batchQueue = sorted;
  folderQueueMode = true;
  batchMeta = safeMeta;
  importedFile('batchEvt', chosen.evt);
  importedFile('batchDataInfo', chosen.datainfo);
  importedFile('batchRecordInfo', chosen.recordinfo);
  $('batchConfirm').checked = false;
  drawBatchMetadata();
  drawBatchQueue();
  const missingTime = items.filter((i) => i.timestamp === null).length;
  if (missingTime)
    messages.push(
      `${missingTime}份未识别出可靠时间，已排在后面并按路径编号排列，请核对；服务器仍会校验时间。`,
    );
  const n = items.length;
  if (n < 2) messages.push('仅发现1份data，多份入口至少需要2份；可改用单份分析或补加同次记录。');
  if (n > 30) messages.push(`候选${n}份全部列出；单次最多30份，请移除不需要的片段后再分析。`);
  if (items.reduce((size, item) => size + item.file.size, 0) > 2 * 1024 ** 3)
    messages.push('本组超过2 GB，请移除片段后再分析。');
  if (safeMeta.datainfo && safeMeta.datainfo.waveform_count !== n)
    messages.push(
      `JSON有${safeMeta.datainfo.waveform_count}份非零波形条目，当前目录找到${n}份data；允许选择子集，请核对。`,
    );
  status(
    'batchFolderStatus',
    `已载入：${record.title}\n已填入${n}份data（按文件头时间排序）；evt ${chosen.evt ? '已匹配' : '待手动添加'}；datainfo ${chosen.datainfo ? '已匹配' : '未采用'}；RecordInfo ${chosen.recordinfo ? '已匹配' : '未采用'}。\n` +
      messages.join('\n'),
  );
  status('batchMetadataStatus', '文件夹的可用JSON已生成安全预览，可在此手动替换或清除。');
  status(
    'batchStatus',
    '候选队列已就绪。可继续移除、调整顺序或逐份补加；核对记录后勾选确认，再开始分析。',
  );
}
async function loadSelectedFolderRecord() {
  if (busy || queueReading) return;
  const record = batchFolderRecords[Number($('batchFolderRecord').value)];
  if ($('batchFolderRecord').value === '' || !record) {
    status('batchFolderStatus', '先选择其中一次记录。', true);
    return;
  }
  queueReading = true;
  lock(true);
  try {
    status('batchFolderStatus', '正在读取本次记录的文件头并匹配JSON……');
    await populateFolderRecord(record);
  } catch (error) {
    status('batchFolderStatus', error.message, true);
  } finally {
    queueReading = false;
    lock(false);
  }
}
$('batchFolder').onchange = async () => {
  if (busy || queueReading) return;
  queueReading = true;
  lock(true);
  $('batchConfirm').checked = false;
  try {
    const files = [...$('batchFolder').files],
      scan = EEGFolderImport.indexFolderFiles(files, inputFormat('batch'));
    if (!scan.records.length)
      throw new Error(
        '未找到勾选格式对应的“数字子文件夹/data.bdf”或“数字子文件夹/fdata.fif”。请选择某次姓名编号记录文件夹，或包含这些记录的上层目录。下方原队列保留。',
      );
    batchFolderRecords = scan.records;
    $('batchFolderSearch').value = '';
    drawFolderChoices();
    const choice = $('batchFolderRecord');
    choice.value = '';
    $('batchFolderChoices').classList.remove('hide');
    if (scan.records.length === 1) {
      choice.value = '0';
      await populateFolderRecord(scan.records[0]);
    } else
      status(
        'batchFolderStatus',
        `发现${scan.records.length}次记录，共${scan.dataFiles}份候选data。请选择一次记录后自动填入；不会合并不同记录。下方在选择前仍保留原队列。`,
      );
  } catch (error) {
    status('batchFolderStatus', error.message, true);
  } finally {
    queueReading = false;
    lock(false);
  }
};
$('batchFolderRecord').onchange = loadSelectedFolderRecord;
$('batchFolderReload').onclick = loadSelectedFolderRecord;
$('batchFolderSearch').oninput = drawFolderChoices;
