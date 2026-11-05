// 事件可选开关、上传与标签展示，不猜测编码含义。
// 阅读顺序见 docs/CODE_READING_GUIDE.md；使用原生JavaScript，无需前端框架。
function eventInfos(result) {
  return [
    ['事件标记', result.event_info],
    ['A 记录事件', result.event_info_a],
    ['B 记录事件', result.event_info_b],
  ];
}
function eventFiles(result) {
  const files = new Map((result.files || []).map((f) => [f.url, f]));
  for (const [name, info] of eventInfos(result)) {
    for (const f of info?.files || [])
      if (!files.has(f.url)) files.set(f.url, { ...f, label: name + ' · ' + f.label });
  }
  return [...files.values()];
}
function eventPanel(parent, info, title) {
  const section = el('section', undefined, 'event-results');
  section.append(
    el('h3', title),
    el(
      'p',
      `总事件 ${info.total_events}；与当前片段相关 ${info.in_segment}；片段外 ${info.outside_segment}；总时间偏移 ${fmt(info.alignment_shift_seconds)} 秒。`,
    ),
  );
  const ul = el('ul', undefined, 'warnings');
  info.warnings.forEach((w) => ul.append(el('li', w)));
  section.append(ul);
  if (info.labels.length) table(section, info.labels, '原始事件标签');
  if (info.rows.length) {
    const details = el('details');
    details.open = true;
    details.append(el('summary', '事件位置与质控状态（最多显示 1000 行）'));
    table(details, info.rows, '事件明细');
    section.append(details);
  } else section.append(el('p', '该文件没有带描述的事件（空时间标记不算事件）。'));
  if (info.table_truncated) section.append(el('p', '页面展示已截断，完整事件请下载 CSV。'));
  parent.append(section);
}
async function eventOptions(prefix, suffix = '') {
  if (!$(prefix + 'UseEvt').checked) return {};
  const file = $(prefix + 'Evt' + suffix).files[0];
  if (!file) {
    if (prefix === 'pair') return {};
    throw new Error('已勾选事件标记，请选择配套 evt.bdf。');
  }
  if (file.size > 20 * 1024 ** 2) throw new Error('evt.bdf 不能超过 20 MB。');
  const offset = Number($(prefix + 'EvtOffset' + suffix).value);
  if (!Number.isFinite(offset) || Math.abs(offset) > 86400)
    throw new Error('事件校正偏移须为 ±86400 秒以内。');
  return { evt_upload: await upload(file, 'evt'), evt_offset: offset };
}
for (const prefix of ['single', 'pair', 'spike']) {
  const toggle = $(prefix + 'UseEvt'),
    box = $(prefix + 'EvtOptions');
  toggle.onchange = () => {
    box.classList.toggle('hide', !toggle.checked);
    box.querySelectorAll('input').forEach((input) => {
      input.disabled = !toggle.checked;
      if (input.type === 'file' && prefix !== 'pair') input.required = toggle.checked;
    });
  };
}
