// 共用工具：找页面元素、发请求、显示图表。这里不计算脑电特征。
// 阅读顺序见 docs/CODE_READING_GUIDE.md；使用原生JavaScript，无需前端框架。
'use strict';
// ===== 单份分析主线（读代码时按此顺序追） =====
// singleForm(表单onsubmit) → analyzeSingleFile() → POST /api/analyze → 后端 _single_analysis() → analyze_eeg_file() → render()
const $ = (id) => document.getElementById(id);
const el = (tag, text, cls) => {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  if (cls) n.className = cls;
  return n;
};
const fmt = (v) =>
  v === null || v === undefined
    ? '无法评估'
    : typeof v === 'number'
      ? Number(v.toPrecision(6)).toLocaleString('zh-CN', { maximumFractionDigits: 5 })
      : String(v);
let busy = false,
  activePair = null;
function inputFormat(prefix) {
  const nodes = document.querySelectorAll(`[name=${prefix}Format]:checked`);
  if (nodes.length !== 1 || !['bdf', 'fif'].includes(nodes[0].value))
    throw new Error('必须且只能勾选BDF或FIF一种格式。');
  return nodes[0].value;
}
function checkFormat(file, format) {
  check(file);
  if (!['bdf', 'fif'].includes(format) || !file.name.toLowerCase().endsWith('.' + format))
    throw new Error('文件与勾选格式不一致，禁止混合BDF/FIF。');
}
async function queueEeg(file, path) {
  return EEGFolderImport.queueItem(file, path, inputFormat('batch'), async (f) =>
    post('/api/fif-header', { upload_id: await upload(f, 'eeg') }),
  );
}

function status(id, text, error = false) {
  $(id).textContent = text;
  $(id).classList.toggle('error', error);
}
function lock(value) {
  busy = value;
  document
    .querySelectorAll('.run')
    .forEach(
      (b) =>
        (b.disabled =
          value || (b.hasAttribute('data-requires-result') && b.dataset.ready !== 'true')),
    );
  $('featureProfile').disabled = value;
  document.querySelectorAll('.format-choice input').forEach((n) => (n.disabled = value));
  for (const id of ['batchFolder', 'batchFolderRecord', 'batchFolderReload', 'batchFolderSearch'])
    $(id).disabled = value;
}
document.querySelectorAll('[data-tab]').forEach((b) =>
  b.addEventListener('click', () => {
    document
      .querySelectorAll('[data-tab]')
      .forEach((t) => t.setAttribute('aria-selected', String(t === b)));
    document
      .querySelectorAll('.tab-panel')
      .forEach((p) => p.classList.toggle('hide', p.id !== b.dataset.tab));
  }),
);
let imageZoom = { scale: 1, x: 0, y: 0, drag: null };
function applyImageZoom() {
  $('viewerImage').style.transform =
    `translate(${imageZoom.x}px,${imageZoom.y}px) scale(${imageZoom.scale})`;
  $('viewerImage').style.cursor = imageZoom.scale > 1 ? 'grab' : 'zoom-in';
  document.querySelector('.viewer-help').textContent =
    `${imageZoom.scale.toFixed(2)}倍 · 滚轮缩放 · 拖动平移 · 双击恢复 · Esc关闭`;
}
function resetImageZoom() {
  imageZoom = { scale: 1, x: 0, y: 0, drag: null };
  applyImageZoom();
}
function closeViewer() {
  $('imageViewer').classList.add('hide');
  $('viewerImage').removeAttribute('src');
  resetImageZoom();
  document.body.style.overflow = '';
}
$('viewerImage').addEventListener('load', resetImageZoom);
$('imageViewer').addEventListener(
  'wheel',
  (e) => {
    e.preventDefault();
    const previous = imageZoom.scale,
      next = Math.max(1, Math.min(12, previous * Math.exp(-Math.sign(e.deltaY) * 0.18))),
      ratio = next / previous,
      rect = $('viewerImage').getBoundingClientRect();
    const x = e.clientX - (rect.left + rect.width / 2 - imageZoom.x),
      y = e.clientY - (rect.top + rect.height / 2 - imageZoom.y);
    imageZoom.x = x - (x - imageZoom.x) * ratio;
    imageZoom.y = y - (y - imageZoom.y) * ratio;
    imageZoom.scale = next;
    if (next === 1) {
      imageZoom.x = 0;
      imageZoom.y = 0;
    }
    applyImageZoom();
  },
  { passive: false },
);
$('viewerImage').ondblclick = resetImageZoom;
$('viewerImage').onpointerdown = (e) => {
  if (e.button !== 0 || imageZoom.scale <= 1) return;
  e.preventDefault();
  imageZoom.drag = { x: e.clientX, y: e.clientY, startX: imageZoom.x, startY: imageZoom.y };
  $('viewerImage').setPointerCapture(e.pointerId);
};
$('viewerImage').onpointermove = (e) => {
  if (!imageZoom.drag) return;
  imageZoom.x = imageZoom.drag.startX + e.clientX - imageZoom.drag.x;
  imageZoom.y = imageZoom.drag.startY + e.clientY - imageZoom.drag.y;
  applyImageZoom();
};
$('viewerImage').onpointerup = $('viewerImage').onpointercancel = () => {
  imageZoom.drag = null;
};
$('viewerClose').onclick = closeViewer;
$('imageViewer').onclick = (e) => {
  if (e.target === $('imageViewer')) closeViewer();
};
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeViewer();
});
// render(): 主线终点。输入=容器id + 后端返回的JSON结果 + 要展示的指标名值对；输出=把指标卡、事件面板、警告、图表缩略图和下载链接摆到页面上。
function render(id, result, metrics) {
  closeViewer();
  const root = $(id);
  root.replaceChildren();
  root.classList.remove('hide');
  const card = el('div', undefined, 'card');
  root.append(card);
  const grid = el('div', undefined, 'metrics');
  for (const [label, value] of metrics) {
    const m = el('div', undefined, 'metric');
    m.append(el('span', label), el('strong', fmt(value)));
    grid.append(m);
  }
  card.append(grid);
  for (const [name, info] of eventInfos(result)) if (info) eventPanel(card, info, name);
  if (result.metadata_info) renderMetadataResult(card, result.metadata_info);
  if (result.montage_info) renderMontage(card, result.montage_info);
  if (result.feature_definition) {
    const details = el('details');
    details.append(el('summary', '查看本次特征计算口径与数据覆盖'));
    result.feature_definition.forEach((s) => details.append(el('p', s)));
    if (result.feature_coverage)
      details.append(el('pre', JSON.stringify(result.feature_coverage, null, 2)));
    card.append(details);
    if (result.kind === 'single') featureTable(card, result.rows);
  }
  if (result.warnings?.length) {
    const ul = el('ul', undefined, 'warnings');
    result.warnings.forEach((w) => ul.append(el('li', w)));
    card.append(ul);
  }
  const charts = el('div', undefined, 'charts'),
    downloads = el('div', undefined, 'downloads');
  for (const f of eventFiles(result)) {
    const a = el('a', f.label);
    a.href = f.url;
    a.download = f.name;
    downloads.append(a);
    if (f.preview_url) {
      const figure = el('figure', undefined, 'chart'),
        img = el('img');
      img.src = f.preview_url;
      img.alt = f.label;
      img.title = '点击放大';
      img.tabIndex = 0;
      const open = () => {
        $('viewerImage').src = f.preview_url;
        $('viewerImage').alt = f.label;
        $('viewerCaption').textContent = f.label;
        $('imageViewer').classList.remove('hide');
        document.body.style.overflow = 'hidden';
        $('viewerClose').focus();
      };
      img.onclick = open;
      img.onkeydown = (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          open();
        }
      };
      const link = el('a', '新窗口查看原图');
      link.href = f.preview_url;
      link.target = '_blank';
      link.rel = 'noopener';
      figure.append(img, el('figcaption', f.label), link);
      if (result.research_appendix) {
        const note = chartResearchNote(f.label, result.research_appendix);
        if (note) figure.append(note);
      }
      charts.append(figure);
    }
  }
  root.append(charts);
  const files = el('div', undefined, 'card');
  files.append(el('h3', '下载结果'), downloads);
  root.append(files);
  if (result.research_appendix) renderResearchAppendix(root, result);
  return card;
}
function table(parent, rows, title) {
  parent.append(el('h3', title));
  if (!rows?.length) {
    parent.append(el('p', '没有符合当前规则的候选记录。'));
    return;
  }
  const wrap = el('div', undefined, 'table-wrap'),
    t = el('table'),
    head = el('thead'),
    hr = el('tr');
  const keys = Object.keys(rows[0]);
  keys.forEach((k) => hr.append(el('th', k)));
  head.append(hr);
  const body = el('tbody');
  rows.forEach((row) => {
    const tr = el('tr');
    keys.forEach((k) => tr.append(el('td', fmt(row[k]))));
    body.append(tr);
  });
  t.append(head, body);
  wrap.append(t);
  parent.append(wrap);
}
// response(): 把 fetch 的响应解析成 JSON；服务端报错时(j.error)抛出错误文字给 run() 显示。
async function response(r) {
  let j;
  try {
    j = await r.json();
  } catch {
    throw new Error('服务没有返回有效结果，请查看本机服务日志并重新启动。');
  }
  if (!r.ok) throw new Error(j.error || `请求失败 ${r.status}`);
  return j;
}
// request(): 实际发网络请求的地方(fetch)；连接失败(TypeError)时提示用户先启动本机服务再访问。
async function request(url, options) {
  try {
    return await response(await fetch(url, options));
  } catch (e) {
    if (e instanceof TypeError)
      throw new Error(
        '连接本机服务失败。请双击“启动脑电特征网页.bat”，再访问 http://127.0.0.1:8765/。',
      );
    throw e;
  }
}
// check(): 上传前的本地校验——必须有文件、非空、不超过 2 GB。
function check(file) {
  if (!file) throw new Error('请选择文件。');
  if (!file.size || file.size > 2 * 1024 ** 3) throw new Error('文件为空或超过 2 GB。');
}
// settings(): 从页面读取分析参数——target_fs 目标采样率、epoch_seconds 片段长度、notch 陷波频率。
function settings(prefix) {
  return {
    input_format: inputFormat(prefix),
    feature_profile: $('featureProfile').value,
    target_fs: Number($(prefix + 'Fs').value),
    epoch_seconds:
      $('featureProfile').value === 'thesis_v1' ? 2 : Number($(prefix + 'Epoch').value),
    notch: $(prefix + 'Notch').value,
    auto_eeg_only: true,
    features: ['time', 'frequency', 'time_frequency', 'network'],
  };
}
const previousEpochs = {};
$('featureProfile').onchange = () => {
  const thesis = $('featureProfile').value === 'thesis_v1';
  for (const prefix of ['single', 'pair', 'batch']) {
    const input = $(prefix + 'Epoch');
    if (thesis) {
      previousEpochs[prefix] = input.value;
      input.value = '2';
    } else if (previousEpochs[prefix]) input.value = previousEpochs[prefix];
    input.readOnly = thesis;
  }
  $('profileNotice').textContent = thesis
    ? '论文方案已启用：固定2秒；时频先汇总频段功率，再计算标准差与CV。全四域每时相287列，缺失留空；结果可搜索、下载字典与宽表。'
    : '原有探索方案已启用。';
  activePair = null;
  $('pairControls').classList.add('hide');
};
function featureTable(parent, rows) {
  const search = el('input');
  search.type = 'search';
  search.placeholder = '搜索特征，如 alpha、TF_cv、FAA、额区';
  search.setAttribute('aria-label', '论文特征搜索');
  const missing = el('input');
  missing.type = 'checkbox';
  const label = el('label', '仅看缺失项');
  label.prepend(missing);
  parent.append(search, label);
  const holder = el('div');
  parent.append(holder);
  const update = () => {
    holder.replaceChildren();
    table(
      holder,
      (rows || []).filter(
        (r) =>
          r['特征'].toLowerCase().includes(search.value.toLowerCase()) &&
          (!missing.checked || r['数值'] === null),
      ),
      `论文方案特征（共${rows?.length || 0}项；缺失不补零）`,
    );
  };
  search.oninput = update;
  missing.onchange = update;
  update();
}

// analyzeSingleFile(): 把【文件本体】作为 POST body、【参数】塞进 X-EEG-Settings 请求头发给 /api/analyze，返回后端 JSON 结果。
function analyzeSingleFile(file, opts) {
  checkFormat(file, opts.input_format);
  return request('/api/analyze?filename=' + encodeURIComponent(file.name), {
    method: 'POST',
    headers: { 'X-EEG-Settings': JSON.stringify(opts) },
    body: file,
  });
}
function post(url, data) {
  return request(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
}
async function run(id, fn) {
  if (busy) return;
  if (location.protocol === 'file:') {
    status(id, '请使用 http://127.0.0.1:8765/ 打开网页。', true);
    return;
  }
  lock(true);
  try {
    const message = await fn();
    status(id, typeof message === 'string' ? message : '分析完成，结果与图表已显示。');
  } catch (e) {
    status(id, e.message, true);
  } finally {
    lock(false);
  }
}
// 主线起点：用户点击“开始分析”触发 singleForm 提交。
// 依次执行：settings('single') 读参数 → analyzeSingleFile() 上传文件与参数 → 得到 r → render('singleResult', r, ...) 把结果显示到页面。

async function upload(file, kind) {
  check(file);
  return (
    await request('/api/upload?kind=' + kind + '&filename=' + encodeURIComponent(file.name), {
      method: 'POST',
      body: file,
    })
  ).upload_id;
}
