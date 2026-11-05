// 结果中文解释、Montage表和论文附录，只展示，不改变计算值。
// 阅读顺序见 docs/CODE_READING_GUIDE.md；使用原生JavaScript，无需前端框架。
function researchNumber(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return '缺失';
  if (value === 0) return '0';
  return Math.abs(value) < 1e-4 || Math.abs(value) >= 1e6
    ? value.toExponential(5)
    : Number(value.toPrecision(7)).toString();
}
function renderMontage(parent, info, title = '本次通道的Montage映射') {
  const details = el('details');
  details.className = 'advanced';
  details.append(
    el('summary', `${title} · ${info.counts.mapped}/${info.counts.total}项可装配`),
    el('p', info.notice, 'notice'),
    el(
      'p',
      `${info.scope}；模板 ${info.template_requested} → ${info.template_resolved}；MNE ${info.mne_version}；${info.coordinate_frame}坐标系，单位${info.unit}。${info.axes}`,
      'muted small',
    ),
  );
  const exportLink = el('a', '下载当前Montage映射CSV');
  const columns = Object.keys(info.rows[0] || {});
  const cell = (v) => '"' + String(v ?? '').replaceAll('"', '""') + '"';
  const csv =
    '\uFEFF' +
    [columns, ...info.rows.map((r) => columns.map((k) => r[k]))]
      .map((row) => row.map(cell).join(','))
      .join('\r\n');
  exportLink.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv);
  exportLink.download = 'Montage_当前通道映射.csv';
  details.append(exportLink);
  const search = el('input');
  search.type = 'search';
  search.placeholder = '搜索原始/标准通道，如 T3、P7、PG、eog';
  search.setAttribute('aria-label', 'Montage通道搜索');
  const unresolved = el('input');
  unresolved.type = 'checkbox';
  const label = el('label', '仅看待核对、非EEG或重复点位');
  label.className = 'check';
  label.prepend(unresolved);
  details.append(search, label);
  const holder = el('div');
  details.append(holder);
  function draw() {
    holder.replaceChildren();
    const rows = info.rows
      .filter(
        (r) =>
          (!unresolved.checked || !r['可装配']) &&
          [
            '原始通道',
            '标准通道',
            'MNE类型',
            '头皮区域',
            '映射状态',
            '分析状态',
            '匹配说明',
            '说明',
          ].some((k) =>
            String(r[k] ?? '')
              .toLowerCase()
              .includes(search.value.toLowerCase()),
          ),
      )
      .map((r) => {
        const row = {};
        for (const k of [
          '记录顺序',
          '原始通道',
          '标准通道',
          'MNE类型',
          '映射状态',
          '分析状态',
          '头皮区域',
          'x_m',
          'y_m',
          'z_m',
          '匹配说明',
          '说明',
        ])
          if (k in r) row[k] = r[k] === null ? '未装配' : r[k];
        return row;
      });
    table(holder, rows, `模板通道映射表（${rows.length}项；坐标与原始信号分开记录）`);
  }
  search.oninput = draw;
  unresolved.onchange = draw;
  draw();
  parent.append(details);
}
$('showMontage').onclick = () =>
  run('montageStatus', async () => {
    status('montageStatus', '正在装配模板映射与点位示意图……');
    const result = await request('/api/montage');
    render('montageResult', result, [
      ['映射条目', result.montage_info.counts.total],
      ['可装配模板点位', result.montage_info.counts.mapped],
      ['待核对/不适用', result.montage_info.counts.unresolved],
    ]);
  });
function chartResearchNote(filename, appendix) {
  const note = el('div', undefined, 'chart-research-note');
  if (filename.startsWith('复核_')) {
    note.append(
      el('strong', '质控失败段 · 原始波形复核'),
      el(
        'p',
        '该图为未滤波原始波形，仅按文件头单位换算并保留极值抽稀显示。此段未纳入特征汇总；本组其他合格段的指标不能用于描述此图。',
      ),
    );
    return note;
  }
  let domain,
    tip,
    keys = [];
  if (/时频/.test(filename)) {
    domain = '时频域';
    tip =
      '亮色表示该时段该频率的功率更高。TF_std和TF_cv先汇总频段功率，再统计时间波动；精确值见结尾附录。图中单通道与全脑汇总不是同一统计范围。';
    keys = ['时频域__alpha_TF_std__全脑均值', '时频域__alpha_TF_cv__全脑均值'];
  } else if (/频域/.test(filename)) {
    domain = '频域';
    tip =
      '左侧为PSD，右侧为相对频段功率。谱熵、θ/β及五对α不对称等未直接绘制的指标均在结尾附录；频谱高低不能直接判断抑郁风险。';
    keys = [
      '频域__alpha_relative_power__全脑均值',
      '频域__spectral_entropy__全脑均值',
      '频域__theta_beta_ratio__全脑均值',
    ];
  } else if (/时域|波形/.test(filename)) {
    domain = '时域';
    tip =
      '波形供查看幅度和变化形态；12项时域指标及7种汇总均列在结尾附录。下列为保留通道汇总，不能当作图中单通道的精确读数。';
    keys = ['时域__std__全脑均值', '时域__rms__全脑均值', '时域__peak_to_peak__全脑均值'];
  } else if (/网络|PLV/.test(filename)) {
    domain = '网络';
    tip =
      '矩阵颜色表示电极间PLV；对角线按定义置0。聚类、度、效率、路径与连接强度的六项指标在结尾附录；PLV不等于脑内解剖连接。';
    const band = ['delta', 'theta', 'alpha', 'beta', 'gamma'].find((b) => filename.includes(b));
    if (band)
      keys = [
        `网络__${band}_clustering`,
        `网络__${band}_global_efficiency`,
        `网络__${band}_connectivity_strength`,
      ];
  } else return null;
  note.append(el('strong', `${domain} · 探索性阅读提示`), el('p', tip));
  const selected = keys
    .map((key) => appendix.rows.find((row) => row['特征'] === key))
    .filter(Boolean);
  for (const row of selected) {
    const value =
      '数值' in row
        ? researchNumber(row['数值'])
        : `A ${researchNumber(row['记录A'])} / B ${researchNumber(row['记录B'])} / 差值 ${researchNumber(row['变化量_B减A'])}`;
    note.append(el('p', `${row['特征']}：${value} ${row['单位']}`, 'small'));
  }
  return note;
}
function renderResearchAppendix(root, result) {
  const appendix = result.research_appendix,
    section = el('section', undefined, 'card research-appendix');
  section.setAttribute('aria-label', '论文探索性分析附录');
  section.append(
    el('h2', '论文探索性分析附录'),
    el('p', appendix.notice, 'notice'),
    el(
      'p',
      `方案 ${appendix.profile} · v${appendix.pipeline_version} · ${appendix.context}`,
      'muted',
    ),
  );
  const valid = appendix.rows.filter((r) =>
    '数值' in r ? r['数值'] !== null : r['记录A'] !== null && r['记录B'] !== null,
  ).length;
  section.append(
    el(
      'p',
      `本次共${appendix.rows.length}项：${Object.entries(appendix.counts)
        .map(([k, v]) => k + v)
        .join(
          '、',
        )}。${valid}项可用于当前记录的数值查看或配对，${appendix.rows.length - valid}项存在缺失；缺失不补零。未勾选的特征域不纳入本表。`,
      'muted',
    ),
  );
  const downloads = el('div', undefined, 'downloads');
  for (const f of (result.files || []).filter(
    (f) =>
      /探索性分析附录\.(html|csv)$/.test(f.name) &&
      (result.kind !== 'comparison' || f.name.includes('_配对_探索性分析附录')),
  )) {
    const a = el('a', f.name.endsWith('.html') ? '下载可打印HTML附录' : '下载完整CSV附录');
    a.href = f.url;
    a.download = f.name;
    downloads.append(a);
  }
  section.append(
    downloads,
    el(
      'p',
      'HTML附录下载后可独立打开，按 Ctrl+P 打印或另存为PDF。图表只展示部分维度，下表覆盖全部已选特征及公式、单位、探索用途和解释界限。',
      'muted small',
    ),
  );
  const search = el('input');
  search.type = 'search';
  search.placeholder = '搜索全部附录：中文指标、alpha、FAA、脑区或公式';
  search.setAttribute('aria-label', '探索性附录搜索');
  const missing = el('input');
  missing.type = 'checkbox';
  const label = el('label', '仅显示存在缺失的指标');
  label.className = 'check';
  label.prepend(missing);
  section.append(search, label);
  const container = el('div');
  section.append(container);
  function draw() {
    container.replaceChildren();
    for (const domain of ['时域', '频域', '时频域', '网络']) {
      const rows = appendix.rows.filter(
        (r) =>
          r['特征域'] === domain &&
          Object.values(r).some((v) =>
            String(v ?? '')
              .toLowerCase()
              .includes(search.value.toLowerCase()),
          ) &&
          (!missing.checked ||
            ('数值' in r ? r['数值'] === null : r['记录A'] === null || r['记录B'] === null)),
      );
      if (!rows.length) continue;
      const details = el('details');
      details.open = true;
      details.append(el('summary', `${domain}附录 · ${rows.length}项`));
      const displayRows = rows.map((r) => {
        const row = { 特征: r['特征'], 中文说明: r['中文说明'], 单位: r['单位'] };
        for (const key of ['数值', '记录A', '记录B', '变化量_B减A'])
          if (key in r) row[key] = researchNumber(r[key]);
        Object.assign(row, {
          公式或定义: r['公式或定义'],
          探索用途: r['探索用途'],
          解释界限: r['解释界限'],
        });
        for (const key of ['缺失原因', 'A缺失原因', 'B缺失原因']) if (key in r) row[key] = r[key];
        return row;
      });
      table(details, displayRows, `${domain}完整特征表`);
      container.append(details);
    }
    if (!container.childElementCount) container.append(el('p', '没有符合筛选条件的特征。'));
  }
  search.oninput = draw;
  missing.onchange = draw;
  draw();
  root.append(section);
}
