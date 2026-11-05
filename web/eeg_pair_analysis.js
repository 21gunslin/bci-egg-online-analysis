// 镇痛前后对比：先分别分析A/B，再用共同通道比较。
// 阅读顺序见 docs/CODE_READING_GUIDE.md；使用原生JavaScript，无需前端框架。
async function compare(channel) {
  status('pairStatus', '正在重算共同通道特征并绘制对比图……');
  const r = await post('/api/compare', { ...activePair, channel });
  const card = render('pairResult', r, [
    ['共同通道', r.metrics.common_channels],
    ['每份有效时长 s', r.metrics.duration_seconds],
    ['统一采样率 Hz', r.metrics.sampling_rate],
    ['时域/时频通道', r.metrics.channel],
  ]);
  card.append(
    el(
      'p',
      `A：${r.label_a}；B：${r.label_b}。全部变化量为 B−A。相对变化分母为 |A|，接近零时留空。`,
      'muted',
    ),
  );
  card.append(
    el(
      'p',
      `原始质控保留：A ${r.quality_a.epoch_count}/${r.quality_a.epoch_total} 片段（${fmt(r.quality_a.keep_ratio * 100)}%）；B ${r.quality_b.epoch_count}/${r.quality_b.epoch_total} 片段（${fmt(r.quality_b.keep_ratio * 100)}%）。`,
      'muted',
    ),
  );
  const select = $('pairChannel');
  select.replaceChildren(
    ...r.labels.map((l) => {
      const o = el('option', l);
      o.value = l;
      return o;
    }),
  );
  select.value = r.metrics.channel;
  $('pairControls').classList.remove('hide');
  const filter = el('select');
  ['全部', '时域', '频域', '时频域', '网络', '汇总'].forEach((v) => {
    const o = el('option', v);
    o.value = v;
    filter.append(o);
  });
  const search = el('input');
  search.placeholder = '搜索特征名';
  search.setAttribute('aria-label', '搜索特征名');
  filter.setAttribute('aria-label', '筛选特征域');
  const row = el('div', undefined, 'row');
  row.append(filter, search);
  card.append(row);
  const holder = el('div');
  card.append(holder);
  function update() {
    const rows = r.rows.filter(
      (x) =>
        (filter.value === '全部' || x['特征域'] === filter.value) &&
        x['特征'].toLowerCase().includes(search.value.toLowerCase()),
    );
    holder.replaceChildren();
    table(holder, rows, '配对特征变化（' + rows.length + ' 项）');
  }
  filter.onchange = update;
  search.oninput = update;
  update();
}
function handlePairSubmit(e) {
  e.preventDefault();
  run('pairStatus', async () => {
    const a = $('pairFileA').files[0],
      b = $('pairFileB').files[0];
    checkFormat(a, inputFormat('pair'));
    checkFormat(b, inputFormat('pair'));
    if ($('pairUseEvt').checked && !$('pairEvtA').files[0] && !$('pairEvtB').files[0])
      throw new Error('已勾选事件标记，请至少选择 A 或 B 的 evt.bdf。');
    activePair = null;
    $('pairControls').classList.add('hide');
    const opts = { ...settings('pair'), cache_for_comparison: true };
    status('pairStatus', '步骤 1/3：分析 A 记录……');
    const ra = await analyzeSingleFile(a, { ...opts, ...(await eventOptions('pair', 'A')) });
    status('pairStatus', '步骤 2/3：分析 B 记录……');
    const rb = await analyzeSingleFile(b, { ...opts, ...(await eventOptions('pair', 'B')) });
    activePair = {
      run_a: ra.run_id,
      run_b: rb.run_id,
      label_a: $('pairLabelA').value,
      label_b: $('pairLabelB').value,
      max_seconds: Number($('pairSeconds').value),
    };
    await compare(null);
  });
}
$('pairForm').onsubmit = handlePairSubmit;
$('pairRedraw').onclick = () => run('pairStatus', () => compare($('pairChannel').value));
