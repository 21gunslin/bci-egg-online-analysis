// 单份分析入口：先看handleSingleSubmit，再跟随analyzeSingleFile找到后端/api/analyze。
// 阅读顺序见 docs/CODE_READING_GUIDE.md；使用原生JavaScript，无需前端框架。
function handleSingleSubmit(event) {
  // 1. 表单本来会直接跳页。先阻止跳页，才能在本页显示异步返回的分析结果。
  event.preventDefault();
  run('singleStatus', async () => {
    // 2. 读到的是“字段名 → 值”的设置对象；现在还没有计算脑电。
    const opts = settings('single');
    opts.features = [...document.querySelectorAll('[name=feature]:checked')].map((n) => n.value);
    opts.auto_eeg_only = $('singleEegOnly').checked;
    if (!opts.features.length) throw new Error('至少选择一个特征类别。');
    status('singleStatus', '正在上传并分析，请等待……');
    // 3. 只有勾选evt时才附加事件参数，事件编码保持原标签。
    Object.assign(opts, await eventOptions('single'));
    // 4. 文件交给本机Python。await表示等待完成，特征公式在Python里计算。
    const result = await analyzeSingleFile($('singleFile').files[0], opts);
    // 5. 返回对象含指标和图片链接，render负责显示它们。
    render('singleResult', result, [
      ['原始采样率 Hz', result.metrics.sampling_rate],
      ['分析通道', result.metrics.analysis_channels],
      ['原始时长 s', result.metrics.duration_seconds],
      ['保留片段', `${result.metrics.kept_epochs} / ${result.metrics.total_epochs}`],
      ['片段保留率', `${fmt(result.metrics.keep_ratio * 100)}%`],
    ]);
    if (result.review_only) return '仅复核：FIF质控未通过，已显示原始标记图片，未计算特征。';
  });
}
$('singleForm').onsubmit = handleSingleSubmit;
