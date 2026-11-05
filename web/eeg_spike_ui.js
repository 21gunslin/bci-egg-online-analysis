// 设备spike计数独立入口，候选波形不等于诊断。
// 阅读顺序见 docs/CODE_READING_GUIDE.md；使用原生JavaScript，无需前端框架。
$('spikeForm').onsubmit = (e) => {
  e.preventDefault();
  run('spikeStatus', async () => {
    const marker = $('spikeFile').files[0],
      eeg = $('spikeEeg').files[0];
    check(marker);
    if (eeg) check(eeg);
    status('spikeStatus', '正在读取 spike 设备计数……');
    const spike_upload = await upload(marker, 'spike');
    let eeg_upload = null;
    if (eeg) {
      status('spikeStatus', '正在上传配套 data.bdf……');
      eeg_upload = await upload(eeg, 'eeg');
    }
    status('spikeStatus', '正在计算计数、候选及图表……');
    const evtSettings = await eventOptions('spike');
    const r = await post('/api/spike', {
        spike_upload,
        eeg_upload,
        ...evtSettings,
        scope: $('spikeScope').value,
        threshold_z: Number($('spikeThreshold').value),
        amplitude_scale: Number($('spikeScale').value),
      }),
      m = r.metrics;
    const card = render('spikeResult', r, [
      ['原始时长 s', m.duration_seconds],
      ['设备计数总和', m.device_count],
      ['设备计数 / 分钟', m.device_count_per_min],
      ['计数非零秒数', m.marked_seconds],
      ['去重候选事件', m.candidate_events],
      ['候选事件 / 分钟', m.candidate_rate_per_min],
      ['有慢波的候选事件', m.slow_wave_candidates],
    ]);
    if (m.candidate_events === null) {
      card.append(el('p', '未提供原始 EEG：出现通道、头皮区域与后续慢波均无法评估。'));
    } else {
      card.append(
        el(
          'p',
          '发生率按整个记录时长计算；头皮区按通道候选计数，跨通道可能重复。未检出候选不等于排除棘波。',
          'muted',
        ),
      );
      table(card, r.regions, '对应头皮区分布');
      table(card, r.events, '去重候选事件（最多显示 200 行）');
      const details = el('details');
      details.append(el('summary', '展开通道级候选明细（最多 500 行）'));
      table(details, r.rows, '通道级候选');
      card.append(details);
      if (r.table_truncated) card.append(el('p', '页面仅展示部分行，完整记录请下载 CSV。'));
    }
  });
};
