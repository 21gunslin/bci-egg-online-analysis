// 启动入口：检查本机服务，绑定BDF/FIF互斥选择。业务步骤在各功能文件。
// 阅读顺序见 docs/CODE_READING_GUIDE.md；使用原生JavaScript，无需前端框架。
if (location.protocol === 'file:') {
  $('startupNotice').textContent =
    '当前为文件方式打开，无法连接分析接口。请启动本机服务并访问 http://127.0.0.1:8765/。';
} else {
  request('/health')
    .then((h) => {
      $('startupNotice').textContent =
        '本机服务 v' + h.version + ' 已连接 · 支持多份 data＋同一 evt · 图表点击可放大';
    })
    .catch((e) => {
      $('startupNotice').textContent = e.message;
    });
}

for (const prefix of ['single', 'pair', 'batch'])
  for (const radio of document.querySelectorAll(`[name=${prefix}Format]`))
    radio.addEventListener('change', () => {
      const format = inputFormat(prefix),
        ids =
          prefix === 'single'
            ? ['singleFile']
            : prefix === 'pair'
              ? ['pairFileA', 'pairFileB']
              : ['batchFiles'];
      for (const id of ids) {
        $(id).value = '';
        $(id).accept = '.' + format;
      }
      $(prefix + 'Result').classList.add('hide');
      if (prefix === 'pair') {
        activePair = null;
        $('pairControls').classList.add('hide');
      }
      if (prefix === 'batch') {
        batchQueue = [];
        batchFolderRecords = [];
        folderQueueMode = false;
        $('batchFolder').value = '';
        $('batchFolderChoices').classList.add('hide');
        $('batchConfirm').checked = false;
        drawBatchQueue();
        status(
          'batchFolderStatus',
          '格式已切换，请重新选择文件夹；仅自动导入' +
            (format === 'fif' ? 'fdata.fif' : 'data.bdf') +
            '。',
        );
      }
      status(
        prefix + 'Status',
        '已选择 ' + format.toUpperCase() + '；原已选EEG清空，禁止混合格式。',
      );
    });
