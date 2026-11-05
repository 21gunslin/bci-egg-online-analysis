/* Folder discovery reads paths and small BDF headers only; no automatic analysis. */
(function (root) {
  'use strict';
  function relativeParts(file) {
    const path = String(file.webkitRelativePath || file.relativePath || '').replace(/\\/g, '/');
    if (!path || path.startsWith('/') || /^[A-Za-z]:/.test(path)) return null;
    const parts = path.split('/');
    if (parts.some((p) => !p || p === '.' || p === '..')) return null;
    return parts;
  }
  function indexFolderFiles(files, format = 'bdf') {
    if (!['bdf', 'fif'].includes(format)) throw new Error('必须选择一种格式：BDF或FIF。');
    const target = format === 'fif' ? 'fdata.fif' : 'data.bdf';
    const entries = [],
      paths = new Set(),
      companionRoots = new Set();
    for (const file of files) {
      const parts = relativeParts(file);
      if (!parts) continue;
      const path = parts.join('/');
      if (paths.has(path)) throw new Error('文件夹内出现重复相对路径，请重新选择。');
      paths.add(path);
      const name = parts[parts.length - 1].toLowerCase();
      const directory = parts.slice(0, -1).join('/');
      entries.push({ file, parts, path, name, directory });
      if (['evt.bdf', 'datainfo.json', 'recordinfo.json'].includes(name))
        companionRoots.add(directory);
    }
    const records = new Map();
    for (const entry of entries) {
      if (
        entry.name !== target ||
        entry.parts.length < 3 ||
        !/^\d+$/.test(entry.parts[entry.parts.length - 2])
      )
        continue;
      let key = entry.parts.slice(0, -2).join('/');
      // Use the nearest record-level companion directory for deeper numeric nesting.
      for (let count = entry.parts.length - 2; count >= 1; count--) {
        if (count < entry.parts.length - 2 && !/^\d+$/.test(entry.parts[count])) break;
        const candidate = entry.parts.slice(0, count).join('/');
        if (companionRoots.has(candidate)) {
          key = candidate;
          break;
        }
      }
      const relativeTitle = key.split('/').slice(1).join('/') || key;
      const title = relativeTitle.replace(
        /_[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i,
        '',
      );
      if (!records.has(key))
        records.set(key, { key, title, data: [], aux: { evt: [], datainfo: [], recordinfo: [] } });
      records.get(key).data.push({
        file: entry.file,
        path: entry.path,
        number: Number(entry.parts[entry.parts.length - 2]),
      });
    }
    for (const record of records.values()) {
      for (const entry of entries) {
        if (entry.directory !== record.key) continue;
        const kind = {
          'evt.bdf': 'evt',
          'datainfo.json': 'datainfo',
          'recordinfo.json': 'recordinfo',
        }[entry.name];
        if (kind) record.aux[kind].push(entry.file);
      }
      record.data.sort((a, b) => a.path.localeCompare(b.path, 'zh-CN', { numeric: true }));
    }
    const result = [...records.values()].sort((a, b) =>
      a.title.localeCompare(b.title, 'zh-CN', { numeric: true }),
    );
    return {
      records: result,
      totalFiles: files.length,
      dataFiles: result.reduce((n, r) => n + r.data.length, 0),
    };
  }
  async function queueItem(file, path, format = 'bdf', fifHeaderReader = null) {
    if (!file || !file.size || file.size > 2 * 1024 ** 3)
      throw new Error('脑电文件为空或超过2 GB。');
    if (!['bdf', 'fif'].includes(format) || !file.name.toLowerCase().endsWith('.' + format))
      throw new Error('文件必须与勾选格式一致，禁止混合BDF/FIF。');
    if (format === 'fif') {
      if (!fifHeaderReader) throw new Error('FIF需通过本机MNE读取头信息。');
      const h = await fifHeaderReader(file);
      let timestamp = null;
      const d = /^(\d{2})\.(\d{2})\.(\d{2})$/.exec(h.start_date),
        t = /^(\d{2})\.(\d{2})\.(\d{2})$/.exec(h.start_time);
      if (d && t)
        timestamp =
          Date.UTC(
            Number(d[3]) + (Number(d[3]) >= 85 ? 1900 : 2000),
            Number(d[2]) - 1,
            Number(d[1]),
            Number(t[1]),
            Number(t[2]),
            Number(t[3]),
          ) +
          (h.first_time || 0) * 1000;
      return {
        file,
        path: path || file.webkitRelativePath || file.name,
        clock:
          timestamp === null
            ? 'FIF未保存可靠时间，须核对元数据'
            : new Date(timestamp).toISOString().replace('T', ' ').replace('Z', '（源时钟）'),
        timestamp,
        duration: h.duration,
      };
    }
    const bytes = await file.slice(0, 256).arrayBuffer();
    if (bytes.byteLength < 256) throw new Error('所选文件头不足256字节。');
    const header = new TextDecoder('ascii').decode(bytes),
      date = header.slice(168, 176).trim(),
      time = header.slice(176, 184).trim();
    let timestamp = null;
    const dm = /^(\d{2})\.(\d{2})\.(\d{2})$/.exec(date),
      tm = /^(\d{2})\.(\d{2})\.(\d{2})$/.exec(time);
    if (dm && tm) {
      const d = Number(dm[1]),
        m = Number(dm[2]),
        y = Number(dm[3]) + (Number(dm[3]) >= 85 ? 1900 : 2000),
        h = Number(tm[1]),
        min = Number(tm[2]),
        s = Number(tm[3]);
      const value = Date.UTC(y, m - 1, d, h, min, s),
        v = new Date(value);
      if (
        v.getUTCFullYear() === y &&
        v.getUTCMonth() === m - 1 &&
        v.getUTCDate() === d &&
        h < 24 &&
        min < 60 &&
        s < 60
      )
        timestamp = value;
    }
    const duration = Number(header.slice(236, 244).trim()) * Number(header.slice(244, 252).trim());
    return {
      file,
      path: path || file.webkitRelativePath || file.name,
      clock: date + ' ' + time,
      timestamp,
      duration: Number.isFinite(duration) && duration >= 0 ? duration : null,
    };
  }
  function sortQueue(items) {
    return items.slice().sort((a, b) => {
      if (a.timestamp !== null && b.timestamp !== null && a.timestamp !== b.timestamp)
        return a.timestamp - b.timestamp;
      if (a.timestamp === null && b.timestamp !== null) return 1;
      if (a.timestamp !== null && b.timestamp === null) return -1;
      return String(a.path || a.file.name).localeCompare(String(b.path || b.file.name), 'zh-CN', {
        numeric: true,
      });
    });
  }
  function matchesRecord(record, query) {
    const title = record.title.toLowerCase();
    return String(query)
      .trim()
      .toLowerCase()
      .split(/\s+/)
      .every((term) => title.includes(term));
  }
  const api = { indexFolderFiles, queueItem, sortQueue, matchesRecord };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.EEGFolderImport = api;
})(typeof window === 'undefined' ? null : window);
