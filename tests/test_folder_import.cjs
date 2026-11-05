'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const {
  indexFolderFiles,
  queueItem,
  sortQueue,
  matchesRecord,
} = require('../web/eeg_folder_import.js');
function file(path, time = '08.30.00', date = '01.01.24') {
  const header = Buffer.alloc(256, 32);
  header.write(date, 168);
  header.write(time, 176);
  header.write('8       ', 236);
  header.write('1       ', 244);
  return {
    name: path.split('/').pop(),
    webkitRelativePath: path,
    size: 3000,
    slice: (start, end) => {
      assert.equal(start, 0);
      assert.equal(end, 256);
      return { arrayBuffer: async () => Uint8Array.from(header).buffer };
    },
  };
}
test('single named record detects numeric children and root companions only', () => {
  const files = [
    'record/11/data.bdf',
    'record/3/DATA.BDF',
    'record/1/data.bdf',
    'record/evt.bdf',
    'record/datainfo.json',
    'record/RecordInfo.json',
    'record/1/spike.bdf',
    'record/video/1.mp4',
    'record/ExamInfo.json',
    'record/notes/data.bdf',
  ].map((p) => file(p));
  const result = indexFolderFiles(files);
  assert.equal(result.records.length, 1);
  assert.equal(result.dataFiles, 3);
  const record = result.records[0];
  assert.deepEqual(
    record.data.map((r) => r.number),
    [1, 3, 11],
  );
  assert.equal(record.aux.evt.length, 1);
  assert.equal(record.aux.datainfo.length, 1);
  assert.equal(record.aux.recordinfo.length, 1);
});
test('parent collection preserves each person and each recording separately', () => {
  const paths = [
    'collection/001_A_before/1/data.bdf',
    'collection/001_A_before/evt.bdf',
    'collection/001_A_after/1/data.bdf',
    'collection/001_A_after/evt.bdf',
    'collection/002_B/5/data.bdf',
    'collection/002_B/evt.bdf',
  ];
  const result = indexFolderFiles(paths.map((p) => file(p)));
  assert.equal(result.records.length, 3);
  assert.ok(result.records.every((r) => r.data.length === 1 && r.aux.evt.length === 1));
});
test('does not borrow auxiliaries from a different record or collection root', () => {
  const result = indexFolderFiles(
    [
      'collection/datainfo.json',
      'collection/A/1/data.bdf',
      'collection/B/1/data.bdf',
      'collection/B/evt.bdf',
    ].map((p) => file(p)),
  );
  assert.equal(result.records.length, 2);
  const a = result.records.find((r) => r.key === 'collection/A');
  assert.equal(a.aux.evt.length, 0);
  assert.equal(a.aux.datainfo.length, 0);
});
test('nested numeric folders resolve to nearest companion root', () => {
  const result = indexFolderFiles(
    ['record/1/3/data.bdf', 'record/1/5/data.bdf', 'record/evt.bdf'].map((p) => file(p)),
  );
  assert.equal(result.records.length, 1);
  assert.equal(result.records[0].key, 'record');
  assert.equal(result.dataFiles, 2);
});
test('invalid paths and nonnumeric data location are not imported', () => {
  const result = indexFolderFiles(
    [
      'record/../1/data.bdf',
      '/record/1/data.bdf',
      'C:/record/1/data.bdf',
      'record/data.bdf',
      'record/notes/data.bdf',
    ].map((p) => file(p)),
  );
  assert.equal(result.records.length, 0);
});
test('duplicate relative paths produce an explicit error', () => {
  assert.throws(
    () => indexFolderFiles([file('record/1/data.bdf'), file('record/1/data.bdf')]),
    /重复/,
  );
});
test('header chronology overrides numeric-folder order and reads only 256 bytes', async () => {
  const a = await queueItem(file('record/1/data.bdf', '08.30.12'));
  const b = await queueItem(file('record/11/data.bdf', '08.30.00'));
  assert.deepEqual(
    sortQueue([a, b]).map((i) => i.path),
    ['record/11/data.bdf', 'record/1/data.bdf'],
  );
  assert.equal(a.duration, 8);
});
test('invalid dates remain unknown rather than silently normalizing', async () => {
  const a = await queueItem(file('record/1/data.bdf', '25.00.00'));
  const b = await queueItem(file('record/3/data.bdf', '08.30.00', '31.02.24'));
  const valid = await queueItem(file('record/5/data.bdf'));
  assert.equal(a.timestamp, null);
  assert.equal(b.timestamp, null);
  assert.equal(sortQueue([a, valid, b])[0], valid);
});
test('manual files without relative paths still produce editable queue items', async () => {
  const f = file('manual.bdf');
  f.webkitRelativePath = '';
  const item = await queueItem(f);
  assert.equal(item.path, 'manual.bdf');
  assert.ok(item.timestamp);
});
test('readable record title omits common collection prefix and trailing UUID', () => {
  const scan = indexFolderFiles([
    file('collection/001_A_2024_01_01_08_30_00_abcdef12-abcd-abcd-abcd-abcdef123456/1/data.bdf'),
  ]);
  assert.equal(scan.records[0].title, '001_A_2024_01_01_08_30_00');
  assert.ok(scan.records[0].key.includes('abcdef12'));
});
test('record search accepts ID, name and multiple time terms', () => {
  const record = { title: '001_演示_2024_01_01_08_30_00' };
  assert.equal(matchesRecord(record, '001 演示 08_30'), true);
  assert.equal(matchesRecord(record, '002'), false);
  assert.equal(matchesRecord(record, ''), true);
});
