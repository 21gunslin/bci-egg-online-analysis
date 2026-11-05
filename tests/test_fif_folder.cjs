const test = require('node:test'),
  assert = require('node:assert/strict');
const { indexFolderFiles, queueItem } = require('../web/eeg_folder_import.js');
const file = (path) => ({ name: path.split('/').at(-1), webkitRelativePath: path, size: 500 });
test('folder format toggle selects only exact primary fdata.fif inside numeric folders', () => {
  const files = [
    'record/1/data.bdf',
    'record/1/fdata.fif',
    'record/1/fdata_group2.fif',
    'record/2/fdata.fif',
    'record/2/data.bdf',
    'record/evt.bdf',
  ].map(file);
  const fif = indexFolderFiles(files, 'fif');
  assert.equal(fif.records.length, 1);
  assert.equal(fif.dataFiles, 2);
  assert.ok(fif.records[0].data.every((r) => r.file.name === 'fdata.fif'));
  assert.equal(fif.records[0].aux.evt.length, 1);
  assert.equal(indexFolderFiles(files, 'bdf').dataFiles, 2);
  assert.throws(() => indexFolderFiles(files, 'both'));
});
test('manual FIF rejects BDF and vice versa; FIF clock is read via MNE callback', async () => {
  await assert.rejects(() => queueItem(file('1/data.bdf'), '', 'fif'), /禁止混合/);
  await assert.rejects(() => queueItem(file('1/fdata.fif'), '', 'bdf'), /禁止混合/);
  const item = await queueItem(file('1/fdata.fif'), 'record/1/fdata.fif', 'fif', async () => ({
    start_date: '24.04.24',
    start_time: '08.30.00',
    duration: 20,
    first_time: 0.5,
  }));
  assert.equal(item.timestamp, Date.UTC(2024, 3, 24, 8, 30, 0) + 500);
  assert.equal(item.duration, 20);
});
