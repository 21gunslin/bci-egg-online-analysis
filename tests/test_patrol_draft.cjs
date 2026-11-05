const test = require('node:test'),
  assert = require('node:assert/strict');
const M = require('../web/eeg_patrol_math.js');
const fixture = () => ({
  duration: 20,
  channels: [
    { name: 'Fp1', eeg: true },
    { name: 'F3', eeg: true },
    { name: 'O1', eeg: true },
  ],
  candidates: [
    { channel: 'Fp1', reason: '平线', persistent: true, start: 0, end: 20 },
    { channel: 'Fp1', reason: '饱和候选', persistent: true, start: 0, end: 20 },
    { channel: 'F3', reason: '跳变', persistent: false, start: 4, end: 6, focus_time: 5 },
    { channel: 'O1', reason: '幅度离群', persistent: false, start: 4, end: 6, focus_time: 5 },
  ],
});
test('persistent channel prompts become names; transient bounds are exact and same-time reasons combined', () => {
  const d = M.buildDraft(fixture());
  assert.equal(d.badsText, 'Fp1');
  assert.match(d.segmentsText, /^4,6,待核候选_/);
  assert.match(d.segmentsText, /F3:跳变/);
  assert.match(d.segmentsText, /O1:幅度离群/);
  assert.equal(d.addedBads, 1);
  assert.equal(d.addedSegments, 1);
  assert.equal(d.allEegBad, false);
});
test('manual text is preserved and repeated clicks are idempotent', () => {
  const d = M.buildDraft(fixture(), 'O1\n', '1,2,manual');
  assert.ok(d.badsText.startsWith('O1\n'));
  assert.ok(d.segmentsText.startsWith('1,2,manual\n'));
  const again = M.buildDraft(fixture(), d.badsText, d.segmentsText);
  assert.equal(again.badsText, d.badsText);
  assert.equal(again.segmentsText, d.segmentsText);
  assert.equal(again.addedBads, 0);
  assert.equal(again.addedSegments, 0);
});
test('persistent gap envelope is never copied as a full-record global bad interval', () => {
  const scan = fixture();
  scan.candidates = scan.candidates.slice(0, 2);
  const d = M.buildDraft(scan);
  assert.equal(d.segmentsText, '');
});
test('manual coverage prevents duplicate intervals and reason delimiters stay within three columns', () => {
  const scan = fixture();
  scan.candidates[2].reason = 'a,b\nc';
  let d = M.buildDraft(scan, '', '0,10,manual');
  assert.equal(d.addedSegments, 0);
  d = M.buildDraft(scan);
  assert.equal(d.segmentsText.split(',').length, 3);
  assert.equal(d.segmentsText.split('\n').length, 1);
});
test('a local transient is still retained when its channel also has a persistent prompt', () => {
  const scan = fixture();
  scan.candidates.push({ channel: 'F3', persistent: true, reason: '工频', start: 0, end: 20 });
  const d = M.buildDraft(scan);
  assert.match(d.badsText, /F3/);
  assert.match(d.segmentsText, /4,6/);
});
test('all EEG marked bad is reported, empty scan preserves manual entries', () => {
  const scan = fixture();
  let d = M.buildDraft(scan, 'F3,O1');
  assert.equal(d.allEegBad, true);
  scan.candidates = [];
  d = M.buildDraft(scan, 'O1', '3,4,user');
  assert.equal(d.badsText, 'O1');
  assert.equal(d.segmentsText, '3,4,user');
  assert.equal(d.addedBads, 0);
});
test('invalid time or foreign channel rejects the entire proposal; input scan is immutable', () => {
  const scan = fixture(),
    original = JSON.stringify(scan);
  M.buildDraft(scan);
  assert.equal(JSON.stringify(scan), original);
  scan.candidates[2].end = 30;
  assert.throws(() => M.buildDraft(scan), /时间/);
  scan.candidates[2].channel = 'unknown';
  assert.throws(() => M.buildDraft(scan), /通道/);
});
