const test = require('node:test'),
  assert = require('node:assert/strict');
const M = require('../web/eeg_patrol_math.js');
test('partial review is partial, not whole-bin clearance', () => {
  const e = [{ channels: ['F3'], start: 0, end: 1, status: 'reviewed' }];
  assert.equal(M.cellFraction(e, 'F3', 0, 2), 0.5);
  assert.equal(M.coverage(e, ['F3', 'O1'], 10).percent, 5);
});
test('overlap and repeat registrations never inflate coverage', () => {
  const e = [
    { channels: ['F3'], start: 0, end: 2 },
    { channels: ['F3'], start: 1, end: 3 },
    { channels: ['F3'], start: 0, end: 2 },
  ];
  assert.equal(M.coverage(e, ['F3'], 10).covered, 3);
  assert.equal(M.cellFraction(e, 'F3', 0, 4), 0.75);
});
test('unreviewed channels remain unreviewed; uncertain is browsing, not cleanliness', () => {
  const e = [{ channels: ['O1'], start: 2, end: 4, status: 'uncertain' }];
  assert.equal(M.cellFraction(e, 'F3', 0, 10), 0);
  assert.equal(M.coverage(e, ['O1'], 10).percent, 20);
});
test('channel pagination is complete and clamps boundaries', () => {
  const a = ['Fp1', 'Fp2', 'F3', 'F4', 'O1'];
  assert.deepEqual(M.pages(a, 4, 0).names, a.slice(0, 4));
  assert.deepEqual(M.pages(a, 4, 9).names, ['O1']);
  assert.equal(M.pages(a, 4, -1).page, 0);
});
