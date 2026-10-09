const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'vision-inspector-test-'));
process.env.VISION_DATA_DIR = dataDir;
const { app, interpret, splitTranslationText } = require('../server');
const nativeFetch = global.fetch;
let server, origin;
before(async () => {
  server = await new Promise((resolve) => {
    const listener = app.listen(0, '127.0.0.1', () => resolve(listener));
  });
  origin = `http://127.0.0.1:${server.address().port}`;
});
after(async () => {
  global.fetch = nativeFetch;
  await new Promise((resolve) => server.close(resolve));
  fs.rmSync(dataDir, { recursive: true, force: true });
});
const request = (url, method = 'GET', body) => nativeFetch(origin + url, {
  method, headers: { 'Content-Type': 'application/json' },
  ...(body === undefined ? {} : { body: JSON.stringify(body) }),
});

test('interpretation handles prototype names without corrupting counts', () => {
  const result = interpret([{ class: '__proto__', score: 0.9 }, { class: 'constructor', score: 0.5 }]);
  assert.equal(result.counts.__proto__, 1);
  assert.equal(result.counts.constructor, 1);
  assert.equal(result.totalObjects, 2);
  assert.equal(result.avgConfidence, 0.7);
  assert.equal(interpret([{ class: 'person', score: 0.9 }]).sceneId, 'general');
});

test('API validates malformed requests and returns JSON errors', async () => {
  assert.equal((await request('/api/health')).status, 200);
  assert.equal((await request('/api/interpret', 'POST', {})).status, 400);
  const invalid = await nativeFetch(origin + '/api/scans', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{',
  });
  assert.equal(invalid.status, 400);
  assert.match((await invalid.json()).error, /JSON/);
  assert.equal((await request('/api/missing')).status, 404);
  assert.equal((await request('/missing.js')).status, 404);
  const result = await (await request('/api/interpret', 'POST', {
    detections: [{ class: 'DOG', score: 4 }, null, { class: '  ', score: 1 }],
    classification: [{ label: 'wrong', prob: 0.1 }, { label: 'dog', prob: 0.9 }],
  })).json();
  assert.equal(result.counts.dog, 1);
  assert.equal(result.avgConfidence, 1);
  assert.match(result.summary, /Primary subject: dog/);
});

test('history round trip sanitizes nested data and deletes only requested records', async () => {
  assert.equal((await request('/api/scans', 'POST', {})).status, 400);
  const scan = await (await request('/api/scans', 'POST', {
    source: 'upload', scene: 'Animals', objects: -20,
    faces: [null, { age: -3, gender: 'test', expression: 'neutral', extra: '<script>' }],
    thumb: 'javascript:alert(1)',
  })).json();
  assert.equal(scan.objects, 0);
  assert.equal(scan.thumb, '');
  assert.deepEqual(scan.faces, [{ age: 0, gender: 'test', expression: 'neutral' }]);
  const second = await (await request('/api/scans', 'POST', { scene: 'Second scan' })).json();
  assert.equal((await (await request('/api/scans')).json()).length, 2);
  assert.ok(fs.existsSync(path.join(dataDir, 'scans.json')));
  await request('/api/scans/' + scan.id, 'DELETE');
  assert.equal((await (await request('/api/scans')).json())[0].id, second.id);
  await request('/api/scans', 'DELETE');
  assert.deepEqual(await (await request('/api/scans')).json(), []);
});

test('UTF-8 translation chunks retain all text and fit the service byte limit', () => {
  const text = 'Bonjour café 世界 🌎 '.repeat(70).trim();
  const chunks = splitTranslationText(text);
  assert.ok(chunks.length > 1);
  for (const chunk of chunks) assert.ok(Buffer.byteLength(chunk) <= 450);
  assert.equal(chunks.join(' ').replace(/\s+/g, ' '), text);
});

test('translation handles English, long text, provider failure and quota errors', async () => {
  const calls = [];
  global.fetch = async (url) => {
    const query = new URL(url).searchParams;
    calls.push(query.get('q'));
    return { ok: true, json: async () => ({ responseStatus: 200, responseData: {
      translatedText: query.get('q'), detectedLanguage: 'fr',
    } }) };
  };
  try {
    assert.equal((await request('/api/translate', 'POST', { text: '  ' })).status, 400);
    const english = await (await request('/api/translate', 'POST', { text: 'Already English', source: 'en' })).json();
    assert.equal(english.translation, 'Already English');
    assert.equal(calls.length, 0);
    const longText = 'Bonjour tout le monde. '.repeat(60).trim();
    const result = await (await request('/api/translate', 'POST', { text: longText, source: 'fr' })).json();
    assert.equal(result.translation, longText);
    assert.ok(calls.length > 1);
    global.fetch = async () => ({ ok: true, json: async () => ({ quotaFinished: true, responseStatus: 403 }) });
    assert.equal((await request('/api/translate', 'POST', { text: 'Bonjour', source: 'fr' })).status, 429);
    global.fetch = async () => ({ ok: false });
    assert.equal((await request('/api/translate', 'POST', { text: 'Bonjour', source: 'fr' })).status, 502);
  } finally { global.fetch = nativeFetch; }
});
