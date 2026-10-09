/**
 * AI Vision Inspector — Backend server
 * -------------------------------------
 * Responsibilities:
 *   1. Serve the static frontend (public/).
 *   2. Expose an interpretation API that turns raw object detections
 *      (produced client-side by the COCO-SSD model) into a scene
 *      classification + human-useful recommendations.
 *   3. Expose a health endpoint for quick checks.
 *
 * Why the AI runs client-side: COCO-SSD via TensorFlow.js runs in the
 * browser, so detection needs no API keys and no heavy native builds.
 * The backend adds the "intelligence on top of raw labels" — scene
 * reasoning and actionable recommendations — which is where a server is
 * genuinely useful and easy to extend.
 */

const express = require('express');
const path = require('path');
const fs = require('fs');
const { randomUUID } = require('crypto');

const app = express();
const PORT = process.env.PORT || 3000;

// --- Middleware -------------------------------------------------------------
app.use(express.json({ limit: '4mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// --- Tiny file-backed database (scan history) -------------------------------
// Zero external dependencies: a JSON file under data/. Good enough to persist
// every inspection (uploads, webcam captures, live snapshots) across restarts.
const DATA_DIR = process.env.VISION_DATA_DIR || path.join(__dirname, 'data');
const DB_FILE = path.join(DATA_DIR, 'scans.json');
function loadDB() {
  try {
    const rows = JSON.parse(fs.readFileSync(DB_FILE, 'utf8'));
    if (!Array.isArray(rows)) throw new Error('Scan history must be an array.');
    return rows.filter((row) => row && typeof row === 'object' && typeof row.id === 'string');
  } catch (err) {
    if (err.code === 'ENOENT') return [];
    throw err; // Preserve damaged history instead of silently overwriting it.
  }
}
function saveDB(rows) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
  const temporary = `${DB_FILE}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, JSON.stringify(rows));
  fs.renameSync(temporary, DB_FILE);
}

// --- Scene reasoning engine -------------------------------------------------
// Each scene is defined by "signal" objects. We score every scene by how many
// of its signals appear in the detections, weighted by detection confidence,
// then pick the best-scoring scene.
const SCENES = [
  {
    id: 'classroom',
    label: 'Classroom / Lecture',
    signals: ['person', 'chair', 'book', 'laptop', 'tv', 'backpack'],
    describe: (c) =>
      `Looks like a learning space: ${c.person || 0} people, ${c.chair || 0} seats` +
      `${c.laptop ? `, ${c.laptop} laptop(s)` : ''}${c.book ? `, ${c.book} book(s)` : ''}.`,
    recommend: (c) => {
      const recs = [];
      if ((c.person || 0) > (c.chair || 0) && c.chair)
        recs.push('More people than visible seats — check seating capacity.');
      if (c.chair && (c.person || 0) < c.chair)
        recs.push(`${c.chair - (c.person || 0)} seat(s) appear empty — room is under capacity.`);
      if (c.laptop || c.book) recs.push('Active study/work environment detected.');
      if (!recs.length) recs.push('Standard classroom layout detected.');
      return recs;
    },
  },
  {
    id: 'office',
    label: 'Office / Workspace',
    signals: ['laptop', 'keyboard', 'mouse', 'tv', 'chair', 'cup', 'cell phone'],
    describe: (c) =>
      `Workspace detected: ${c.laptop || 0} laptop(s), ` +
      `${c.keyboard || 0} keyboard(s), ${c.chair || 0} chair(s).`,
    recommend: (c) => {
      const recs = [];
      if (c.cup || c.bottle) recs.push('Beverages present — remind to keep liquids away from devices.');
      if ((c.laptop || 0) >= 2) recs.push('Multi-device setup — verify cable management / ergonomics.');
      if (c['cell phone']) recs.push('Phone on desk — possible focus distraction.');
      if (!recs.length) recs.push('Tidy single-user workstation.');
      return recs;
    },
  },
  {
    id: 'meeting',
    label: 'Meeting / Collaboration',
    signals: ['person', 'laptop', 'chair', 'cup', 'tv', 'dining table'],
    minPeople: 3,
    describe: (c) =>
      `Group setting: ${c.person || 0} people gathered` +
      `${c.laptop ? ` with ${c.laptop} laptop(s)` : ''}.`,
    recommend: (c) => {
      const recs = [`${c.person || 0} participants detected.`];
      if (c.tv) recs.push('Display/screen present — likely a presentation in progress.');
      if (!c.laptop) recs.push('No laptops visible — may be a discussion-only session.');
      return recs;
    },
  },
  {
    id: 'traffic',
    label: 'Street / Traffic',
    signals: ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'traffic light', 'person', 'stop sign'],
    describe: (c) => {
      const vehicles = (c.car || 0) + (c.truck || 0) + (c.bus || 0) + (c.motorcycle || 0);
      return `Outdoor/traffic scene: ${vehicles} vehicle(s)` +
        `${c.person ? `, ${c.person} pedestrian(s)` : ''}.`;
    },
    recommend: (c) => {
      const recs = [];
      const vehicles = (c.car || 0) + (c.truck || 0) + (c.bus || 0);
      if (vehicles >= 5) recs.push('High vehicle density — possible congestion.');
      if (c.person && vehicles) recs.push('Pedestrians near vehicles — flag for safety monitoring.');
      if (c.bicycle || c.motorcycle) recs.push('Two-wheelers present — watch for vulnerable road users.');
      if (!recs.length) recs.push('Light traffic conditions.');
      return recs;
    },
  },
  {
    id: 'kitchen',
    label: 'Kitchen / Dining',
    signals: ['dining table', 'cup', 'bottle', 'bowl', 'fork', 'knife', 'spoon', 'wine glass', 'microwave', 'oven', 'refrigerator', 'sink', 'banana', 'apple', 'pizza'],
    describe: (c) =>
      `Food/dining area detected with ${Object.keys(c).length} item type(s).`,
    recommend: (c) => {
      const recs = [];
      if (c.knife || c.fork) recs.push('Utensils detected — table appears set for a meal.');
      if (c.refrigerator || c.microwave || c.oven) recs.push('Appliances present — this is a kitchen.');
      if (!recs.length) recs.push('Dining items detected.');
      return recs;
    },
  },
  {
    id: 'living_room',
    label: 'Living Room',
    signals: ['couch', 'tv', 'remote', 'potted plant', 'chair', 'book', 'vase', 'clock'],
    describe: (c) => `Relaxation space: ${c.couch ? 'couch' : 'seating'}${c.tv ? ' + TV' : ''} detected.`,
    recommend: (c) => {
      const recs = [];
      if (c.tv && c.remote) recs.push('Entertainment setup detected.');
      if (c['potted plant']) recs.push('Greenery present — pleasant ambiance.');
      if (!recs.length) recs.push('Comfortable indoor living space.');
      return recs;
    },
  },
  {
    id: 'animals',
    label: 'Animals / Pets',
    signals: ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
    describe: (c) => {
      const animals = Object.keys(c).filter((k) =>
        ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'].includes(k));
      return `Animal(s) detected: ${animals.join(', ')}.`;
    },
    recommend: () => ['Animal subject detected — suitable for wildlife/pet monitoring.'],
  },
];

/**
 * Turn a detections array into a scene interpretation.
 * @param {Array<{class:string, score:number}>} detections
 * @param {Array<{label:string, prob:number}>} [classification] - MobileNet top labels
 */
function interpret(detections, classification = []) {
  // Count by class and track average confidence.
  const counts = Object.create(null);
  const confidenceByClass = Object.create(null);
  let confidenceSum = 0;
  for (const d of detections) {
    counts[d.class] = (counts[d.class] || 0) + 1;
    confidenceByClass[d.class] = (confidenceByClass[d.class] || 0) + d.score;
    confidenceSum += d.score || 0;
  }
  const uniqueTypes = Object.keys(counts);

  // Score each scene.
  let best = null;
  for (const scene of SCENES) {
    let score = 0;
    let matched = 0;
    for (const sig of scene.signals) {
      if (counts[sig]) {
        matched += 1;
        score += confidenceByClass[sig];
      }
    }
    if (scene.minPeople && (counts.person || 0) < scene.minPeople) continue;
    if (matched === 0) continue;
    // A person or chair alone cannot establish the type of room.
    if (matched === 1 && ['person', 'chair', 'book', 'cup', 'tv'].some((label) =>
      counts[label] && scene.signals.includes(label))) continue;
    const scaled = score + matched * 0.5; // reward breadth of matching signals
    if (!best || scaled > best.score) best = { scene, score: scaled, matched };
  }

  const avgConfidence = detections.length ? confidenceSum / detections.length : 0;

  // MobileNet's top label names the overall subject — prepend it for a
  // richer "what is in the image" read.
  const top = Array.isArray(classification) && classification[0];
  const subjectLine = top
    ? `Primary subject: ${top.label} (${Math.round((top.prob || 0) * 100)}% confidence). `
    : '';

  if (!best) {
    return {
      scene: top ? 'Subject Identified' : 'General Scene',
      sceneId: 'general',
      summary:
        subjectLine +
        (uniqueTypes.length
          ? `Detected ${detections.length} object(s) across ${uniqueTypes.length} type(s): ${uniqueTypes.join(', ')}.`
          : (top ? 'No distinct objects were localized, but the image was classified above.'
                 : 'No recognizable objects were detected in this image.')),
      recommendations: uniqueTypes.length
        ? ['Review the localized objects and the classification labels above.']
        : top
          ? [`Image recognized as "${top.label}" — lower the confidence threshold to surface smaller objects.`]
          : ['Try a clearer image, or one containing common objects (people, furniture, vehicles, electronics).'],
      counts,
      uniqueTypes: uniqueTypes.length,
      totalObjects: detections.length,
      avgConfidence,
    };
  }

  return {
    scene: best.scene.label,
    sceneId: best.scene.id,
    summary: subjectLine + best.scene.describe(counts),
    recommendations: best.scene.recommend(counts),
    counts,
    uniqueTypes: uniqueTypes.length,
    totalObjects: detections.length,
    avgConfidence,
  };
}

// --- Routes -----------------------------------------------------------------
app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', service: 'ai-vision-inspector', time: new Date().toISOString() });
});

app.post('/api/interpret', (req, res) => {
  const { detections, classification } = req.body || {};
  if (!Array.isArray(detections)) {
    return res.status(400).json({ error: 'Request body must include a "detections" array.' });
  }
  // Validate / sanitize at the boundary.
  const clean = detections.slice(0, 1000)
    .filter((d) => d && typeof d.class === 'string' && d.class.trim())
    .map((d) => ({ class: d.class.trim().toLowerCase().slice(0, 100), score: Math.max(0, Math.min(1, Number(d.score) || 0)) }));
  const cleanClasses = Array.isArray(classification)
    ? classification.slice(0, 20)
        .filter((c) => c && typeof c.label === 'string' && c.label.trim())
        .map((c) => ({ label: c.label.trim().slice(0, 200), prob: Math.max(0, Math.min(1, Number(c.prob) || 0)) }))
        .sort((a, b) => b.prob - a.prob)
    : [];

  try {
    const result = interpret(clean, cleanClasses);
    res.json(result);
  } catch (err) {
    console.error('Interpretation error:', err);
    res.status(500).json({ error: 'Failed to interpret detections.' });
  }
});

// --- Scan history DB API ----------------------------------------------------
app.get('/api/scans', (_req, res) => {
  res.json(loadDB().slice(0, 100));
});

app.post('/api/scans', (req, res) => {
  const b = req.body || {};
  if (!b || Array.isArray(b) || typeof b !== 'object' || !Object.keys(b).length) {
    return res.status(400).json({ error: 'A scan record is required.' });
  }
  const s = (v, n) => (v == null ? '' : String(v).slice(0, n));
  const number = (v, max) => Number.isFinite(Number(v)) ? Math.max(0, Math.min(max, Number(v))) : 0;
  const row = {
    id: randomUUID(),
    time: new Date().toISOString(),
    source: s(b.source || 'upload', 20),
    scene: s(b.scene || '—', 80),
    subject: s(b.subject, 80),
    summary: s(b.summary, 600),
    objects: Math.floor(number(b.objects, 1000)),
    faces: Array.isArray(b.faces) ? b.faces.filter((f) => f && typeof f === 'object').slice(0, 10).map((f) => ({
      age: Math.round(number(f.age, 120)), gender: s(f.gender, 20), expression: s(f.expression, 30),
    })) : [],
    translation: s(b.translation, 600),
    thumb: typeof b.thumb === 'string' && b.thumb.length <= 500000 &&
      /^data:image\/(?:png|jpeg|webp);base64,[A-Za-z0-9+/=]+$/.test(b.thumb) ? b.thumb : '',
  };
  const rows = loadDB();
  rows.unshift(row);
  saveDB(rows.slice(0, 200));
  res.json(row);
});

app.delete('/api/scans/:id', (req, res) => {
  const rows = loadDB().filter((r) => r.id !== req.params.id);
  saveDB(rows);
  res.json({ ok: true, count: rows.length });
});

app.delete('/api/scans', (_req, res) => {
  saveDB([]);
  res.json({ ok: true });
});

// --- Translation proxy (MyMemory, no key) -----------------------------------
function splitTranslationText(text, maxBytes = 450) {
  const chunks = [];
  let chunk = '';
  for (const character of text) {
    if (Buffer.byteLength(chunk + character, 'utf8') > maxBytes) {
      // Prefer word boundaries so a chunk never splits an OCR word mid-way.
      const boundary = Math.max(chunk.lastIndexOf(' '), chunk.lastIndexOf('\n'));
      if (boundary > chunk.length / 2) {
        chunks.push(chunk.slice(0, boundary).trim());
        chunk = chunk.slice(boundary + 1);
      } else {
        chunks.push(chunk.trim());
        chunk = '';
      }
    }
    chunk += character;
  }
  if (chunk.trim()) chunks.push(chunk.trim());
  return chunks;
}

app.post('/api/translate', async (req, res) => {
  const { text, source } = req.body || {};
  if (typeof text !== 'string' || !text.trim()) {
    return res.status(400).json({ error: 'A non-empty "text" string is required.' });
  }
  if (text.length > 12000) {
    return res.status(413).json({ error: 'Translate up to 12,000 characters at a time.' });
  }
  const src = typeof source === 'string' && /^[a-z]{2}$/i.test(source) ? source.toLowerCase() : 'autodetect';
  if (src === 'en') return res.json({ translation: text.trim(), source: 'en', status: 200 });
  try {
    const chunks = splitTranslationText(text.trim());
    const translations = [];
    let detectedSource = src;
    for (let i = 0; i < chunks.length; i += 3) {
      const batch = await Promise.all(chunks.slice(i, i + 3).map(async (chunk) => {
        const params = new URLSearchParams({ q: chunk, langpair: `${src}|en` });
        const r = await fetch(`https://api.mymemory.translated.net/get?${params}`, {
          signal: AbortSignal.timeout(15000),
        });
        if (!r.ok) throw new Error('Translation service unavailable.');
        const j = await r.json();
        if (Number(j.responseStatus) !== 200 || j.quotaFinished || !j.responseData?.translatedText) {
          const err = new Error(j.quotaFinished
            ? 'The free translation quota has been reached. Try again later.'
            : 'Translation is unavailable for this language.');
          err.status = j.quotaFinished ? 429 : 502;
          throw err;
        }
        if (j.responseData.detectedLanguage) detectedSource = j.responseData.detectedLanguage;
        return j.responseData.translatedText;
      }));
      translations.push(...batch);
    }
    res.json({
      translation: translations.join(' '),
      source: detectedSource,
      status: 200,
    });
  } catch (err) {
    console.error('Translate error:', err);
    res.status(err.status || 502).json({ error: err.status ? err.message : 'Translation service unavailable. Please try again.' });
  }
});

app.use('/api', (_req, res) => res.status(404).json({ error: 'API endpoint not found.' }));

// Fallback to the SPA entry point.
app.get('*', (_req, res) => {
  if (path.extname(_req.path)) return res.status(404).send('File not found.');
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.use((err, _req, res, _next) => {
  if (err.type === 'entity.parse.failed') return res.status(400).json({ error: 'Invalid JSON request body.' });
  if (err.type === 'entity.too.large') return res.status(413).json({ error: 'Request body is too large.' });
  console.error('Request failed:', err);
  res.status(500).json({ error: 'Unable to complete the request. Please try again.' });
});

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`\n  👁️  AI Vision Inspector running at http://localhost:${PORT}\n`);
  });
}

module.exports = { app, interpret, splitTranslationText };
