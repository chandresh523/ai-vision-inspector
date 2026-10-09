/* AI Vision Inspector — application logic
 * CNNs:  COCO-SSD (detection) + MobileNet (ImageNet classification)
 * SLM:   DistilBERT-MNLI zero-shot intent routing (nlu.js) + keyword fallback
 * Dash:  command console → real canvas edits, live webcam, history, advisories
 */

const $ = (id) => document.getElementById(id);
const els = {
  status: $('modelStatus'), dropzone: $('dropzone'), fileInput: $('fileInput'),
  dropPrompt: $('dropPrompt'), canvasWrap: $('canvasWrap'), canvas: $('canvas'),
  spinner: $('spinner'), webcam: $('webcam'),
  browseBtn: $('browseBtn'), sampleBtn: $('sampleBtn'), camBtn: $('camBtn'),
  liveBtn: $('liveBtn'), shotBtn: $('shotBtn'), resetBtn: $('resetBtn'),
  toolbar: $('toolbar'), threshold: $('threshold'), threshVal: $('threshVal'),
  boxToggle: $('boxToggle'), downloadBtn: $('downloadBtn'),
  emptyResults: $('emptyResults'), results: $('results'),
  statObjects: $('statObjects'), statTypes: $('statTypes'), statConfidence: $('statConfidence'),
  sceneBadge: $('sceneBadge'), sceneSummary: $('sceneSummary'),
  subjectBlock: $('subjectBlock'), classChips: $('classChips'),
  recommendations: $('recommendations'), detectionList: $('detectionList'),
  nluStatus: $('nluStatus'), faceStatus: $('faceStatus'), quickActions: $('quickActions'),
  cmdInput: $('cmdInput'), cmdSend: $('cmdSend'), cmdResponse: $('cmdResponse'), historyList: $('historyList'),
  ocrBtn: $('ocrBtn'), revealBtn: $('revealBtn'),
  ocrPanel: $('ocrPanel'), ocrStatus: $('ocrStatus'), ocrLang: $('ocrLang'), ocrText: $('ocrText'), ocrTranslation: $('ocrTranslation'),
  dbBtn: $('dbBtn'), dbModal: $('dbModal'), dbList: $('dbList'), dbClose: $('dbClose'), dbClear: $('dbClear'),
  personPopup: $('personPopup'), popupBody: $('popupBody'), popupClose: $('popupClose'),
};

const ctx = els.canvas.getContext('2d');
const base = document.createElement('canvas');     // clean scaled source
const bctx = base.getContext('2d', { willReadFrequently: true });
const BOX_COLORS = ['#7afcff', '#ff6edb', '#ffd24d', '#5effa0', '#b58cff', '#ff8a3d'];

const COCO = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'];

const NOTES = {
  person: 'People present — consider occupancy & safety monitoring.',
  car: 'Vehicle — relevant for traffic/parking analysis.', truck: 'Large vehicle — traffic/logistics context.',
  bus: 'Public transport vehicle detected.', motorcycle: 'Motorcyclist — vulnerable road user.',
  bicycle: 'Cyclist — vulnerable road user.', knife: 'Sharp object — potential safety hazard.',
  scissors: 'Sharp object — potential safety hazard.', 'cell phone': 'Phone in frame — distraction / privacy note.',
  laptop: 'Electronics — workspace/asset context.', keyboard: 'Electronics — workspace context.',
  tv: 'Display — presentation/entertainment context.', bottle: 'Container — spill risk near electronics.',
  cup: 'Beverage — spill risk near devices.', 'wine glass': 'Glassware — fragile item.',
  backpack: 'Bag — left-luggage / security relevance.', handbag: 'Bag — security relevance.',
  suitcase: 'Luggage — security/transit relevance.', dog: 'Animal — pet/wildlife subject.',
  cat: 'Animal — pet subject.', 'stop sign': 'Traffic sign — navigation context.',
  'traffic light': 'Traffic signal — road-scene context.', chair: 'Seating — occupancy/layout context.',
};

const INTENTS = [
  { id: 'blur_bg',   label: 'blur the background',        keys: ['blur', 'background', 'bokeh', 'defocus'] },
  { id: 'grayscale', label: 'convert to grayscale',       keys: ['grayscale', 'greyscale', 'black and white', 'desaturate', 'mono'] },
  { id: 'brighten',  label: 'brighten the image',         keys: ['brighten', 'brighter', 'lighten', 'light up'] },
  { id: 'darken',    label: 'darken the image',           keys: ['darken', 'darker', 'dim'] },
  { id: 'contrast',  label: 'increase contrast',          keys: ['contrast', 'punchy', 'pop'] },
  { id: 'highlight', label: 'highlight objects',          keys: ['highlight', 'spotlight', 'focus', 'emphasize', 'show me the'] },
  { id: 'redact',    label: 'redact or hide objects',     keys: ['redact', 'hide', 'censor', 'blackout', 'mask', 'anonymize'] },
  { id: 'crop',      label: 'crop to the main subject',   keys: ['crop', 'zoom', 'cut to', 'close up'] },
  { id: 'edges',     label: 'detect edges or outline',    keys: ['edge', 'outline', 'sketch', 'contour', 'lines'] },
  { id: 'count',     label: 'count objects',              keys: ['count', 'how many', 'number of', 'tally'] },
  { id: 'describe',  label: 'describe the image',         keys: ['describe', 'what is', 'what do you see', 'explain', 'tell me about'] },
  { id: 'reset',     label: 'reset the image',            keys: ['reset', 'restore', 'original', 'undo', 'clear edit'] },
];

let detector = null, classifier = null;
let lastPredictions = [], lastClasses = [], lastFaces = [], lastData = null;
let showBoxes = true, threshold = 0.5, hasImage = false;
let currentEdit = null;                 // active image-op renderer (or null)
let camStream = null, camLoop = null, liveOn = false, liveBusy = false;
let cameraStarting = false, previewPaused = false, liveTimer = null;
let sourceVersion = 0, analysisVersion = 0, interpretationVersion = 0, commandVersion = 0;
let cameraVersion = 0, liveVersion = 0, ocrVersion = 0, ocrBusy = false;
let popupOpen = false, popupBusy = false, popupVersion = 0, popupTimer = null, frameN = 0;
let modelsReady = null;
const history = [];

// ---- Models ----------------------------------------------------------------
async function loadModels() {
  setStatus('loading', 'loading CNNs…');
  const loaded = await Promise.allSettled([
    Promise.resolve().then(() => cocoSsd.load({ base: 'mobilenet_v2' })),
    Promise.resolve().then(() => mobilenet.load({ version: 2, alpha: 1.0 })),
  ]);
  detector = loaded[0].status === 'fulfilled' ? loaded[0].value : null;
  classifier = loaded[1].status === 'fulfilled' ? loaded[1].value : null;
  loaded.forEach((result) => { if (result.status === 'rejected') console.error('CNN load failed:', result.reason); });
  showModelStatus();
  initOptionalModels();
  loadDbCount();
}
function showModelStatus() {
  if (detector && classifier) setStatus('ready', 'COCO-SSD + MobileNet ready');
  else if (detector) setStatus('ready', 'detection ready · classification unavailable');
  else if (classifier) setStatus('ready', 'classification ready · detection unavailable');
  else setStatus('error', 'CNNs unavailable — check connection and reload');
}
function initOptionalModels() {
  if (window.NLU) {
    window.NLU.onstatus = (s) => (els.nluStatus.textContent = s);
    window.NLU.load();
  }
  if (window.VisionExtras) {
    window.VisionExtras.onfacestatus = (s) => (els.faceStatus.textContent = s);
    window.VisionExtras.loadFaces();
  }
}
function setStatus(state, label) {
  els.status.className = `status status--${state}`;
  els.status.querySelector('.label').textContent = label;
}

// ---- Image intake ----------------------------------------------------------
function beginSource() {
  sourceVersion++; analysisVersion++; interpretationVersion++; commandVersion++; ocrVersion++;
  lastPredictions = []; lastClasses = []; lastFaces = []; lastData = null;
  currentEdit = null; hasImage = false;
  els.spinner.hidden = true;
  els.toolbar.hidden = true; els.ocrBtn.hidden = true; els.ocrPanel.hidden = true;
  els.results.hidden = true; els.emptyResults.hidden = false;
  return sourceVersion;
}
function snapshot(source = base) {
  const copy = document.createElement('canvas');
  copy.width = source.width; copy.height = source.height;
  copy.getContext('2d').drawImage(source, 0, 0);
  return copy;
}
function handleFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) { respond('Choose an image file such as a JPEG, PNG, or WebP.'); return; }
  const url = URL.createObjectURL(file);
  loadImage(url, () => URL.revokeObjectURL(url));
}
function loadImage(src, done, source = 'upload') {
  stopCam();
  const version = beginSource();
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = async () => {
    try {
      if (version !== sourceVersion) return;
      drawToBase(img);
      await analyze(source);
    } catch (error) {
      if (version === sourceVersion) { setStatus('error', 'could not read image'); respond('This image could not be read. Try another image.'); }
      console.error(error);
    } finally { if (done) done(); }
  };
  img.onerror = () => {
    if (version === sourceVersion) { setStatus('error', 'could not load image'); respond('The image could not be loaded. Try a different file.'); }
    if (done) done();
  };
  img.src = src;
}
function drawToBase(source, w, h) {
  const sw = w || source.naturalWidth || source.videoWidth;
  const sh = h || source.naturalHeight || source.videoHeight;
  if (!sw || !sh) throw new Error('Image has no usable dimensions');
  const scale = Math.min(1, 900 / Math.max(sw, sh));
  base.width = els.canvas.width = Math.max(1, Math.round(sw * scale));
  base.height = els.canvas.height = Math.max(1, Math.round(sh * scale));
  bctx.drawImage(source, 0, 0, base.width, base.height);
  currentEdit = null;
  hasImage = true;
  redraw();
  els.dropPrompt.hidden = true;
  els.canvasWrap.hidden = false;
  els.resetBtn.hidden = false;
  els.ocrBtn.hidden = false;
}
function sizeCanvasTo(w, h) {
  els.canvas.width = w; els.canvas.height = h;
}
function redraw() {
  if (!hasImage) return;
  if (currentEdit) { currentEdit(); return; }
  if (els.canvas.width !== base.width || els.canvas.height !== base.height) sizeCanvasTo(base.width, base.height);
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  ctx.drawImage(base, 0, 0);
  if (showBoxes) { drawBoxes(filtered()); drawFaces(lastFaces); }
}
function filtered() { return lastPredictions.filter((p) => p.score >= threshold); }

// ---- Inference -------------------------------------------------------------
async function analyze(source = 'upload', pushHistory = true) {
  const version = sourceVersion, run = ++analysisVersion;
  const input = snapshot();
  els.spinner.hidden = false;
  respond('Analyzing image…');
  try {
    await modelsReady;
    if (version !== sourceVersion || run !== analysisVersion) return;
    if (!detector && !classifier) throw new Error('CNN models are unavailable');
    const [preds, classes, faces] = await Promise.all([
      detector ? detector.detect(input, 50, 0.01) : Promise.resolve([]),
      classifier ? classifier.classify(input, 3).catch((error) => { console.warn('Classification failed:', error); return []; }) : Promise.resolve([]),
      (window.VisionExtras && window.VisionExtras.faceReady)
        ? window.VisionExtras.detectFaces(input).catch(() => []) : Promise.resolve([]),
    ]);
    if (version !== sourceVersion || run !== analysisVersion) return;
    lastPredictions = preds;
    lastClasses = (classes || []).map((c) => ({ label: c.className.split(',')[0], prob: c.probability }));
    lastFaces = faces || [];
    redraw();
    renderDetections(filtered());
    renderClasses(lastClasses);
    await refreshInterpretation();
    if (version !== sourceVersion || run !== analysisVersion) return;
    els.toolbar.hidden = false;
    els.ocrBtn.hidden = false;
    respond(lastFaces.length
      ? `Image analyzed — ${lastFaces.length} face(s) detected. Tell me what to do with it.`
      : 'Image analyzed — tell me what to do with it.');
    if (pushHistory) addHistory();
    saveScan(source);
    if (window.VisionScene) window.VisionScene.flash();
    showModelStatus();
  } catch (err) {
    if (version !== sourceVersion || run !== analysisVersion) return;
    console.error('Inference error:', err);
    setStatus('error', 'inference failed — try another image');
    respond('Analysis could not finish. Check the model status and try another image. Image edits and text reading are still available.');
    els.toolbar.hidden = false;
  } finally {
    if (version === sourceVersion && run === analysisVersion) els.spinner.hidden = true;
  }
}
async function refreshInterpretation() {
  const version = sourceVersion, request = ++interpretationVersion;
  const preds = filtered(), classes = lastClasses;
  const data = await fetchInterpretation(preds, classes);
  if (version !== sourceVersion || request !== interpretationVersion || !hasImage) return;
  lastData = data;
  renderResults(preds, data);
}

function drawBoxes(preds) {
  preds.forEach((p, i) => {
    const [x, y, w, h] = p.bbox;
    const color = BOX_COLORS[i % BOX_COLORS.length];
    ctx.lineWidth = 3; ctx.strokeStyle = color; ctx.shadowColor = color; ctx.shadowBlur = 12;
    ctx.strokeRect(x, y, w, h); ctx.shadowBlur = 0;
    const label = `${p.class} ${(p.score * 100).toFixed(0)}%`;
    ctx.font = '600 15px system-ui, sans-serif';
    const tw = ctx.measureText(label).width;
    const labelY = Math.max(0, y - 22), labelX = Math.max(0, Math.min(x, els.canvas.width - tw - 12));
    ctx.fillStyle = color; ctx.fillRect(labelX, labelY, tw + 12, 22);
    ctx.fillStyle = '#0a0a12'; ctx.fillText(label, labelX + 6, labelY + 16);
  });
}
function drawFaces(faces) {
  (faces || []).forEach((f) => {
    const g = '#ff6edb';
    ctx.lineWidth = 2.5; ctx.strokeStyle = g; ctx.shadowColor = g; ctx.shadowBlur = 14;
    // rounded-ish face frame
    ctx.strokeRect(f.x, f.y, f.width, f.height); ctx.shadowBlur = 0;
    const bits = [`~${f.age}y`];
    if (f.gender) bits.push(f.gender);
    if (f.expression) bits.push(f.expression);
    const label = bits.join(' · ');
    ctx.font = '600 14px system-ui, sans-serif';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = g; ctx.fillRect(f.x, f.y + f.height, tw + 12, 22);
    ctx.fillStyle = '#0a0a12'; ctx.fillText(label, f.x + 6, f.y + f.height + 16);
  });
}

// ---- Backend interpretation -----------------------------------------------
async function fetchInterpretation(preds, classes) {
  const detections = preds.map((p) => ({ class: p.class, score: p.score }));
  try {
    const res = await fetch('/api/interpret', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ detections, classification: classes }),
      signal: AbortSignal.timeout(12000),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error('Interpret API error:', err);
    return { scene: 'Offline', summary: 'Detection ran; interpretation service unavailable.',
      recommendations: ['Start the server with `npm start` and reload.'],
      uniqueTypes: new Set(detections.map((d) => d.class)).size, totalObjects: detections.length,
      avgConfidence: detections.reduce((s, d) => s + d.score, 0) / (detections.length || 1) };
  }
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
}

// ---- Rendering -------------------------------------------------------------
function renderResults(preds, data) {
  els.emptyResults.hidden = true; els.results.hidden = false;
  els.statObjects.textContent = data.totalObjects ?? preds.length;
  els.statTypes.textContent = data.uniqueTypes ?? new Set(preds.map((p) => p.class)).size;
  els.statConfidence.textContent = `${Math.round((data.avgConfidence || 0) * 100)}%`;
  els.sceneBadge.textContent = data.scene || '—';
  els.sceneSummary.textContent = data.summary || '';
  els.recommendations.innerHTML = '';
  (data.recommendations || []).forEach((r) => {
    const li = document.createElement('li'); li.textContent = r; els.recommendations.appendChild(li);
  });
}
function renderClasses(classes) {
  els.classChips.innerHTML = '';
  if (!classes.length) { els.subjectBlock.hidden = true; return; }
  els.subjectBlock.hidden = false;
  classes.forEach((c, i) => {
    const chip = document.createElement('span');
    chip.className = 'chip' + (i === 0 ? ' chip--top' : '');
    chip.innerHTML = `<b>${escapeHtml(c.label)}</b><span>${Math.round(c.prob * 100)}%</span>`;
    els.classChips.appendChild(chip);
  });
}
function renderDetections(preds) {
  els.detectionList.innerHTML = '';
  if (!preds.length) {
    const li = document.createElement('li'); li.className = 'detection';
    li.innerHTML = '<span class="name">no objects above threshold</span>';
    els.detectionList.appendChild(li);
  }
  [...preds].sort((a, b) => b.score - a.score).forEach((p, i) => {
    const pct = Math.round(p.score * 100);
    const li = document.createElement('li'); li.className = 'detection';
    li.innerHTML =
      `<span class="swatch" style="background:${BOX_COLORS[i % BOX_COLORS.length]}"></span>` +
      `<span class="name">${p.class}</span>` +
      `<span class="bar"><span style="width:${pct}%"></span></span><span class="pct">${pct}%</span>`;
    els.detectionList.appendChild(li);
  });
  // Per-object advisories (deduped by class).
  const seen = new Set();
  preds.forEach((p) => {
    if (seen.has(p.class)) return; seen.add(p.class);
    const note = NOTES[p.class]; if (!note) return;
    const li = document.createElement('li'); li.className = 'advisory';
    li.innerHTML = `<span class="adv-key">${p.class}</span><span>${note}</span>`;
    els.detectionList.appendChild(li);
  });
  // Face rows (age / gender / emotion).
  lastFaces.forEach((f, i) => {
    const li = document.createElement('li'); li.className = 'advisory advisory--face';
    const g = f.gender ? `${f.gender} ${Math.round((f.genderProb || 0) * 100)}%` : '';
    li.innerHTML = `<span class="adv-key">face ${i + 1}</span><span>age ~${f.age}${g ? ' · ' + g : ''}${f.expression ? ' · ' + f.expression : ''}</span>`;
    els.detectionList.appendChild(li);
  });
}

// ---- Webcam (capture + live) ----------------------------------------------
async function startCam() {
  if (cameraStarting || camStream) return;
  cameraStarting = true;
  const request = ++cameraVersion;
  beginSource();
  els.camBtn.textContent = 'cancel camera';
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error('Camera requires localhost or HTTPS');
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
    if (request !== cameraVersion) { stream.getTracks().forEach((track) => track.stop()); return; }
    camStream = stream;
    els.webcam.srcObject = stream;
    await els.webcam.play();
    if (request !== cameraVersion) return;
    els.dropPrompt.hidden = true; els.canvasWrap.hidden = false;
    els.shotBtn.hidden = false; els.liveBtn.hidden = false; els.resetBtn.hidden = false;
    els.revealBtn.hidden = false;
    els.camBtn.textContent = 'stop cam';
    const w = els.webcam.videoWidth, h = els.webcam.videoHeight, scale = Math.min(1, 900 / Math.max(w, h));
    sizeCanvasTo(Math.max(1, Math.round(w * scale)), Math.max(1, Math.round(h * scale)));
    previewPaused = false;
    respond('Camera ready. Capture an image or go live to analyze it.');
    (function preview() {
      if (!camStream || request !== cameraVersion) return;
      if (!liveOn && !previewPaused && els.webcam.readyState >= 2) ctx.drawImage(els.webcam, 0, 0, els.canvas.width, els.canvas.height);
      camLoop = requestAnimationFrame(preview);
    })();
  } catch (err) {
    if (request !== cameraVersion) return;
    console.error(err); stopCam(); setStatus('error', 'camera unavailable');
    respond(err.name === 'NotAllowedError' ? 'Camera permission was denied. Allow camera access in your browser, then try again.' : 'Camera unavailable. Connect a camera or upload an image.');
  } finally { if (request === cameraVersion) cameraStarting = false; }
}
function pauseLive() {
  liveOn = false; liveVersion++; previewPaused = true;
  if (liveTimer) clearTimeout(liveTimer);
  liveTimer = null;
  els.liveBtn.textContent = 'go live'; els.liveBtn.classList.remove('active');
}
function stopCam() {
  cameraVersion++; cameraStarting = false;
  pauseLive();
  if (camLoop) cancelAnimationFrame(camLoop);
  camLoop = null;
  if (camStream) camStream.getTracks().forEach((t) => t.stop());
  camStream = null; els.webcam.srcObject = null;
  els.camBtn.textContent = 'camera'; els.shotBtn.hidden = true; els.liveBtn.hidden = true;
  els.revealBtn.hidden = true; closePopup();
  if (hasImage) redraw();
  else { els.canvasWrap.hidden = true; els.dropPrompt.hidden = false; }
}
function capture() {
  if (!camStream || !els.webcam.videoWidth) return;
  const wasLive = liveOn;
  pauseLive(); beginSource();
  drawToBase(els.webcam, els.webcam.videoWidth, els.webcam.videoHeight);
  stopCam();
  analyze(wasLive ? 'live' : 'webcam');
}
async function toggleLive() {
  if (!camStream) return;
  if (liveOn) { pauseLive(); respond('Live analysis paused. You can edit or capture the frame.'); return; }
  const camera = cameraVersion;
  await modelsReady;
  if (!camStream || camera !== cameraVersion) return;
  if (!detector && !classifier) { respond('CNN models are unavailable. Check your connection and reload.'); return; }
  liveOn = true; previewPaused = true;
  els.liveBtn.textContent = 'stop live'; els.liveBtn.classList.add('active');
  currentEdit = null; frameN = 0;
  liveLoop(++liveVersion);
}
async function liveLoop(version) {
  if (!liveOn || !camStream || version !== liveVersion) return;
  if (liveBusy) { liveTimer = setTimeout(() => liveLoop(version), 150); return; }
  liveBusy = true;
  const input = document.createElement('canvas');
  const scale = Math.min(1, 900 / Math.max(els.webcam.videoWidth, els.webcam.videoHeight));
  input.width = Math.max(1, Math.round(els.webcam.videoWidth * scale));
  input.height = Math.max(1, Math.round(els.webcam.videoHeight * scale));
  input.getContext('2d').drawImage(els.webcam, 0, 0, input.width, input.height);
  try {
    const [preds, faces, classes] = await Promise.all([
      detector ? detector.detect(input, 50, 0.01) : Promise.resolve([]),
      window.VisionExtras && window.VisionExtras.faceReady ? window.VisionExtras.detectFaces(input).catch(() => []) : Promise.resolve([]),
      classifier && frameN % 3 === 0 ? classifier.classify(input, 3).catch(() => null) : Promise.resolve(null),
    ]);
    if (!liveOn || !camStream || version !== liveVersion) return;
    base.width = input.width; base.height = input.height; bctx.drawImage(input, 0, 0);
    hasImage = true; currentEdit = null;
    lastPredictions = preds; lastFaces = faces || [];
    if (classes) lastClasses = classes.map((c) => ({ label: c.className.split(',')[0], prob: c.probability }));
    redraw(); renderDetections(filtered()); renderClasses(lastClasses);
    els.toolbar.hidden = false; els.ocrBtn.hidden = false;
    await refreshInterpretation();
    frameN++;
  } catch (error) {
    if (version === liveVersion) { console.warn('Live analysis failed:', error); respond('A camera frame could not be analyzed. Retrying…'); }
  } finally { liveBusy = false; }
  if (liveOn && camStream && version === liveVersion) liveTimer = setTimeout(() => liveLoop(version), 320);
}

// ---- Image operations (command targets) ------------------------------------
function classesPresent() { return [...new Set(lastPredictions.map((p) => p.class))]; }
function extractClass(text) {
  let t = text.toLowerCase();
  const aliases = { people: 'person', persons: 'person', humans: 'person', men: 'person', women: 'person', man: 'person', woman: 'person', children: 'person', child: 'person', bikes: 'bicycle', bike: 'bicycle', phones: 'cell phone', phone: 'cell phone', mice: 'mouse', sofas: 'couch', sofa: 'couch' };
  t = t.replace(/\b(people|persons|humans|men|women|man|woman|children|child|bikes|bike|phones|phone|mice|sofas|sofa)\b/g, (word) => aliases[word]);
  const candidates = [...new Set([...classesPresent(), ...COCO])].sort((a, b) => b.length - a.length);
  for (const cls of candidates) {
    const escaped = cls.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    if (new RegExp(`\\b${escaped}(?:s|es)?\\b`).test(t)) return cls;
  }
  return null;
}
function boxesOf(cls) {
  const f = filtered();
  return cls ? f.filter((p) => p.class === cls) : f;
}
function unionBox(boxes) {
  if (!boxes.length) return null;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  boxes.forEach((p) => { const [x, y, w, h] = p.bbox; x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x + w); y1 = Math.max(y1, y + h); });
  return [x0, y0, x1 - x0, y1 - y0];
}
const OPS = {
  grayscale: () => { full(); ctx.filter = 'grayscale(1)'; ctx.drawImage(base, 0, 0); ctx.filter = 'none'; },
  brighten: () => { full(); ctx.filter = 'brightness(1.45)'; ctx.drawImage(base, 0, 0); ctx.filter = 'none'; },
  darken: () => { full(); ctx.filter = 'brightness(0.55)'; ctx.drawImage(base, 0, 0); ctx.filter = 'none'; },
  contrast: () => { full(); ctx.filter = 'contrast(1.7) saturate(1.2)'; ctx.drawImage(base, 0, 0); ctx.filter = 'none'; },
  blur_bg: () => {
    full(); ctx.filter = 'blur(7px) brightness(0.8)'; ctx.drawImage(base, 0, 0); ctx.filter = 'none';
    boxesOf(null).forEach((p) => { const [x, y, w, h] = p.bbox; ctx.drawImage(base, x, y, w, h, x, y, w, h); });
  },
  highlight: (cls) => {
    full(); ctx.drawImage(base, 0, 0);
    ctx.fillStyle = 'rgba(5,6,14,0.62)'; ctx.fillRect(0, 0, base.width, base.height);
    boxesOf(cls).forEach((p, i) => {
      const [x, y, w, h] = p.bbox; ctx.drawImage(base, x, y, w, h, x, y, w, h);
      ctx.strokeStyle = BOX_COLORS[i % BOX_COLORS.length]; ctx.lineWidth = 3;
      ctx.shadowColor = ctx.strokeStyle; ctx.shadowBlur = 16; ctx.strokeRect(x, y, w, h); ctx.shadowBlur = 0;
    });
  },
  redact: (cls) => {
    full(); ctx.drawImage(base, 0, 0);
    boxesOf(cls || 'person').forEach((p) => {
      const [x, y, w, h] = p.bbox; ctx.fillStyle = '#05060c'; ctx.fillRect(x, y, w, h);
      ctx.fillStyle = '#ff6edb'; ctx.font = '600 12px system-ui';
      ctx.fillText('REDACTED', x + 6, y + 16);
    });
  },
  crop: (cls) => {
    const box = unionBox(boxesOf(cls));
    if (!box) { full(); ctx.drawImage(base, 0, 0); return; }
    const pad = 12;
    let [x, y, w, h] = box;
    const right = Math.min(base.width, x + w + pad), bottom = Math.min(base.height, y + h + pad);
    x = Math.max(0, x - pad); y = Math.max(0, y - pad);
    w = Math.max(1, right - x); h = Math.max(1, bottom - y);
    sizeCanvasTo(Math.round(w), Math.round(h));
    ctx.clearRect(0, 0, w, h); ctx.drawImage(base, x, y, w, h, 0, 0, w, h);
  },
  edges: () => {
    sizeCanvasTo(base.width, base.height);
    const src = bctx.getImageData(0, 0, base.width, base.height);
    const out = ctx.createImageData(base.width, base.height);
    const W = base.width, H = base.height, d = src.data, o = out.data;
    const gray = (i) => (d[i] * 0.299 + d[i + 1] * 0.587 + d[i + 2] * 0.114);
    for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
      const i = (y * W + x) * 4;
      const gx = -gray(i - 4 - W * 4) - 2 * gray(i - 4) - gray(i - 4 + W * 4)
                 + gray(i + 4 - W * 4) + 2 * gray(i + 4) + gray(i + 4 + W * 4);
      const gy = -gray(i - W * 4 - 4) - 2 * gray(i - W * 4) - gray(i - W * 4 + 4)
                 + gray(i + W * 4 - 4) + 2 * gray(i + W * 4) + gray(i + W * 4 + 4);
      const m = Math.min(255, Math.hypot(gx, gy));
      o[i] = m * 0.5; o[i + 1] = m; o[i + 2] = Math.min(255, m * 1.3); o[i + 3] = 255;
    }
    ctx.putImageData(out, 0, 0);
  },
};
function full() { sizeCanvasTo(base.width, base.height); ctx.clearRect(0, 0, base.width, base.height); }

// ---- Command routing (SLM + keyword) --------------------------------------
function keywordRoute(text) {
  const t = text.toLowerCase();
  let best = null, bestScore = 0;
  for (const it of INTENTS) {
    let s = 0; for (const k of it.keys) if (new RegExp(`\\b${k}\\b`).test(t)) s += k.length;
    if (s > bestScore) { bestScore = s; best = it.id; }
  }
  return best;
}
async function runCommand(text) {
  text = (text || '').trim(); if (!text) return;
  if (!hasImage) { respond('Load an image first, then I can edit or analyze it.'); return; }
  if (camStream) pauseLive();
  const version = sourceVersion, request = ++commandVersion;
  let id = keywordRoute(text);
  let via = 'keyword';
  if (!id && window.NLU && window.NLU.ready) {
    const r = await window.NLU.route(text, INTENTS.map((i) => i.label));
    if (r && r.score > 0.35) { const m = INTENTS.find((i) => i.label === r.label); if (m) { id = m.id; via = `SLM ${(r.score * 100) | 0}%`; } }
  }
  if (version !== sourceVersion || request !== commandVersion || !hasImage) return;
  if (!id) { respond(`I couldn't map "${text}" to an action. Try: blur bg, grayscale, highlight people, crop to the dog, count cars, edges, reset.`); return; }
  applyIntent(id, extractClass(text), via);
}
function applyIntent(id, cls, via) {
  const tag = via ? ` (via ${via})` : '';
  if (id === 'count') {
    const n = cls ? boxesOf(cls).length : filtered().length;
    respond(`I count ${n} ${cls || 'object'}${n === 1 ? '' : 's'} above the current threshold.${tag}`); return;
  }
  if (id === 'describe') {
    const subj = lastClasses[0] ? `Looks like ${lastClasses[0].label}. ` : '';
    respond(`${subj}${lastData ? lastData.summary : ''}${tag}`); return;
  }
  if (id === 'reset') { currentEdit = null; redraw(); respond(`Restored the original with detections.${tag}`); return; }
  const op = OPS[id];
  if (!op) { respond(`No handler for that yet.`); return; }
  const targetClass = id === 'redact' ? (cls || 'person') : cls;
  if (['crop', 'highlight', 'redact'].includes(id) && !boxesOf(targetClass).length) {
    respond(`No ${targetClass || 'objects'} detected above the current threshold. Lower the confidence threshold or try another image.${tag}`); return;
  }
  currentEdit = () => op(cls);
  currentEdit();
  const msgs = {
    blur_bg: `Blurred the background, keeping ${boxesOf(null).length} detected object(s) sharp.`,
    grayscale: 'Converted the image to grayscale.', brighten: 'Brightened the image.',
    darken: 'Darkened the image.', contrast: 'Boosted contrast & saturation.',
    highlight: `Highlighted ${boxesOf(cls).length} ${cls || 'object'}(s), dimmed the rest.`,
    redact: `Redacted ${boxesOf(cls || 'person').length} ${cls || 'person'}(s).`,
    crop: `Cropped to ${cls || 'the main subject'}.`,
    edges: 'Rendered an edge map of the image.',
  };
  respond(`${msgs[id] || 'Done.'}${tag}`);
}
function respond(msg) { els.cmdResponse.textContent = msg; }

// ---- History ---------------------------------------------------------------
function addHistory() {
  const tc = document.createElement('canvas'); const s = 140 / base.width;
  tc.width = 140; tc.height = Math.round(base.height * s);
  tc.getContext('2d').drawImage(base, 0, 0, tc.width, tc.height);
  history.unshift({
    thumb: tc.toDataURL('image/jpeg', 0.6), src: base.toDataURL('image/jpeg', 0.85),
    preds: lastPredictions, classes: lastClasses, faces: lastFaces, data: lastData,
    scene: lastData ? lastData.scene : '—',
  });
  if (history.length > 8) history.pop();
  renderHistory();
}
function renderHistory() {
  els.historyList.innerHTML = '';
  if (!history.length) { els.historyList.innerHTML = '<span class="history-empty">inspections you run appear here</span>'; return; }
  history.forEach((h, idx) => {
    const d = document.createElement('div'); d.className = 'hist'; d.title = h.scene;
    d.innerHTML = `<img src="${h.thumb}" alt="" /><span>${escapeHtml(h.scene)}</span>`;
    d.addEventListener('click', () => restoreHistory(idx));
    els.historyList.appendChild(d);
  });
}
function restoreHistory(idx) {
  const h = history[idx]; if (!h) return;
  stopCam(); const version = beginSource();
  const img = new Image();
  img.onload = async () => {
    if (version !== sourceVersion) return;
    drawToBase(img);
    lastPredictions = h.preds; lastClasses = h.classes; lastFaces = h.faces || [];
    redraw(); renderDetections(filtered()); renderClasses(lastClasses);
    await refreshInterpretation();
    if (version !== sourceVersion) return;
    els.toolbar.hidden = false; els.ocrBtn.hidden = false;
    respond('Restored a previous inspection.');
  };
  img.onerror = () => { if (version === sourceVersion) respond('This history image could not be restored.'); };
  img.src = h.src;
}

// ---- Database (server-persisted scans) ------------------------------------
function thumbData(max = 160) {
  const tc = document.createElement('canvas'); const s = max / base.width;
  tc.width = max; tc.height = Math.round(base.height * s);
  tc.getContext('2d').drawImage(base, 0, 0, tc.width, tc.height);
  return tc.toDataURL('image/jpeg', 0.5);
}
async function saveScan(source) {
  try {
    const faces = lastFaces.map((f) => ({ age: f.age, gender: f.gender, expression: f.expression }));
    const response = await fetch('/api/scans', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        source, scene: lastData ? lastData.scene : '—',
        subject: lastClasses[0] ? lastClasses[0].label : '',
        summary: lastData ? lastData.summary : '', objects: filtered().length,
        faces, thumb: thumbData(),
      }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    loadDbCount();
  } catch (e) { console.warn('saveScan failed', e); respond('Analysis is ready, but this scan could not be saved to the database.'); }
}
async function fetchScans() {
  const response = await fetch('/api/scans', { signal: AbortSignal.timeout(10000) });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const rows = await response.json();
  if (!Array.isArray(rows)) throw new Error('Invalid scan response');
  return rows;
}
async function loadDbCount() {
  try { const rows = await fetchScans(); els.dbBtn.querySelector('span').textContent = `database (${rows.length})`; }
  catch { /* The database can be temporarily offline while local inspection still works. */ }
}
let dbVersion = 0;
async function openDb() {
  const request = ++dbVersion;
  els.dbModal.hidden = false;
  els.dbList.innerHTML = '<span class="history-empty">loading…</span>';
  try { const rows = await fetchScans(); if (request === dbVersion) renderDb(rows); }
  catch (e) { if (request === dbVersion) els.dbList.innerHTML = '<span class="history-empty">could not load database — please try again</span>'; }
}
function renderDb(rows) {
  if (!rows.length) { els.dbList.innerHTML = '<span class="history-empty">no scans stored yet</span>'; return; }
  els.dbList.innerHTML = '';
  rows.forEach((r) => {
    const d = document.createElement('div'); d.className = 'db-row';
    const faces = Array.isArray(r.faces) && r.faces.length ? ` · ${r.faces.length} face(s)` : '';
    const when = new Date(r.time).toLocaleString();
    const thumbnail = typeof r.thumb === 'string' && /^data:image\/(?:jpeg|png|webp);base64,[A-Za-z0-9+/=]+$/.test(r.thumb) ? r.thumb : '';
    d.innerHTML =
      `<img alt="Scan thumbnail" />` +
      `<div class="db-meta"><span class="db-scene">${escapeHtml(r.scene)}</span>` +
      `<span class="db-sub">${escapeHtml(r.subject || '—')} · ${Number(r.objects) || 0} obj${faces}</span>` +
      `<span class="db-time">${escapeHtml(r.source)} · ${escapeHtml(when)}</span>` +
      (r.translation ? `<span class="db-tr">“${escapeHtml(r.translation)}”</span>` : '') + `</div>` +
      `<button type="button" class="db-del" aria-label="Delete scan">✕</button>`;
    const img = d.querySelector('img');
    if (thumbnail) img.src = thumbnail;
    else img.style.visibility = 'hidden';
    img.addEventListener('error', () => { img.style.visibility = 'hidden'; });
    d.querySelector('.db-del').addEventListener('click', async (e) => {
      e.stopPropagation(); e.currentTarget.disabled = true;
      try {
        const response = await fetch('/api/scans/' + encodeURIComponent(r.id), { method: 'DELETE' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        openDb(); loadDbCount();
      } catch (error) { e.target.disabled = false; respond('This scan could not be deleted. Please try again.'); }
    });
    els.dbList.appendChild(d);
  });
}
async function clearDb() {
  els.dbClear.disabled = true;
  try {
    const response = await fetch('/api/scans', { method: 'DELETE' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    openDb(); loadDbCount();
  } catch (error) { respond('The database could not be cleared. Please try again.'); }
  finally { els.dbClear.disabled = false; }
}

// ---- OCR + translate ------------------------------------------------------
async function runOcr() {
  if (!hasImage) { respond('Load or capture an image before reading text.'); return; }
  if (!window.VisionExtras) { respond('Text reading is unavailable. Reload the page and try again.'); return; }
  if (ocrBusy) { respond('Text reading is already in progress.'); return; }
  if (camStream) pauseLive();
  const version = sourceVersion, request = ++ocrVersion, input = snapshot();
  const thumbnail = thumbData();
  ocrBusy = true; els.ocrBtn.disabled = true;
  els.ocrPanel.hidden = false; els.ocrStatus.textContent = 'working…';
  els.ocrText.textContent = '…'; els.ocrTranslation.textContent = '…'; els.ocrLang.textContent = '—';
  try {
    const r = await window.VisionExtras.ocrTranslate(input, (message) => {
      if (version === sourceVersion && request === ocrVersion) els.ocrStatus.textContent = message;
    });
    if (version !== sourceVersion || request !== ocrVersion) return;
    els.ocrLang.textContent = r.source || 'auto';
    els.ocrText.textContent = r.text || (r.note || 'no text found');
    els.ocrTranslation.textContent = r.translation || (r.note || '—');
    els.ocrStatus.textContent = r.note && !r.translation ? 'text ready' : 'done';
    if (r.translation) {
      try {
        const response = await fetch('/api/scans', { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ source: 'text', scene: 'Text / OCR', subject: r.source,
            summary: (r.text || '').slice(0, 200), translation: r.translation, objects: 0, thumb: thumbnail }) });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        loadDbCount();
      } catch (error) { console.warn('OCR save failed:', error); }
    }
  } catch (error) {
    console.error(error);
    if (version === sourceVersion && request === ocrVersion) { els.ocrStatus.textContent = 'ocr failed'; els.ocrTranslation.textContent = 'Text reading failed. Check your connection and try again.'; }
  } finally { ocrBusy = false; els.ocrBtn.disabled = false; }
}

// ---- Live person popup (for-fun scores) -----------------------------------
function openPopup() {
  popupOpen = true; popupVersion++; els.personPopup.hidden = false;
  els.revealBtn.classList.add('active'); updatePersonPopup();
}
function closePopup() {
  popupOpen = false; popupVersion++;
  if (popupTimer) clearTimeout(popupTimer);
  popupTimer = null;
  els.personPopup.hidden = true; els.revealBtn.classList.remove('active');
}
async function updatePersonPopup() {
  if (!popupOpen || popupBusy) return;
  const version = popupVersion;
  popupBusy = true;
  try {
    if (!window.VisionExtras) throw new Error('Face analysis is unavailable');
    if (!window.VisionExtras.faceReady) {
      els.popupBody.textContent = 'Loading face AI…';
      await window.VisionExtras.loadFaces();
    }
    if (!popupOpen || version !== popupVersion) return;
    if (!window.VisionExtras.faceReady) throw new Error(window.VisionExtras.faceError || 'Face models unavailable. Check your connection and reload.');
    const input = document.createElement('canvas');
    const source = camStream && els.webcam.readyState >= 2 ? els.webcam : base;
    input.width = source.videoWidth || source.width;
    input.height = source.videoHeight || source.height;
    input.getContext('2d').drawImage(source, 0, 0, input.width, input.height);
    const people = await window.VisionExtras.analyzePersons(input);
    if (popupOpen && version === popupVersion) renderPersonCards(people);
  } catch (error) {
    if (popupOpen && version === popupVersion) els.popupBody.textContent = error.message || 'Face analysis unavailable.';
  } finally {
    popupBusy = false;
    if (popupOpen) popupTimer = setTimeout(updatePersonPopup, 1000);
  }
}
function bar(v, color) { return `<span class="pbar"><span style="width:${v}%;background:${color}"></span></span>`; }
function renderPersonCards(people) {
  if (!people || !people.length) { els.popupBody.innerHTML = '<span class="history-empty">point the camera at a person…</span>'; return; }
  els.popupBody.innerHTML = '';
  people.forEach((p, i) => {
    const card = document.createElement('div'); card.className = 'person-card';
    card.innerHTML =
      `<div class="person-top"><span class="person-id">person ${i + 1}</span>` +
      `<span class="person-sub">~${p.age}y · ${p.gender} · ${p.expression || 'neutral'}</span></div>` +
      `<div class="pscore"><span>facial features</span>${bar(p.faceScore, 'linear-gradient(90deg,#7afcff,#5effa0)')}<b>${p.faceScore}</b></div>` +
      `<div class="pscore"><span>symmetry</span>${bar(p.symmetry, 'linear-gradient(90deg,#b58cff,#7afcff)')}<b>${p.symmetry}</b></div>` +
      `<div class="pscore"><span>fashion sense</span>${bar(p.fashionScore, 'linear-gradient(90deg,#ff6edb,#ffd24d)')}<b>${p.fashionScore}</b></div>` +
      `<div class="person-style">style read · <b>${p.style}</b></div>`;
    els.popupBody.appendChild(card);
  });
}

// ---- Samples ---------------------------------------------------------------
const SAMPLES = [
  'https://images.unsplash.com/photo-1524178232363-1fb2b075b655?w=900&q=80',
  'https://images.unsplash.com/photo-1497366216548-37526070297c?w=900&q=80',
  'https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=900&q=80',
  'https://images.unsplash.com/photo-1517849845537-4d257902454a?w=900&q=80',
];
let sampleIdx = 0;
async function loadSample() {
  stopCam(); const version = beginSource();
  setStatus('loading', 'fetching sample…');
  const url = SAMPLES[sampleIdx++ % SAMPLES.length];
  try {
    const response = await fetch(url, { signal: AbortSignal.timeout(20000) });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const blob = await response.blob();
    if (version !== sourceVersion) return;
    const obj = URL.createObjectURL(blob);
    loadImage(obj, () => URL.revokeObjectURL(obj), 'sample');
  } catch (err) {
    if (version !== sourceVersion) return;
    console.error(err); setStatus('error', 'sample fetch failed — upload your own');
    respond('The sample could not be downloaded. Try again or choose an image from your device.');
  }
}

// ---- Reset -----------------------------------------------------------------
function reset() {
  stopCam(); closePopup(); beginSource();
  els.fileInput.value = '';
  els.canvasWrap.hidden = true; els.dropPrompt.hidden = false;
  els.resetBtn.hidden = true; els.toolbar.hidden = true;
  els.ocrBtn.hidden = true; els.revealBtn.hidden = true; els.ocrPanel.hidden = true;
  els.results.hidden = true; els.emptyResults.hidden = false;
  lastPredictions = []; lastClasses = []; lastFaces = []; currentEdit = null; hasImage = false;
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  respond('Load an image, then tell me what to do with it.');
}

// ---- Living eye ------------------------------------------------------------
function initEye() {
  const eye = $('heroEye'), iris = $('iris'), pupil = $('pupil'), inner = $('eyeInner');
  if (!eye) return;
  const target = { x: 0, y: 0 }, cur = { x: 0, y: 0 };
  window.addEventListener('pointermove', (e) => {
    const r = eye.getBoundingClientRect();
    const dx = (e.clientX - (r.left + r.width / 2)) / window.innerWidth;
    const dy = (e.clientY - (r.top + r.height / 2)) / window.innerHeight;
    target.x = Math.max(-1, Math.min(1, dx * 3)); target.y = Math.max(-1, Math.min(1, dy * 3));
  }, { passive: true });
  (function follow() {
    cur.x += (target.x - cur.x) * 0.08; cur.y += (target.y - cur.y) * 0.08;
    iris.style.transform = `translate(${cur.x * 14}px, ${cur.y * 14}px)`;
    pupil.style.transform = `translate(${cur.x * 8}px, ${cur.y * 8}px)`;
    requestAnimationFrame(follow);
  })();
  (function blink() {
    inner.classList.add('blink'); setTimeout(() => inner.classList.remove('blink'), 150);
    setTimeout(blink, 2600 + Math.random() * 4200);
  })();
}

// ---- Wiring ----------------------------------------------------------------
window.addEventListener('nluavailable', initOptionalModels);

els.dropzone.addEventListener('click', () => { if (!camStream) els.fileInput.click(); });
els.browseBtn.addEventListener('click', (e) => { e.stopPropagation(); els.fileInput.click(); });
els.fileInput.addEventListener('change', (e) => { handleFile(e.target.files[0]); e.target.value = ''; });
els.sampleBtn.addEventListener('click', (e) => { e.stopPropagation(); loadSample(); });
els.camBtn.addEventListener('click', (e) => { e.stopPropagation(); (camStream || cameraStarting) ? stopCam() : startCam(); });
els.liveBtn.addEventListener('click', (e) => { e.stopPropagation(); toggleLive(); });
els.shotBtn.addEventListener('click', (e) => { e.stopPropagation(); capture(); });
els.resetBtn.addEventListener('click', (e) => { e.stopPropagation(); reset(); });

els.threshold.addEventListener('input', (e) => {
  threshold = Number(e.target.value) / 100; els.threshVal.textContent = `${e.target.value}%`;
  if (hasImage) {
    redraw(); renderDetections(filtered());
    refreshInterpretation();
  }
});
els.boxToggle.addEventListener('click', (e) => {
  e.stopPropagation(); showBoxes = !showBoxes;
  els.boxToggle.classList.toggle('active', showBoxes);
  els.boxToggle.textContent = showBoxes ? '◻ boxes' : '◼ boxes';
  if (!currentEdit) redraw();
});
els.downloadBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  const a = document.createElement('a'); a.download = 'vision-inspection.png';
  a.href = els.canvas.toDataURL('image/png'); a.click();
});

els.quickActions.querySelectorAll('.qa').forEach((q) =>
  q.addEventListener('click', () => { els.cmdInput.value = q.dataset.cmd; runCommand(q.dataset.cmd); }));
els.cmdSend.addEventListener('click', () => runCommand(els.cmdInput.value));
els.cmdInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') runCommand(els.cmdInput.value); });

els.ocrBtn.addEventListener('click', (e) => { e.stopPropagation(); runOcr(); });
els.revealBtn.addEventListener('click', (e) => { e.stopPropagation(); popupOpen ? closePopup() : openPopup(); });
els.popupClose.addEventListener('click', (e) => { e.stopPropagation(); closePopup(); });
els.dbBtn.addEventListener('click', openDb);
els.dbClose.addEventListener('click', () => { els.dbModal.hidden = true; });
els.dbClear.addEventListener('click', clearDb);
els.dbModal.addEventListener('click', (e) => { if (e.target === els.dbModal) els.dbModal.hidden = true; });

['dragover', 'dragenter'].forEach((ev) =>
  els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.add('dragover'); }));
['dragleave', 'drop'].forEach((ev) =>
  els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.remove('dragover'); }));
els.dropzone.addEventListener('drop', (e) => handleFile(e.dataTransfer.files[0]));

window.addEventListener('pagehide', stopCam);
initEye();
modelsReady = loadModels();
