/* extras.js — show-off vision add-ons
 * - Face detection + AGE / GENDER / EMOTION via face-api.js (TinyFaceDetector,
 *   AgeGenderNet, FaceExpressionNet) running fully in-browser.
 * - OCR (Tesseract.js) + language detection (franc) + translation to English
 *   (backend MyMemory proxy).
 * All loaded lazily with graceful fallback so the core app never blocks.
 */
window.VisionExtras = {
  faceReady: false,
  faceLoading: false,
  faceError: '',
  _facePromise: null,
  landmarksReady: false,
  expressionsReady: false,
  onfacestatus: null,

  _s(msg) { if (typeof this.onfacestatus === 'function') this.onfacestatus(msg); },

  loadFaces() {
    if (this.faceReady) return Promise.resolve(true);
    if (this._facePromise) return this._facePromise;
    if (!window.faceapi) {
      this.faceError = 'Face AI could not load. Check your connection and reload.';
      this._s('face AI unavailable');
      return Promise.resolve(false);
    }
    this.faceLoading = true;
    this.faceError = '';
    this._s('loading face models…');
    const MODEL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.13/model';
    this._facePromise = (async () => {
      try {
        await faceapi.tf.ready();
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL),
          faceapi.nets.ageGenderNet.loadFromUri(MODEL),
        ]);
        const optional = await Promise.allSettled([
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL),
          faceapi.nets.faceExpressionNet.loadFromUri(MODEL),
        ]);
        this.landmarksReady = optional[0].status === 'fulfilled';
        this.expressionsReady = optional[1].status === 'fulfilled';
        this.faceReady = true;
        this._s(this.landmarksReady && this.expressionsReady ? 'face AI ready' : 'face AI ready · limited');
        return true;
      } catch (err) {
        console.warn('face-api load failed:', err);
        this.faceError = 'Face AI could not load. Check your connection and try again.';
        this._s('face AI unavailable');
        return false;
      } finally {
        this.faceLoading = false;
        this._facePromise = null;
      }
    })();
    return this._facePromise;
  },

  /** Detect faces on a canvas/image → [{x,y,width,height,age,gender,genderProb,expression}] */
  async detectFaces(input) {
    if (this.faceLoading) await this._facePromise;
    if (!this.faceReady || !window.faceapi) return [];
    try {
      const opts = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 });
      // Canonical face-api task order: detect → expressions → ageGender.
      let chain = faceapi.detectAllFaces(input, opts);
      if (this.expressionsReady) chain = chain.withFaceExpressions();
      chain = chain.withAgeAndGender();
      const results = await chain;
      return results.map((r) => {
        const b = r.detection.box;
        let expression = '';
        if (r.expressions) {
          const top = Object.entries(r.expressions).sort((a, b2) => b2[1] - a[1])[0];
          if (top && top[1] > 0.4) expression = top[0];
        }
        return {
          x: b.x, y: b.y, width: b.width, height: b.height,
          age: Math.round(r.age),
          gender: r.gender,
          genderProb: r.genderProbability,
          expression,
        };
      });
    } catch (err) {
      console.warn('detectFaces failed:', err);
      return [];
    }
  },

  /** Rich per-person analysis with FOR-FUN scores (landmarks + colour). */
  async analyzePersons(input) {
    if (this.faceLoading) await this._facePromise;
    if (!this.faceReady || !window.faceapi) return [];
    try {
      const opts = new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 });
      // Canonical face-api task order: detect → landmarks → expressions → ageGender.
      let chain = faceapi.detectAllFaces(input, opts);
      if (this.landmarksReady) chain = chain.withFaceLandmarks();
      if (this.expressionsReady) chain = chain.withFaceExpressions();
      chain = chain.withAgeAndGender();
      const res = await chain;
      return res.map((r) => {
        const b = (r.detection ? r.detection.box : r.box);
        const expr = topExpr(r.expressions);
        const sym = (r.landmarks) ? symmetry(r.landmarks) : 0.7;
        const faceScore = scoreFace(sym, (r.detection ? r.detection.score : 0.8), expr.pos);
        const fash = fashion(input, b);
        return {
          x: b.x, y: b.y, width: b.width, height: b.height,
          age: Math.round(r.age), gender: r.gender, genderProb: r.genderProbability,
          expression: expr.name, faceScore, symmetry: Math.round(sym * 100),
          fashionScore: fash.score, style: fash.label,
        };
      });
    } catch (err) { console.warn('analyzePersons failed:', err); return []; }
  },

  // ---- OCR + translation ---------------------------------------------------
  async ocrTranslate(input, onprogress) {
    if (!window.Tesseract) throw new Error('Text recognition could not load. Check your connection and reload.');
    onprogress && onprogress('reading text…');
    let worker, data;
    try {
      worker = await Tesseract.createWorker(['eng', 'spa', 'fra', 'deu', 'ita', 'por'], 1, {
        logger: (message) => {
          if (onprogress && message.status === 'recognizing text') {
            onprogress(`reading text… ${Math.round(message.progress * 100)}%`);
          }
        },
      });
      ({ data } = await worker.recognize(input));
    } finally {
      if (worker) await worker.terminate();
    }
    const text = (data.text || '').trim();
    if (!text) return { text: '', source: '', translation: '', note: 'No readable text found.' };

    // Language detection (franc) → ISO-639-1 for the translator.
    let source = '';
    try {
      const { franc } = await import('https://cdn.jsdelivr.net/npm/franc-min@6.2.0/+esm');
      // Tiny snippets are unreliable language samples; let the translation
      // service detect their language rather than guessing confidently.
      source = text.length >= 40 ? ISO3[franc(text)] || '' : '';
    } catch (e) { /* detection optional */ }

    if (source === 'en') return { text, source, translation: text, note: 'Text is already in English.' };
    onprogress && onprogress('translating…');
    try {
      const r = await fetch('/api/translate', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, source }),
        signal: AbortSignal.timeout(120000),
      });
      const j = await r.json();
      if (!r.ok || !j.translation) throw new Error(j.error || 'Translation unavailable.');
      return { text, source: j.source || source || 'auto', translation: j.translation };
    } catch (e) {
      return { text, source: source || 'auto', translation: '', note: e.message || 'Translation unavailable.' };
    }
  },
};

// ---- For-fun scoring helpers (NOT a real judgment of anyone) --------------
function topExpr(expressions) {
  if (!expressions || !Object.keys(expressions).length) return { name: '', pos: 0.5 };
  const entries = Object.entries(expressions).sort((a, b) => b[1] - a[1]);
  const [name, val] = entries[0];
  const positive = (expressions.happy || 0) + (expressions.surprised || 0) * 0.5;
  return { name: val > 0.4 ? name : 'neutral', pos: Math.min(1, positive) };
}
function symmetry(landmarks) {
  const p = landmarks.positions; if (!p || p.length < 68) return 0.7;
  const cx = p[30].x; // nose tip as the mirror axis
  const pairs = [[0, 16], [1, 15], [2, 14], [3, 13], [36, 45], [39, 42], [17, 26], [21, 22], [48, 54], [31, 35]];
  const w = Math.abs(p[16].x - p[0].x) || 1;
  let acc = 0, n = 0;
  for (const [a, b] of pairs) {
    acc += Math.abs((cx - p[a].x) - (p[b].x - cx)) / w; // horizontal mismatch
    acc += Math.abs(p[a].y - p[b].y) / w;               // vertical mismatch
    n += 2;
  }
  return Math.max(0, Math.min(1, 1 - (acc / n) * 3));
}
function scoreFace(sym, det, pos) {
  const s = 52 + sym * 38 + (det || 0.8) * 5 + (pos || 0) * 5; // generous, playful
  return Math.round(Math.max(45, Math.min(99, s)));
}
function fashion(input, b) {
  const W = input.width || input.videoWidth, H = input.height || input.videoHeight;
  const rx = Math.max(0, b.x - b.width * 0.25);
  const ry = Math.min(H - 1, b.y + b.height * 1.05);
  const rw = Math.min(W - rx, b.width * 1.5);
  const rh = Math.min(H - ry, b.height * 2.0);
  if (rw < 4 || rh < 4) return { score: 60, label: 'smart casual' };
  const c = document.createElement('canvas'); c.width = 40; c.height = 40;
  const cx = c.getContext('2d'); cx.drawImage(input, rx, ry, rw, rh, 0, 0, 40, 40);
  const d = cx.getImageData(0, 0, 40, 40).data;
  let rgA = 0, ybA = 0, rg2 = 0, yb2 = 0, sat = 0, bri = 0, n = 0;
  for (let i = 0; i < d.length; i += 4) {
    const R = d[i], G = d[i + 1], B = d[i + 2];
    const rg = R - G, yb = 0.5 * (R + G) - B;
    rgA += rg; ybA += yb; rg2 += rg * rg; yb2 += yb * yb;
    const mx = Math.max(R, G, B), mn = Math.min(R, G, B);
    sat += mx ? (mx - mn) / mx : 0; bri += mx / 255; n++;
  }
  const mrg = rgA / n, myb = ybA / n;
  const srg = Math.sqrt(Math.max(0, rg2 / n - mrg * mrg)), syb = Math.sqrt(Math.max(0, yb2 / n - myb * myb));
  const colorful = Math.sqrt(srg * srg + syb * syb) + 0.3 * Math.sqrt(mrg * mrg + myb * myb);
  const S = sat / n, Bv = bri / n;
  let score = 50 + Math.min(30, colorful * 0.5) + (Bv > 0.2 ? 10 : 0) + (S > 0.25 ? 8 : 0);
  score = Math.round(Math.max(45, Math.min(98, score)));
  let label;
  if (colorful < 18 && S < 0.2) label = Bv < 0.35 ? 'moody monochrome' : 'clean minimalist';
  else if (colorful > 45) label = 'bold & eclectic';
  else if (S > 0.45) label = 'vibrant & expressive';
  else label = 'smart casual';
  return { score, label };
}

// Minimal ISO 639-3 → 639-1 map for the languages we care about.
const ISO3 = {
  eng: 'en', spa: 'es', fra: 'fr', deu: 'de', ita: 'it', por: 'pt', nld: 'nl',
  rus: 'ru', jpn: 'ja', cmn: 'zh', kor: 'ko', ara: 'ar', hin: 'hi', ben: 'bn',
  tur: 'tr', pol: 'pl', swe: 'sv', ukr: 'uk', ell: 'el', heb: 'he', tha: 'th', vie: 'vi',
};
