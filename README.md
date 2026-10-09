# 👁️ AI Vision Inspector

A full-stack, in-browser **AI vision dashboard**. Upload / drag / sample / webcam an
image → it detects objects, classifies the subject, reads & translates text,
analyses faces, and lets you **direct edits in natural language**. Everything is
wrapped in a slowed cinematic video background with lower exposure and zoom blur,
transparent refractive glass shards, a custom glowing cursor, and GSAP scroll motion.

> Hackathon brief: **AI Vision Inspector** — accept an image, identify ≥3 object
> types, display results, give a useful interpretation. This build targets the
> 🔴 *Advanced* tier (detection + confidence + recommendation) and then some.

---

## Quick start

```bash
cd hackathon
npm install      # installs express
npm start        # → http://localhost:3000
```

Open **http://localhost:3000**. The AI model weights download from public CDNs on
first load (no API keys, no training). First load takes a few seconds; after that
the browser caches them.

---

## Architecture at a glance

```
Browser (frontend)                              Node/Express (backend)
──────────────────                              ──────────────────────
index.html  …… structure (hero/dashboard/…)     server.js
styles.css  …… liquid-glass + video theme         ├─ static file serving (public/)
scene.js    …… WebGL video + glass shards         ├─ POST /api/interpret   scene reasoning
app.js      …… app logic, dashboard, image ops    ├─ GET/POST/DELETE /api/scans  history DB
extras.js   …… face AI + OCR/translate             ├─ POST /api/translate   MyMemory proxy
nlu.js      …… SLM intent routing (zero-shot)      └─ GET  /api/health
                                                  data/scans.json  (file-backed DB)

In-browser AI (all pretrained, CDN-loaded):
  • COCO-SSD (mobilenet_v2)  — object detection, 80 classes, boxes + confidence
  • MobileNet v2             — ImageNet classification, 1000 classes (names subject)
  • face-api.js              — face detect + age + gender + emotion + landmarks
  • Tesseract.js             — OCR (eng/spa/fra/deu/ita/por)
  • DistilBERT-MNLI (xenova) — zero-shot command → action routing (SLM)
  • franc-min                — language detection for translation
```

The **CNNs run client-side** (no server GPU needed); the **backend** adds the
reasoning layer (scene interpretation + recommendations), persistence (scan
history DB) and the translation proxy.

---

## Backend (`server.js`)

Plain Express, one dependency. Zero-config file database.

| Method & route            | Purpose |
|---------------------------|---------|
| `GET  /api/health`        | Liveness check. |
| `POST /api/interpret`     | Body `{ detections:[{class,score}], classification:[{label,prob}] }` → `{ scene, summary, recommendations[], counts, uniqueTypes, totalObjects, avgConfidence }`. Scores a set of scene signatures (classroom, office, meeting, traffic, kitchen, living-room, animals) and prepends the MobileNet subject. |
| `GET  /api/scans`         | Last 100 stored scans (newest first). |
| `POST /api/scans`         | Store a scan `{ source, scene, subject, summary, objects, faces[], translation, thumb }`. Capped at 200 rows in `data/scans.json`. |
| `DELETE /api/scans/:id`   | Delete one scan. |
| `DELETE /api/scans`       | Clear all scans. |
| `POST /api/translate`     | Body `{ text, source }` → proxies MyMemory (`source|en`) → `{ translation, source, status }`. No API key. |

**Interpretation engine** — `interpret(detections, classification)` in `server.js`
counts detections by class, scores each scene by weighted signal matches, and
returns a summary + actionable recommendations. All request bodies are sanitised
at the boundary (clamped scores, string length caps).

**Database** — a tiny JSON store (`data/scans.json`) via `fs`. No native modules,
survives restarts. Thumbnails are small JPEG data-URIs so the file stays light.

---

## Frontend

### `scene.js` — the living world (Three.js r160)
- **Video background**: `public/media/domain-background.mp4` is a silent H.264
  excerpt (source seconds 9–52), cropped from 3840×2160 to the top 3840×1680 to
  remove burned-in subtitles, then scaled to 1920×840. Playback at **0.55×** makes
  the 43-second asset loop in about **78 seconds**. The cover-fitted video fills
  the viewport with an additional 1.12× zoom, lower exposure, a vignette, and a
  pointer-reactive radial zoom blur.
- **Glass shards**: 16 clear faceted `MeshPhysicalMaterial` meshes (10 on smaller
  screens; 8 with reduced motion) use `transmission: 1`, low thickness and
  restrained reflections. A **live cube-map** samples the same blurred video,
  so the transparent shards refract the scene and reflect its changing lights.
- **Fallbacks**: the native video remains visible when WebGL is unavailable;
  `public/media/domain-poster.jpg` provides its loading/failure poster.
- **2D glowing cursor**: a DOM ring + dot whose colour / size / glow shift by what
  it hovers (void, control, stage, text, status, tool).
- **Scroll motion**: GSAP + ScrollTrigger move each scene on x/y/z; the shard field
  travels on z through the scroll; pointer parallax on floating elements.
- Honours `prefers-reduced-motion` with a paused video and reduced scene motion;
  pauses playback/rendering while the tab is hidden.

### `app.js` — application + dashboard logic
- Loads COCO-SSD + MobileNet; warms up the SLM and face models in the background.
- **Image intake**: upload, drag-drop, sample (CORS-safe blobs), webcam.
- **Detection view**: boxes + confidence, MobileNet subject chips, metrics,
  scene interpretation, recommendations, per-object advisories, face rows.
- **Command console** (natural-language → canvas ops): `blur the background`,
  `grayscale`, `brighten/darken/contrast`, `highlight <class>`, `redact <class>`,
  `crop to <subject/class>`, `edges`, `count <class>`, `describe`, `reset`.
- **Live webcam**: continuous detection + face boxes (age/gender/emotion).
- **Reveal popup**: for-fun live person scoring — facial-feature score, symmetry,
  fashion-sense score + style label (clearly labelled as playful, not a judgment).
- **Session history** (thumbnails, click to restore) + **server database** modal.

### `extras.js` — face AI + OCR/translate
- `loadFaces()` — TinyFaceDetector + AgeGenderNet + (optional) landmarks &
  expressions from the face-api model CDN.
- `detectFaces()` — fast per-frame age/gender/emotion boxes.
- `analyzePersons()` — richer landmark-based scoring for the reveal popup.
- `ocrTranslate()` — Tesseract OCR → franc language detect → `/api/translate`.

### `nlu.js` — the SLM
Lazy-loads Transformers.js **DistilBERT-MNLI** and routes a free-text command to
the closest action label via zero-shot classification. An instant keyword router
is always the fallback, so the console works smoothly even before the model loads.

---

## Feature checklist vs. brief

| Requirement | Status |
|---|---|
| Accept an image | ✅ upload / drag / sample / webcam |
| Identify ≥3 object types | ✅ COCO-SSD (80) + MobileNet (1000) |
| Display detected result | ✅ labelled boxes + confidence + chips |
| Useful interpretation | ✅ scene reasoning + recommendations |
| Advanced: detection + confidence + action | ✅ |
| Extras | live webcam, face age/gender/emotion, OCR+translation, NL image editing, SLM routing, scan database, person scoring |

---

## Notes & limitations
- All models are **pretrained, open-source**, loaded from public CDNs — nothing is
  trained here (training would be out of scope for a live demo).
- Webcam needs permission in a modern browser. Model downloads and translation
  need network access; the keyword command router remains available if the SLM
  cannot load. Model loading time depends on the connection and device.
- The **person scoring** (facial-feature / fashion) is a lighthearted heuristic
  from landmarks and clothing-colour statistics. It is **not** an authoritative or
  scientific judgment of any person and is labelled as such in the UI.
- Translation uses the free MyMemory endpoint (rate-limited); OCR covers
  Latin-script languages by default.
