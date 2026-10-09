# AI Vision Inspector — Discussion & Decisions Log

A running record of what was asked, what was decided, and why — captured as the
project was built end-to-end with Claude Code (ruflo workflow).

---

## 1. Brief
Hackathon prompt **#1 — AI Vision Inspector 👁️**: accept an image, identify ≥3
object types, display the result, give a useful interpretation. Target the
Advanced tier (detection + confidence + recommendation) in ~90 minutes, hosted
on localhost with a backend + frontend.

## 2. Core technical decisions
- **Vision = client-side pretrained CNNs**, not a trained-from-scratch model:
  - **COCO-SSD** for localized object detection (80 classes, boxes, confidence).
  - **MobileNet v2** (ImageNet, 1000 classes) added later to *name the subject*,
    so the inspector says "golden retriever / espresso / library", not just
    generic boxes.
  - Rationale: no API keys, no GPU server, runs immediately, reliable for a demo.
- **Backend = Node/Express** serving the SPA + a reasoning API. Kept to a single
  dependency (express). The server turns raw detections + the MobileNet subject
  into a **scene classification + recommendations** (`/api/interpret`).
- **No manual dataset downloads**: the networks are already trained on COCO /
  ImageNet; their weights auto-download from CDNs. Downloading raw datasets
  client-side would add nothing.

## 3. Visual direction (user-led)
Iterated live with the user reviewing in the browser:
- **Mood**: liquid chrome / monochrome → later **playful & colourful**.
- **Background**: floating **3D refractive + reflective glass shards** → then a
  **cursor-reactive plasma** sky whose light the shards refract & reflect (via a
  live cube-map captured from the plasma on an isolated layer).
- **Cursor**: started as a 3D glass shard → changed to a **2D glowing cursor**
  that shifts colour / size / glow by hover target.
- **Liquid glass UI**: deep, refractive panels (bevel + specular sheen + chromatic
  edges). Fixed a readability problem (glass over plasma washed out text) by
  darkening/opacifying panels, lowering WebGL exposure, and adding a scrim.
- **Living eye**: the hero eye tracks the pointer, blinks on an organic cadence,
  and pulses.
- Bug fixed: shards looked **opaque** — caused by `transparent:true` + very high
  `envMapIntensity` + full iridescence pushing them into the alpha pass. Fixed by
  removing `transparent`, `transmission:1`, and restrained reflections so light
  bends *through* each facet.

## 4. Dashboard & interactivity
- Turned the stage into an **interactive dashboard**: image + controls on the
  left, a **command console** + history on the right.
- **Natural-language image editing** (no chatbot): commands map to real canvas
  operations — blur background, grayscale, brighten/darken/contrast, highlight
  class, redact class, crop to subject, edge-detect, count, describe, reset.
- **SLM**: an in-browser **DistilBERT-MNLI zero-shot** router (Transformers.js)
  maps free text → action, with an instant **keyword fallback** so the UI never
  blocks while the model downloads.

## 5. Webcam, faces, OCR, database (final additions)
- **Live webcam detection**: continuous object + face detection with a capture
  button.
- **Face AI** (face-api.js, pretrained): live **age, gender, emotion**, framed
  face boxes; works on uploads and the live feed.
- **OCR + translation**: Tesseract.js reads text, franc detects the language, and
  a backend MyMemory proxy translates it to English.
- **Reveal popup** (live, no capture): for-fun **facial-feature score, symmetry,
  and fashion-sense score + style label**, derived from face landmarks and
  clothing-colour statistics. Explicitly labelled as a playful heuristic, not a
  real judgment of anyone.
- **Database**: a file-backed JSON store (`data/scans.json`) persists every scan
  (uploads, webcam captures, live snapshots, OCR) with a **"database" button** to
  browse, inspect, and clear history.
- **Theme consistency**: all new surfaces (modal, popup, OCR panel, person cards)
  use the same liquid-glass + neon theme.

## 6. Verified results
- `POST /api/interpret` correctly classifies scenes (e.g., classroom, animals)
  and prepends the MobileNet subject + confidence.
- `POST/GET/DELETE /api/scans` persist and read back from `data/scans.json`.
- `POST /api/translate` returns English (e.g., *bonjour le monde → hello world*).
- All static assets serve `200`; all JS files pass `node -c` syntax checks.
- Visual/interaction layers (WebGL, GSAP, model inference) are reviewed live in
  the browser by the user (not headless-screenshottable).

## 7. Open limitations
- Model weights and webcam need a modern browser + network.
- Translation uses the free MyMemory tier (rate-limited); OCR defaults to
  Latin-script languages.
- face-api.js bundles its own TF build alongside tfjs 4.x — coexists, may log
  warnings; face features are guarded so the app continues if they fail to load.
- Person scoring is intentionally lighthearted and not a scientific measure.
```
