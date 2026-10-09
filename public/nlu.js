/* nlu.js — optional in-browser SLM for intent routing.
 * Uses Transformers.js zero-shot classification (DistilBERT-MNLI) to map a
 * free-text command to one of the dashboard's action labels. Loads lazily in
 * the background; app.js always has an instant keyword fallback, so the UI
 * stays smooth whether or not the model finishes downloading.
 */
window.NLU = {
  ready: false,
  loading: false,
  _pipe: null,
  _loadPromise: null,
  _routing: Promise.resolve(),
  onstatus: null,

  _setStatus(s) { if (typeof this.onstatus === 'function') this.onstatus(s); },

  load() {
    if (this.ready) return Promise.resolve(true);
    if (this._loadPromise) return this._loadPromise;
    this.loading = true;
    this._setStatus('loading SLM…');
    this._loadPromise = (async () => {
      try {
        const { pipeline, env } = await import(
          'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js'
        );
        env.allowLocalModels = false;
        env.useBrowserCache = true;
        // Localhost is not cross-origin isolated: avoid worker/SAB failures.
        env.backends.onnx.wasm.numThreads = 1;
        this._pipe = await pipeline(
          'zero-shot-classification',
          'Xenova/distilbert-base-uncased-mnli',
          { quantized: true }
        );
        this.ready = true;
        this._setStatus('SLM ready');
        return true;
      } catch (err) {
        console.warn('NLU model unavailable, using keyword fallback:', err);
        this._setStatus('keyword mode');
        return false;
      } finally {
        this.loading = false;
        this._loadPromise = null;
      }
    })();
    return this._loadPromise;
  },

  /** Returns the best-matching label, or null if the model isn't ready. */
  async route(text, labels) {
    if (!this.ready || !this._pipe || typeof text !== 'string' || !text.trim() || !Array.isArray(labels) || !labels.length) return null;
    // ONNX sessions must not run overlapping inference requests.
    const request = this._routing.then(async () => {
      try {
        const out = await this._pipe(text.slice(0, 1000), labels, { truncation: true });
        return { label: out.labels[0], score: out.scores[0] };
      } catch (err) {
        console.warn('NLU route failed:', err);
        return null;
      }
    });
    this._routing = request.then(() => undefined, () => undefined);
    return request;
  },
};

window.dispatchEvent(new Event('nluavailable'));
