/* AI Vision Inspector — WebGL scene
 * -----------------------------------
 * - Slowed, cropped video with exposure control and radial zoom blur.
 * - Floating 3D glass shards that REFRACT (transmission) and REFLECT
 *   (live cube-map captured from the video) its changing light.
 * - 2D glowing cursor (DOM) that changes color/size/glow by hover target.
 * - Everything moves on x/y/z with scroll + pointer parallax.
 */
(() => {
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const THREE = window.THREE;
  const video = document.getElementById('backgroundVideo');
  // Playback rate applies even when WebGL is unavailable: the <video> is also
  // the fallback background. The asset itself has subtitles cropped away.
  video.defaultPlaybackRate = video.playbackRate = 1.0;
  const playBackground = () => {
    if (!reduce && !document.hidden) video.play().catch(() => {});
  };
  video.addEventListener('loadedmetadata', () => {
    video.playbackRate = 1.0;
    if (reduce) { video.currentTime = 2; video.pause(); }
    else playBackground();
  });
  window.addEventListener('pointerdown', playBackground, { passive: true });
  playBackground();

  if (!THREE) return;
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({
      canvas: document.getElementById('bg'),
      antialias: true, alpha: true, powerPreference: 'low-power',
    });
  } catch (error) {
    console.warn('Video background is using its CSS fallback.', error);
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.25));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.8;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(58, window.innerWidth / window.innerHeight, 0.1, 200);
  camera.position.set(0, 0, 10);
  scene.add(camera);

  const texture = new THREE.VideoTexture(video);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  const videoUniforms = {
    uVideo: { value: texture },
    uAspect: { value: window.innerWidth / window.innerHeight },
    uVideoAspect: { value: 1920 / 840 },
    uExposure: { value: 0.42 },
    uZoom: { value: 1.12 },
    uBlur: { value: 0.075 },
    uFocus: { value: new THREE.Vector2(0.5, 0.5) },
  };
  video.addEventListener('loadedmetadata', () => {
    videoUniforms.uVideoAspect.value = video.videoWidth / video.videoHeight;
  });
  // A radial, multi-sample zoom blur is applied to the actual video pixels.
  // The identical shader lights the glass, so refractions follow each frame.
  const videoMat = new THREE.ShaderMaterial({
    uniforms: videoUniforms,
    depthWrite: false,
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: `
      uniform sampler2D uVideo;
      uniform float uAspect;
      uniform float uVideoAspect;
      uniform float uExposure;
      uniform float uZoom;
      uniform float uBlur;
      uniform vec2 uFocus;
      varying vec2 vUv;
      void main() {
        vec2 cover = vec2(min(1.0, uAspect / uVideoAspect), min(1.0, uVideoAspect / uAspect));
        vec2 uv = vUv + (uFocus - vec2(0.5)) * 0.025;
        vec2 offset = (uv - 0.5) * cover / uZoom;
        vec3 color = vec3(0.0);
        for (int i = 0; i < 5; i++) {
          float progress = float(i) / 4.0;
          color += texture2D(uVideo, 0.5 + offset * (1.0 - progress * uBlur)).rgb;
        }
        color *= uExposure / 5.0;
        float vignette = 1.0 - 0.24 * smoothstep(0.1, 0.72, distance(vUv, vec2(0.5)));
        gl_FragColor = vec4(color * vignette, 1.0);
        #include <colorspace_fragment>
      }`,
  });
  const background = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), videoMat);
  background.position.z = -85;
  background.renderOrder = -10;
  camera.add(background);
  function sizeBackground() {
    const height = 2 * 85 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    background.scale.set(height * camera.aspect, height, 1);
    videoUniforms.uAspect.value = camera.aspect;
  }
  sizeBackground();

  // Only the cube camera sees this sphere; the main view uses a cover-fitted
  // plane, avoiding stretched panorama video and making phone layouts work.
  const envMat = videoMat.clone();
  envMat.side = THREE.BackSide;
  envMat.uniforms.uVideo.value = texture;
  envMat.uniforms.uAspect.value = 2;
  envMat.uniforms.uExposure.value = 0.8;
  const sky = new THREE.Mesh(new THREE.SphereGeometry(60, 24, 16), envMat);
  sky.layers.set(1);
  scene.add(sky);
  const cubeRT = new THREE.WebGLCubeRenderTarget(32, {
    minFilter: THREE.LinearMipmapLinearFilter, generateMipmaps: true,
  });
  const cubeCam = new THREE.CubeCamera(1, 120, cubeRT);
  cubeCam.layers.set(1);
  scene.add(cubeCam);
  scene.environment = cubeRT.texture;
  scene.add(new THREE.AmbientLight(0xffffff, 0.15));
  const key = new THREE.DirectionalLight(0xffffff, 0.45);
  key.position.set(5, 6, 8);
  scene.add(key);

  // Thin, clear facets transmit the video; restrained highlights keep them
  // transparent rather than looking like solid chrome objects.
  const TINTS = [0xffffff, 0xfff2ff, 0xf2ffff, 0xfffbe6, 0xf3f0ff];
  function glassMat(tint) {
    // Real refractive glass: transmission 1, NO `transparent` flag (that would
    // move it to the alpha pass and kill refraction), restrained reflections
    // so the video visibly bends through each facet.
    return new THREE.MeshPhysicalMaterial({
      color: tint, metalness: 0, roughness: 0.0,
      transmission: 1.0, thickness: 0.65, ior: 1.22,
      clearcoat: 0.18, clearcoatRoughness: 0.05,
      envMap: cubeRT.texture, envMapIntensity: 0.42,
      iridescence: 0.08, iridescenceIOR: 1.3, iridescenceThicknessRange: [100, 400],
      attenuationColor: new THREE.Color(0xeaf2ff), attenuationDistance: 40,
      specularIntensity: 0.35, flatShading: true,
    });
  }
  const GEOS = [
    () => new THREE.OctahedronGeometry(1, 0),
    () => new THREE.TetrahedronGeometry(1.25, 0),
    () => new THREE.IcosahedronGeometry(1, 0),
    () => new THREE.ConeGeometry(0.95, 2.0, 5, 1),
    () => new THREE.DodecahedronGeometry(1, 0),
    () => new THREE.TorusKnotGeometry(0.6, 0.26, 32, 6),
  ];
  const field = new THREE.Group();
  scene.add(field);
  const shards = [];
  const COUNT = reduce ? 6 : (window.innerWidth < 700 ? 8 : 12);
  for (let i = 0; i < COUNT; i++) {
    const mesh = new THREE.Mesh(GEOS[i % GEOS.length](), glassMat(TINTS[i % TINTS.length]));
    const s = 0.65 + Math.random() * 1.3;
    mesh.scale.setScalar(s);
    mesh.position.set(
      (Math.random() - 0.5) * 20,
      (Math.random() - 0.5) * 14,
      3 - Math.random() * 20                        // some close to camera
    );
    mesh.rotation.set(Math.random() * 6, Math.random() * 6, Math.random() * 6);
    mesh.userData.spin = {
      x: (Math.random() - 0.5) * 0.006,
      y: (Math.random() - 0.5) * 0.006,
      z: (Math.random() - 0.5) * 0.004,
      amp: 0.4 + Math.random() * 0.9,
      spd: 0.3 + Math.random() * 0.7,
      phase: Math.random() * 6.28,
      baseY: 0,
    };
    mesh.userData.spin.baseY = mesh.position.y;
    field.add(mesh);
    shards.push(mesh);
  }

  // ---- 2D glowing cursor (DOM) --------------------------------------------
  const ring = document.getElementById('cursorRing');
  const dot = document.getElementById('cursor');
  const STATES = {
    void:    { color: '#9fe8ff', size: 30,  glow: 0.5, blur: 6 },
    control: { color: '#ffd24d', size: 60,  glow: 1.0, blur: 10 },
    stage:   { color: '#7afcff', size: 74,  glow: 1.0, blur: 12 },
    text:    { color: '#ff6edb', size: 50,  glow: 0.9, blur: 10 },
    status:  { color: '#5effa0', size: 46,  glow: 0.9, blur: 9 },
    tool:    { color: '#b58cff', size: 50,  glow: 0.9, blur: 9 },
  };
  function classify(el) {
    if (!el) return STATES.void;
    if (el.closest('.status')) return STATES.status;
    if (el.closest('.toolbar, .tool')) return STATES.tool;
    if (el.closest('.control')) return STATES.control;
    if (el.closest('.stage, .drop-prompt, .canvas-wrap')) return STATES.stage;
    if (el.closest('.mega, h1, h2, .whisper, .eyebrow, .scene-badge, .m-val, .recs li, .detection, .read-summary')) return STATES.text;
    if (el.closest('[data-magnetic]')) return STATES.control;
    return STATES.void;
  }
  const cur = { x: 0, y: 0, tx: 0, ty: 0 };
  let st = STATES.void, curSize = 30, curGlow = 0.5;

  // ---- Pointer + scroll ---------------------------------------------------
  const ptr = { x: 0, y: 0, tx: 0, ty: 0 };
  const mouseUV = { x: 0.5, y: 0.5, px: 0.5, py: 0.5, v: 0 };
  let scrollT = 0, scrollTarget = 0;
  window.addEventListener('pointermove', (e) => {
    cur.tx = e.clientX; cur.ty = e.clientY;
    if (dot) { dot.style.left = e.clientX + 'px'; dot.style.top = e.clientY + 'px'; }
    ptr.tx = (e.clientX / window.innerWidth) * 2 - 1;
    ptr.ty = -((e.clientY / window.innerHeight) * 2 - 1);
    // Pointer coordinates also drive the glass parallax.
    const nx = e.clientX / window.innerWidth;
    const ny = 1 - e.clientY / window.innerHeight;
    mouseUV.v = Math.min(1, mouseUV.v + Math.hypot(nx - mouseUV.px, ny - mouseUV.py) * 6);
    mouseUV.x = nx; mouseUV.y = ny; mouseUV.px = nx; mouseUV.py = ny;
    st = classify(document.elementFromPoint(e.clientX, e.clientY));
  }, { passive: true });
  window.addEventListener('pointerdown', () => { if (ring) ring.style.transform = 'translate(-50%,-50%) scale(0.7)'; });
  window.addEventListener('pointerup', () => { if (ring) ring.style.transform = 'translate(-50%,-50%) scale(1)'; });

  function updateScroll() {
    const max = document.body.scrollHeight - window.innerHeight;
    scrollTarget = max > 0 ? window.scrollY / max : 0;
  }
  window.addEventListener('scroll', updateScroll, { passive: true });
  updateScroll();

  // ---- Render loop --------------------------------------------------------
  const clock = new THREE.Clock();
  let lastReflection = -1, lastFrame = 0, raf;
  let sceneReady = false;
  function tick(now = 0) {
    if (document.hidden) return;
    raf = requestAnimationFrame(tick);
    if (now - lastFrame < (reduce ? 120 : 1000 / 30)) return;
    const delta = Math.min((now - lastFrame) / 16.667, 3);
    lastFrame = now;
    const t = reduce ? 0 : clock.getElapsedTime();
    if (video.readyState < 2) return;
    if (!sceneReady) {
      sceneReady = true;
      document.body.classList.add('video-scene-ready');
    }
    if (!reduce) {
      videoUniforms.uFocus.value.x += (mouseUV.x - videoUniforms.uFocus.value.x) * 0.08;
      videoUniforms.uFocus.value.y += (mouseUV.y - videoUniforms.uFocus.value.y) * 0.08;
      videoUniforms.uBlur.value = 0.075 + mouseUV.v * 0.035;
      mouseUV.v *= 0.92;
      envMat.uniforms.uFocus.value.copy(videoUniforms.uFocus.value);
      envMat.uniforms.uBlur.value = videoUniforms.uBlur.value;
    }
    ptr.x += (ptr.tx - ptr.x) * 0.06;
    ptr.y += (ptr.ty - ptr.y) * 0.06;
    scrollT += (scrollTarget - scrollT) * 0.07;

    if (now - lastReflection > 300) {
      cubeCam.update(renderer, scene);
      lastReflection = now;
    }

    field.rotation.y = ptr.x * 0.4 + t * 0.03;
    field.rotation.x = ptr.y * 0.25;
    field.position.z = scrollT * 14;
    field.position.x = ptr.x * 1.6;
    field.position.y = -ptr.y * 1.2 - scrollT * 2.5;
    for (const m of shards) {
      const u = m.userData.spin;
      if (!reduce) { m.rotation.x += u.x * delta; m.rotation.y += u.y * delta; m.rotation.z += u.z * delta; }
      m.position.y = u.baseY + Math.sin(t * u.spd + u.phase) * u.amp;
    }

    camera.position.z = 10 - scrollT * 2;
    camera.lookAt(0, -scrollT * 1.5, 0);

    // 2D cursor: trail + morph color/size/glow.
    cur.x += (cur.tx - cur.x) * 0.2;
    cur.y += (cur.ty - cur.y) * 0.2;
    curSize += (st.size - curSize) * 0.15;
    curGlow += (st.glow - curGlow) * 0.15;
    if (ring) {
      ring.style.left = cur.x + 'px';
      ring.style.top = cur.y + 'px';
      ring.style.width = ring.style.height = curSize + 'px';
      ring.style.background = `radial-gradient(circle, ${hexA(st.color, 0.55)} 0%, ${hexA(st.color, 0.12)} 45%, transparent 72%)`;
      ring.style.boxShadow = `0 0 ${18 * curGlow}px ${hexA(st.color, 0.8 * curGlow)}, 0 0 ${40 * curGlow}px ${hexA(st.color, 0.4 * curGlow)}`;
      ring.style.borderColor = hexA(st.color, 0.6);
    }
    if (dot) dot.style.background = st.color;

    renderer.render(scene, camera);
  }

  function hexA(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }

  tick();
  document.addEventListener('visibilitychange', () => {
    cancelAnimationFrame(raf);
    if (document.hidden) video.pause();
    else { playBackground(); tick(performance.now()); }
  });
  renderer.domElement.addEventListener('webglcontextlost', (event) => {
    event.preventDefault();
    cancelAnimationFrame(raf);
    document.body.classList.remove('video-scene-ready');
  });
  renderer.domElement.addEventListener('webglcontextrestored', () => {
    sceneReady = false;
    tick(performance.now());
  });

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    sizeBackground();
    updateScroll();
  });

  // ---- GSAP scroll motion for DOM layers ----------------------------------
  if (window.gsap && window.ScrollTrigger && !reduce) {
    gsap.registerPlugin(ScrollTrigger);
    gsap.utils.toArray('.scene:not(.scene--hero)').forEach((el) => {
      gsap.fromTo(el,
        { autoAlpha: 0, z: -500, rotationX: 14, y: 90 },
        { autoAlpha: 1, z: 0, rotationX: 0, y: 0, ease: 'power2.out',
          scrollTrigger: { trigger: el, start: 'top 88%', end: 'top 35%', scrub: 1 } });
    });
    gsap.utils.toArray('[data-reveal]').forEach((el) => {
      gsap.fromTo(el,
        { autoAlpha: 0, y: 45, z: -140 },
        { autoAlpha: 1, y: 0, z: 0, ease: 'power3.out', duration: 1,
          scrollTrigger: { trigger: el, start: 'top 92%' } });
    });
    const floats = gsap.utils.toArray('[data-float]');
    (function floatLoop() {
      floats.forEach((el, i) => {
        const d = (i % 3 + 1) * 7;
        el.style.transform = `translate3d(${ptr.x * d}px, ${ptr.y * d}px, 0)`;
      });
      requestAnimationFrame(floatLoop);
    })();
  } else {
    document.querySelectorAll('.scene, [data-reveal]').forEach((el) => {
      el.style.opacity = 1; el.style.visibility = 'visible';
    });
  }

  window.VisionScene = {
    flash() {
      shards.forEach((m) => { m.userData.spin.x *= 2.4; m.userData.spin.y *= 2.4; });
      setTimeout(() => shards.forEach((m) => { m.userData.spin.x /= 2.4; m.userData.spin.y /= 2.4; }), 650);
    },
  };
})();
