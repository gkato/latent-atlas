/* ==========================================================================
   LATENT ATLAS — viz.js
   Kit de visualização: loop de canvas com ciclo de vida, cores do tema,
   primitivas de desenho (matrizes, setas, texto, brilho), mapas de cor,
   controles de UI (slider, segmentado, toggle, stepper, readout),
   tensor 3D em CSS, e as animações da home e dos cabeçalhos.
   ========================================================================== */
(function () {
  'use strict';
  const LA = (window.LA = window.LA || {});
  const V = (LA.viz = LA.viz || {});
  const UI = (LA.ui = LA.ui || {});

  /* ----------------------------------------------------------- math utils */
  V.clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  V.lerp = (a, b, t) => a + (b - a) * t;
  V.inv = (a, b, x) => (x - a) / (b - a || 1);
  V.smooth = (t) => t * t * (3 - 2 * t);
  V.ease = {
    inOut: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
    out: (t) => 1 - Math.pow(1 - t, 3),
    in: (t) => t * t * t,
    back: (t) => { const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2); },
    elastic: (t) => (t === 0 || t === 1 ? t : Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * ((2 * Math.PI) / 3)) + 1),
  };
  /** onda triangular/pingue-pongue 0..1 com período p */
  V.pingpong = (t, p) => { const x = (t % p) / p; return x < 0.5 ? x * 2 : 2 - x * 2; };
  /** fase 0..1 dentro de um ciclo de período p, e índice do ciclo */
  V.phase = (t, p) => ({ f: (t % p) / p, i: Math.floor(t / p) });
  /** gerador pseudoaleatório determinístico (mulberry32) */
  V.rng = function (seed) {
    let a = (seed >>> 0) || 1;
    const r = function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    r.normal = () => { let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    r.int = (n) => Math.floor(r() * n);
    r.pick = (arr) => arr[Math.floor(r() * arr.length)];
    return r;
  };
  V.softmax = (xs, temp = 1) => { const m = Math.max(...xs); const e = xs.map((x) => Math.exp((x - m) / temp)); const s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); };
  V.fmt = (x, d = 2) => (Math.abs(x) >= 1e4 || (Math.abs(x) < 1e-3 && x !== 0) ? x.toExponential(d) : (+x).toFixed(d));
  V.bytes = (b) => { const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB']; let i = 0; while (Math.abs(b) >= 1024 && i < u.length - 1) { b /= 1024; i++; } return (i ? b.toFixed(b < 10 ? 2 : 1) : b.toFixed(0)) + ' ' + u[i]; };

  /* ---------------------------------------------------------------- cores */
  let cache = null;
  function readColors() {
    const cs = getComputedStyle(document.documentElement);
    const g = (n) => cs.getPropertyValue(n).trim();
    cache = {
      a: g('--viz-a'), b: g('--viz-b'), c: g('--viz-c'), d: g('--viz-d'), e: g('--viz-e'), f: g('--viz-f'),
      ink: g('--viz-ink'), muted: g('--viz-muted'), grid: g('--viz-grid'), bg: g('--viz-bg'), panel: g('--viz-panel'),
      line: g('--line-2'), text: g('--text'), text2: g('--text-2'), text3: g('--text-3'),
      glm: g('--glm'), qwen: g('--qwen'), ds: g('--ds'), red: g('--red'),
      mint: g('--mint'), lavender: g('--lavender'), pink: g('--pink'), purple: g('--purple'), magenta: g('--magenta'), amber: g('--amber'),
      dark: LA.theme ? LA.theme.get() === 'dark' : true,
    };
    cache.series = [cache.a, cache.b, cache.c, cache.d, cache.e, cache.f];
    return cache;
  }
  /** Cores atuais (sensíveis ao tema). Use dentro do draw: const C = LA.viz.C(); */
  V.C = () => cache || readColors();
  if (LA.theme) LA.theme.onChange(() => { cache = null; });

  /** converte '#rrggbb' + alpha em rgba() */
  V.alpha = function (color, a) {
    if (!color) return `rgba(128,128,128,${a})`;
    color = color.trim();
    if (color.startsWith('#')) {
      let c = color.slice(1);
      if (c.length === 3) c = c.split('').map((x) => x + x).join('');
      const n = parseInt(c.slice(0, 6), 16);
      return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
    }
    if (color.startsWith('rgb')) { const p = color.match(/[\d.]+/g); return `rgba(${p[0]},${p[1]},${p[2]},${a})`; }
    return color;
  };
  function hex2rgb(c) { c = c.replace('#', ''); if (c.length === 3) c = c.split('').map((x) => x + x).join(''); const n = parseInt(c, 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  V.mix = function (c1, c2, t) {
    const a = hex2rgb(c1), b = hex2rgb(c2);
    return `rgb(${Math.round(V.lerp(a[0], b[0], t))},${Math.round(V.lerp(a[1], b[1], t))},${Math.round(V.lerp(a[2], b[2], t))})`;
  };
  /**
   * Mapa de cor. v em [0,1] (seq) ou [-1,1] (div).
   * 'seq'  : fundo → verde → menta
   * 'heat' : roxo → magenta → âmbar
   * 'div'  : lavanda (neg) ← fundo → menta (pos)
   */
  V.cmap = function (v, name = 'seq') {
    const C = V.C();
    const bg = C.dark ? '#0b1a13' : '#f2f3f7';
    if (name === 'div') {
      v = V.clamp(v, -1, 1);
      return v >= 0 ? V.mix(bg, C.dark ? '#3FF49B' : '#12985A', Math.pow(v, 0.8)) : V.mix(bg, C.dark ? '#BE8EFD' : '#5A12C4', Math.pow(-v, 0.8));
    }
    v = V.clamp(v, 0, 1);
    if (name === 'heat') {
      const stops = C.dark ? ['#1a0b33', '#440099', '#D62598', '#FC93D6', '#FCBB00'] : ['#f5f0ff', '#BE8EFD', '#D62598', '#C21F88', '#B77900'];
      const x = v * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x));
      return V.mix(stops[i], stops[i + 1], x - i);
    }
    const stops = C.dark ? [bg, '#0f4d2f', '#169D5A', '#3FF49B', '#E9FAF2'] : [bg, '#c8f3dc', '#26D07C', '#169D5A', '#0A4D2B'];
    const x = v * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x));
    return V.mix(stops[i], stops[i + 1], x - i);
  };

  /* ------------------------------------------------------------ canvas I/O */
  /**
   * Loop de animação com ciclo de vida automático.
   *   LA.viz.loop(canvas, (g, w, h, t, dt, S) => { ... }, { static:false, speed:1 })
   * - Ajusta DPR e redimensiona com o container.
   * - Pausa fora da tela, em aba oculta e no botão global de pausa.
   * - Limpeza automática ao trocar de capítulo.
   * Retorna um handle: { stop(), redraw(), S (estado livre), pointer, canvas }.
   */
  V.loop = function (canvas, draw, opts = {}) {
    if (!canvas) throw new Error('LA.viz.loop: canvas inexistente');
    const g = canvas.getContext('2d');
    const H = { canvas, g, S: opts.state || {}, t: 0, w: 1, h: 1, speed: opts.speed || 1, running: true };
    let raf = 0, visible = true, last = performance.now();
    const pointer = (H.pointer = { x: -1, y: -1, down: false, inside: false, clicked: false });
    function fit() {
      const r = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      H.w = Math.max(1, r.width); H.h = Math.max(1, r.height);
      const W = Math.round(H.w * dpr), Hh = Math.round(H.h * dpr);
      if (canvas.width !== W || canvas.height !== Hh) { canvas.width = W; canvas.height = Hh; }
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    function frame(now) {
      raf = 0;
      if (!H.running) return;
      const dt = Math.min(0.05, Math.max(0, (now - last) / 1000));
      last = now;
      if (!LA.paused || opts.static) H.t += dt * H.speed;
      if (H.w < 8 || H.h < 8) { fit(); if (!opts.static) schedule(); return; }
      try {
        g.save();
        g.clearRect(0, 0, H.w, H.h);
        draw(g, H.w, H.h, H.t, dt, H.S, H);
        g.restore();
      } catch (e) {
        H.running = false;
        console.error('[viz] erro no draw', e);
        const fig = canvas.closest('.la-fig');
        if (fig && !fig.querySelector('.la-fig-error')) fig.appendChild(Object.assign(document.createElement('div'), { className: 'la-fig-error', textContent: (LA.t ? LA.t('viz.err') : 'Erro na animação: ') + e.message }));
        return;
      }
      pointer.clicked = false;
      if (!opts.static && visible && !document.hidden) schedule();
    }
    function schedule() { if (!raf && H.running) raf = requestAnimationFrame(frame); }
    H.redraw = () => { last = performance.now(); schedule(); };
    fit();
    const ro = new ResizeObserver(() => { fit(); H.redraw(); });
    ro.observe(canvas);
    const io = new IntersectionObserver((es) => { visible = es[es.length - 1].isIntersecting; if (visible) { last = performance.now(); schedule(); } }, { rootMargin: '80px' });
    io.observe(canvas);
    const onVis = () => { if (!document.hidden) { last = performance.now(); schedule(); } };
    document.addEventListener('visibilitychange', onVis);
    const pos = (e) => { const r = canvas.getBoundingClientRect(); const p = e.touches ? e.touches[0] : e; pointer.x = p.clientX - r.left; pointer.y = p.clientY - r.top; };
    const onMove = (e) => { pos(e); pointer.inside = true; if (opts.static) H.redraw(); };
    const onLeave = () => { pointer.inside = false; pointer.x = pointer.y = -1; if (opts.static) H.redraw(); };
    const onDown = (e) => { pos(e); pointer.down = true; if (opts.static) H.redraw(); };
    const onUp = () => { if (pointer.down) pointer.clicked = true; pointer.down = false; H.redraw(); };
    canvas.addEventListener('pointermove', onMove);
    canvas.addEventListener('pointerleave', onLeave);
    canvas.addEventListener('pointerdown', onDown);
    window.addEventListener('pointerup', onUp);
    H.stop = () => {
      H.running = false; if (raf) cancelAnimationFrame(raf);
      ro.disconnect(); io.disconnect(); document.removeEventListener('visibilitychange', onVis);
      canvas.removeEventListener('pointermove', onMove); canvas.removeEventListener('pointerleave', onLeave);
      canvas.removeEventListener('pointerdown', onDown); window.removeEventListener('pointerup', onUp);
    };
    if (LA.onCleanup) LA.onCleanup(H.stop);
    if (LA.theme) LA.onCleanup(LA.theme.onChange(() => setTimeout(H.redraw, 0)));
    schedule();
    return H;
  };

  /** Atalho: encontra/cria o canvas dentro de uma figura e inicia o loop. */
  V.fig = function (figOrSel, draw, opts) {
    const fig = typeof figOrSel === 'string' ? document.querySelector(figOrSel) : figOrSel;
    if (!fig) throw new Error('LA.viz.fig: figura não encontrada: ' + figOrSel);
    let cv = fig.querySelector('canvas');
    if (!cv) {
      let st = fig.querySelector('.la-fig-stage');
      if (!st) { st = document.createElement('div'); st.className = 'la-fig-stage'; st.style.height = (opts && opts.height || 320) + 'px'; fig.insertBefore(st, fig.querySelector('.la-fig-controls, figcaption')); }
      cv = document.createElement('canvas'); st.appendChild(cv);
    }
    return V.loop(cv, draw, opts);
  };

  /** Interval/timeout com limpeza automática. */
  V.every = (ms, fn) => { const id = setInterval(() => { if (!LA.paused) fn(); }, ms); LA.onCleanup && LA.onCleanup(() => clearInterval(id)); return id; };
  V.after = (ms, fn) => { const id = setTimeout(fn, ms); LA.onCleanup && LA.onCleanup(() => clearTimeout(id)); return id; };

  /* --------------------------------------------------------- 2D primitives */
  V.rr = function (g, x, y, w, h, r = 6) {
    r = Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2);
    g.beginPath();
    g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r);
    g.closePath();
  };
  V.box = function (g, x, y, w, h, o = {}) {
    V.rr(g, x, y, w, h, o.r == null ? 8 : o.r);
    if (o.fill) { g.fillStyle = o.fill; g.fill(); }
    if (o.stroke) { g.strokeStyle = o.stroke; g.lineWidth = o.lw || 1.5; g.stroke(); }
    if (o.label) V.text(g, o.label, x + w / 2, y + h / 2, { size: o.size || 12, color: o.color || V.C().ink, align: 'center', base: 'middle', font: o.font || 'sans', weight: o.weight || 500 });
  };
  V.font = (size = 12, font = 'sans', weight = 400) => {
    const fam = font === 'mono' ? getComputedStyle(document.documentElement).getPropertyValue('--font-mono') : font === 'display' ? getComputedStyle(document.documentElement).getPropertyValue('--font-display') : getComputedStyle(document.documentElement).getPropertyValue('--font-sans');
    return `${weight} ${size}px ${fam}`;
  };
  let fontCache = {};
  V.text = function (g, s, x, y, o = {}) {
    const key = (o.size || 12) + (o.font || 'sans') + (o.weight || 400);
    g.font = fontCache[key] || (fontCache[key] = V.font(o.size || 12, o.font || 'sans', o.weight || 400));
    g.fillStyle = o.color || V.C().ink;
    g.textAlign = o.align || 'left';
    g.textBaseline = o.base || 'alphabetic';
    if (o.alpha != null) g.globalAlpha = o.alpha;
    let str = String(s);
    if ((o.font || 'sans') === 'mono') str = V.noLig(str);
    if (o.maxW) g.fillText(str, x, y, o.maxW); else g.fillText(str, x, y);
    if (o.alpha != null) g.globalAlpha = 1;
  };
  /** evita ligaduras da fonte mono (<| |> -> != ...) inserindo ZWNJ entre símbolos */
  V.noLig = (str) => str.replace(/([<>|=!\-:*+./&~^])(?=[<>|=!\-:*+./&~^])/g, '$1\u200C');
  /**
   * Texto com quebra de linha no canvas.
   *   LA.viz.wrap(g, texto, x, y, maxW, { lineH: 16, size, font, weight, color, align, base, maxLines })
   * Retorna a altura ocupada (px).
   */
  V.wrap = function (g, s, x, y, maxW, o = {}) {
    const size = o.size || 12, lh = o.lineH || Math.round(size * 1.35);
    g.font = V.font(size, o.font || 'sans', o.weight || 400);
    const words = String(s).split(/\s+/);
    const lines = [];
    let cur = '';
    words.forEach((wd) => {
      const t = cur ? cur + ' ' + wd : wd;
      if (g.measureText(t).width > maxW && cur) { lines.push(cur); cur = wd; } else cur = t;
    });
    if (cur) lines.push(cur);
    const L = o.maxLines ? lines.slice(0, o.maxLines) : lines;
    if (o.maxLines && lines.length > o.maxLines) L[L.length - 1] = L[L.length - 1].replace(/\s*\S*$/, '…');
    L.forEach((ln, i) => V.text(g, ln, x, y + i * lh, Object.assign({}, o, { maxW: undefined })));
    return L.length * lh;
  };
  if (LA.theme) LA.theme.onChange(() => { fontCache = {}; });
  document.fonts && document.fonts.ready && document.fonts.ready.then(() => { fontCache = {}; });

  V.line = function (g, x1, y1, x2, y2, o = {}) {
    g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2);
    g.strokeStyle = o.color || V.C().muted; g.lineWidth = o.lw || 1.5;
    if (o.dash) g.setLineDash(o.dash); if (o.dashOffset) g.lineDashOffset = o.dashOffset;
    g.stroke(); g.setLineDash([]);
  };
  V.arrow = function (g, x1, y1, x2, y2, o = {}) {
    const col = o.color || V.C().muted, lw = o.lw || 1.8, hs = o.head || 8;
    const a = Math.atan2(y2 - y1, x2 - x1);
    g.beginPath();
    if (o.curve) {
      const mx = (x1 + x2) / 2 - Math.sin(a) * o.curve, my = (y1 + y2) / 2 + Math.cos(a) * o.curve;
      g.moveTo(x1, y1); g.quadraticCurveTo(mx, my, x2, y2);
      const a2 = Math.atan2(y2 - my, x2 - mx);
      g.strokeStyle = col; g.lineWidth = lw; if (o.dash) g.setLineDash(o.dash); g.stroke(); g.setLineDash([]);
      head(a2);
    } else {
      g.moveTo(x1, y1); g.lineTo(x2 - Math.cos(a) * hs * 0.6, y2 - Math.sin(a) * hs * 0.6);
      g.strokeStyle = col; g.lineWidth = lw; if (o.dash) g.setLineDash(o.dash); g.stroke(); g.setLineDash([]);
      head(a);
    }
    function head(an) {
      g.beginPath(); g.moveTo(x2, y2);
      g.lineTo(x2 - hs * Math.cos(an - 0.45), y2 - hs * Math.sin(an - 0.45));
      g.lineTo(x2 - hs * Math.cos(an + 0.45), y2 - hs * Math.sin(an + 0.45));
      g.closePath(); g.fillStyle = col; g.fill();
    }
  };
  V.glow = function (g, x, y, r, color, a = 1) {
    const gr = g.createRadialGradient(x, y, 0, x, y, r);
    gr.addColorStop(0, V.alpha(color, 0.9 * a)); gr.addColorStop(0.35, V.alpha(color, 0.35 * a)); gr.addColorStop(1, V.alpha(color, 0));
    g.fillStyle = gr; g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fill();
  };
  V.dot = function (g, x, y, r, color) { g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fillStyle = color; g.fill(); };
  /** ponto viajando ao longo de um segmento (t 0..1) */
  V.along = (x1, y1, x2, y2, t) => [V.lerp(x1, x2, t), V.lerp(y1, y2, t)];
  V.bez = (p0, p1, p2, p3, t) => { const u = 1 - t; return [u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0], u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1]]; };
  V.grid = function (g, w, h, step = 24) {
    g.strokeStyle = V.C().grid; g.lineWidth = 1; g.beginPath();
    for (let x = 0.5; x < w; x += step) { g.moveTo(x, 0); g.lineTo(x, h); }
    for (let y = 0.5; y < h; y += step) { g.moveTo(0, y); g.lineTo(w, y); }
    g.stroke();
  };

  /**
   * Desenha uma matriz como heatmap.
   * opts: { x, y, cell (px) | w,h, data: number[][], min, max, cmap:'seq'|'div'|'heat',
   *         gap, r, values:bool (escreve números), digits, hiRow, hiCol, hiCells:[[i,j]],
   *         label, rowLabels, colLabels, stroke }
   * Retorna { x, y, w, h, cw, ch, cellAt(px,py) -> [i,j] | null }
   */
  V.matrix = function (g, o) {
    const C = V.C();
    const data = o.data, R = data.length, K = data[0] ? data[0].length : 0;
    const gap = o.gap == null ? 2 : o.gap;
    const cw = o.cell || (o.w ? (o.w - gap * (K - 1)) / K : 18), ch = o.cellH || o.cell || (o.h ? (o.h - gap * (R - 1)) / R : cw);
    let mn = o.min, mx = o.max;
    if (mn == null || mx == null) { let a = Infinity, b = -Infinity; data.forEach((r) => r.forEach((v) => { if (v < a) a = v; if (v > b) b = v; })); if (mn == null) mn = a; if (mx == null) mx = b; }
    const cm = o.cmap || 'seq';
    for (let i = 0; i < R; i++) for (let j = 0; j < K; j++) {
      const v = data[i][j];
      const x = o.x + j * (cw + gap), y = o.y + i * (ch + gap);
      let t = cm === 'div' ? v / Math.max(Math.abs(mn), Math.abs(mx), 1e-9) : (v - mn) / (mx - mn || 1);
      g.fillStyle = v == null || Number.isNaN(v) ? C.panel : (o.colorFn ? o.colorFn(v, i, j) : V.cmap(t, cm));
      V.rr(g, x, y, cw, ch, o.r == null ? Math.min(4, cw / 5) : o.r); g.fill();
      if (o.stroke) { g.strokeStyle = o.stroke; g.lineWidth = 1; g.stroke(); }
      if (o.values && cw >= 15 && ch >= 12) {
        const light = cm === 'div' ? Math.abs(t) > 0.55 : t > 0.6;
        V.text(g, o.fmt ? o.fmt(v) : (+v).toFixed(o.digits == null ? 1 : o.digits), x + cw / 2, y + ch / 2 + 0.5, { size: Math.max(9, Math.min(12, cw * 0.4)), font: 'mono', align: 'center', base: 'middle', color: light && C.dark ? '#00210F' : light ? '#fff' : C.ink });
      }
    }
    const W = K * cw + (K - 1) * gap, Ht = R * ch + (R - 1) * gap;
    const hl = (x, y, w, h) => { g.strokeStyle = o.hiColor || C.c; g.lineWidth = 2.5; V.rr(g, x - 2, y - 2, w + 4, h + 4, 5); g.stroke(); };
    if (o.hiRow != null && o.hiRow >= 0 && o.hiRow < R) hl(o.x, o.y + o.hiRow * (ch + gap), W, ch);
    if (o.hiCol != null && o.hiCol >= 0 && o.hiCol < K) hl(o.x + o.hiCol * (cw + gap), o.y, cw, Ht);
    (o.hiCells || []).forEach(([i, j]) => hl(o.x + j * (cw + gap), o.y + i * (ch + gap), cw, ch));
    if (o.label) V.text(g, o.label, o.x + W / 2, o.y - 10, { size: 12, align: 'center', color: C.muted, font: 'mono' });
    if (o.rowLabels) o.rowLabels.forEach((l, i) => V.text(g, l, o.x - 6, o.y + i * (ch + gap) + ch / 2, { size: 11, align: 'right', base: 'middle', color: C.muted, font: 'mono' }));
    if (o.colLabels) o.colLabels.forEach((l, j) => V.text(g, l, o.x + j * (cw + gap) + cw / 2, o.y + Ht + 14, { size: 11, align: 'center', color: C.muted, font: 'mono' }));
    return {
      x: o.x, y: o.y, w: W, h: Ht, cw, ch, gap,
      cellAt(px, py) { const j = Math.floor((px - o.x) / (cw + gap)), i = Math.floor((py - o.y) / (ch + gap)); return i >= 0 && i < R && j >= 0 && j < K ? [i, j] : null; },
      center(i, j) { return [o.x + j * (cw + gap) + cw / 2, o.y + i * (ch + gap) + ch / 2]; },
    };
  };
  /** Gera uma matriz R×K com valores aleatórios (seed) — útil para demonstrações. */
  V.randMatrix = (R, K, seed = 1, scale = 1) => { const r = V.rng(seed); return Array.from({ length: R }, () => Array.from({ length: K }, () => r.normal() * scale)); };
  V.matmul = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)));
  V.transpose = (A) => A[0].map((_, j) => A.map((r) => r[j]));

  /** Barras (ex.: distribuição de probabilidade). */
  V.bars = function (g, o) {
    const C = V.C();
    const n = o.values.length, gap = o.gap == null ? 4 : o.gap;
    const bw = (o.w - gap * (n - 1)) / n;
    const mx = o.max == null ? Math.max(...o.values, 1e-9) : o.max;
    o.values.forEach((v, i) => {
      const bh = (v / mx) * o.h;
      const x = o.x + i * (bw + gap), y = o.y + o.h - bh;
      g.fillStyle = o.colorFn ? o.colorFn(v, i) : (o.hi === i ? C.c : C.a);
      V.rr(g, x, y, bw, Math.max(1, bh), Math.min(5, bw / 3)); g.fill();
      if (o.labels) V.text(g, o.labels[i], x + bw / 2, o.y + o.h + 14, { size: o.labelSize || 11, align: 'center', color: C.muted, font: o.labelFont || 'mono', maxW: bw + gap });
      if (o.showValues) V.text(g, o.fmt ? o.fmt(v) : v.toFixed(2), x + bw / 2, y - 5, { size: 10.5, align: 'center', color: C.ink, font: 'mono' });
    });
  };

  /* ================================================================== UI */
  const h = (...a) => LA.h(...a);
  /** container de controles da figura (cria se faltar) */
  UI.controls = function (fig) {
    let c = fig.querySelector('.la-fig-controls');
    if (!c) { c = h('div', { class: 'la-fig-controls' }); const cap = fig.querySelector('figcaption'); fig.insertBefore(c, cap); }
    return c;
  };
  UI.slider = function (parent, o) {
    const fmt = o.fmt || ((v) => String(v));
    const input = h('input', { type: 'range', class: 'la-range', min: o.min, max: o.max, step: o.step || 1, value: o.value });
    const val = h('span', { class: 'val' }, fmt(+o.value));
    const el = h('div', { class: 'la-ctl' }, h('label', null, o.label || ''), input, val);
    input.addEventListener('input', () => { val.textContent = fmt(+input.value); o.onInput && o.onInput(+input.value); });
    parent.appendChild(el);
    return { el, input, get: () => +input.value, set: (v, silent) => { input.value = v; val.textContent = fmt(+v); if (!silent) o.onInput && o.onInput(+v); } };
  };
  UI.seg = function (parent, o) {
    const s = h('div', { class: 'la-seg' });
    let cur = o.value;
    const btns = o.options.map((op) => {
      const v = typeof op === 'object' ? op.v : op, l = typeof op === 'object' ? op.label : op;
      const b = h('button', { class: v === cur ? 'on' : '', type: 'button' }, l);
      b.addEventListener('click', () => { cur = v; btns.forEach((x) => x.classList.remove('on')); b.classList.add('on'); o.onChange && o.onChange(v); });
      s.appendChild(b);
      return b;
    });
    const el = o.label ? h('div', { class: 'la-ctl' }, h('label', null, o.label), s) : s;
    parent.appendChild(el);
    return { el, get: () => cur, set: (v, silent) => { const i = o.options.findIndex((op) => (typeof op === 'object' ? op.v : op) === v); if (i < 0) return; if (silent) { cur = v; btns.forEach((x) => x.classList.remove('on')); btns[i].classList.add('on'); } else btns[i].click(); } };
  };
  UI.toggle = function (parent, o) {
    const inp = h('input', { type: 'checkbox' }); inp.checked = !!o.value;
    const el = h('label', { class: 'la-toggle' }, inp, h('i'), h('span', null, o.label || ''));
    inp.addEventListener('change', () => o.onChange && o.onChange(inp.checked));
    parent.appendChild(el);
    return { el, get: () => inp.checked, set: (v) => { inp.checked = !!v; o.onChange && o.onChange(!!v); } };
  };
  UI.button = function (parent, label, onClick, o = {}) {
    const b = h('button', { class: 'la-b' + (o.primary ? ' primary' : ''), type: 'button', html: label });
    b.addEventListener('click', onClick);
    parent.appendChild(b);
    return b;
  };
  UI.readout = function (parent, label, value = '—') {
    const b = h('b', null, value);
    const el = h('div', { class: 'la-readout' }, h('span', null, label), b);
    parent.appendChild(el);
    return { el, set: (v, state) => { b.textContent = v; el.classList.remove('ok', 'bad', 'warn'); if (state) el.classList.add(state); } };
  };
  /** mostra/esconde um grupo de controles (útil em figuras com vários modos) */
  UI.group = function (parent) { const gdiv = LA.h('div', { style: { display: 'contents' } }); parent.appendChild(gdiv); return { el: gdiv, show(on) { gdiv.style.display = on ? 'contents' : 'none'; } }; };
  UI.select = function (parent, o) {
    const s = h('select', { class: 'la-select' }, o.options.map((op) => h('option', { value: typeof op === 'object' ? op.v : op }, typeof op === 'object' ? op.label : op)));
    s.value = o.value;
    s.addEventListener('change', () => o.onChange && o.onChange(s.value));
    const el = h('div', { class: 'la-ctl' }, o.label ? h('label', null, o.label) : null, s);
    parent.appendChild(el);
    return { el, get: () => s.value, set: (v) => { s.value = v; o.onChange && o.onChange(v); } };
  };
  UI.legend = function (parent, items) {
    const el = h('div', { class: 'la-legend' }, items.map(([c, l]) => h('span', null, h('i', { style: { '--c': c } }), l)));
    parent.appendChild(el);
    return el;
  };

  /**
   * Stepper: explicação guiada em passos, sincronizada com uma animação.
   *   const st = LA.ui.stepper(fig, [{t:'Título', d:'texto html'}, ...], { autoplay: 6000, onStep:(i)=>{} })
   *   st.i  -> passo atual.   st.go(i)
   */
  UI.stepper = function (fig, steps, o = {}) {
    const wrap = h('div', { class: 'la-stepper' });
    const bar = h('div', { class: 'la-stepper-bar' });
    const text = h('div', { class: 'la-stepper-text' });
    const prev = h('button', { class: 'la-b', type: 'button' }, LA.t ? LA.t('st.prev') : '← anterior');
    const next = h('button', { class: 'la-b primary', type: 'button' }, LA.t ? LA.t('st.next') : 'próximo →');
    const play = h('button', { class: 'la-b', type: 'button' }, o.autoplay ? LA.t('st.pause') : LA.t('st.auto'));
    const cnt = h('span', { class: 'cnt' });
    const nav = h('div', { class: 'la-stepper-nav' }, prev, next, play, cnt);
    const dots = steps.map((s, i) => { const b = h('button', { type: 'button', title: s.t }); b.addEventListener('click', () => { stopAuto(); go(i); }); bar.appendChild(b); return b; });
    wrap.append(bar, text, nav);
    const cap = fig.querySelector('figcaption');
    fig.insertBefore(wrap, cap || null);
    const api = { i: 0, go, t0: performance.now() };
    let timer = null;
    function go(i) {
      api.i = (i + steps.length) % steps.length;
      api.t0 = performance.now();
      dots.forEach((d, k) => { d.classList.toggle('on', k === api.i); d.classList.toggle('past', k < api.i); });
      text.innerHTML = `<b class="st">${api.i + 1}. ${steps[api.i].t}</b>${steps[api.i].d || ''}`;
      text.querySelectorAll('a[data-ch]').forEach((a) => { a.href = '#/c/' + a.dataset.ch + (a.dataset.sec ? '/' + a.dataset.sec : ''); });
      cnt.textContent = `${api.i + 1} / ${steps.length}`;
      if (LA.renderMath) LA.renderMath(text);
      o.onStep && o.onStep(api.i, steps[api.i]);
    }
    function startAuto() { stopAuto(); timer = setInterval(() => { if (!LA.paused) go(api.i + 1); }, o.autoplay || 5000); play.textContent = LA.t('st.pause'); }
    function stopAuto() { if (timer) clearInterval(timer); timer = null; play.textContent = LA.t('st.auto'); }
    prev.addEventListener('click', () => { stopAuto(); go(api.i - 1); });
    next.addEventListener('click', () => { stopAuto(); go(api.i + 1); });
    play.addEventListener('click', () => (timer ? stopAuto() : startAuto()));
    LA.onCleanup && LA.onCleanup(stopAuto);
    go(0);
    if (o.autoplay) startAuto();
    api.elapsed = () => (performance.now() - api.t0) / 1000;
    api.stop = stopAuto;
    return api;
  };

  /**
   * Tensor 3D em CSS (pilha de matrizes em perspectiva).
   *   LA.ui.tensor3d(container, { slabs: 4, rows: 6, cols: 8, cell: 16, depthGap: 34,
   *       color: (s,i,j)=>'#...', slabColor: (s)=>'#...', labels:['camada 0',...], rotate:true })
   *   Retorna { el, setCell(s,i,j,color), explode(k), rotate(x,y), highlightSlab(s) }
   */
  UI.tensor3d = function (container, o = {}) {
    const S = o.slabs || 4, R = o.rows || 5, K = o.cols || 7, cell = o.cell || 16, gap = 2, dz = o.depthGap || 34;
    const wrap = h('div', { class: 'la-t3d-wrap' });
    const t3 = h('div', { class: 'la-t3d' });
    const W = K * (cell + gap) + 4, Hh = R * (cell + gap) + 4;
    t3.style.width = W + 'px'; t3.style.height = Hh + 'px';
    const slabs = [];
    const cells = [];
    for (let s = 0; s < S; s++) {
      const slab = h('div', { class: 'slab' });
      slab.style.gridTemplateColumns = `repeat(${K}, ${cell}px)`;
      slab.style.gridAutoRows = cell + 'px';
      slab.style.setProperty('--sc', o.slabColor ? o.slabColor(s) : 'var(--viz-a)');
      cells[s] = [];
      for (let i = 0; i < R; i++) {
        cells[s][i] = [];
        for (let j = 0; j < K; j++) {
          const c = h('div', { class: 'cell' });
          c.style.background = o.color ? o.color(s, i, j) : 'color-mix(in srgb, var(--viz-a) 40%, transparent)';
          slab.appendChild(c); cells[s][i][j] = c;
        }
      }
      if (o.labels && o.labels[s]) slab.appendChild(h('div', { class: 'slab-label' }, o.labels[s]));
      t3.appendChild(slab); slabs.push(slab);
    }
    let spread = 1, rx = o.rx == null ? -18 : o.rx, ry = o.ry == null ? 32 : o.ry;
    function layout() {
      slabs.forEach((sl, s) => { sl.style.transform = `translateZ(${(s - (S - 1) / 2) * dz * spread}px)`; });
      t3.style.transform = `rotateX(${rx}deg) rotateY(${ry}deg)`;
    }
    layout();
    wrap.appendChild(t3);
    container.appendChild(wrap);
    let raf = 0, t0 = performance.now();
    if (o.rotate) {
      const tick = (now) => { if (!LA.paused) { ry = (o.ry == null ? 32 : o.ry) + Math.sin((now - t0) / 2600) * 22; rx = (o.rx == null ? -18 : o.rx) + Math.sin((now - t0) / 3900) * 6; layout(); } raf = requestAnimationFrame(tick); };
      raf = requestAnimationFrame(tick);
      LA.onCleanup && LA.onCleanup(() => cancelAnimationFrame(raf));
    }
    return {
      el: wrap, slabs, cells,
      setCell(s, i, j, color) { if (cells[s] && cells[s][i] && cells[s][i][j]) cells[s][i][j].style.background = color; },
      explode(k) { spread = k; layout(); },
      rotate(x, y) { rx = x; ry = y; layout(); },
      highlightSlab(s) { slabs.forEach((sl, k) => { sl.style.opacity = s == null || k === s ? 1 : 0.25; }); },
    };
  };

  /* ============================================================ HOME HERO */
  /** Constelação latente: nuvem de pontos 3D com clusters e conceitos flutuando. */
  V.homeHero = function (canvas) {
    if (!canvas) return;
    const r = V.rng(7);
    const words = ['Atenção', 'MoE', 'KV cache', 'NVFP4', 'MLA', 'RoPE', 'Softmax', 'Tensor Parallel', 'EXL3', 'Speculative', 'PagedAttention', 'Gated DeltaNet', 'Top-k', 'FP8', 'NCCL', 'Prefill', 'Decode', 'MTP', 'YaRN', 'Router', 'Hadamard', 'RDMA', 'CUDA Graphs', 'Embedding'];
    const clusters = Array.from({ length: 6 }, () => [r.normal() * 1.1, r.normal() * 0.7, r.normal() * 1.1]);
    const P = [];
    for (let i = 0; i < 420; i++) {
      const c = clusters[i % clusters.length];
      P.push({ x: c[0] + r.normal() * 0.33, y: c[1] + r.normal() * 0.33, z: c[2] + r.normal() * 0.33, k: i % clusters.length, s: 0.6 + r() * 1.4 });
    }
    const W = words.map((w, i) => ({ w, c: clusters[i % clusters.length], o: [r.normal() * 0.5, r.normal() * 0.5, r.normal() * 0.5] }));
    const pulses = [];
    V.loop(canvas, (g, w, h, t) => {
      const C = V.C();
      const cols = ['#3FF49B', '#BE8EFD', '#FC93D6', '#26D07C', '#FCBB00', '#AF71FF'];
      const cx = w * (w > 900 ? 0.74 : 0.6), cy = h * 0.5, sc = Math.min(w * 0.5, h) * 0.42;
      const a = t * 0.12, b = 0.35 + Math.sin(t * 0.07) * 0.15;
      const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
      const proj = (x, y, z) => { const X = x * ca - z * sa, Z = x * sa + z * ca; const Y = y * cb - Z * sb, Z2 = y * sb + Z * cb; const f = Math.min(1.5, 6 / (6 + Z2)); return [cx + X * sc * f, cy + Y * sc * f, f, Z2]; };
      const pts = P.map((p) => { const q = proj(p.x, p.y, p.z); return { q, p }; });
      // conexões (vizinhos no mesmo cluster)
      g.lineWidth = 0.6;
      for (let i = 0; i < pts.length; i += 3) {
        const A = pts[i];
        for (let j = i + 3; j < Math.min(pts.length, i + 40); j += 6) {
          const B = pts[j];
          if (A.p.k !== B.p.k) continue;
          const dx = A.q[0] - B.q[0], dy = A.q[1] - B.q[1];
          const d = dx * dx + dy * dy;
          if (d < 2600) { g.strokeStyle = V.alpha(cols[A.p.k], 0.14 * A.q[2]); g.beginPath(); g.moveTo(A.q[0], A.q[1]); g.lineTo(B.q[0], B.q[1]); g.stroke(); }
        }
      }
      pts.sort((x, y) => y.q[3] - x.q[3]);
      pts.forEach(({ q, p }) => {
        const rr = Math.min(4, p.s * q[2] * 1.7);
        g.fillStyle = V.alpha(cols[p.k], Math.max(0.15, Math.min(0.9, 0.3 + 0.55 * (q[2] - 0.7))));
        g.beginPath(); g.arc(q[0], q[1], rr, 0, Math.PI * 2); g.fill();
      });
      // pulsos viajando entre clusters (atenção)
      if (!LA.paused && Math.random() < 0.06) { const i = Math.floor(Math.random() * clusters.length), j = (i + 1 + Math.floor(Math.random() * (clusters.length - 1))) % clusters.length; pulses.push({ i, j, t: 0 }); }
      for (let k = pulses.length - 1; k >= 0; k--) {
        const pu = pulses[k]; pu.t += 0.012;
        if (pu.t >= 1) { pulses.splice(k, 1); continue; }
        const A = proj(...clusters[pu.i]), B = proj(...clusters[pu.j]);
        const mx = (A[0] + B[0]) / 2, my = Math.min(A[1], B[1]) - 60;
        g.strokeStyle = V.alpha(cols[pu.i], 0.25 * (1 - pu.t)); g.lineWidth = 1.2;
        g.beginPath(); g.moveTo(A[0], A[1]); g.quadraticCurveTo(mx, my, B[0], B[1]); g.stroke();
        const u = pu.t, x = (1 - u) * (1 - u) * A[0] + 2 * (1 - u) * u * mx + u * u * B[0], y = (1 - u) * (1 - u) * A[1] + 2 * (1 - u) * u * my + u * u * B[1];
        V.glow(g, x, y, 10, cols[pu.i], 1);
      }
      // palavras
      W.forEach((wd, i) => {
        const q = proj(wd.c[0] + wd.o[0], wd.c[1] + wd.o[1] + Math.sin(t * 0.5 + i) * 0.05, wd.c[2] + wd.o[2]);
        if (w <= 900 || q[0] < w * 0.52) return;
        V.text(g, wd.w, q[0], q[1], { size: 9 + q[2] * 5, font: 'mono', color: V.alpha('#E9FAF2', Math.max(0, Math.min(0.85, (q[2] - 0.8) * 2.2))), align: 'center' });
      });
      // vinheta à esquerda para legibilidade do texto
      if (w <= 900) { g.fillStyle = 'rgba(5,13,9,.62)'; g.fillRect(0, 0, w, h); }
      else {
        const lg = g.createLinearGradient(0, 0, w * 0.62, 0);
        lg.addColorStop(0, 'rgba(5,13,9,.92)'); lg.addColorStop(0.6, 'rgba(5,13,9,.55)'); lg.addColorStop(1, 'rgba(5,13,9,0)');
        g.fillStyle = lg; g.fillRect(0, 0, w, h);
      }
    });
  };

  /** Fundo animado discreto do cabeçalho de capítulo. */
  V.heroBg = function (canvas, part) {
    if (!canvas) return;
    const r = V.rng(100 + (part || 0) * 13);
    const N = 46;
    const P = Array.from({ length: N }, () => ({ x: r(), y: r(), vx: (r() - 0.5) * 0.02, vy: (r() - 0.5) * 0.02, s: 1 + r() * 2 }));
    V.loop(canvas, (g, w, h, t, dt) => {
      const C = V.C();
      const col = getComputedStyle(canvas.closest('.la-chapter') || document.documentElement).getPropertyValue('--pc').trim();
      const c = col.startsWith('var(') ? C.a : col || C.a;
      const cc = (c.startsWith('#') || c.startsWith('rgb')) ? c : C.a;
      P.forEach((p) => { p.x += p.vx * dt; p.y += p.vy * dt; if (p.x < 0 || p.x > 1) p.vx *= -1; if (p.y < 0 || p.y > 1) p.vy *= -1; });
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
        const dx = (P[i].x - P[j].x) * w, dy = (P[i].y - P[j].y) * h, d = Math.sqrt(dx * dx + dy * dy);
        if (d < 120) { g.strokeStyle = V.alpha(cc, 0.12 * (1 - d / 120)); g.lineWidth = 1; g.beginPath(); g.moveTo(P[i].x * w, P[i].y * h); g.lineTo(P[j].x * w, P[j].y * h); g.stroke(); }
      }
      P.forEach((p, i) => { g.fillStyle = V.alpha(cc, 0.35 + 0.25 * Math.sin(t * 1.5 + i)); g.beginPath(); g.arc(p.x * w, p.y * h, p.s, 0, Math.PI * 2); g.fill(); });
      const lg = g.createLinearGradient(0, 0, w * 0.7, 0);
      const bg = C.dark ? 'rgba(10,20,15,' : 'rgba(247,247,250,';
      lg.addColorStop(0, bg + '.85)'); lg.addColorStop(1, bg + '0)');
      g.fillStyle = lg; g.fillRect(0, 0, w, h);
    });
  };

  /** Glifo animado dos cartões de receita. */
  V.recipeGlyph = function (canvas, kind) {
    V.loop(canvas, (g, w, h, t) => {
      const C = V.C();
      const col = kind === 'glm' ? C.glm : kind === 'qwen' ? C.qwen : C.ds;
      if (kind === 'glm') { // grade de especialistas acendendo (MoE)
        const n = 6, s = 13, ox = w - n * (s + 4) - 12, oy = 16;
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
          const k = i * n + j; const on = Math.sin(t * 2 + k * 1.7) > 0.72 || (k * 7 + Math.floor(t * 3)) % 23 === 0;
          g.fillStyle = on ? col : V.alpha(col, 0.14); V.rr(g, ox + j * (s + 4), oy + i * (s + 4), s, s, 3); g.fill();
        }
      } else if (kind === 'qwen') { // faixas híbridas (linear x plena)
        for (let i = 0; i < 12; i++) {
          const full = i % 4 === 3; const y = 14 + i * 9;
          const len = 40 + 50 * (0.5 + 0.5 * Math.sin(t * 2 + i));
          g.fillStyle = full ? C.b : V.alpha(col, 0.55); V.rr(g, w - 14 - len, y, len, 6, 3); g.fill();
        }
      } else { // padrão de atenção esparsa
        const n = 10, s = 9, ox = w - n * (s + 2) - 12, oy = 14;
        for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
          const sel = j === i || i - j < 2 || ((i * 31 + j * 17 + Math.floor(t * 2)) % 7 === 0);
          g.fillStyle = sel ? col : V.alpha(col, 0.1); V.rr(g, ox + j * (s + 2), oy + i * (s + 2), s, s, 2); g.fill();
        }
      }
    });
  };
})();
