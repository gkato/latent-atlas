/* ==========================================================================
   LATENT ATLAS — num.js
   Aritmética de formatos numéricos para as figuras de quantização:
   minifloats genéricos (FP32/FP16/BF16/FP8 E4M3/E5M2/FP6/FP4 E2M1/E8M0),
   arredondamento RNE, quantização inteira (sim/assimétrica, por grupo),
   MXFP4/MXFP8 (OCP Microscaling), NVFP4, transformada de Hadamard.
   ========================================================================== */
(function () {
  'use strict';
  const LA = (window.LA = window.LA || {});
  const N = (LA.num = LA.num || {});

  /**
   * Formatos. e = bits de expoente, m = bits de mantissa, bias.
   * kind: 'ieee' (expoente máximo reservado p/ Inf/NaN),
   *       'fn'   (sem Inf; NaN só com todos os bits 1 — ex.: E4M3FN),
   *       'fin'  (sem Inf/NaN — ex.: FP4 E2M1, FP6),
   *       'e8m0' (só expoente, sem sinal: escala potência de 2).
   */
  N.formats = {
    fp32: { name: 'FP32', e: 8, m: 23, bias: 127, kind: 'ieee' },
    fp16: { name: 'FP16', e: 5, m: 10, bias: 15, kind: 'ieee' },
    bf16: { name: 'BF16', e: 8, m: 7, bias: 127, kind: 'ieee' },
    e4m3: { name: 'FP8 E4M3', e: 4, m: 3, bias: 7, kind: 'fn' },
    e5m2: { name: 'FP8 E5M2', e: 5, m: 2, bias: 15, kind: 'ieee' },
    e3m2: { name: 'FP6 E3M2', e: 3, m: 2, bias: 3, kind: 'fin' },
    e2m3: { name: 'FP6 E2M3', e: 2, m: 3, bias: 1, kind: 'fin' },
    e2m1: { name: 'FP4 E2M1', e: 2, m: 1, bias: 1, kind: 'fin' },
    e8m0: { name: 'E8M0', e: 8, m: 0, bias: 127, kind: 'e8m0' },
  };
  Object.keys(N.formats).forEach((k) => { const f = N.formats[k]; f.id = k; f.bits = f.kind === 'e8m0' ? 8 : 1 + f.e + f.m; });

  /** decodifica um padrão de bits (inteiro) no valor real */
  N.decode = function (fmt, bits) {
    const f = typeof fmt === 'string' ? N.formats[fmt] : fmt;
    if (f.kind === 'e8m0') return bits === 255 ? NaN : Math.pow(2, bits - 127);
    if (f.id === 'fp32') { const b = new DataView(new ArrayBuffer(4)); b.setUint32(0, bits >>> 0); return b.getFloat32(0); }
    const s = (bits >> (f.e + f.m)) & 1;
    const E = (bits >> f.m) & ((1 << f.e) - 1);
    const M = bits & ((1 << f.m) - 1);
    const maxE = (1 << f.e) - 1;
    if (f.kind === 'ieee' && E === maxE) return M ? NaN : s ? -Infinity : Infinity;
    if (f.kind === 'fn' && E === maxE && M === (1 << f.m) - 1) return NaN;
    const v = E === 0 ? (M / Math.pow(2, f.m)) * Math.pow(2, 1 - f.bias) : (1 + M / Math.pow(2, f.m)) * Math.pow(2, E - f.bias);
    return s ? -v : v;
  };

  /** separa os campos de bits (para desenhar) */
  N.fields = function (fmt, bits) {
    const f = typeof fmt === 'string' ? N.formats[fmt] : fmt;
    if (f.kind === 'e8m0') return { s: null, E: bits, M: 0, f };
    return { s: (bits >> (f.e + f.m)) & 1, E: (bits >> f.m) & ((1 << f.e) - 1), M: bits & ((1 << f.m) - 1), f };
  };

  /** maior valor finito */
  N.maxFinite = function (fmt) {
    const f = typeof fmt === 'string' ? N.formats[fmt] : fmt;
    if (f.kind === 'e8m0') return Math.pow(2, 127);
    const maxE = (1 << f.e) - 1;
    if (f.kind === 'ieee') return (2 - Math.pow(2, -f.m)) * Math.pow(2, maxE - 1 - f.bias);
    if (f.kind === 'fn') return (1 + ((1 << f.m) - 2) / Math.pow(2, f.m)) * Math.pow(2, maxE - f.bias);
    return (1 + ((1 << f.m) - 1) / Math.pow(2, f.m)) * Math.pow(2, maxE - f.bias);
  };
  /** menor subnormal positivo e menor normal */
  N.minSub = (fmt) => { const f = typeof fmt === 'string' ? N.formats[fmt] : fmt; return f.kind === 'e8m0' ? Math.pow(2, -127) : Math.pow(2, 1 - f.bias - f.m); };
  N.minNormal = (fmt) => { const f = typeof fmt === 'string' ? N.formats[fmt] : fmt; return Math.pow(2, 1 - f.bias); };

  const tables = {};
  /** lista ordenada de {v, bits} não-negativos finitos (formatos ≤ 16 bits) */
  N.table = function (fmt) {
    const f = typeof fmt === 'string' ? N.formats[fmt] : fmt;
    if (tables[f.id]) return tables[f.id];
    const out = [];
    if (f.kind === 'e8m0') { for (let b = 0; b < 255; b++) out.push({ v: Math.pow(2, b - 127), bits: b }); }
    else {
      const half = 1 << (f.e + f.m);
      for (let b = 0; b < half; b++) { const v = N.decode(f, b); if (Number.isFinite(v)) out.push({ v, bits: b }); }
    }
    out.sort((a, b) => a.v - b.v);
    tables[f.id] = out;
    return out;
  };
  /** todos os valores representáveis (com sinal), útil para desenhar a "régua" do formato */
  N.values = function (fmt) {
    const t = N.table(fmt).map((x) => x.v);
    const f = typeof fmt === 'string' ? N.formats[fmt] : fmt;
    if (f.kind === 'e8m0') return t;
    const neg = t.filter((v) => v > 0).map((v) => -v).reverse();
    return neg.concat(t);
  };

  /**
   * Codifica x no formato com arredondamento ao par mais próximo (RNE).
   * mode 'sat' (padrão): satura no máximo finito; 'ieee': estoura para ±Inf quando aplicável.
   * Retorna { bits, v (valor representado), err (v - x) }
   */
  N.encode = function (fmt, x, mode = 'sat') {
    const f = typeof fmt === 'string' ? N.formats[fmt] : fmt;
    if (f.id === 'fp32') { const b = new DataView(new ArrayBuffer(4)); b.setFloat32(0, x); const bits = b.getUint32(0); const v = b.getFloat32(0); return { bits, v, err: v - x }; }
    if (Number.isNaN(x)) {
      const nanBits = f.kind === 'e8m0' ? 255 : f.kind === 'fin' ? 0 : ((1 << (f.e + f.m)) - 1);
      return { bits: nanBits, v: NaN, err: NaN };
    }
    if (f.kind === 'e8m0') {
      if (x <= 0) return { bits: 0, v: Math.pow(2, -127), err: Math.pow(2, -127) - x };
      let e = Math.round(Math.log2(x)); e = Math.max(-127, Math.min(127, e));
      return { bits: e + 127, v: Math.pow(2, e), err: Math.pow(2, e) - x };
    }
    const signBit = 1 << (f.e + f.m);
    const s = x < 0 || Object.is(x, -0) ? 1 : 0;
    const a = Math.abs(x);
    const T = N.table(f);
    const mx = T[T.length - 1];
    let pick;
    if (a >= mx.v) {
      if (mode === 'ieee' && f.kind === 'ieee') {
        // limite de estouro = max + meio ulp
        const ulp = Math.pow(2, ((1 << f.e) - 2) - f.bias - f.m);
        if (a >= mx.v + ulp / 2) { const infBits = ((1 << f.e) - 1) << f.m; return { bits: (s ? signBit : 0) | infBits, v: s ? -Infinity : Infinity, err: Infinity }; }
      }
      pick = mx;
    } else {
      // busca binária pelo vizinho
      let lo = 0, hi = T.length - 1;
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (T[mid].v <= a) lo = mid; else hi = mid; }
      const A = T[lo], B = T[hi];
      const da = a - A.v, db = B.v - a;
      if (da < db) pick = A; else if (db < da) pick = B; else pick = (A.bits & 1) === 0 ? A : B; // empate → par
    }
    const v = s ? -pick.v : pick.v;
    return { bits: (s ? signBit : 0) | pick.bits, v, err: v - x };
  };
  N.round = (fmt, x, mode) => N.encode(fmt, x, mode).v;
  N.toBin = (bits, n) => (bits >>> 0).toString(2).padStart(n, '0').slice(-n);

  /* -------------------------------------------------- quantização inteira */
  /**
   * Quantização inteira uniforme de um vetor.
   *   opts: { bits:4, symmetric:true, group:0 (0 = tensor todo) }
   * Retorna { q: int[], deq: float[], scales: [], zeros: [], mse, maxErr }
   */
  N.quantInt = function (xs, o = {}) {
    const bits = o.bits || 4, sym = o.symmetric !== false, G = o.group || xs.length;
    const q = new Array(xs.length), deq = new Array(xs.length), scales = [], zeros = [];
    for (let g0 = 0; g0 < xs.length; g0 += G) {
      const seg = xs.slice(g0, g0 + G);
      let scale, zero = 0;
      if (sym) {
        const qmax = (1 << (bits - 1)) - 1; // ex.: 4 bits → 7
        const amax = Math.max(...seg.map(Math.abs), 1e-12);
        scale = amax / qmax;
        seg.forEach((x, i) => { const v = Math.max(-qmax - 1, Math.min(qmax, Math.round(x / scale))); q[g0 + i] = v; deq[g0 + i] = v * scale; });
      } else {
        const qmax = (1 << bits) - 1;
        const mn = Math.min(...seg), mx = Math.max(...seg);
        scale = (mx - mn) / qmax || 1e-12;
        zero = Math.round(-mn / scale);
        seg.forEach((x, i) => { const v = Math.max(0, Math.min(qmax, Math.round(x / scale) + zero)); q[g0 + i] = v; deq[g0 + i] = (v - zero) * scale; });
      }
      scales.push(scale); zeros.push(zero);
    }
    return Object.assign({ q, deq, scales, zeros }, N.errStats(xs, deq));
  };

  /** estatísticas de erro */
  N.errStats = function (xs, ys) {
    let se = 0, me = 0, sx = 0;
    xs.forEach((x, i) => { const d = ys[i] - x; se += d * d; me = Math.max(me, Math.abs(d)); sx += x * x; });
    const mse = se / xs.length;
    return { mse, rmse: Math.sqrt(mse), maxErr: me, sqnr: 10 * Math.log10((sx / xs.length) / (mse || 1e-30)) };
  };

  /* ------------------------------------------------------ microscaling */
  /**
   * MX (OCP Microscaling): blocos de 32 elementos compartilham uma escala E8M0 (potência de 2).
   *   X = floor(log2(amax)) - emax_elem ;  escala = 2^X ;  elem = round_fmt(x / 2^X) (saturando)
   *   elemFmt: 'e2m1' (MXFP4, emax=2), 'e4m3' (MXFP8, emax=8), 'e5m2' (emax=15), 'e3m2'/'e2m3' (MXFP6)
   */
  N.mx = function (xs, elemFmt = 'e2m1', block = 32) {
    const f = N.formats[elemFmt];
    const emax = Math.floor(Math.log2(N.maxFinite(f)));
    const blocks = [], deq = new Array(xs.length), codes = new Array(xs.length);
    for (let b0 = 0; b0 < xs.length; b0 += block) {
      const seg = xs.slice(b0, b0 + block);
      const amax = Math.max(...seg.map(Math.abs));
      const X = amax === 0 ? -127 : Math.max(-127, Math.min(127, Math.floor(Math.log2(amax)) - emax));
      const scale = Math.pow(2, X);
      seg.forEach((x, i) => { const e = N.encode(f, x / scale); codes[b0 + i] = e; deq[b0 + i] = e.v * scale; });
      blocks.push({ start: b0, amax, X, scale, scaleBits: X + 127 });
    }
    return Object.assign({ blocks, deq, codes, bitsPerElem: f.bits + 8 / block }, N.errStats(xs, deq));
  };

  /**
   * NVFP4: blocos de 16 elementos FP4 (E2M1) + escala por bloco em FP8 E4M3 + escala global FP32.
   *   globalScale = amax_tensor / (6 * 448)
   *   s_bloco(E4M3) = e4m3( amax_bloco / 6 / globalScale )
   *   x ≈ e2m1 * s_bloco * globalScale
   */
  N.nvfp4 = function (xs, block = 16) {
    const amaxT = Math.max(...xs.map(Math.abs), 1e-30);
    const globalScale = amaxT / (6 * 448);
    const blocks = [], deq = new Array(xs.length), codes = new Array(xs.length);
    for (let b0 = 0; b0 < xs.length; b0 += block) {
      const seg = xs.slice(b0, b0 + block);
      const amax = Math.max(...seg.map(Math.abs));
      const sEnc = N.encode('e4m3', amax / 6 / globalScale);
      const s = sEnc.v * globalScale || 1e-30;
      seg.forEach((x, i) => { const e = N.encode('e2m1', x / s); codes[b0 + i] = e; deq[b0 + i] = e.v * s; });
      blocks.push({ start: b0, amax, scaleE4M3: sEnc.v, scaleBits: sEnc.bits, scale: s });
    }
    return Object.assign({ blocks, deq, codes, globalScale, bitsPerElem: 4 + 8 / block }, N.errStats(xs, deq));
  };

  /** quantização FP8 por tensor (escala = amax / 448 para E4M3) */
  N.fp8Tensor = function (xs, fmt = 'e4m3') {
    const amax = Math.max(...xs.map(Math.abs), 1e-30);
    const scale = amax / N.maxFinite(fmt);
    const deq = xs.map((x) => N.encode(fmt, x / scale).v * scale);
    return Object.assign({ scale, deq }, N.errStats(xs, deq));
  };

  /* ---------------------------------------------------------- Hadamard */
  /** Walsh–Hadamard rápida normalizada (n potência de 2). Ortogonal: H·H = I. */
  N.fwht = function (xs) {
    const a = xs.slice(); const n = a.length;
    if (n & (n - 1)) throw new Error('fwht: tamanho deve ser potência de 2');
    for (let hh = 1; hh < n; hh *= 2) for (let i = 0; i < n; i += hh * 2) for (let j = i; j < i + hh; j++) { const x = a[j], y = a[j + hh]; a[j] = x + y; a[j + hh] = x - y; }
    const s = 1 / Math.sqrt(n);
    return a.map((v) => v * s);
  };
  /** matriz de Hadamard n×n normalizada */
  N.hadamard = function (n) { return Array.from({ length: n }, (_, i) => N.fwht(Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)))); };
  /** sinais aleatórios ±1 (para a rotação "randomizada" usada em incoerência) */
  N.randSigns = function (n, seed = 3) { const r = LA.viz ? LA.viz.rng(seed) : Math.random; return Array.from({ length: n }, () => (r() < 0.5 ? -1 : 1)); };

  /** gera pesos sintéticos "tipo LLM": gaussianos com alguns outliers */
  N.synthWeights = function (n, seed = 11, outliers = 0.02, outScale = 8) {
    const r = LA.viz ? LA.viz.rng(seed) : null;
    const rn = r ? r.normal : () => { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const rr = r || Math.random;
    return Array.from({ length: n }, () => { const x = rn() * 0.02; return rr() < outliers ? x * outScale + (rr() < 0.5 ? -1 : 1) * 0.04 : x; });
  };

  /** memória de pesos: parâmetros × bits por parâmetro → bytes */
  N.weightBytes = (params, bpw) => (params * bpw) / 8;
})();
