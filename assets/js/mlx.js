/* ==========================================================================
   LATENT ATLAS — mlx.js
   Copilot local via Apple MLX: botão na barra do topo + modal que escolhe o
   modelo (com sugestão pela RAM), adiciona modelos pelo id do Hugging Face,
   baixa, sobe o mlx_lm.server e liga o copilot sozinho.
   Fala com o helper tools/atlas_server.py (API JSON em /api/mlx/*).
   Sem o helper (file:// ou outro servidor estático) o modal funciona em modo
   manual: escolha do modelo, comandos para copiar e "verificar conexão".
   ========================================================================== */
(function () {
  'use strict';
  const LA = (window.LA = window.LA || {});
  const h = (...a) => LA.h(...a);
  const T = (k, v) => LA.t('mlx.' + k, v);
  const GiB = 1073741824;
  const RAMS = [8, 16, 24, 32, 48, 64, 96, 128];
  const ID_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/;
  const CIRC = ['①', '②', '③', '④'];

  const ICON = {
    chip: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2.5"/><rect x="9.5" y="9.5" width="5" height="5" rx="1" fill="currentColor" stroke="none" opacity=".55"/><path d="M9.5 2.8v2.4M14.5 2.8v2.4M9.5 18.8v2.4M14.5 18.8v2.4M2.8 9.5h2.4M2.8 14.5h2.4M18.8 9.5h2.4M18.8 14.5h2.4"/></svg>',
  };

  /* ---------------------------------------------------------------- estado */
  const store = (k, d) => (LA.store ? LA.store.get(k, d) : d);
  const save = (k, v) => { if (LA.store) LA.store.set(k, v); };
  const S = {
    helper: null,            // null = detectando · true/false
    lost: 0,                 // falhas seguidas do polling
    st: null,                // último /api/mlx/status
    open: false,
    sel: store('mlx:sel', null),
    custom: store('mlx:custom', []),
    ramGB: store('mlx:ram', 16),
    port: +store('mlx:port', 8080) || 8080,
    portSet: store('mlx:port', null) != null, // a porta já foi escolhida (senão usa a do helper)
    adv: Object.assign({ maxTokens: 2048, temp: 0.4 }, store('mlx:adv', {})),
    ext: store('mlx:ext', null), // servidor de fora do helper ligado ao copilot ("usar esse servidor")
    userStart: null,         // id que o usuário mandou subir (toast ao ficar pronto)
    busyOp: null,            // POST em andamento
    err: null,               // {op, code, message, data} do último POST que falhou
    portBusy: null,          // {port, openai, models}
    manual: { state: 'idle', models: [], port: null, err: '' },
    add: { kind: '', msgs: [] },
    flash: null,
    noOpen: false,           // grupo "grande demais" aberto
    opener: null,
  };
  if (!Array.isArray(S.custom)) S.custom = [];
  S.custom = S.custom.filter((c) => c && typeof c.id === 'string' && ID_RE.test(c.id));
  if (!RAMS.includes(+S.ramGB)) S.ramGB = 16;
  if (typeof S.sel !== 'string') S.sel = null;
  const validPort = (p) => Number.isInteger(p) && p >= 1024 && p <= 65535;
  if (!(S.ext && validPort(S.ext.port) && typeof S.ext.id === 'string')) S.ext = null;

  /* ------------------------------------------------------------ formatação */
  const NF = {};
  const NB = ' '; // número e unidade nunca quebram de linha ("8,9 GB")
  function num(x, d) {
    const k = LA.lang + ':' + d;
    const f = NF[k] || (NF[k] = new Intl.NumberFormat(LA.lang === 'en' ? 'en-US' : 'pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d }));
    return f.format(+x || 0);
  }
  /** bytes → GB (1e9, uma casa) ou MB abaixo de 1 GB */
  const gb = (b) => { b = Math.max(0, +b || 0); return b >= 1e9 ? num(b / 1e9, 1) + NB + 'GB' : num(b / 1e6, 0) + NB + 'MB'; };
  const ramTxt = (b) => Math.round(b / GiB) + NB + 'GB';
  const speedTxt = (bps) => (bps >= 1e6 ? num(bps / 1e6, 1) + NB + 'MB/s' : num((bps || 0) / 1e3, 0) + NB + 'KB/s');
  const tokTxt = (t) => num(t, 0) + NB + 'tok/s';
  /** duração legível; arredonda os minutos uma vez só (nunca "60 min" nem "1 h 60 min") */
  function dur(s) {
    s = Math.max(0, Math.round(+s || 0));
    if (s < 60) return T('t.s', { s: Math.max(1, s) });
    const min = Math.round(s / 60);
    if (min < 60) return T('t.m', { m: min });
    const hh = Math.floor(min / 60), mm = min % 60;
    return mm ? T('t.h', { h: hh, m: mm }) : T('t.h0', { h: hh });
  }
  const pct = (a, b) => (b > 0 ? Math.max(0, Math.min(100, Math.floor((100 * a) / b))) : 0);
  const short = (id) => String(id || '').split('/').pop();
  const orgOf = (id) => { const p = String(id || '').split('/'); return p.length > 1 ? p[0] : ''; };
  const reduced = () => !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  const helperPort = () => +location.port || (location.protocol === 'https:' ? 443 : 80);

  /* ------------------------------------------------------- modelos / seleção */
  const ramBytes = () => (S.helper && S.st && S.st.mac && S.st.mac.ram_bytes) || S.ramGB * GiB;
  const locals = () => (S.helper && S.st && Array.isArray(S.st.local_models) ? S.st.local_models : []);
  const localOf = (id) => locals().find((l) => l.id === id) || null;
  const curatedOf = (id) => LA.mlxModels.find((m) => m.id.toLowerCase() === String(id).toLowerCase()) || null;
  const customOf = (id) => S.custom.find((m) => m.id.toLowerCase() === String(id).toLowerCase()) || null;
  function localOnly() {
    return locals().filter((l) => !curatedOf(l.id) && !customOf(l.id))
      .map((l) => ({ id: l.id, bytes: l.complete ? l.bytes : 0, diskBytes: l.bytes, localOnly: true }));
  }
  function findModel(id) {
    if (!id) return null;
    return curatedOf(id) || customOf(id) || localOnly().find((m) => m.id === id) || null;
  }
  function selModel() {
    let m = findModel(S.sel);
    if (!m) m = LA.mlxRecommend(ramBytes()) || LA.mlxModels[0];
    return m;
  }
  const selId = () => selModel().id;
  function select(id) {
    S.sel = id; save('mlx:sel', id);
    S.err = null; S.portBusy = null;
    update();
  }
  /** a arquitetura roda no mlx-lm instalado? (só quando o helper conhece os tipos) */
  function supportedOf(m) {
    if (m.supported === false) return false;
    const mlx = S.helper && S.st && S.st.mlx;
    if (!mlx || !mlx.installed || !Array.isArray(mlx.model_types) || !mlx.model_types.length || !m.type) return m.supported == null ? null : m.supported;
    const t = (mlx.remap && mlx.remap[m.type]) || m.type;
    return mlx.model_types.includes(t);
  }
  /** download parcial que vale a pena retomar (≥ 1%): só config/tokenizer no cache não conta */
  function partialOf(m, loc) {
    if (!loc || loc.complete || !(loc.bytes > 0)) return false;
    return m.bytes ? loc.bytes / m.bytes >= 0.01 : loc.bytes >= 5e7;
  }

  /* intenção "baixar e iniciar": fica no localStorage para sobreviver a um reload
     (ou a outra aba) enquanto o download corre */
  const AUTO_GRACE = 20; // s entre o POST do download e o 1º status que mostra o download
  function autoGet() {
    const a = store('mlx:autostart', null);
    return a && typeof a.id === 'string' && ID_RE.test(a.id) && validPort(a.port) ? a : null;
  }
  const autoSet = (a) => save('mlx:autostart', a || null);

  /** servidor externo (não gerenciado) que o copilot está usando de fato */
  function extLive() {
    return S.ext && S.helper && S.lost < 3 && copilotOn(S.ext.port) ? S.ext : null;
  }
  function setExt(e) { S.ext = e || null; save('mlx:ext', S.ext); }
  /** confere se o servidor externo continua no ar (senão esquece) */
  async function extCheck() {
    if (!S.ext || !S.helper) return;
    const port = S.ext.port;
    try {
      const pr = await api('/api/mlx/probe?port=' + port);
      if (S.ext && S.ext.port === port && !(pr && pr.listening && pr.openai && !pr.managed)) { setExt(null); update(); }
    } catch (e) { /* helper fora: tenta na próxima abertura */ }
  }

  /* ------------------------------------------------------------------- API */
  async function api(path, body) {
    const opt = { headers: { 'X-Atlas': '1' }, cache: 'no-store' };
    if (body !== undefined) { opt.method = 'POST'; opt.headers['Content-Type'] = 'application/json'; opt.body = JSON.stringify(body); }
    let r;
    try { r = await fetch(path, opt); } catch (e) { const er = new Error(T('e.net')); er.code = 'net'; throw er; }
    let j = null;
    try { j = await r.json(); } catch (e) { /* corpo vazio ou não-JSON */ }
    if (!r.ok) {
      const er = new Error((j && j.message) || 'HTTP ' + r.status);
      er.status = r.status; er.code = (j && j.error) || 'http'; er.data = j || {};
      throw er;
    }
    return j;
  }

  /** o helper se identifica no cabeçalho Server ("LatentAtlas/1.0"); outro servidor estático
      (python -m http.server, nginx…) nem recebe o GET /api/health — assim não há 404 no console */
  async function serverIsHelper() {
    try {
      const r = await fetch(location.pathname, { method: 'HEAD', cache: 'no-store' });
      const sv = r.headers.get('Server');
      return sv == null ? null : /^LatentAtlas\//.test(sv);
    } catch (e) { return null; }
  }
  async function detect() {
    if (location.protocol === 'file:' || !window.fetch) { S.helper = false; update(); return; }
    if (await serverIsHelper() === false) { S.helper = false; update(); return; }
    try {
      const r = await fetch('/api/health', { cache: 'no-store' });
      const j = r.ok ? await r.json() : null;
      S.helper = !!(j && j.app === 'latent-atlas');
    } catch (e) { S.helper = false; }
    update();
    if (S.helper) { poll(); extCheck(); }
  }

  /* --------------------------------------------------------------- polling */
  let pollT = 0, polling = false, pollAgain = false;
  function busy() {
    const st = S.st;
    if (!st) return false;
    return (st.mlx && st.mlx.install && st.mlx.install.state === 'running') ||
      (st.download && st.download.state === 'running') || (st.server && st.server.state === 'starting');
  }
  function delay() {
    // helper sumido: com o modal aberto tenta a cada 4 s (volta rápido); fechado, a cada 20 s
    if (S.lost >= 3) return S.open && !document.hidden ? 4000 : 20000;
    let d = busy() ? 1000 : S.open ? 4000 : 15000;
    if (document.hidden) d = Math.max(d, busy() ? 5000 : 30000);
    return d;
  }
  function schedule(ms) {
    if (!S.helper) return;
    clearTimeout(pollT);
    pollT = setTimeout(poll, ms == null ? delay() : ms);
  }
  async function poll() {
    if (!S.helper) return;
    if (polling) { pollAgain = true; return; }
    polling = true;
    try {
      const st = await api('/api/mlx/status');
      const wasLost = S.lost >= 3;
      S.lost = 0;
      onStatus(st, wasLost);
    } catch (e) {
      S.lost++;
      if (S.lost === 3) update();
    } finally {
      polling = false;
      if (pollAgain) { pollAgain = false; schedule(0); } else schedule();
    }
  }

  function onStatus(st, force) {
    if (!st || typeof st !== 'object') return;
    const prev = S.st;
    if (prev && st.seq === prev.seq && !force) { updateLive(); return; }
    S.st = st;
    // porta nunca escolhida: começa pela do helper (--mlx-port), não por um 8080 fixo
    if (!prev && !S.portSet && st.server && validPort(+st.server.port) && +st.server.port !== helperPort()) {
      S.port = +st.server.port;
      if (M) M.portIn.value = S.port;
    }
    transitions(prev, st);
    update();
  }

  function transitions(prev, st) {
    const srv = st.server || {}, ps = (prev && prev.server) || {};
    if (srv.state === 'ready' && !(ps.state === 'ready' && ps.id === srv.id && ps.port === srv.port)) {
      const announce = !!prev && (ps.state === 'starting' || S.userStart === srv.id);
      const cfg = LA.copilot && LA.copilot.config;
      // um servidor externo escolhido pelo usuário ("usar esse servidor") não é trocado sem pedido
      const extOn = !!(S.ext && S.ext.port !== srv.port && copilotOn(S.ext.port));
      if (announce || (cfg && cfg.provider === 'mlx' && !extOn)) connectCopilot(srv.id, srv.port, announce);
      if (S.ext && (announce || S.ext.port === srv.port)) setExt(null);
      S.userStart = null;
    }
    if (srv.state === 'error' && S.userStart === srv.id) S.userStart = null;
    const dl = st.download || {}, pd = (prev && prev.download) || {};
    const A = autoGet();
    if (A) {
      const loc = (st.local_models || []).find((l) => l.id === A.id);
      const mine = dl.id === A.id;
      const nowS = Date.now() / 1000;
      if (mine && (dl.state === 'error' || dl.state === 'canceled')) autoSet(null);
      else if (mine && dl.state === 'running') { /* espera o fim */ }
      else if (mine && dl.state === 'done') {
        const since = nowS - (+dl.finished || nowS);
        if (loc && loc.complete && since < 3600) { autoSet(null); setTimeout(() => startServer(A.id, true, A), 0); }
        else if (since > AUTO_GRACE) autoSet(null); // terminou há muito tempo (ou o cache não fechou)
      } else if (nowS - (+A.t || 0) > AUTO_GRACE) autoSet(null); // intenção velha (helper reiniciado…)
    } else if (pd.state === 'running' && dl.state === 'done') toast(T('dlDone', { name: short(dl.id) }));
    const pi = prev && prev.mlx && prev.mlx.install, ni = st.mlx && st.mlx.install;
    if (pi && ni && pi.state === 'running' && ni.state === 'done') toast(T('toastInstalled'));
  }

  /** liga o copilot ao servidor local */
  function connectCopilot(label, port, announce) {
    if (!LA.copilot || !LA.copilot.configure) return;
    LA.copilot.configure({ provider: 'mlx', base: 'http://127.0.0.1:' + port + '/v1', model: '', label: label || '', noThink: true, ctxChars: 9000, tokenField: 'max_tokens' });
    if (announce) toast(T('toast', { name: short(label) || ':' + port }));
    update();
  }
  function copilotOn(port) {
    const cfg = LA.copilot && LA.copilot.config;
    return !!(cfg && cfg.provider === 'mlx' && cfg.base === 'http://127.0.0.1:' + port + '/v1');
  }

  /* ---------------------------------------------------------------- ações */
  async function post(op, path, body) {
    S.busyOp = op; S.err = null;
    if (op !== 'start') S.portBusy = null;
    update();
    try {
      const j = await api(path, body || {});
      if (j && typeof j.seq === 'number' && j.mac) onStatus(j, true);
      return j || {};
    } catch (e) {
      if (e.code === 'port_busy') S.portBusy = { port: body && body.port, openai: !!(e.data && e.data.openai), models: (e.data && e.data.models) || [] };
      else S.err = { op, code: e.code, message: e.message, data: e.data || {} };
      if (op === 'start') S.userStart = null;
      if (op === 'download') autoSet(null);
      return null;
    } finally {
      S.busyOp = null;
      update();
      schedule(250);
    }
  }
  function fitOk(m) {
    if (!m.bytes) return true;
    const f = LA.mlxFit(m.bytes, ramBytes());
    if (f.fit === 'ok') return true;
    return window.confirm(T('fitConfirm', { name: short(m.id), need: gb(f.need), budget: gb(f.budget) }));
  }
  function supportOk(m) {
    return supportedOf(m) !== false || window.confirm(T('unsupConfirm', { name: short(m.id), t: m.type || '?' }));
  }
  function install(upgrade) { return post('install', '/api/mlx/install', { upgrade: !!upgrade }); }
  async function download(m) {
    if (!fitOk(m) || !supportOk(m)) return;
    if (!portOk()) { update(); return; }
    // a intenção de subir o servidor no fim vai para o localStorage (sobrevive a reload) e
    // também no corpo, para um helper que saiba subir sozinho mesmo com a aba fechada
    const start = { port: S.port, max_tokens: S.adv.maxTokens, temp: S.adv.temp };
    autoSet(Object.assign({ id: m.id, t: Date.now() / 1000 }, start));
    const r = await post('download', '/api/mlx/download', { id: m.id, start });
    if (!r) autoSet(null);
  }
  /** sobe o servidor; `o` = {port, max_tokens, temp} da intenção salva (auto-start) */
  async function startServer(id, auto, o) {
    const m = findModel(id) || { id };
    const port = o && validPort(+o.port) ? +o.port : S.port;
    const maxTokens = o && o.max_tokens ? +o.max_tokens : S.adv.maxTokens;
    const temp = o && o.temp != null ? +o.temp : S.adv.temp;
    if (!(o ? validPort(port) && !(S.helper && port === helperPort()) : portOk())) { update(); return; }
    if (!auto && (!fitOk(m) || !supportOk(m))) return;
    S.portBusy = null; S.err = null;
    const srv = (S.st && S.st.server) || {};
    const ours = srv.port === port && (srv.state === 'ready' || srv.state === 'starting');
    if (!ours) {
      try {
        const pr = await api('/api/mlx/probe?port=' + port);
        if (pr && pr.listening && !pr.managed) { S.portBusy = { port, openai: !!pr.openai, models: pr.models || [] }; update(); return; }
        if (S.ext && S.ext.port === port && !(pr && pr.listening)) setExt(null); // o externo saiu da porta
      } catch (e) { /* o start valida de novo */ }
    }
    S.userStart = id;
    await post('start', '/api/mlx/start', { id, port, max_tokens: maxTokens, temp });
  }
  const stopServer = () => post('stop', '/api/mlx/stop', {});
  const cancelDl = () => { autoSet(null); return post('cancel', '/api/mlx/download/cancel', {}); };
  async function del(m, bytes) {
    if (!window.confirm(T('c.delConfirm', { id: m.id, s: gb(bytes) }))) return;
    const r = await post('delete', '/api/mlx/delete', { id: m.id });
    if (r) toast(T('c.deleted', { s: gb(r.freed_bytes != null ? r.freed_bytes : bytes) }));
  }

  /* ---------------------------------------------------- porta / avançado */
  function portOk() {
    const p = S.port;
    return Number.isInteger(p) && p >= 1024 && p <= 65535 && !(S.helper && p === helperPort());
  }

  /* ------------------------------------------ adicionar do Hugging Face */
  function parseId(raw) {
    let s = String(raw || '').trim();
    const m = s.match(/^(?:https?:\/\/)?(?:www\.)?(?:huggingface\.co|hf\.co)\/(.*)$/i);
    if (m) s = m[1];
    s = s.replace(/^models\//, '');
    const parts = s.split(/[/?#]/).filter(Boolean);
    if (parts.length < 2) return null;
    const id = parts[0] + '/' + parts[1];
    if (!ID_RE.test(id) || id.includes('--') || id.includes('..')) return null;
    return id;
  }
  const globRe = (p) => new RegExp('^' + p.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*').replace(/\?/g, '.') + '$');
  let PATS = null;
  const allowed = (name) => (PATS || (PATS = LA.mlxPatterns.map(globRe))).some((re) => re.test(name));

  /** info do modelo direto da API pública do Hugging Face (modo manual) */
  async function hfInfo(id) {
    let r;
    const url = 'https://huggingface.co/api/models/' + id.split('/').map(encodeURIComponent).join('/') + '?blobs=true';
    try { r = await fetch(url, { cache: 'no-store' }); } catch (e) { const er = new Error(e.message || 'network'); er.code = 'hf_unreachable'; throw er; }
    if (r.status === 401 || r.status === 403 || r.status === 404) { const er = new Error('HTTP ' + r.status); er.code = 'not_found'; throw er; }
    if (!r.ok) { const er = new Error('HTTP ' + r.status); er.code = 'hf_unreachable'; throw er; }
    const j = await r.json();
    const cfg = j.config || {};
    const q = cfg.quantization || cfg.quantization_config || {};
    const files = (j.siblings || []).filter((s) => s && s.rfilename && allowed(s.rfilename))
      .map((s) => ({ name: s.rfilename, size: (s.lfs && s.lfs.size) || s.size || 0 }));
    return {
      id: j.id || j.modelId || id, sha: j.sha || null, files,
      download_bytes: files.reduce((a, f) => a + (+f.size || 0), 0),
      model_type: cfg.model_type || null, mapped_type: null, supported: null,
      library: j.library_name || null, pipeline_tag: j.pipeline_tag || null, tags: j.tags || [],
      gated: j.gated || false, params: (j.safetensors && j.safetensors.total) || null,
      quant_bits: q.bits || null,
      moe: ['num_experts', 'num_local_experts', 'n_routed_experts', 'moe_intermediate_size'].some((k) => cfg[k] != null),
    };
  }
  function fmtParams(n) {
    if (!n) return null;
    if (n >= 1e9) return (n >= 1e10 ? Math.round(n / 1e9) : Math.round(n / 1e8) / 10) + 'B';
    return Math.round(n / 1e6) + 'M';
  }
  function toEntry(info) {
    const id = info.id;
    const name = short(id);
    const tags = Array.isArray(info.tags) ? info.tags : [];
    const files = Array.isArray(info.files) ? info.files : [];
    const isMlx = info.library === 'mlx' || tags.includes('mlx');
    const size = name.match(/(?:^|[-_.])(\d+(?:\.\d+)?)[Bb](?=$|[-_.])/);
    const act = name.match(/[-_.][Aa](\d+(?:\.\d+)?)[Bb](?=$|[-_.])/);
    const bitsName = name.match(/(\d{1,2})-?bit/i);
    let bits = null;
    if (/mxfp4/i.test(name)) bits = 'MXFP4';
    else if (info.quant_bits) bits = info.quant_bits + '-bit';
    else if (bitsName) bits = bitsName[1] + '-bit';
    const quantized = !!(bits || isMlx);
    return {
      id, bytes: +info.download_bytes || 0,
      params: size ? size[1] + 'B' : (!quantized ? fmtParams(info.params) : null),
      active: act ? '~' + act[1] + 'B' : null,
      type: info.model_type || null,
      moe: !!info.moe || !!act || /moe/i.test(name),
      think: /think|qwq|-r1\b|gpt-oss|reason/i.test(name) || (/^qwen3/i.test(name) && !/instruct|coder/i.test(name)),
      vision: info.pipeline_tag === 'image-text-to-text' || tags.includes('vision') || tags.includes('image-text-to-text'),
      harmony: info.model_type === 'gpt_oss' || /gpt-oss/i.test(name),
      qat: /qat/i.test(name),
      bits, gated: !!info.gated, library: info.library || null, mlxLib: isMlx,
      supported: info.supported == null ? null : !!info.supported,
      weights: files.some((f) => /^model.*\.safetensors$/.test(f.name)),
      added: Date.now(), custom: true,
    };
  }
  async function addModel() {
    const raw = M.addIn.value;
    if (!raw.trim()) { M.addIn.focus(); return; }
    const id = parseId(raw);
    if (!id) { setAdd('bad', [T('add.bad')]); return; }
    const ex = curatedOf(id) || customOf(id);
    if (ex) { M.addIn.value = ''; flash(ex.id); setAdd('info', [T('add.dup', { id: ex.id })]); select(ex.id); return; }
    setAdd('busy', [T('add.busy', { id })]);
    M.addBtn.disabled = true;
    try {
      // helper fora do ar (ou caiu agora): consulta a API pública do Hugging Face direto
      let info = null;
      if (S.helper && S.lost < 3) {
        try { info = await api('/api/mlx/info?id=' + encodeURIComponent(id)); } catch (e) { if (e.code !== 'net') throw e; }
      }
      if (!info) info = await hfInfo(id);
      const e = toEntry(Object.assign({ id }, info));
      if (!e.weights) { setAdd('bad', [T('add.noWeights', { id: e.id })]); return; }
      const dup = curatedOf(e.id) || customOf(e.id);
      if (dup) { M.addIn.value = ''; flash(dup.id); setAdd('info', [T('add.dup', { id: dup.id })]); select(dup.id); return; }
      delete e.weights;
      S.custom.push(e); save('mlx:custom', S.custom.map(stripEntry));
      M.addIn.value = '';
      const msgs = [T('add.ok', { id: e.id, s: gb(e.bytes) })];
      if (!e.mlxLib) msgs.push(T('add.notMlx', { lib: e.library || '?' }));
      if (e.supported === false) msgs.push(T('add.unsup', { t: e.type || '?', v: (S.st && S.st.mlx && S.st.mlx.version) || '?' }));
      if (e.gated) msgs.push(T('add.gated'));
      setAdd(msgs.length > 1 ? 'warn' : 'ok', msgs);
      flash(e.id);
      select(e.id);
    } catch (err) {
      if (err.code === 'not_found' || err.status === 404) setAdd('bad', [T('add.notFound', { id })]);
      else if (err.code === 'bad_id' || err.status === 400) setAdd('bad', [T('add.bad')]);
      else setAdd('bad', [T('add.net', { m: err.message })]);
    } finally {
      M.addBtn.disabled = false;
    }
  }
  const stripEntry = (e) => {
    const o = {};
    ['id', 'bytes', 'params', 'active', 'type', 'moe', 'think', 'vision', 'harmony', 'qat', 'bits', 'gated', 'library', 'mlxLib', 'supported', 'added'].forEach((k) => { o[k] = e[k] == null ? null : e[k]; });
    return o;
  };
  S.custom.forEach((c) => { c.custom = true; });
  function removeCustom(id) {
    S.custom = S.custom.filter((c) => c.id !== id);
    save('mlx:custom', S.custom.map(stripEntry));
    if (S.sel === id) { S.sel = null; save('mlx:sel', null); }
    setAdd('', []);
    update();
  }
  function setAdd(kind, msgs) { S.add = { kind, msgs }; update(); }
  function flash(id) { S.flash = id; setTimeout(() => { if (S.flash === id) S.flash = null; }, 1600); }

  /* -------------------------------------------------------- modo manual */
  async function manualCheck() {
    if (!portOk()) { update(); return; }
    const port = S.port;
    S.manual = { state: 'busy', models: [], port, err: '' };
    update();
    const ctl = window.AbortController ? new AbortController() : null;
    const tm = setTimeout(() => ctl && ctl.abort(), 5000);
    try {
      const r = await fetch('http://127.0.0.1:' + port + '/v1/models', { cache: 'no-store', signal: ctl ? ctl.signal : undefined });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const j = await r.json();
      const ids = (j.data || j.models || []).map((x) => x && (x.id || x.name)).filter(Boolean);
      S.manual = { state: 'ok', models: ids, port, err: '' };
      connectCopilot(selId(), port, true);
    } catch (e) {
      S.manual = { state: 'bad', models: [], port, err: String(e && e.message || e) };
    } finally {
      clearTimeout(tm);
      update();
    }
  }

  /* ---------------------------------------------------------------- toast */
  let toastEl = null, toastT = 0;
  function toast(msg) {
    if (!toastEl) { toastEl = h('div', { class: 'la-mlx-toast', role: 'status', 'aria-live': 'polite' }); document.body.appendChild(toastEl); }
    toastEl.textContent = msg;
    toastEl.classList.remove('on');
    void toastEl.offsetWidth;
    toastEl.classList.add('on');
    clearTimeout(toastT);
    toastT = setTimeout(() => toastEl.classList.remove('on'), 3600);
  }

  /* ------------------------------------------------------------- status */
  function status() {
    if (S.helper === null) return { k: 'wait' };
    if (!S.helper) return S.manual.state === 'ok' ? { k: 'ready', manual: true, port: S.manual.port, id: selId() } : { k: 'manual' };
    if (S.lost >= 3) return { k: 'lost' };
    const st = S.st;
    if (!st) return { k: 'wait' };
    const ins = (st.mlx && st.mlx.install) || {}, dl = st.download || {}, srv = st.server || {};
    if (dl.state === 'running') return { k: 'dl', p: pct(dl.done_bytes, dl.total_bytes), id: dl.id };
    if (ins.state === 'running') return { k: 'install' };
    if (srv.state === 'starting') return { k: 'loading', id: srv.id };
    if (srv.state === 'ready') return { k: 'ready', id: srv.id, tok: srv.tok_s, port: srv.port };
    const ex = extLive();
    if (ex) return { k: 'ready', ext: true, port: ex.port, id: ex.id };
    if (srv.state === 'error' || ins.state === 'error' || dl.state === 'error') return { k: 'error' };
    return { k: 'off' };
  }
  const DOT = { wait: 'off', manual: 'off', off: 'off', dl: 'busy', install: 'busy', loading: 'busy', ready: 'ready', error: 'error', lost: 'error' };
  function pillText(s) {
    if (s.k === 'ready' && (s.manual || s.ext)) return T('pill.manualOk', { port: s.port });
    return T('pill.' + s.k, { p: s.p });
  }

  /** fase do CTA principal (modo helper) */
  function phase() {
    const st = S.st;
    if (S.helper === null) return 'wait';
    if (!S.helper) return 'manual';
    if (S.lost >= 3) return 'lost';
    if (!st) return 'wait';
    const mac = st.mac || {}, mlx = st.mlx || {}, ins = mlx.install || {}, dl = st.download || {}, srv = st.server || {};
    if (!(mac.os === 'Darwin' && mac.arm64)) return 'unsupported';
    if (ins.state === 'running') return 'installing';
    if (!mlx.installed) return mlx.detecting ? 'checking' : 'install';
    const id = selId();
    if (dl.state === 'running') return dl.id === id ? 'downloading' : 'otherdl';
    if (srv.id === id && srv.state === 'starting') return 'starting';
    if (srv.id === id && srv.state === 'ready') return 'ready';
    if (srv.id === id && srv.state === 'error') return 'srverror';
    const loc = localOf(id);
    if (loc && loc.complete) return srv.state === 'ready' || srv.state === 'starting' ? 'switch' : 'start';
    if (dl.id === id && dl.state === 'error') return 'dlerror';
    if (partialOf(selModel(), loc)) return 'resume';
    return 'download';
  }

  /* ========================================================= TOPBAR */
  let btn = null;
  function topButton() {
    btn = h('button', { class: 'la-iconbtn la-mlxbtn', id: 'la-mlx-btn', type: 'button', 'aria-haspopup': 'dialog', html: ICON.chip + '<span class="lbl">MLX</span><i class="la-mlx-dot" data-state="off" aria-hidden="true"></i>' });
    btn.addEventListener('click', open);
    updateTop();
    return btn;
  }
  function updateTop() {
    if (!btn) return;
    const s = status();
    const dot = btn.querySelector('.la-mlx-dot');
    const ds = DOT[s.k] || 'off';
    if (dot && dot.getAttribute('data-state') !== ds) dot.setAttribute('data-state', ds);
    btn.setAttribute('data-state', ds);
    let title;
    if (s.k === 'dl') title = T('btn.dl', { p: s.p });
    else if (s.k === 'loading') title = T('btn.loading', { name: short(s.id) });
    else if (s.k === 'ready') title = s.manual || s.ext ? T('btn.manual', { port: s.port }) : T('btn.ready', { name: short(s.id), tok: s.tok ? ' · ' + tokTxt(s.tok) : '' });
    else if (s.k === 'install') title = T('btn.install');
    else if (s.k === 'error' || s.k === 'lost') title = T('btn.error');
    else title = T('btn.off');
    if (btn.title !== title) { btn.title = title; btn.setAttribute('aria-label', title); }
    if (s.k === 'dl') { btn.setAttribute('data-dl', ''); btn.style.setProperty('--p', String(s.p)); }
    else if (btn.hasAttribute('data-dl')) { btn.removeAttribute('data-dl'); btn.style.removeProperty('--p'); }
  }

  /* ========================================================== MODAL */
  let M = null;
  function sec(n, title) {
    const body = h('div', { class: 'la-mlx-secb' });
    const el = h('section', { class: 'la-mlx-sec', 'aria-labelledby': 'la-mlx-s' + n },
      h('h3', { id: 'la-mlx-s' + n }, h('span', { class: 'n', 'aria-hidden': 'true' }, CIRC[n - 1]), h('span', null, title)), body);
    return { el, body };
  }
  function build() {
    const close = h('button', { class: 'la-mlx-x', type: 'button', 'aria-label': T('close'), title: T('close') }, '✕');
    close.addEventListener('click', closeModal);
    const pill = h('span', { class: 'la-mlx-pill', 'data-state': 'off' }, h('i'), h('span'));
    const head = h('header', { class: 'la-mlx-head' },
      h('div', { class: 'la-mlx-htxt' },
        h('span', { class: 'la-mlx-eyebrow' }, h('i'), T('eyebrow')),
        h('h2', { id: 'la-mlx-title' }, T('title')),
        h('p', { class: 'la-mlx-lead' }, T('lead'))),
      h('div', { class: 'la-mlx-hside' }, pill, close));

    const cv = h('canvas', { 'aria-hidden': 'true' });
    const pipe = h('div', { class: 'la-mlx-pipe', role: 'img', 'aria-label': T('pipeAria') }, cv);

    const sMac = sec(1, T('s1')), sMod = sec(2, T('s2')), sRun = sec(3, T('s3')), sCp = sec(4, T('s4'));

    // ② adicionar do Hugging Face (persistente: nunca reconstruído enquanto se digita)
    const addIn = h('input', { class: 'la-input', type: 'text', id: 'la-mlx-add', placeholder: T('add.ph'), spellcheck: 'false', autocomplete: 'off', autocapitalize: 'off', 'aria-describedby': 'la-mlx-addmsg' });
    const addBtn = h('button', { class: 'la-b', type: 'button', 'aria-label': T('add.b') }, h('span', { 'aria-hidden': 'true' }, '+'), h('span', { class: 't' }, T('add.b')));
    addBtn.addEventListener('click', addModel);
    addIn.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); addModel(); } });
    const addMsg = h('div', { class: 'la-mlx-addmsg', id: 'la-mlx-addmsg', 'aria-live': 'polite' });
    const add = h('div', { class: 'la-mlx-add' },
      h('label', { for: 'la-mlx-add' }, T('add.t')),
      h('div', { class: 'row' }, addIn, addBtn), addMsg);
    const list = h('div', { class: 'la-mlx-list' });
    sMod.body.append(add, list);

    // ③ porta + avançado (persistente)
    const portIn = h('input', { class: 'la-input', type: 'number', id: 'la-mlx-port', min: 1024, max: 65535, step: 1, inputmode: 'numeric' });
    portIn.value = S.port;
    portIn.addEventListener('input', () => {
      S.port = Math.round(+portIn.value);
      S.portSet = true;
      if (portOk()) save('mlx:port', S.port);
      if (S.portBusy && S.portBusy.port !== S.port) S.portBusy = null;
      update();
    });
    const advField = (key, label, attrs, lo, hi) => {
      const inp = h('input', Object.assign({ class: 'la-input', type: 'number', id: 'la-mlx-' + key }, attrs));
      inp.value = S.adv[key];
      inp.addEventListener('input', () => {
        const v = +inp.value;
        const ok = inp.value !== '' && v >= lo && v <= hi;
        inp.classList.toggle('bad', !ok);
        if (ok) { S.adv[key] = key === 'maxTokens' ? Math.round(v) : v; save('mlx:adv', S.adv); update(); }
      });
      return h('label', { class: 'la-mlx-fld', for: 'la-mlx-' + key }, h('span', null, label), inp);
    };
    const adv = h('details', { class: 'la-mlx-adv' }, h('summary', null, T('adv')),
      h('div', { class: 'grid' },
        advField('maxTokens', T('maxTok'), { min: 16, max: 32768, step: 64 }, 16, 32768),
        advField('temp', T('temp'), { min: 0, max: 2, step: 0.1 }, 0, 2)));
    const portRow = h('div', { class: 'la-mlx-cfg' },
      h('label', { class: 'la-mlx-fld port', for: 'la-mlx-port' }, h('span', null, T('port')), portIn), adv);
    const portMsg = h('div', { class: 'la-mlx-portmsg' });
    const runSum = h('div', { class: 'la-mlx-sum' });
    // sem aria-live aqui: o progresso muda a cada segundo. Quem anuncia é o `sr` (mudanças de fase
    // e degraus de 25% do download)
    const runMain = h('div', { class: 'la-mlx-main' });
    const sr = h('div', { class: 'la-visually-hidden', role: 'status', 'aria-live': 'polite' });
    const runSec = h('div', { class: 'la-mlx-sec2' });
    const logPre = h('pre'); // focável (tabindex) só com o log aberto — ver o toggle
    const logKindEl = h('span', { class: 'k' });
    const log = h('details', { class: 'la-mlx-log' }, h('summary', null, T('c.log'), logKindEl), logPre);
    log.addEventListener('toggle', () => {
      if (log.open) { logPre.setAttribute('tabindex', '0'); logPoll(); } else { logPre.removeAttribute('tabindex'); clearTimeout(logT); }
    });
    sRun.body.append(runSum, portRow, portMsg, runMain, runSec, log, sr);

    const cpBody = h('div');
    const foot = h('p', { class: 'la-mlx-foot' });
    sCp.body.append(cpBody);

    const colA = h('div', { class: 'la-mlx-col a' }, sMac.el, sMod.el);
    const colB = h('div', { class: 'la-mlx-col b' }, sRun.el, sCp.el, foot);
    const scroll = h('div', { class: 'la-mlx-scroll' }, pipe, h('div', { class: 'la-mlx-grid' }, colA, colB));

    // atalho do celular: o status vem primeiro (nunca cortado); o nome do modelo é que encolhe
    const dockSt = h('span', { class: 'st', 'data-state': 'off' });
    const dockTxt = h('span', { class: 'tx' });
    const dock = h('div', { class: 'la-mlx-dock' }, dockSt, dockTxt,
      h('button', { class: 'la-b primary', type: 'button', onclick: () => sRun.el.scrollIntoView({ behavior: reduced() ? 'auto' : 'smooth', block: 'start' }) }, T('dock.go')));

    const dlg = h('div', { class: 'la-mlx-dlg', role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'la-mlx-title', tabindex: '-1' }, head, scroll, dock);
    const root = h('div', { class: 'la-mlx', id: 'la-mlx' }, dlg);
    let downOnRoot = false;
    root.addEventListener('mousedown', (e) => { downOnRoot = e.target === root; });
    root.addEventListener('click', (e) => { if (e.target === root && downOnRoot) closeModal(); });
    document.addEventListener('keydown', onKey, true);
    document.body.appendChild(root);

    M = { root, dlg, head, pill, close, cv, pipe, scroll, macBody: sMac.body, list, addIn, addBtn, addMsg,
      portIn, portMsg, runSum, runMain, runSec, log, logPre, logKindEl, cpBody, foot, dock, dockSt, dockTxt, sr, srKey: '', secRun: sRun.el, prog: null, elapsed: null, tail: null };

    if ('IntersectionObserver' in window) {
      const io = new IntersectionObserver((es) => { const e = es[es.length - 1]; dock.classList.toggle('hide', e.isIntersecting); }, { root: scroll, threshold: 0.02 });
      io.observe(sRun.el);
    } else dock.classList.add('hide');
    if ('ResizeObserver' in window) {
      new ResizeObserver(() => { if (S.open) { pipeFit(); pipeKick(); } }).observe(pipe);
      const ro = new ResizeObserver(() => stickB());
      ro.observe(colB); ro.observe(scroll);
    }
    M.colB = colB;
  }
  /** coluna da direita "gruda" pelo topo; se for mais alta que a área visível, gruda pela base */
  function stickB() {
    if (!M || !S.open) return;
    const st = getComputedStyle(M.scroll);
    const avail = M.scroll.clientHeight - parseFloat(st.paddingTop) - parseFloat(st.paddingBottom);
    const top = Math.min(0, Math.floor(avail - M.colB.offsetHeight));
    M.colB.style.top = top + 'px';
  }

  /** teclado com o modal aberto (fase de captura no document): Esc fecha, Tab fica preso no diálogo
      e os atalhos globais (⌘K, ⌘J, Esc do copilot) não vazam para trás do modal */
  function onKey(e) {
    if (!S.open) return;
    const inside = M.dlg.contains(document.activeElement);
    if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); closeModal(); return; }
    if (e.key === 'Tab') {
      // só o que o Tab nativo alcança: fora de <details> fechado (exceto o próprio summary),
      // de [hidden] e do atalho do celular escondido
      const f = Array.from(M.dlg.querySelectorAll('button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea, summary, [tabindex]:not([tabindex="-1"])'))
        .filter((el) => el.getClientRects().length && !el.closest('.la-mlx-dock.hide, [hidden]') &&
          !((el.tagName === 'SUMMARY' && el.parentElement && el.parentElement.parentElement) || el).closest('details:not([open])') &&
          (!el.checkVisibility || el.checkVisibility()));
      if (!f.length) { e.preventDefault(); M.dlg.focus(); return; }
      const first = f[0], last = f[f.length - 1], a = document.activeElement;
      if (!inside) { e.preventDefault(); (e.shiftKey ? last : first).focus(); }
      else if (e.shiftKey && (a === first || a === M.dlg)) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && a === last) { e.preventDefault(); first.focus(); }
      e.stopPropagation();
      return;
    }
    if ((e.metaKey || e.ctrlKey) && /^[kj]$/i.test(e.key)) { e.preventDefault(); e.stopPropagation(); }
  }

  function open() {
    if (!M) build();
    if (S.open) return;
    S.open = true;
    S.opener = document.activeElement && document.activeElement !== document.body ? document.activeElement : null;
    M.root.classList.add('on');
    document.documentElement.classList.add('la-mlx-lock');
    update();
    showSelected();
    pipeFit();
    pipeStart();
    setTimeout(() => { if (S.open) M.dlg.focus({ preventScroll: true }); }, 30);
    if (S.helper) { schedule(0); extCheck(); }
    if (M.log.open) logPoll();
  }
  /** ao abrir, o cartão escolhido (ou o que está rodando) fica visível na lista, sem animação */
  function showSelected() {
    const card = M.list.querySelector('.la-mlx-card.on');
    if (!card || card.closest('[data-g="sugg"]')) return; // a sugestão já é o 1º cartão da lista
    const sc = M.scroll;
    let top = 0;
    for (let n = card; n && n !== sc; n = n.offsetParent) top += n.offsetTop;
    const bot = top + card.offsetHeight;
    if (top >= sc.scrollTop && bot <= sc.scrollTop + sc.clientHeight) return;
    sc.scrollTop = Math.max(0, top - Math.max(24, (sc.clientHeight - card.offsetHeight) / 2));
  }
  function closeModal() {
    if (!S.open) return;
    S.open = false;
    M.root.classList.remove('on');
    document.documentElement.classList.remove('la-mlx-lock');
    pipeStop();
    M.srKey = ''; M.sr.textContent = '';
    clearTimeout(logT); if (tailT > 0) clearTimeout(tailT); tailT = 0; clearInterval(tickT); tickT = 0;
    const back = S.opener && document.body.contains(S.opener) ? S.opener : document.getElementById('la-mlx-btn');
    if (back && back.focus) back.focus({ preventScroll: true });
    S.opener = null;
    schedule();
  }

  /* ---------------------------------------------------- render helpers */
  /** reconstrói el só quando a assinatura muda; preserva o foco (data-k) */
  function part(el, sig, build) {
    if (el._sig === sig) return false;
    el._sig = sig;
    const ae = document.activeElement;
    const had = !!(ae && el.contains(ae));
    const key = had ? ae.getAttribute('data-k') : null;
    const kids = [].concat(build()).flat(Infinity).filter((x) => x != null && x !== false);
    el.replaceChildren(...kids.map((k) => (typeof k === 'string' ? document.createTextNode(k) : k)));
    if (had) {
      // o foco nunca cai no body: volta ao mesmo controle ou ao próprio diálogo
      const n = key ? Array.from(el.querySelectorAll('[data-k]')).find((x) => x.getAttribute('data-k') === key && !x.disabled) : null;
      if (n) n.focus({ preventScroll: true });
      else if (M && S.open) M.dlg.focus({ preventScroll: true });
    }
    return true;
  }
  const ic = (k) => h('i', { class: 'ic', 'data-ic': k, 'aria-hidden': 'true' }, { ok: '✓', bad: '✕', off: '○', warn: '!', busy: '' }[k] || '');
  function row(k, label, value, extra, opts) {
    return h('div', { class: 'la-mlx-row', 'data-ic': k },
      ic(k), h('span', { class: 'k' }, label),
      h('span', Object.assign({ class: 'v' + (opts && opts.mono ? ' mono' : '') }, opts && opts.title ? { title: opts.title } : {}), value),
      extra ? h('span', { class: 'act' }, extra) : null);
  }
  function note(kind, ...kids) { return h('div', { class: 'la-mlx-note', 'data-k2': kind }, kids); }
  function button(label, onclick, o) {
    o = o || {};
    const b = h('button', { class: 'la-b' + (o.primary ? ' primary' : '') + (o.cls ? ' ' + o.cls : ''), type: 'button', 'data-k': o.k || label, disabled: o.disabled || null, title: o.title || null }, label);
    if (onclick) b.addEventListener('click', onclick);
    return b;
  }
  function cta(label, onclick, o) {
    o = o || {};
    const b = h('button', { class: 'la-mlx-cta' + (o.cls ? ' ' + o.cls : ''), type: 'button', 'data-k': 'cta', disabled: o.disabled || S.busyOp ? true : null }, o.icon ? h('span', { class: 'i', 'aria-hidden': 'true' }, o.icon) : null, h('span', null, label));
    if (onclick) b.addEventListener('click', onclick);
    return b;
  }
  function copyText(text, done) {
    const ok = () => done(true), fb = () => {
      try {
        const ta = h('textarea', { style: { position: 'fixed', opacity: '0', top: '0', left: '0' } });
        ta.value = text; document.body.appendChild(ta); ta.select();
        const r = document.execCommand('copy'); ta.remove(); done(r);
      } catch (e) { done(false); }
    };
    if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(ok, fb); else fb();
  }
  function codeBox(text, k) {
    const b = h('button', { class: 'la-mlx-copy', type: 'button', 'data-k': 'copy:' + (k || text.slice(0, 24)) }, LA.t('code.copy'));
    b.addEventListener('click', () => copyText(text, (ok) => { b.textContent = ok ? LA.t('code.copied') : '✕'; setTimeout(() => { b.textContent = LA.t('code.copy'); }, 1400); }));
    return h('div', { class: 'la-mlx-code' }, h('pre', null, h('code', null, text)), b);
  }
  /** frase traduzida para um código de erro do helper: {error, message} da API ou os campos
      *_code / error do status (download, server, install). err = {code, message, data} */
  function errText(err) {
    if (!err) return '';
    const map = LA.t('mlx.e') || {};
    const d = err.data || {};
    let code = err.code;
    if (code === 'bad_param' && d.param === 'port') code = 'bad_port';
    let s = map[code];
    if (typeof s === 'string') s = s.replace(/\{(\w+)\}/g, (m, k) => (k === 'need' ? gb(d.need_bytes) : k === 'free' ? gb(d.free_bytes) : k === 'param' ? String(d.param || '?') : m));
    return s || err.message || String(code || '');
  }
  /** detalhe cru do helper (inglês, com nome de arquivo etc.) como linha secundária */
  function errDetail(err) {
    const raw = err && err.message ? String(err.message) : '';
    if (!raw || err.code === 'net' || !/\s/.test(raw) || raw === errText(err)) return null;
    return h('p', { class: 'la-mlx-detail' }, raw);
  }
  function errNote(err, head, extra) {
    return note('err', head ? h('b', null, head) : null, h('p', null, errText(err)), errDetail(err), extra || null);
  }
  /** erro guardado no status (download/server/install): o código traduz, a mensagem vira detalhe */
  const stErr = (o) => ({ code: (o && (o.error_code || o.error)) || 'internal', message: (o && o.error) || '' });
  function spinLine(text, extra) { return h('div', { class: 'la-mlx-spin' }, h('i', { class: 'sp', 'aria-hidden': 'true' }), h('span', null, text), extra || null); }

  /* ------------------------------------------------------------- update */
  function update() {
    updateTop();
    if (!M || !S.open) return;
    const s = status();
    const ps = DOT[s.k] || 'off';
    M.pill.setAttribute('data-state', ps);
    M.pill.lastChild.textContent = pillText(s);
    renderMac();
    renderModels();
    renderRun();
    renderCp();
    renderDock(s);
    updateLive();
    ticker();
    tailKick();
    pipeKick();
    stickB();
  }

  /* ① Seu Mac */
  function renderMac() {
    const st = S.st || {};
    const sel = selModel();
    const sig = JSON.stringify([S.helper, S.lost >= 3, S.helper ? [st.mac, st.mlx && [st.mlx.installed, st.mlx.detecting, st.mlx.version, st.mlx.mlx_version, st.mlx.install && [st.mlx.install.state, st.mlx.install.method, st.mlx.install.error, st.mlx.install.error_code], st.mlx.installers], st.hf, Math.round(((st.mac && st.mac.disk_free_bytes) || 0) / 1e8)] : S.ramGB, sel.bytes, S.busyOp === 'install', S.err && S.err.op === 'install' && S.err]);
    part(M.macBody, sig, () => (S.helper ? macHelper(st, sel) : macManual()));
  }
  function macManual() {
    const isFile = location.protocol === 'file:';
    const out = [];
    if (S.helper === null) return [spinLine(T('detecting'))];
    // via http já existe um servidor estático nesta porta: o ./serve.sh vai para a próxima livre e
    // imprime a URL — por isso o link direto só aparece no file://
    out.push(note('info', h('b', null, T('nohelper.t')),
      h('p', null, isFile ? T('nohelper.file') : T('nohelper.http')),
      h('p', null, isFile ? T('nohelper.do') : T('nohelper.doHttp')),
      codeBox('./serve.sh', 'serve'),
      isFile ? h('a', { class: 'la-b', href: 'http://localhost:8765', 'data-k': 'go8765' }, T('nohelper.open') + ' ↗') : null));
    const seg = h('div', { class: 'la-seg la-mlx-ram', role: 'group', 'aria-label': T('ram') });
    RAMS.forEach((g) => {
      const b = h('button', { type: 'button', class: g === +S.ramGB ? 'on' : '', 'aria-pressed': g === +S.ramGB ? 'true' : 'false', 'data-k': 'ram' + g }, g + NB + 'GB');
      b.addEventListener('click', () => { S.ramGB = g; save('mlx:ram', g); update(); });
      seg.appendChild(b);
    });
    const f = LA.mlxFit(0, S.ramGB * GiB);
    out.push(h('div', { class: 'la-mlx-ramrow' }, h('span', { class: 'lbl' }, T('ram')), seg,
      h('small', null, T('ramHint', { budget: gb(f.budget) }))));
    out.push(h('p', { class: 'la-mlx-muted' }, T('needArm')));
    return out;
  }
  function macHelper(st, sel) {
    const out = [];
    if (S.lost >= 3) out.push(note('err', T('lost')));
    if (!S.st) return out.concat([spinLine(T('connecting'))]);
    const mac = st.mac || {}, mlx = st.mlx || {}, ins = mlx.install || {}, hf = st.hf || {};
    out.push(row(S.lost >= 3 ? 'bad' : 'ok', T('r.helper'), S.lost >= 3 ? T('r.helperLost') : T('r.helperOk', { origin: location.host })));
    const arm = mac.os === 'Darwin' && mac.arm64;
    const f = LA.mlxFit(0, mac.ram_bytes || 16 * GiB);
    out.push(row(arm ? 'ok' : 'bad', T('r.chip'),
      arm ? T('r.chipV', { chip: mac.chip || 'Apple Silicon', ram: ramTxt(mac.ram_bytes || 0), budget: gb(f.budget) }) : [mac.chip || mac.os || '?', mac.arm64 ? 'arm64' : 'x86_64', mac.ram_bytes ? ramTxt(mac.ram_bytes) + ' RAM' : null].filter(Boolean).join(' · '),
      null, { title: arm ? T('r.budgetTip') : null }));
    if (!arm) out.push(note('err', T('r.notArm')));
    // mlx-lm
    if (ins.state === 'running') {
      out.push(row('busy', 'mlx-lm', T('r.installing', { m: ins.method || 'uv' })));
      M.tail = h('pre', { class: 'la-mlx-tail' }, S.tailText || '…');
      out.push(M.tail);
    } else {
      M.tail = null;
      if (mlx.installed) {
        out.push(row('ok', 'mlx-lm', 'v' + (mlx.version || '?') + (mlx.mlx_version ? ' · mlx ' + mlx.mlx_version : ''),
          arm ? button(T('b.update'), () => install(true), { k: 'upd', disabled: !!S.busyOp, title: T('b.updateTip') }) : null, { mono: true }));
        // falha de atualização: o ③ não mostra a fase de instalação, então o detalhe fica aqui
        if (ins.state === 'error') out.push(errNote(stErr(ins), T('r.updateErr')));
      } else if (mlx.detecting) {
        out.push(row('busy', 'mlx-lm', T('detectingMlx')));
      } else {
        // o detalhe do erro de instalação aparece uma vez só, no ③, ao lado do "Tentar de novo"
        out.push(row(ins.state === 'error' ? 'bad' : 'off', 'mlx-lm', ins.state === 'error' ? T('r.installErr') : T('r.mlxMissing'),
          arm ? button(T('c.install'), () => install(false), { k: 'inst', disabled: !!S.busyOp }) : null));
      }
    }
    if (S.err && S.err.op === 'install' && mlx.installed) out.push(installErr());
    // disco
    const free = mac.disk_free_bytes;
    if (free != null) {
      const low = sel.bytes && free < sel.bytes + GiB;
      out.push(row(low ? 'warn' : 'ok', T('r.disk'), T('r.diskV', { s: gb(free) }) + (low ? ' — ' + T('r.diskLow', { s: gb(sel.bytes) }) : '')));
    }
    if (hf.cache_dir) out.push(row('ok', T('r.cache'), hf.cache_dir, null, { mono: true, title: hf.cache_dir }));
    out.push(row(hf.token ? 'ok' : 'off', T('r.token'), hf.token ? T('r.tokenYes') : T('r.tokenNo')));
    return out;
  }
  function installErr() {
    const e = S.err;
    if (e.code === 'no_installer') {
      const hint = (e.data && Array.isArray(e.data.hint) && e.data.hint.length) ? e.data.hint : ['curl -LsSf https://astral.sh/uv/install.sh | sh', 'brew install mlx-lm'];
      return note('err', h('p', null, T('noInstaller')), codeBox(hint.join('\n'), 'hint'));
    }
    return errNote(e);
  }

  /* ② Modelos */
  function renderModels() {
    const ram = ramBytes();
    const st = S.st || {};
    const srv = st.server || {}, dl = st.download || {};
    const loc = locals().map((l) => [l.id, l.complete, dl.state === 'running' && dl.id === l.id ? -1 : Math.round(l.bytes / 1e7)]);
    const sel = selId();
    const mt = st.mlx && st.mlx.installed ? (st.mlx.model_types || []).length : 0;
    const sig = JSON.stringify([ram, sel, S.custom.map((c) => c.id), loc, [srv.id, srv.state], [dl.id, dl.state === 'running'], mt, S.flash, S.helper]);
    if (part(M.list, sig, () => listView(ram, sel))) {
      // as barras de memória só "crescem" na primeira vez e quando a RAM muda
      if (M.animRam !== ram) {
        M.animRam = ram;
        M.list.classList.remove('settled');
        clearTimeout(M.settleT);
        M.settleT = setTimeout(() => M.list.classList.add('settled'), 1100);
      }
      if (S.flash) {
        const c = M.list.querySelector('.la-mlx-card.flash');
        if (c) setTimeout(() => c.scrollIntoView({ block: 'nearest', behavior: reduced() ? 'auto' : 'smooth' }), 40);
      }
    }
    part(M.addMsg, JSON.stringify(S.add), () => (S.add.msgs.length ? h('div', { class: 'la-mlx-addres', 'data-k2': S.add.kind },
      S.add.kind === 'busy' ? h('i', { class: 'sp', 'aria-hidden': 'true' }) : null,
      h('div', null, S.add.msgs.map((m, i) => h('p', { class: i ? 'w' : '' }, m)))) : h('p', { class: 'la-mlx-hint' }, T('add.hint'))));
  }
  function listView(ram, sel) {
    const rec = LA.mlxRecommend(ram);
    const fitOf = (m) => LA.mlxFit(m.bytes, ram).fit;
    // dentro de cada grupo, o que já está no Mac vem antes: rodando > baixado > resto (por rank)
    const srv = (S.st && S.st.server) || {};
    const onMac = (m) => (srv.id === m.id && (srv.state === 'ready' || srv.state === 'starting') ? 2 : (localOf(m.id) || {}).complete ? 1 : 0);
    const byRank = (a, b) => onMac(b) - onMac(a) || b.rank - a.rank;
    const cur = LA.mlxModels.filter((m) => m !== rec);
    const groups = [];
    if (rec) groups.push({ k: 'sugg', t: T('g.sugg', { ram: ramTxt(ram) }), items: [rec] });
    if (S.custom.length) groups.push({ k: 'custom', t: T('g.custom'), items: S.custom.slice().sort((a, b) => (b.added || 0) - (a.added || 0)) });
    const lo = localOnly();
    if (lo.length) groups.push({ k: 'local', t: T('g.local'), items: lo });
    const ok = cur.filter((m) => fitOf(m) === 'ok').sort(byRank);
    const tight = cur.filter((m) => fitOf(m) === 'tight').sort(byRank);
    const no = cur.filter((m) => fitOf(m) === 'no').sort((a, b) => onMac(b) - onMac(a) || a.bytes - b.bytes);
    if (ok.length) groups.push({ k: 'ok', t: T('g.ok'), items: ok });
    if (tight.length) groups.push({ k: 'tight', t: T('g.tight'), items: tight });
    const out = [];
    groups.forEach((g) => {
      out.push(h('div', { class: 'la-mlx-group', 'data-g': g.k, role: 'group', 'aria-label': g.t },
        h('h4', null, g.k === 'sugg' ? h('span', { class: 'star', 'aria-hidden': 'true' }, '★') : null, g.t),
        g.items.map((m) => card(m, { rec: m === rec, sel, ram }))));
    });
    if (no.length) {
      const d = h('details', { class: 'la-mlx-group big', 'data-g': 'no' },
        h('summary', { 'data-k': 'bigsum' }, T('g.no', { n: no.length })),
        no.map((m) => card(m, { rec: false, sel, ram })));
      if (S.noOpen || no.some((m) => m.id === sel)) d.open = true;
      d.addEventListener('toggle', () => { S.noOpen = d.open; });
      out.push(d);
    }
    return out;
  }
  function bd(cls, text, title) { return h('span', { class: 'la-mlx-bd ' + cls, title: title || null }, text); }
  function card(m, o) {
    const st = S.st || {};
    const srv = st.server || {}, dl = st.download || {};
    const loc = localOf(m.id);
    const on = m.id === o.sel;
    const f = m.bytes ? LA.mlxFit(m.bytes, o.ram) : null;
    const sup = supportedOf(m);
    const b = [];
    if (o.rec) b.push(bd('sugg', '★ ' + T('bd.sugg')));
    if (srv.id === m.id && srv.state === 'ready') b.push(bd('run', [h('i'), T('bd.run')]));
    else if (srv.id === m.id && srv.state === 'starting') b.push(bd('busy', T('bd.loading')));
    if (dl.state === 'running' && dl.id === m.id) b.push(h('span', { class: 'la-mlx-bd busy dlp' }, T('bd.dling', { p: pct(dl.done_bytes, dl.total_bytes) })));
    else if (loc && loc.complete) b.push(bd('ok', T('bd.dl')));
    else if (partialOf(m, loc)) b.push(bd('warn', m.bytes ? T('bd.partial', { p: pct(loc.bytes, m.bytes) }) : T('bd.partialU')));
    if (m.custom) b.push(bd('custom', T('bd.custom')));
    if (m.localOnly) b.push(bd('custom', T('bd.local')));
    if (m.moe) b.push(bd('moe', 'MoE' + (m.active ? ' · ' + T('bd.active', { a: m.active }) : ''), T('tip.moe')));
    if (m.think) b.push(bd('think', T('bd.think'), T('tip.think')));
    if (m.vision) b.push(bd('vision', T('bd.vision'), T('tip.vision')));
    if (m.qat) b.push(bd('qat', 'QAT', T('tip.qat')));
    if (m.harmony) b.push(bd('warn', T('bd.harmony'), T('tip.harmony')));
    if (m.gated) b.push(bd('warn', T('bd.gated'), T('add.gated')));
    if (m.custom && m.library && !m.mlxLib) b.push(bd('warn', T('bd.notmlx'), T('add.notMlx', { lib: m.library })));
    if (sup === false) b.push(bd('bad', T('bd.unsup'), T('tip.unsup', { t: m.type || '?' })));
    const spec = [m.params ? T('spec.params', { p: m.params }) : null, m.bits || null,
      m.localOnly ? (loc ? T('spec.disk', { s: gb(loc.bytes) }) : null) : m.bytes ? T('spec.dl', { s: gb(m.bytes) }) : null].filter(Boolean).join(' · ');
    const pick = h('button', { class: 'pick', type: 'button', 'aria-pressed': on ? 'true' : 'false', 'data-k': 'pick:' + m.id },
      h('span', { class: 'nm' }, h('b', null, short(m.id)), h('span', { class: 'org' }, orgOf(m.id)), h('i', { class: 'radio', 'aria-hidden': 'true' })),
      b.length ? h('span', { class: 'bds' }, b) : null,
      spec ? h('span', { class: 'spec' }, spec) : null,
      f ? memBar(m, f) : null,
      m.note ? h('span', { class: 'note' }, m.note[LA.lang === 'en' ? 'en' : 'pt']) : null);
    pick.addEventListener('click', () => { if (S.sel !== m.id || !findModel(S.sel)) select(m.id); });
    const rm = m.custom ? h('button', { class: 'rm', type: 'button', title: T('rm'), 'aria-label': T('rm') + ': ' + m.id, 'data-k': 'rm:' + m.id }, '✕') : null;
    if (rm) rm.addEventListener('click', () => removeCustom(m.id));
    return h('div', { class: 'la-mlx-card' + (on ? ' on' : '') + (S.flash === m.id ? ' flash' : ''), 'data-fit': f ? f.fit : 'na', 'data-id': m.id }, pick, rm);
  }
  function memBar(m, f) {
    const scale = Math.max(f.budget, f.need) || 1;
    const w = (x) => Math.max(0, Math.min(100, (100 * x) / scale)).toFixed(2) + '%';
    const over = f.need > f.budget;
    const cap = over ? T('memOver', { need: gb(f.need), budget: gb(f.budget) }) : T('mem', { need: gb(f.need), budget: gb(f.budget) });
    return h('span', { class: 'la-mlx-mem', 'data-fit': f.fit },
      h('span', { class: 'track', role: 'img', 'aria-label': cap },
        h('span', { class: 'fill' },
          h('i', { class: 'w', style: { width: w(Math.min(m.bytes, scale)) } }),
          h('i', { class: 'o', style: { width: w(f.need - m.bytes) } })),
        over ? h('b', { class: 'lim', style: { left: w(f.budget) } }) : null),
      h('small', null, cap));
  }

  /* ③ Baixar e rodar */
  function renderRun() {
    const m = selModel();
    const ram = ramBytes();
    const loc = localOf(m.id);
    const hfTok = !!(S.st && S.st.hf && S.st.hf.token);
    part(M.runSum, JSON.stringify([m.id, m.bytes, ram, loc && [loc.complete, Math.round(loc.bytes / 1e7)], supportedOf(m), m.gated && [S.helper, hfTok]]), () => summary(m, ram, hfTok));
    const pOk = portOk();
    M.portIn.classList.toggle('bad', !pOk);
    M.portIn.setAttribute('aria-invalid', pOk ? 'false' : 'true');
    part(M.portMsg, String(pOk), () => (pOk ? [] : [h('p', { class: 'la-mlx-bad' }, T('portBad'))]));
    if (S.helper === null) {
      part(M.runMain, 'detecting', () => spinLine(T('detecting')));
      part(M.runSec, 'x', () => []);
      M.log.hidden = true;
      return;
    }
    if (!S.helper) {
      const sig = JSON.stringify(['manual', S.helper, m.id, S.port, S.adv, S.manual, pOk]);
      part(M.runMain, sig, () => manualRun(m, pOk));
      part(M.runSec, 'x', () => []);
      M.log.hidden = true;
      return;
    }
    M.log.hidden = false;
    const ph = phase();
    const st = S.st || {};
    const srv = st.server || {}, dl = st.download || {}, ins = (st.mlx && st.mlx.install) || {};
    const A = autoGet();
    const extra = ph === 'ready' ? [srv.load_s, srv.tok_s, srv.port] : ph === 'srverror' ? [srv.error, srv.error_code, (srv.log_tail || []).slice(-6)] :
      ph === 'dlerror' ? [dl.error, dl.error_code] : ph === 'resume' ? [loc && Math.floor(loc.bytes / 1e7)] : ph === 'otherdl' ? [dl.id] :
        ph === 'switch' ? [srv.id, srv.port] : ph === 'install' ? [ins.state, ins.error, ins.error_code, (st.mlx || {}).installers] : ph === 'starting' ? [srv.started] :
          ph === 'downloading' ? [A && A.id === m.id && A.port] : [];
    const sig = JSON.stringify([ph, m.id, m.bytes, S.port, pOk, S.busyOp, S.err, S.portBusy, extra, copilotOn(srv.port), LA.lang]);
    part(M.runMain, sig, () => runMain(ph, m, pOk));
    const canDel = loc && loc.bytes > 0 && !['downloading', 'starting', 'ready', 'installing', 'wait', 'lost', 'checking'].includes(ph) && !(dl.state === 'running' && dl.id === m.id) && !(srv.id === m.id && (srv.state === 'ready' || srv.state === 'starting'));
    part(M.runSec, JSON.stringify([canDel, loc && loc.bytes, m.id, S.busyOp]), () => (canDel ? [button(T('c.del', { s: gb(loc.bytes) }), () => del(m, loc.bytes), { k: 'del', cls: 'danger', disabled: !!S.busyOp })] : []));
    // o chip mostra o tipo de log no idioma da página; a API continua recebendo server|download|install
    const kinds = LA.t('mlx.logKinds') || {};
    const kind = logKind(ph);
    const kt = kinds[kind] || kind;
    if (M.logKindEl.textContent !== kt) M.logKindEl.textContent = kt;
  }
  function summary(m, ram, hfTok) {
    const f = m.bytes ? LA.mlxFit(m.bytes, ram) : null;
    const out = [h('div', { class: 'la-mlx-sumh' }, h('span', { class: 'lbl' }, T('sel')),
      h('b', null, short(m.id)), h('span', { class: 'org' }, orgOf(m.id)))];
    const spec = [m.params ? T('spec.params', { p: m.params }) : null, m.bits, m.bytes ? T(m.localOnly ? 'spec.disk' : 'spec.dl', { s: gb(m.bytes) }) : null].filter(Boolean).join(' · ');
    if (spec) out.push(h('p', { class: 'la-mlx-spec' }, spec));
    if (f && f.fit === 'tight') out.push(note('warn', T('fitTight', { need: gb(f.need), budget: gb(f.budget) })));
    if (f && f.fit === 'no') out.push(note('err', T('fitNo', { need: gb(f.need), budget: gb(f.budget) })));
    if (supportedOf(m) === false) out.push(note('warn', T('tip.unsup', { t: m.type || '?' })));
    // modelo restrito sem token HF: o download falharia com "gated" — avisa antes
    if (m.gated && !(S.helper && hfTok)) out.push(note('warn', T('add.gated')));
    return out;
  }
  function portBusyNote() {
    const pb = S.portBusy;
    if (!pb) return null;
    const busyPort = pb.port || S.port;
    const alt = busyPort + 1;
    const useThat = () => {
      S.portBusy = null;
      const label = (pb.models || []).length === 1 ? pb.models[0] : ':' + busyPort;
      setExt({ port: busyPort, id: label, models: (pb.models || []).slice(0, 8) });
      connectCopilot(label, busyPort, true);
    };
    return note('warn', h('p', null, T('busy', { port: busyPort }) + (pb.openai ? ' ' + T('busyOpenai', { models: (pb.models || []).slice(0, 3).map(short).join(', ') || '—' }) : '')),
      h('div', { class: 'la-mlx-acts' },
        button(T('useAlt', { port: alt }), () => { S.port = alt; S.portSet = true; M.portIn.value = alt; save('mlx:port', alt); S.portBusy = null; startServer(selId()); }, { k: 'alt' }),
        pb.openai ? button(T('useThat'), useThat, { k: 'usethat' }) : null));
  }
  function progressBlock() {
    const bar = h('i');
    // texto que muda a cada segundo fica fora da árvore de acessibilidade: o progressbar
    // (aria-valuenow/valuetext) e o anúncio por degraus já informam o leitor de tela
    const pctEl = h('b', { class: 'pct', 'aria-hidden': 'true' }, '0%');
    const track = h('div', { class: 'track', role: 'progressbar', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': '0', 'aria-label': T('p.aria') }, bar);
    const line = h('p', { class: 'line', 'aria-hidden': 'true' });
    const file = h('p', { class: 'file', 'aria-hidden': 'true' });
    M.prog = { bar, pctEl, track, line, file };
    return h('div', { class: 'la-mlx-prog' }, h('div', { class: 'top' }, track, pctEl), line, file);
  }
  function runMain(ph, m, pOk) {
    M.prog = null; M.elapsed = null;
    const st = S.st || {};
    const srv = st.server || {}, dl = st.download || {}, ins = (st.mlx && st.mlx.install) || {};
    const loc = localOf(m.id);
    const out = [];
    const working = !!S.busyOp;
    if (S.err && S.err.op !== 'install') out.push(errNote(S.err));
    if (S.err && S.err.op === 'install' && !(st.mlx && st.mlx.installed)) out.push(installErr());
    out.push(portBusyNote());
    switch (ph) {
      case 'wait': out.push(spinLine(S.helper === null ? T('detecting') : T('connecting'))); break;
      case 'lost': out.push(note('err', T('lost'))); break;
      case 'checking': out.push(spinLine(T('detectingMlx'))); break;
      case 'unsupported': out.push(cta(T('c.unsupported'), null, { disabled: true })); break;
      case 'install':
        if (ins.state === 'error') out.push(errNote(stErr(ins), T('r.installErr')));
        out.push(cta(ins.state === 'error' ? T('c.retryInstall') : T('c.install'), () => install(false), { icon: '⬇' }));
        out.push(h('p', { class: 'la-mlx-muted' }, T('installNote', { m: (st.mlx && st.mlx.installers && st.mlx.installers.uv) ? 'uv' : (st.mlx && st.mlx.installers && st.mlx.installers.brew) ? 'Homebrew' : 'uv/Homebrew' })));
        break;
      case 'installing': out.push(spinLine(T('r.installing', { m: ins.method || 'uv' }))); break;
      case 'download':
        out.push(cta(m.bytes ? T('c.dl', { s: gb(m.bytes) }) : T('c.dlU'), () => download(m), { icon: '⬇', disabled: !pOk }));
        out.push(h('p', { class: 'la-mlx-muted' }, T('dlNote')));
        break;
      case 'resume':
        out.push(cta(m.bytes ? T('c.resume', { p: pct(loc.bytes, m.bytes) }) : T('c.resumeU'), () => download(m), { icon: '⬇', disabled: !pOk }));
        out.push(h('p', { class: 'la-mlx-muted' }, T('p.kept')));
        break;
      case 'dlerror':
        out.push(errNote(stErr(dl), T('err.dl')));
        out.push(cta(T('c.retry'), () => download(m), { icon: '↻', disabled: !pOk }));
        break;
      case 'downloading': {
        const A = autoGet();
        out.push(progressBlock());
        out.push(h('div', { class: 'la-mlx-acts' }, button(T('c.cancel'), cancelDl, { k: 'cancel', disabled: working }),
          h('span', { class: 'la-mlx-muted' }, A && A.id === m.id ? T('p.then', { port: A.port }) : T('p.kept'))));
        break;
      }
      case 'otherdl':
        out.push(note('info', T('otherDl', { name: short(dl.id), p: pct(dl.done_bytes, dl.total_bytes) })));
        out.push(h('div', { class: 'la-mlx-acts' }, button(T('c.selectIt'), () => select(dl.id), { k: 'selit' }), button(T('c.cancel'), cancelDl, { k: 'cancel', disabled: working })));
        break;
      case 'start':
        out.push(cta(T('c.start', { port: S.port }), () => startServer(m.id), { icon: '▶', disabled: !pOk }));
        out.push(h('p', { class: 'la-mlx-muted' }, T('startNote')));
        break;
      case 'switch':
        out.push(note('info', T('otherRun', { name: short(srv.id), port: srv.port })));
        out.push(cta(T('c.switch'), () => startServer(m.id), { icon: '⇄', disabled: !pOk }));
        break;
      case 'starting': {
        M.elapsed = h('span', { class: 'el mono', 'aria-hidden': 'true' });
        out.push(h('div', { class: 'la-mlx-loading' }, h('i', { class: 'sp', 'aria-hidden': 'true' }), h('div', { class: 'tx' }, h('b', null, T('loading')), h('p', null, T('loadingSub'))), M.elapsed));
        out.push(h('div', { class: 'la-mlx-acts' }, button(T('c.stop'), stopServer, { k: 'stop', disabled: working })));
        break;
      }
      case 'ready': {
        const bits = [T('readyLoaded', { s: srv.load_s != null ? num(srv.load_s, 0) : '?' })];
        if (srv.tok_s) bits.push(T('readyTok', { t: num(srv.tok_s, 0) }));
        out.push(h('div', { class: 'la-mlx-ready' }, h('i', { class: 'ok', 'aria-hidden': 'true' }, '✓'), h('div', null, h('b', null, T('ready')), h('p', { class: 'mono' }, bits.join(' · ')))));
        out.push(cta(T('c.open'), openCopilot, { icon: '✦' }));
        out.push(h('div', { class: 'la-mlx-acts' }, button(T('c.stopSrv'), stopServer, { k: 'stop', disabled: working })));
        break;
      }
      case 'srverror': {
        const tail = (srv.log_tail || []).slice(-6).join('\n');
        out.push(errNote(stErr(srv), T('err.srv'), tail ? h('pre', { class: 'la-mlx-tail' }, tail) : null));
        out.push(cta(T('c.retry'), () => startServer(m.id), { icon: '↻', disabled: !pOk }));
        break;
      }
      default: break;
    }
    return out;
  }
  function manualRun(m, pOk) {
    const port = pOk ? S.port : 8080;
    // um comando por linha (com "\"): o id do modelo e as flags ficam legíveis sem rolar de lado
    const out = [h('p', { class: 'la-mlx-muted' }, T('m.lead')),
      h('ol', { class: 'la-mlx-steps' },
        h('li', null, h('span', null, T('m.step1')), codeBox('uv tool install --python 3.12 mlx-lm\n# ' + T('m.orBrew'), 'inst')),
        h('li', null, h('span', null, T('m.step2')), codeBox('mlx_lm.server \\\n  --model ' + m.id + ' \\\n  --port ' + port + ' \\\n  --max-tokens ' + S.adv.maxTokens + ' --temp ' + S.adv.temp, 'srv')),
        h('li', null, h('span', null, T('m.step3'))))];
    const mm = S.manual;
    out.push(cta(mm.state === 'busy' ? T('m.checking') : T('m.check'), manualCheck, { icon: '⚡', disabled: !pOk || mm.state === 'busy' }));
    if (mm.state === 'ok') out.push(note('ok', T('m.ok', { port: mm.port, n: mm.models.length })));
    if (mm.state === 'bad') out.push(note('err', T('m.fail', { port: mm.port }), location.protocol === 'https:' ? h('p', null, T('m.https')) : null));
    return out;
  }
  function logKind(ph) {
    const st = S.st || {};
    const ins = (st.mlx && st.mlx.install) || {};
    if (ph === 'installing' || ph === 'install' || (ins.state === 'error' && !(st.mlx && st.mlx.installed))) return 'install';
    if (ph === 'downloading' || ph === 'dlerror' || ph === 'resume' || ph === 'download') return 'download';
    return 'server';
  }
  function openCopilot() { closeModal(); if (LA.copilot && LA.copilot.open) LA.copilot.open(); }

  /* ④ Copilot */
  function renderCp() {
    const st = S.st || {};
    const srv = st.server || {};
    const lost = !!S.helper && S.lost >= 3;
    const ex = extLive();
    let live = null;
    if (S.helper && !lost && srv.state === 'ready') live = { port: srv.port, id: srv.id, tok: srv.tok_s };
    else if (ex) live = { port: ex.port, id: ex.id, ext: true };
    else if (!S.helper && S.manual.state === 'ok') live = { port: S.manual.port, id: selId() };
    const on = !!live && copilotOn(live.port);
    part(M.cpBody, JSON.stringify([live, on, S.helper, lost, LA.lang]), () => {
      // helper fora do ar: o servidor gerenciado normalmente parou junto — não dá para afirmar nada
      if (lost) return [h('p', { class: 'la-mlx-muted' }, T('cp.unknown'))];
      if (!live) return [h('p', { class: 'la-mlx-muted' }, S.helper ? T('cp.wait') : T('cp.waitManual'))];
      const base = 'http://127.0.0.1:' + live.port + '/v1';
      return [
        h('div', { class: 'la-mlx-kv' },
          h('span', null, T('cp.endpoint')), h('code', null, base),
          h('span', null, T('cp.model')), h('code', null, live.id || '—')),
        live.ext ? h('p', { class: 'la-mlx-muted la-mlx-extnote' }, T('cp.ext', { port: live.port })) : null,
        on ? h('p', { class: 'la-mlx-okline' }, '✓ ' + T('cp.on'))
          : note('info', h('p', null, T('cp.elsewhere')), h('div', { class: 'la-mlx-acts' }, button(T('cp.use'), () => connectCopilot(live.id, live.port, true), { k: 'cpuse', primary: true }))),
        h('div', { class: 'la-mlx-acts' },
          button('✦ ' + T('c.open'), openCopilot, { k: 'cpopen', primary: on }),
          button(T('cp.test'), () => { if (!copilotOn(live.port)) connectCopilot(live.id, live.port, false); closeModal(); if (LA.copilot && LA.copilot.ask) LA.copilot.ask(T('testQ')); }, { k: 'cptest' })),
      ];
    });
    const ft = S.helper ? T('foot') : T('footManual');
    if (M.foot.textContent !== ft) M.foot.textContent = ft;
  }

  function renderDock(s) {
    const ds = DOT[s.k] || 'off';
    const pt = pillText(s);
    if (M.dockSt.textContent !== pt) M.dockSt.textContent = pt;
    if (M.dockSt.getAttribute('data-state') !== ds) M.dockSt.setAttribute('data-state', ds);
    const t = short(s.id || selId()); // o modelo a que o status se refere (baixando/rodando), senão o escolhido
    if (M.dockTxt.textContent !== t) M.dockTxt.textContent = t;
  }
  /** anúncio para leitor de tela: só mudanças de fase e degraus de 25% do download */
  function announce(s) {
    const key = s.k + ':' + (s.id || '') + (s.k === 'dl' ? ':' + Math.floor((s.p || 0) / 25) : '');
    if (M.srKey === key) return;
    const first = !M.srKey || M.srKey.startsWith('wait:');
    M.srKey = key;
    if (first) return; // ao abrir (ou ao achar o helper), o diálogo já é lido inteiro
    M.sr.textContent = (s.id ? short(s.id) + ' — ' : '') + pillText(s);
  }

  /* ---------------------------------------- atualizações em tempo real */
  function updateLive() {
    if (!M || !S.open) return;
    const st = S.st || {};
    const dl = st.download || {}, srv = st.server || {};
    if (M.prog && dl.state === 'running') {
      const p = pct(dl.done_bytes, dl.total_bytes);
      M.prog.bar.style.width = (dl.total_bytes ? (100 * dl.done_bytes) / dl.total_bytes : 0).toFixed(2) + '%';
      M.prog.track.setAttribute('aria-valuenow', String(p));
      M.prog.pctEl.textContent = p + '%';
      const parts = [T('p.of', { a: gb(dl.done_bytes), b: gb(dl.total_bytes) })];
      if (dl.speed_bps) parts.push(speedTxt(dl.speed_bps));
      if (dl.eta_s != null && dl.speed_bps) parts.push(T('p.eta', { t: dur(dl.eta_s) }));
      M.prog.line.textContent = parts.join(' · ');
      M.prog.track.setAttribute('aria-valuetext', p + '% · ' + parts[0]);
      // até 3 arquivos baixam ao mesmo tempo: contagem de concluídos + o arquivo em andamento
      // (sem "arquivo 8 de 9", que não corresponderia ao nome mostrado)
      const fl = [];
      if (dl.files_total) fl.push(T('p.files', { a: dl.files_done || 0, b: dl.files_total }));
      if (dl.current_file) fl.push(T('p.now', { f: dl.current_file }));
      M.prog.file.textContent = fl.join(' · ');
    }
    if (M.elapsed && srv.state === 'starting') M.elapsed.textContent = dur(Date.now() / 1000 - (srv.started || Date.now() / 1000));
    if (dl.state === 'running') M.list.querySelectorAll('.dlp').forEach((b) => { b.textContent = T('bd.dling', { p: pct(dl.done_bytes, dl.total_bytes) }); });
    announce(status());
  }
  let tickT = 0;
  function ticker() {
    const want = S.open && S.st && S.st.server && S.st.server.state === 'starting';
    if (want && !tickT) tickT = setInterval(updateLive, 1000);
    else if (!want && tickT) { clearInterval(tickT); tickT = 0; }
  }
  /* cauda do log de instalação no ① */
  let tailT = 0;
  function tailKick() { if (S.open && M && M.tail && S.helper && !tailT) tailPoll(); }
  async function tailPoll() {
    tailT = 0;
    if (!S.open || !M.tail || !S.helper) return;
    tailT = -1; // em andamento
    try {
      const j = await api('/api/mlx/logs?kind=install&n=8');
      S.tailText = (j.lines || []).join('\n') || '…';
      if (M.tail) { M.tail.textContent = S.tailText; M.tail.scrollTop = M.tail.scrollHeight; }
    } catch (e) { /* tenta de novo */ }
    tailT = S.open && M.tail ? setTimeout(tailPoll, 1200) : 0;
  }
  /* log completo (details "Mostrar log") */
  let logT = 0;
  async function logPoll() {
    clearTimeout(logT);
    if (!S.open || !M.log.open || !S.helper) return;
    const kind = logKind(phase());
    try {
      const j = await api('/api/mlx/logs?kind=' + kind + '&n=200');
      const pre = M.logPre;
      const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 12;
      const txt = (j.lines || []).join('\n') || T('logEmpty');
      if (pre.textContent !== txt) { pre.textContent = txt; if (atBottom) pre.scrollTop = pre.scrollHeight; }
    } catch (e) { /* tenta de novo */ }
    logT = setTimeout(logPoll, busy() ? 1000 : 2500);
  }

  /* ===================================================== PIPELINE (canvas) */
  const PIPE = { raf: 0, pt: 0, t: 0, last: 0, w: 0, h: 0, dpr: 1, parts: [], acc: [0, 0, 0, 0], lit: [0.3, 0.3, 0.3, 0.3, 0.3], disk: 0, mem: 0, seed: 1 };
  function pipeFit() {
    if (!M) return;
    // tamanho de layout (clientWidth), não o retângulo transformado: o modal abre com animação de escala
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    PIPE.w = Math.max(1, M.cv.clientWidth); PIPE.h = Math.max(1, M.cv.clientHeight); PIPE.dpr = dpr;
    const W = Math.round(PIPE.w * dpr), H = Math.round(PIPE.h * dpr);
    if (M.cv.width !== W || M.cv.height !== H) { M.cv.width = W; M.cv.height = H; }
  }
  function pipeStart() {
    if (!M || !LA.viz) return;
    if (reduced()) { pipeDraw(true); return; }
    if (!PIPE.raf) { PIPE.last = performance.now(); PIPE.raf = requestAnimationFrame(pipeFrame); }
  }
  function pipeStop() { if (PIPE.raf) cancelAnimationFrame(PIPE.raf); PIPE.raf = 0; clearTimeout(PIPE.pt); }
  /** redesenha já (movimento reduzido / pausado / mudança de estado) */
  function pipeKick() { if (S.open && M && LA.viz && (reduced() || LA.paused || !PIPE.raf)) pipeDraw(true); }
  function pipeFrame(now) {
    PIPE.raf = 0;
    if (!S.open) return;
    const dt = Math.min(0.05, Math.max(0, (now - PIPE.last) / 1000));
    PIPE.last = now;
    if (LA.paused || document.hidden) {
      pipeDraw(true);
      clearTimeout(PIPE.pt);
      PIPE.pt = setTimeout(() => { if (S.open && !PIPE.raf) { PIPE.last = performance.now(); PIPE.raf = requestAnimationFrame(pipeFrame); } }, 400);
      return;
    }
    PIPE.t += dt;
    try { pipeDraw(false, dt); } catch (e) { pipeStop(); return; }
    PIPE.raf = requestAnimationFrame(pipeFrame);
  }
  // erros de download que são do Hugging Face (rede, gated, não achado) → nó HF fica vermelho; o resto → disco
  const HF_ERRS = ['network', 'gated', 'not_found', 'hf_unreachable', 'bad_metadata', 'no_files'];
  function pipeModel() {
    const sel = selModel();
    const ram = ramBytes();
    const f = sel.bytes ? LA.mlxFit(sel.bytes, ram) : LA.mlxFit(0, ram);
    const st = S.st || {};
    const dl = st.download || {}, srv = st.server || {}, ins = (st.mlx && st.mlx.install) || {};
    const loc = localOf(sel.id);
    const capT = LA.t('mlx.cap') || {};
    const o = { mode: 'idle', pct: 0, speed: 0, tok: 0, err: -1, disk: 0, mem: 0, caps: [], port: S.port };
    o.caps = [sel.bytes ? gb(sel.bytes) : '—', capT.cache, T('cap.gpu', { v: gb(f.budget) }), ':' + S.port, capT.off];
    if (loc && loc.complete) { o.disk = 1; o.caps[1] = '✓ ' + gb(loc.bytes); }
    else if (partialOf(sel, loc) && sel.bytes) { o.disk = loc.bytes / sel.bytes; o.caps[1] = pct(loc.bytes, sel.bytes) + '%'; }
    const memFrac = Math.min(1, f.need / f.budget);
    if (S.helper && S.st && S.lost < 3) {
      // a porta do servidor só vale enquanto ele existe; parado, mostra a que o CTA vai usar
      if (srv.port && (srv.state === 'starting' || srv.state === 'ready' || srv.state === 'error')) o.caps[3] = ':' + srv.port;
      if (dl.state === 'running') {
        o.mode = 'download'; o.pct = dl.total_bytes ? dl.done_bytes / dl.total_bytes : 0; o.speed = dl.speed_bps || 0;
        o.caps[0] = dl.speed_bps ? speedTxt(dl.speed_bps) : capT.connecting;
        o.disk = o.pct; o.caps[1] = pct(dl.done_bytes, dl.total_bytes) + '%';
        if (dl.id !== sel.id) o.caps[0] = short(dl.id).slice(0, 14);
      } else if (srv.state === 'starting') {
        o.mode = 'starting'; o.mem = memFrac; o.caps[2] = capT.loading; o.caps[4] = capT.wait;
        o.elapsed = Date.now() / 1000 - (srv.started || Date.now() / 1000);
      } else if (srv.state === 'ready') {
        const rm = findModel(srv.id) || sel;
        const rf = rm.bytes ? LA.mlxFit(rm.bytes, ram) : f;
        o.mode = 'ready'; o.tok = srv.tok_s || 0; o.mem = Math.min(1, rf.need / rf.budget); o.disk = 1;
        o.caps[2] = '~' + gb(rf.need); o.caps[4] = srv.tok_s ? tokTxt(srv.tok_s) : capT.on;
        if (rm.bytes) { o.caps[0] = gb(rm.bytes); o.caps[1] = '✓ ' + gb(rm.bytes); }
      } else if (ins.state === 'running') { o.mode = 'install'; o.caps[3] = capT.install; }
      else if (extLive()) { o.mode = 'ready'; o.mem = memFrac; o.caps[3] = ':' + S.ext.port; o.caps[4] = capT.on; }
      if (srv.state === 'error') { o.mode = 'error'; o.err = 3; o.caps[3] = capT.err; }
      else if (dl.state === 'error' && dl.id === sel.id && o.mode === 'idle') { o.mode = 'error'; o.err = HF_ERRS.includes(dl.error_code || dl.error) ? 0 : 1; o.caps[o.err] = capT.err; }
      else if (ins.state === 'error' && o.mode === 'idle') { o.mode = 'error'; o.err = 3; o.caps[3] = capT.err; }
    } else if (S.helper && S.lost >= 3) { o.mode = 'error'; o.err = 3; o.caps[3] = capT.err; }
    else if (!S.helper) {
      if (S.manual.state === 'ok') { o.mode = 'ready'; o.mem = memFrac; o.caps[3] = ':' + S.manual.port; o.caps[4] = capT.on; o.disk = 1; }
      else if (S.manual.state === 'bad') { o.mode = 'error'; o.err = 3; o.caps[3] = ':' + S.manual.port + ' ✕'; }
    }
    return o;
  }
  function pipeDraw(still, dt) {
    if (!M || !LA.viz) return;
    if (!PIPE.w || PIPE.w < 8) pipeFit();
    const V = LA.viz, C = V.C();
    const g = M.cv.getContext('2d');
    const w = PIPE.w, hh = PIPE.h;
    g.setTransform(PIPE.dpr, 0, 0, PIPE.dpr, 0, 0);
    g.clearRect(0, 0, w, hh);
    const o = pipeModel();
    const t = PIPE.t;
    const mob = w < 560;
    const cols = [C.d, C.b, C.a, C.c, C.e];
    const n = 5, padX = mob ? 4 : 16;
    const colW = (w - 2 * padX) / n;
    // no celular os nós são estreitos para sobrar uma faixa ≥ 20 px entre eles (setas e partículas)
    const bw = mob ? V.clamp(colW - 24, 30, 40) : Math.max(34, Math.min(84, colW - 34));
    const isc = mob ? 0.72 : 1; // escala dos ícones
    const low = mob || hh < 112; // canvas baixo (celular, ou tela deitada): métricas verticais compactas
    const bh = low ? 40 : 52;
    const by = Math.max(6, Math.round((hh - (bh + (low ? 31 : 37))) / 2));
    const cx = (i) => padX + colW * (i + 0.5);
    const cy = by + bh / 2;

    // alvos de brilho por nó
    const tgt = { idle: [0.42, 0.42, 0.32, 0.32, 0.32], download: [1, 0.85, 0.3, 0.3, 0.3], install: [0.35, 0.4, 0.35, 0.9, 0.3], starting: [0.5, 0.9, 1, 0.9, 0.45], ready: [0.75, 0.85, 1, 1, 1], error: [0.4, 0.4, 0.4, 0.4, 0.35] }[o.mode];
    const k = still ? 1 : Math.min(1, (dt || 0.016) * 4);
    for (let i = 0; i < n; i++) PIPE.lit[i] += (tgt[i] - PIPE.lit[i]) * k;
    PIPE.disk += (o.disk - PIPE.disk) * (still ? 1 : Math.min(1, (dt || 0.016) * 3));
    let memT = o.mem;
    if (o.mode === 'starting') memT = o.mem * (1 - Math.exp(-Math.max(0, o.elapsed || 0) / 14));
    PIPE.mem += (memT - PIPE.mem) * (still ? 1 : Math.min(1, (dt || 0.016) * 2.5));

    // conectores
    for (let i = 0; i < n - 1; i++) {
      const x1 = cx(i) + bw / 2 + 5, x2 = cx(i + 1) - bw / 2 - 5;
      if (x2 - x1 < 4) continue;
      const act = (o.mode === 'download' && i === 0) || (o.mode === 'starting' && (i === 1 || i === 2)) || (o.mode === 'ready' && (i === 2 || i === 3));
      g.save();
      g.setLineDash([3, 5]);
      g.lineDashOffset = still ? 0 : -t * (act ? 26 : 7);
      g.strokeStyle = V.alpha(act ? cols[i + 1] : C.muted, act ? 0.75 : 0.4);
      g.lineWidth = act ? 1.8 : 1.2;
      g.beginPath(); g.moveTo(x1, cy); g.lineTo(x2, cy); g.stroke();
      g.restore();
      // seta
      g.fillStyle = V.alpha(act ? cols[i + 1] : C.muted, act ? 0.85 : 0.45);
      g.beginPath(); g.moveTo(x2 + 1, cy); g.lineTo(x2 - 4, cy - 3.5); g.lineTo(x2 - 4, cy + 3.5); g.closePath(); g.fill();
    }

    // partículas
    const rates = [0, 0, 0, 0];
    if (o.mode === 'download') rates[0] = V.clamp((o.speed || 4e5) / 2.5e5, 2, 14);
    if (o.mode === 'starting') { rates[1] = 7; rates[2] = 2; }
    if (o.mode === 'ready') { rates[2] = 1.6; rates[3] = V.clamp((o.tok || 20) / 3, 3, 16); }
    if (mob) for (let i = 0; i < 4; i++) rates[i] *= 0.4; // faixa curta no celular: menos partículas, sem "bolo"
    const segX = (i) => [cx(i) + bw / 2 + 3, cx(i + 1) - bw / 2 - 3];
    if (!still) {
      const rnd = Math.random;
      for (let i = 0; i < 4; i++) {
        PIPE.acc[i] += rates[i] * (dt || 0.016);
        while (PIPE.acc[i] >= 1) { PIPE.acc[i] -= 1; PIPE.parts.push({ s: i, u: 0, v: 0.9 + rnd() * 0.5, j: (rnd() - 0.5) * (mob ? 8 : 12), a: rnd() * 6.28 }); }
      }
      PIPE.parts.forEach((p) => { p.u += (dt || 0.016) * p.v; });
      PIPE.parts = PIPE.parts.filter((p) => p.u < 1 && rates[p.s] > 0).slice(-160);
    } else if (!PIPE.parts.length || reduced()) {
      // quadro estático: alguns pacotes parados mostram o sentido do fluxo
      PIPE.parts = [];
      for (let i = 0; i < 4; i++) if (rates[i]) for (let q = 0; q < 3; q++) PIPE.parts.push({ s: i, u: 0.2 + q * 0.3, v: 0, j: (q - 1) * 4, a: q });
    }
    PIPE.parts.forEach((p) => {
      const [x1, x2] = segX(p.s);
      if (x2 - x1 < 4) return;
      const x = V.lerp(x1, x2, p.u);
      const y = cy + p.j * Math.sin(p.u * Math.PI) + (p.s === 3 ? Math.sin(p.u * 6.28 + p.a) * 2 : 0);
      const col = cols[p.s === 0 ? 0 : p.s === 3 ? 4 : p.s + 1];
      const fade = Math.min(1, p.u * 6, (1 - p.u) * 6);
      if (p.s === 3) {
        g.globalAlpha = fade;
        const pw = mob ? 5 : 8, ph2 = mob ? 4 : 6;
        V.rr(g, x - pw / 2, y - ph2 / 2, pw, ph2, 2); g.fillStyle = V.alpha(col, 0.9); g.fill();
        g.globalAlpha = 1;
      } else {
        V.glow(g, x, y, mob ? 4.5 : 8, col, 0.55 * fade);
        V.dot(g, x, y, mob ? 1.5 : 2, V.alpha(col, fade));
      }
    });

    // nós
    const labels = LA.t(mob ? 'mlx.pipeShort' : 'mlx.pipe');
    for (let i = 0; i < n; i++) {
      const x = cx(i), L = PIPE.lit[i], col = cols[i];
      const isErr = o.err === i;
      const ecol = isErr ? C.red : col;
      if (L > 0.6 && !isErr) V.glow(g, x, cy, bw * 0.75, col, (L - 0.6) * 0.45);
      if (isErr) {
        const pulse = still ? 0.6 : 0.5 + 0.5 * Math.sin(t * 5);
        V.glow(g, x, cy, bw * 0.8, C.red, 0.25 + 0.35 * pulse);
      }
      V.rr(g, x - bw / 2, by, bw, bh, mob ? 9 : 12);
      g.fillStyle = C.panel; g.fill();
      g.fillStyle = V.alpha(ecol, 0.06 + 0.1 * L); g.fill();
      g.strokeStyle = V.alpha(ecol, isErr ? 0.95 : 0.25 + 0.6 * L);
      g.lineWidth = isErr ? 1.8 : 1.2 + 0.6 * L; g.stroke();
      g.save();
      g.translate(x, cy);
      g.globalAlpha = isErr ? 1 : 0.45 + 0.55 * L;
      drawIcon(g, V, C, i, ecol, isc, o, t, still);
      g.restore();
      // anel de "trabalhando" (start/instalação) no servidor
      if (i === 3 && (o.mode === 'starting' || o.mode === 'install') && !still) {
        const ph = (t * 0.9) % 1;
        V.rr(g, x - bw / 2 - 6 * ph, by - 6 * ph, bw + 12 * ph, bh + 12 * ph, 14);
        g.strokeStyle = V.alpha(o.mode === 'install' ? C.d : col, 0.6 * (1 - ph)); g.lineWidth = 1.5; g.stroke();
      }
      // rótulos
      const ly = by + bh + (low ? 14 : 17);
      V.text(g, labels[i], x, ly, { size: low ? 10.5 : 11.5, weight: 600, align: 'center', color: L > 0.6 ? C.text : C.text2, maxW: colW - 4 });
      V.text(g, o.caps[i] || '', x, ly + (low ? 14 : 16), { size: low ? 9.5 : 10.5, font: 'mono', align: 'center', color: isErr ? C.red : L > 0.6 ? cols[i] : C.text3, maxW: colW - 4 });
    }
    // brilho em varredura (idle): uma luz percorre o pipeline devagar
    if (o.mode === 'idle' && !still) {
      const u = (t / 5.5) % 1;
      const x = padX + u * (w - 2 * padX);
      V.glow(g, x, cy, mob ? 22 : 30, C.a, 0.12);
    }
  }
  function drawIcon(g, V, C, i, col, s, o, t, still) {
    g.lineWidth = 1.6; g.strokeStyle = col; g.fillStyle = col; g.lineCap = 'round'; g.lineJoin = 'round';
    if (i === 0) {
      // Hugging Face: carinha sorridente
      const r = 11 * s;
      g.beginPath(); g.arc(0, 0, r, 0, Math.PI * 2); g.stroke();
      V.dot(g, -4 * s, -2.5 * s, 1.5 * s, col); V.dot(g, 4 * s, -2.5 * s, 1.5 * s, col);
      g.beginPath(); g.arc(0, 1.5 * s, 5 * s, 0.15 * Math.PI, 0.85 * Math.PI); g.stroke();
      if (o.mode === 'download' && !still) { const b = Math.sin(t * 8) * 1.2; g.beginPath(); g.arc(-r * 0.95, 3 * s + b, 2.4 * s, 0, 6.28); g.arc(r * 0.95, 3 * s - b, 2.4 * s, 0, 6.28); g.fill(); }
    } else if (i === 1) {
      // disco: cilindro que enche
      const rw = 11 * s, rh = 3.6 * s, top = -9 * s, bot = 9 * s;
      const fillY = bot - (bot - top) * V.clamp(PIPE.disk, 0, 1);
      g.save();
      g.beginPath(); g.moveTo(-rw, top); g.lineTo(-rw, bot); g.ellipse(0, bot, rw, rh, 0, Math.PI, 0, true); g.lineTo(rw, top); g.closePath(); g.clip();
      g.fillStyle = V.alpha(col, 0.55); g.fillRect(-rw - 2, fillY, rw * 2 + 4, bot - fillY + rh + 2);
      g.restore();
      g.beginPath(); g.ellipse(0, top, rw, rh, 0, 0, Math.PI * 2); g.stroke();
      g.beginPath(); g.moveTo(-rw, top); g.lineTo(-rw, bot); g.ellipse(0, bot, rw, rh, 0, Math.PI, 0, true); g.lineTo(rw, top); g.stroke();
    } else if (i === 2) {
      // memória unificada: chip com grade de células
      const cw = 24 * s, chh = 18 * s;
      V.rr(g, -cw / 2, -chh / 2, cw, chh, 3 * s); g.stroke();
      for (let p = -1; p <= 1; p++) { g.beginPath(); g.moveTo(-cw / 2 - 3 * s, p * 5 * s); g.lineTo(-cw / 2, p * 5 * s); g.moveTo(cw / 2, p * 5 * s); g.lineTo(cw / 2 + 3 * s, p * 5 * s); g.stroke(); }
      const nx = 5, ny = 3, gx = (cw - 6 * s) / nx, gy = (chh - 6 * s) / ny;
      const lit = Math.round(V.clamp(PIPE.mem, 0, 1) * nx * ny);
      for (let yy = 0; yy < ny; yy++) for (let xx = 0; xx < nx; xx++) {
        const idx = yy * nx + xx;
        const on = idx < lit;
        g.fillStyle = on ? V.alpha(col, 0.95) : V.alpha(C.muted, 0.28);
        g.fillRect(-cw / 2 + 3 * s + xx * gx + 0.6, -chh / 2 + 3 * s + yy * gy + 0.6, gx - 1.2, gy - 1.2);
      }
    } else if (i === 3) {
      // servidor: duas unidades de rack com LEDs
      const rw = 24 * s, rh = 8 * s;
      for (let u = 0; u < 2; u++) {
        const y = -rh - 1.5 * s + u * (rh + 3 * s);
        V.rr(g, -rw / 2, y, rw, rh, 2.2 * s); g.stroke();
        const ready = o.mode === 'ready';
        const blink = still ? 1 : (Math.sin(t * (ready ? 9 : 3) + u * 2) > 0 ? 1 : 0.25);
        V.dot(g, rw / 2 - 4 * s, y + rh / 2, 1.6 * s, ready || o.mode === 'starting' ? V.alpha(col, blink) : V.alpha(C.muted, 0.6));
        g.beginPath(); g.moveTo(-rw / 2 + 3.5 * s, y + rh / 2); g.lineTo(-rw / 2 + 11 * s, y + rh / 2); g.stroke();
      }
    } else {
      // copilot: brilho de quatro pontas
      const r = (o.mode === 'ready' && !still ? 10 + Math.sin(t * 3) * 1.2 : 10) * s;
      g.beginPath();
      g.moveTo(0, -r); g.quadraticCurveTo(r * 0.12, -r * 0.12, r, 0); g.quadraticCurveTo(r * 0.12, r * 0.12, 0, r);
      g.quadraticCurveTo(-r * 0.12, r * 0.12, -r, 0); g.quadraticCurveTo(-r * 0.12, -r * 0.12, 0, -r); g.closePath();
      g.globalAlpha *= 0.9; g.fill();
      V.dot(g, r * 0.95, -r * 0.8, 1.8 * s, col);
    }
  }

  /* ============================================================ INIT */
  LA.mlx = {
    topButton,
    open,
    close: closeModal,
    toggle: () => (S.open ? closeModal() : open()),
    get isOpen() { return S.open; },
    refresh: () => { if (S.helper) schedule(0); else update(); },
  };
  document.addEventListener('la:ready', detect);
  document.addEventListener('visibilitychange', () => { if (!document.hidden && S.helper) schedule(300); });
  if (LA.theme && LA.theme.onChange) LA.theme.onChange(() => setTimeout(pipeKick, 0));
})();
