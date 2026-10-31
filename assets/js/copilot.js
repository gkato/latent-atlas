/* ==========================================================================
   LATENT ATLAS — copilot.js
   Tutor flutuante sensível ao contexto da página.
   Fala com qualquer endpoint no contrato OpenAI Chat Completions
   (OpenAI, OpenRouter, vLLM — inclusive a sua própria receita —, LM Studio,
   Ollama) e com modelos MLX locais no Mac via mlx_lm.server.
   ========================================================================== */
(function () {
  'use strict';
  const LA = (window.LA = window.LA || {});
  const h = (...a) => LA.h(...a);

  /* ------------------------------------------------------------ provedores */
  const PV = LA.t('cp.prov');
  const HEAD = LA.lang === 'en' ? 'YOUR-HEAD' : 'SEU-HEAD';
  const PROVIDERS = [
    { id: 'mlx', label: PV.mlx[0], desc: PV.mlx[1], base: 'http://127.0.0.1:8080/v1', needsKey: false, model: '', ctx: 9000, tokenField: 'max_tokens' },
    { id: 'openai', label: PV.openai[0], desc: PV.openai[1], base: 'https://api.openai.com/v1', needsKey: true, model: '', ctx: 24000, tokenField: 'max_completion_tokens' },
    { id: 'openrouter', label: PV.openrouter[0], desc: PV.openrouter[1], base: 'https://openrouter.ai/api/v1', needsKey: true, model: '', ctx: 24000, tokenField: 'max_tokens' },
    { id: 'vllm', label: PV.vllm[0], desc: PV.vllm[1], base: 'http://' + HEAD + ':8888/v1', needsKey: false, model: '', ctx: 24000, tokenField: 'max_tokens' },
    { id: 'lmstudio', label: PV.lmstudio[0], desc: PV.lmstudio[1], base: 'http://localhost:1234/v1', needsKey: false, model: '', ctx: 9000, tokenField: 'max_tokens' },
    { id: 'ollama', label: PV.ollama[0], desc: PV.ollama[1], base: 'http://localhost:11434/v1', needsKey: false, model: '', ctx: 9000, tokenField: 'max_tokens' },
    { id: 'custom', label: PV.custom[0], desc: PV.custom[1], base: '', needsKey: false, model: '', ctx: 16000, tokenField: 'max_tokens' },
  ];
  const DEFAULTS = {
    provider: 'mlx', base: PROVIDERS[0].base, key: '', model: PROVIDERS[0].model,
    temperature: 0.4, maxTokens: 2048, stream: true, showThinking: true, ctxChars: PROVIDERS[0].ctx,
    ctxMode: 'chapter', extraSystem: '', tokenField: 'max_tokens', extraHeaders: '', noThink: true,
  };
  let cfg = Object.assign({}, DEFAULTS, LA.store.get('copilot', {}));
  function saveCfg() { LA.store.set('copilot', cfg); updateModelTag(); }

  /* ----------------------------------------------------------- estado */
  const threads = {}; // chave da página -> mensagens [{role, content, att, reasoning}]
  let attachment = null; // {kind, label, text}
  let aborter = null;
  let els = {};

  const pageKey = () => { const r = LA.route || { type: 'home' }; return r.type === 'chapter' ? 'c:' + r.id : r.type; };
  const thread = () => (threads[pageKey()] = threads[pageKey()] || []);

  /* ============================================================== INIT */
  LA.copilot = {
    init,
    open: () => open(true),
    close: () => open(false),
    toggle: () => open(!document.body.classList.contains('cp-open')),
    ask: (text, att) => { if (att) setAttachment(att); open(true); els.ta.value = text || ''; if (text) send(); },
    askAbout,
    get config() { return Object.assign({}, cfg); },
    configure,
  };

  /** mescla um patch na configuração (usado pelo assistente MLX) e atualiza a UI */
  function configure(patch) {
    Object.assign(cfg, patch || {});
    saveCfg();
    if (els.panel && els.panel.classList.contains('show-settings')) buildSettings();
    updateModelTag();
    return Object.assign({}, cfg);
  }

  const SVG = {
    spark: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2.5c.4 3.9 1.9 6.1 5.8 6.9-3.9.8-5.4 3-5.8 6.9-.4-3.9-1.9-6.1-5.8-6.9 3.9-.8 5.4-3 5.8-6.9Z"/><path d="M19 14.5c.2 1.9.9 2.9 2.8 3.3-1.9.4-2.6 1.4-2.8 3.3-.2-1.9-.9-2.9-2.8-3.3 1.9-.4 2.6-1.4 2.8-3.3Z" opacity=".75"/><path d="M5.5 15c.15 1.4.7 2.2 2.1 2.5-1.4.3-1.95 1.1-2.1 2.5-.15-1.4-.7-2.2-2.1-2.5 1.4-.3 1.95-1.1 2.1-2.5Z" opacity=".6"/></svg>',
    send: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h13M13 6l6 6-6 6"/></svg>',
    stop: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>',
  };

  function init() {
    const root = document.getElementById('la-copilot-root') || document.body;
    const fab = h('button', { class: 'cp-fab', title: LA.t('cp.fab'), 'aria-label': LA.t('cp.open'), html: SVG.spark + '<span class="cp-badge"></span>' });
    fab.addEventListener('click', () => open(true));

    const ctxLabel = h('span', null, '—');
    const orb = h('div', { class: 'cp-orb' });
    const bSettings = h('button', { class: 'cp-hbtn', title: LA.t('cp.settings'), html: '⚙' });
    const bWide = h('button', { class: 'cp-hbtn', title: LA.t('cp.wide'), html: '⇔' });
    const bClear = h('button', { class: 'cp-hbtn', title: LA.t('cp.clear'), html: '⌫' });
    const bClose = h('button', { class: 'cp-hbtn', title: LA.t('cp.close'), html: '✕' });
    const head = h('div', { class: 'cp-head' }, orb, h('div', { class: 'cp-title' }, h('b', null, 'Copilot'), ctxLabel), bSettings, bWide, bClear, bClose);

    const seg = h('div', { class: 'la-seg' });
    [['section', LA.t('cp.ctx.section')], ['chapter', LA.t('cp.ctx.chapter')], ['none', LA.t('cp.ctx.none')]].forEach(([v, l]) => {
      const b = h('button', { class: cfg.ctxMode === v ? 'on' : '', type: 'button' }, l);
      b.addEventListener('click', () => { cfg.ctxMode = v; saveCfg(); Array.from(seg.children).forEach((x) => x.classList.remove('on')); b.classList.add('on'); });
      seg.appendChild(b);
    });
    const modelTag = h('span', { class: 'cp-model', title: LA.t('cp.modelTip') });
    modelTag.addEventListener('click', () => showSettings(true));
    const ctxbar = h('div', { class: 'cp-ctxbar' }, h('span', null, LA.t('cp.ctx')), seg, modelTag);

    const msgs = h('div', { class: 'cp-msgs' });
    const chips = h('div', { class: 'cp-chips' });
    const attach = h('div', { class: 'cp-attach' }, h('span', null, '📎'), h('div', { class: 'tx' }), h('button', { title: LA.t('cp.rmAtt') }, '✕'));
    attach.querySelector('button').addEventListener('click', () => setAttachment(null));
    const ta = h('textarea', { rows: 1, placeholder: LA.t('cp.ph'), title: LA.t('cp.phTip') });
    const sendBtn = h('button', { class: 'cp-send', title: LA.t('cp.send'), html: SVG.send });
    const input = h('div', { class: 'cp-input' }, ta, sendBtn);
    const foot = h('div', { class: 'cp-foot' }, h('span', null, LA.t('cp.foot')), h('span', null, '⌘/Ctrl J'));
    const settings = h('div', { class: 'cp-settings' });
    const panel = h('aside', { class: 'cp-panel', 'aria-label': 'Copilot' }, head, ctxbar, msgs, settings, chips, attach, input, foot);
    const selAsk = h('button', { class: 'cp-selask', html: LA.t('cp.selAsk') });

    root.append(fab, panel);
    document.body.appendChild(selAsk);
    els = { fab, panel, msgs, chips, attach, ta, sendBtn, settings, ctxLabel, orb, modelTag, selAsk };

    bSettings.addEventListener('click', () => showSettings(!panel.classList.contains('show-settings')));
    bWide.addEventListener('click', () => panel.classList.toggle('wide'));
    bClear.addEventListener('click', () => { threads[pageKey()] = []; renderThread(); });
    bClose.addEventListener('click', () => open(false));
    sendBtn.addEventListener('click', () => (aborter ? stop() : send()));
    ta.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (!aborter) send(); } });
    ta.addEventListener('input', autosize);

    document.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'j') { e.preventDefault(); LA.copilot.toggle(); }
      else if (e.key === 'Escape' && document.body.classList.contains('cp-open') && !document.getElementById('la-search').classList.contains('on')) open(false);
    });
    document.addEventListener('la:route', () => { updateCtxLabel(); renderThread(); setAttachment(null); });

    // seleção de texto → botão flutuante
    document.addEventListener('mouseup', (e) => {
      if (e.target.closest && (e.target.closest('.cp-panel') || e.target.closest('.cp-selask'))) return;
      setTimeout(() => {
        const s = window.getSelection();
        const txt = s ? String(s).trim() : '';
        if (txt.length > 3 && s.rangeCount && s.anchorNode && s.anchorNode.parentElement && s.anchorNode.parentElement.closest('#la-main')) {
          const r = s.getRangeAt(0).getBoundingClientRect();
          selAsk.style.left = Math.min(window.innerWidth - 200, r.left + window.scrollX + r.width / 2 - 80) + 'px';
          selAsk.style.top = r.top + window.scrollY - 44 + 'px';
          selAsk.classList.add('on');
          selAsk.dataset.txt = txt.slice(0, 4000);
        } else selAsk.classList.remove('on');
      }, 10);
    });
    selAsk.addEventListener('mousedown', (e) => e.preventDefault());
    selAsk.addEventListener('click', () => {
      selAsk.classList.remove('on');
      setAttachment({ kind: 'selection', label: LA.t('cp.selLabel'), text: selAsk.dataset.txt });
      open(true);
      els.ta.focus();
      if (!els.ta.value) els.ta.value = LA.t('cp.selQ');
    });

    buildSettings();
    updateModelTag();
    updateCtxLabel();
    renderThread();
  }

  function open(on) {
    document.body.classList.toggle('cp-open', on);
    if (on) { updateCtxLabel(); setTimeout(() => els.ta && els.ta.focus(), 250); const p = PROVIDERS.find((x) => x.id === cfg.provider); if (!cfg.base || (p && p.needsKey && (!cfg.key || !cfg.model))) showSettings(true); }
  }
  function autosize() { const t = els.ta; t.style.height = 'auto'; t.style.height = Math.min(180, t.scrollHeight) + 'px'; }
  function updateCtxLabel() {
    if (!els.ctxLabel) return;
    const c = LA.getContext();
    els.ctxLabel.textContent = c.type === 'chapter' ? c.title : c.title || LA.t('ctx.home');
    buildChips(c);
  }
  function updateModelTag() {
    if (!els.modelTag) return;
    const p = PROVIDERS.find((x) => x.id === cfg.provider);
    els.modelTag.textContent = (p ? p.label : 'API') + ' · ' + (cfg.model || cfg.label || (p && !p.needsKey ? LA.t('cp.serverModel') : LA.t('cp.pickModel')));
  }

  function buildChips(c) {
    const base = c.type === 'chapter' ? LA.t('cp.chips.chapter') : c.type === 'catalogo' ? LA.t('cp.chips.catalog') : LA.t('cp.chips.home');
    els.chips.innerHTML = '';
    base.slice(0, 4).forEach((q) => { const b = h('button', null, q); b.addEventListener('click', () => { els.ta.value = q; send(); }); els.chips.appendChild(b); });
  }

  function setAttachment(a) {
    attachment = a;
    els.attach.classList.toggle('on', !!a);
    if (a) els.attach.querySelector('.tx').textContent = `${a.label}: ${a.text.slice(0, 240)}${a.text.length > 240 ? '…' : ''}`;
  }

  function askAbout(kind, el) {
    if (kind === 'figure' && el) {
      const title = (el.dataset.title || '').trim();
      const cap = el.querySelector('figcaption');
      let prevText = '';
      let p = el.previousElementSibling, n = 0;
      while (p && n < 2) { if (p.innerText) prevText = p.innerText + '\n' + prevText; p = p.previousElementSibling; n++; }
      const stepper = el.querySelector('.la-stepper-text');
      const ctrls = Array.from(el.querySelectorAll('.la-ctl')).map((c) => c.innerText.replace(/\s+/g, ' ')).join(' | ');
      const txt = LA.t('cp.figCtx', { t: title, c: cap ? cap.innerText : '', s: stepper ? LA.t('cp.figStep') + stepper.innerText + '\n' : '', k: ctrls ? LA.t('cp.figCtl') + ctrls + '\n' : '', p: prevText.slice(0, 1800) });
      setAttachment({ kind: 'figure', label: LA.t('cp.figLabel', { t: title || LA.t('cp.figDefault') }), text: txt });
      open(true);
      els.ta.value = LA.t('cp.figQ');
      autosize();
    }
  }

  /* ============================================================ SETTINGS */
  function showSettings(on) { els.panel.classList.toggle('show-settings', on); if (on) buildSettings(); }

  function buildSettings() {
    const S = els.settings;
    S.innerHTML = '';
    S.appendChild(h('h3', null, LA.t('cp.s.h3')));
    S.appendChild(h('p', { style: { fontSize: '13px', color: 'var(--text-2)', margin: '0 0 10px' } }, LA.t('cp.s.intro')));
    const grid = h('div', { class: 'cp-providers' });
    PROVIDERS.forEach((p) => {
      const b = h('button', { class: 'cp-prov' + (cfg.provider === p.id ? ' on' : ''), type: 'button' }, h('b', null, p.label), h('span', null, p.desc));
      b.addEventListener('click', () => {
        const changed = cfg.provider !== p.id;
        cfg.provider = p.id;
        if (changed) { cfg.base = p.base; cfg.model = p.model; cfg.label = ''; cfg.ctxChars = p.ctx; cfg.tokenField = p.tokenField; cfg.noThink = p.id === 'mlx' || p.id === 'lmstudio' || p.id === 'ollama'; if (!p.needsKey) cfg.key = cfg.key || ''; }
        saveCfg(); buildSettings();
      });
      grid.appendChild(b);
    });
    S.appendChild(grid);

    const prov = PROVIDERS.find((x) => x.id === cfg.provider) || PROVIDERS[PROVIDERS.length - 1];
    S.appendChild(h('h4', null, LA.t('cp.s.conn')));
    const fBase = field(LA.t('cp.s.base'), 'text', cfg.base, LA.t('cp.s.basePh'), (v) => { cfg.base = v.trim().replace(/\/+$/, ''); });
    S.appendChild(fBase);
    const fKey = field(LA.t('cp.s.key') + (prov.needsKey ? '' : LA.t('cp.s.optional')), 'password', cfg.key, prov.needsKey ? 'sk-…' : LA.t('cp.s.keyPhOpt'), (v) => { cfg.key = v.trim(); });
    const eye = h('button', { class: 'la-b', type: 'button', title: LA.t('cp.s.eye') }, '👁');
    eye.addEventListener('click', () => { const i = fKey.querySelector('input'); i.type = i.type === 'password' ? 'text' : 'password'; });
    fKey.querySelector('.row').appendChild(eye);
    fKey.appendChild(h('small', null, LA.t('cp.s.keyNote')));
    S.appendChild(fKey);

    const fModel = field(LA.t('cp.s.model'), 'text', cfg.model, prov.id === 'mlx' ? LA.t('cp.s.modelPhMlx') : LA.t('cp.s.modelPh'), (v) => { cfg.model = v.trim(); cfg.label = ''; });
    const dl = h('datalist', { id: 'cp-models' });
    fModel.querySelector('input').setAttribute('list', 'cp-models');
    const bList = h('button', { class: 'la-b', type: 'button' }, LA.t('cp.s.list'));
    bList.addEventListener('click', listModels);
    fModel.querySelector('.row').appendChild(bList);
    fModel.appendChild(dl);
    if (prov.id === 'mlx') fModel.appendChild(h('small', null, LA.t('cp.s.mlxNote')));
    S.appendChild(fModel);

    S.appendChild(h('h4', null, LA.t('cp.s.gen')));
    const g2 = h('div', { class: 'cp-grid2' });
    g2.appendChild(field(LA.t('cp.s.temp'), 'number', cfg.temperature, '0.4', (v) => { cfg.temperature = v === '' ? '' : +v; }, { step: 0.1, min: 0, max: 2 }));
    g2.appendChild(field(LA.t('cp.s.maxTok'), 'number', cfg.maxTokens, '1800', (v) => { cfg.maxTokens = +v || 1800; }, { step: 100, min: 64 }));
    g2.appendChild(field(LA.t('cp.s.ctxBudget'), 'number', cfg.ctxChars, '16000', (v) => { cfg.ctxChars = +v || 12000; }, { step: 1000, min: 1000 }));
    g2.appendChild(selectField(LA.t('cp.s.tokField'), cfg.tokenField, [['max_tokens', 'max_tokens'], ['max_completion_tokens', 'max_completion_tokens']], (v) => { cfg.tokenField = v; }));
    S.appendChild(g2);
    const tg = h('div', { style: { display: 'flex', gap: '18px', flexWrap: 'wrap', marginTop: '12px' } });
    tg.appendChild(toggle(LA.t('cp.s.stream'), cfg.stream, (v) => { cfg.stream = v; }));
    tg.appendChild(toggle(LA.t('cp.s.showThink'), cfg.showThinking, (v) => { cfg.showThinking = v; }));
    tg.appendChild(toggle(LA.t('cp.s.noThink'), cfg.noThink, (v) => { cfg.noThink = v; }));
    S.appendChild(tg);
    S.appendChild(field(LA.t('cp.s.extra'), 'textarea', cfg.extraSystem, LA.t('cp.s.extraPh'), (v) => { cfg.extraSystem = v; }));
    S.appendChild(field(LA.t('cp.s.headers'), 'text', cfg.extraHeaders, '{"HTTP-Referer":"http://localhost"}', (v) => { cfg.extraHeaders = v; }));

    const status = h('div', { class: 'cp-status' });
    const bTest = h('button', { class: 'la-b primary', type: 'button' }, LA.t('cp.s.test'));
    bTest.addEventListener('click', () => testConnection(status));
    const bBack = h('button', { class: 'la-b', type: 'button' }, LA.t('cp.s.back'));
    bBack.addEventListener('click', () => showSettings(false));
    const bReset = h('button', { class: 'la-b', type: 'button' }, LA.t('cp.s.reset'));
    bReset.addEventListener('click', () => { cfg = Object.assign({}, DEFAULTS); saveCfg(); buildSettings(); });
    S.appendChild(h('div', { class: 'cp-actions' }, bTest, bBack, bReset));
    S.appendChild(status);
    S.appendChild(helpFor(prov.id));
  }

  function field(label, type, value, ph, onChange, attrs = {}) {
    const inp = type === 'textarea' ? h('textarea', { placeholder: ph }) : h('input', Object.assign({ type, placeholder: ph, spellcheck: 'false', autocomplete: 'off' }, attrs));
    inp.value = value == null ? '' : value;
    inp.addEventListener('change', () => { onChange(inp.value); saveCfg(); });
    return h('div', { class: 'cp-field' }, h('label', null, label), h('div', { class: 'row' }, inp));
  }
  function selectField(label, value, opts, onChange) {
    const s = h('select', null, opts.map(([v, l]) => h('option', { value: v }, l)));
    s.value = value;
    s.addEventListener('change', () => { onChange(s.value); saveCfg(); });
    return h('div', { class: 'cp-field' }, h('label', null, label), h('div', { class: 'row' }, s));
  }
  function toggle(label, value, onChange) {
    const inp = h('input', { type: 'checkbox' }); inp.checked = !!value;
    inp.addEventListener('change', () => { onChange(inp.checked); saveCfg(); });
    return h('label', { class: 'la-toggle' }, inp, h('i'), h('span', null, label));
  }
  function pre(code) {
    const p = h('pre', null, h('code', null, code));
    const b = h('button', { class: 'copy', type: 'button' }, LA.t('code.copy'));
    b.addEventListener('click', () => { navigator.clipboard && navigator.clipboard.writeText(code); b.textContent = 'ok ✓'; setTimeout(() => (b.textContent = LA.t('code.copy')), 1200); });
    p.appendChild(b);
    return p;
  }
  function helpFor(id) {
    const box = h('div', { class: 'cp-help' });
    if (id === 'mlx') {
      const T = LA.t('cp.h.mlx'), C = LA.t('cp.h.mlxCmd');
      box.append(h('b', null, T[0]));
      if (LA.mlx && LA.mlx.open) {
        const bMlx = h('button', { class: 'la-b primary cp-mlxsetup', type: 'button' }, LA.t('mlx.cpSetup'));
        bMlx.addEventListener('click', () => LA.mlx.open());
        box.append(h('p', null, bMlx));
      }
      box.append(h('p', null, T[1]), pre(C[0]), h('p', null, T[2]), pre(C[1]), h('p', null, T[3]), pre(C[2]), h('p', null, T[4]), h('p', null, T[5]), h('p', null, T[6]), pre(LA.t('cp.h.serve')));
    } else if (id === 'openai' || id === 'openrouter') {
      box.append(h('b', null, id === 'openai' ? 'OpenAI' : 'OpenRouter'), h('p', null, LA.t('cp.h.openai')));
    } else if (id === 'vllm') {
      box.append(h('b', null, LA.t('cp.h.vllmT')), h('p', null, LA.t('cp.h.vllm')), pre(LA.t('cp.h.vllmCmd')));
    } else if (id === 'lmstudio') {
      box.append(h('b', null, 'LM Studio'), h('p', null, LA.t('cp.h.lmstudio')));
    } else if (id === 'ollama') {
      box.append(h('b', null, 'Ollama'), h('p', null, LA.t('cp.h.ollama')), pre('OLLAMA_ORIGINS="*" ollama serve\nollama pull qwen3:8b'));
    } else {
      box.append(h('b', null, LA.t('cp.h.customT')), h('p', null, LA.t('cp.h.custom')));
    }
    return box;
  }

  function headers() {
    const hd = { 'Content-Type': 'application/json' };
    if (cfg.key) hd.Authorization = 'Bearer ' + cfg.key;
    if (cfg.extraHeaders) { try { Object.assign(hd, JSON.parse(cfg.extraHeaders)); } catch (e) { /* ignora JSON inválido */ } }
    return hd;
  }

  async function listModels() {
    const dl = document.getElementById('cp-models');
    const status = els.settings.querySelector('.cp-status');
    try {
      const r = await fetch(cfg.base + '/models', { headers: headers() });
      if (!r.ok) throw new Error('HTTP ' + r.status + ' ' + (await r.text()).slice(0, 300));
      const j = await r.json();
      const ids = (j.data || j.models || []).map((m) => m.id || m.name).filter(Boolean);
      dl.innerHTML = '';
      ids.forEach((id) => dl.appendChild(h('option', { value: id })));
      status.className = 'cp-status on ok';
      status.textContent = ids.length ? LA.t('cp.models', { n: ids.length }) + ids.slice(0, 30).join('\n') + (ids.length > 30 ? '\n…' : '') : LA.t('cp.noModels');
      if (!cfg.model && ids.length) { cfg.model = ids[0]; saveCfg(); const inp = els.settings.querySelector('input[list="cp-models"]'); if (inp) inp.value = ids[0]; }
    } catch (e) {
      status.className = 'cp-status on bad';
      status.textContent = explainError(e);
    }
  }

  async function testConnection(status) {
    status.className = 'cp-status on';
    status.textContent = LA.t('cp.testing');
    try {
      const body = buildBody([{ role: 'user', content: LA.t('cp.testQ') }], false, 16);
      const { json } = await postWithRetry(body, null);
      const txt = json && json.choices && json.choices[0] && (json.choices[0].message && (json.choices[0].message.content || json.choices[0].message.reasoning_content) || json.choices[0].text);
      status.className = 'cp-status on ok';
      status.textContent = LA.t('cp.connected') + (json.model || cfg.model) + LA.t('cp.reply') + String(txt || LA.t('cp.empty')).slice(0, 200);
    } catch (e) {
      status.className = 'cp-status on bad';
      status.textContent = explainError(e);
    }
  }

  function explainError(e) {
    const m = String(e && e.message || e);
    if (e && e.name === 'AbortError') return LA.t('cp.e.cancel');
    if (/Failed to fetch|NetworkError|Load failed|fetch/i.test(m) && !/HTTP/.test(m)) {
      return LA.t('cp.e.net', { base: cfg.base });
    }
    if (/HTTP 401|HTTP 403/.test(m)) return LA.t('cp.e.auth') + m;
    if (/HTTP 404/.test(m)) return LA.t('cp.e.404') + m;
    if (/HTTP 429/.test(m)) return LA.t('cp.e.429') + m;
    return '✕ ' + m;
  }

  /* ============================================================ PROMPT */
  function systemPrompt() {
    const c = LA.getContext();
    const L = [];
    LA.t('cp.sys').forEach((x) => L.push(x));
    if (cfg.extraSystem) L.push(LA.t('cp.sys.extra') + cfg.extraSystem);
    if (cfg.ctxMode !== 'none') {
      L.push(LA.t('cp.sys.trail') + c.outline);
      L.push(LA.t('cp.sys.page') + c.title + (c.subtitle ? ' — ' + c.subtitle : '') + (c.part ? '\n' + c.part : '') + (c.recipes ? LA.t('cp.sys.recipes') + c.recipes : ''));
      if (c.section) L.push(LA.t('cp.sys.section') + c.section);
      const budget = Math.max(1000, cfg.ctxChars || 12000);
      let body = '';
      if (cfg.ctxMode === 'section' && c.sectionText) body = c.sectionText;
      else body = c.pageText || '';
      if (c.summary) body += LA.t('cp.sys.summary') + c.summary;
      if (body.length > budget) {
        // mantém a região ao redor da seção visível quando truncar
        let i = c.section ? body.indexOf(c.section) : -1;
        if (i < 0) i = 0;
        const start = Math.max(0, Math.min(body.length - budget, i - Math.floor(budget * 0.25)));
        body = (start > 0 ? '[…]\n' : '') + body.slice(start, start + budget) + (start + budget < body.length ? '\n[…]' : '');
      }
      L.push(LA.t('cp.sys.content') + '"""\n' + body + '\n"""');
    }
    return L.join('\n');
  }

  function buildBody(messages, stream, maxTok) {
    const b = { model: cfg.model || undefined, messages, stream: !!stream };
    const lim = Math.max(16, Math.min(32768, Math.round(+(maxTok || cfg.maxTokens) || 2048)));
    b[cfg.tokenField || 'max_tokens'] = lim;
    // números sempre válidos: o mlx_lm.server fecha o socket (parece CORS) com valores inválidos
    if (cfg.temperature !== '' && cfg.temperature != null && !Number.isNaN(+cfg.temperature)) b.temperature = Math.max(0, Math.min(2, +cfg.temperature));
    if (cfg.noThink) b.chat_template_kwargs = { enable_thinking: false };
    return b;
  }

  async function postWithRetry(body, signal) {
    let attempt = 0;
    let lastErr;
    while (attempt < 3) {
      attempt++;
      const r = await fetch(cfg.base + '/chat/completions', { method: 'POST', headers: headers(), body: JSON.stringify(body), signal });
      if (r.ok) {
        if (body.stream) return { res: r };
        return { json: await r.json() };
      }
      const txt = await r.text();
      lastErr = new Error('HTTP ' + r.status + ': ' + txt.slice(0, 600));
      if (r.status === 400 || r.status === 422) {
        const low = txt.toLowerCase();
        let fixed = false;
        if (low.includes('max_tokens') && body.max_tokens != null) { body.max_completion_tokens = body.max_tokens; delete body.max_tokens; cfg.tokenField = 'max_completion_tokens'; fixed = true; }
        else if (low.includes('max_completion_tokens') && body.max_completion_tokens != null) { body.max_tokens = body.max_completion_tokens; delete body.max_completion_tokens; cfg.tokenField = 'max_tokens'; fixed = true; }
        if (low.includes('temperature') && body.temperature != null) { delete body.temperature; fixed = true; }
        if (low.includes('chat_template_kwargs') && body.chat_template_kwargs) { delete body.chat_template_kwargs; fixed = true; }
        if (fixed) { saveCfg(); continue; }
      }
      throw lastErr;
    }
    throw lastErr;
  }

  /* ============================================================ CHAT */
  async function send() {
    const text = els.ta.value.trim();
    if (!text) return;
    if (!cfg.base) { showSettings(true); return; }
    els.ta.value = ''; autosize();
    const th = thread();
    const userMsg = { role: 'user', content: text, att: attachment ? { label: attachment.label, text: attachment.text } : null };
    th.push(userMsg);
    const bot = { role: 'assistant', content: '', reasoning: '', pending: true };
    th.push(bot);
    const att = attachment; setAttachment(null);
    renderThread();

    const history = th.slice(0, -2).filter((m) => !m.error).slice(-10).map((m) => ({ role: m.role, content: m.role === 'user' && m.att ? `[${m.att.label}]\n${m.att.text}\n\n${m.content}` : m.content }));
    const userContent = att ? `[${att.label}]\n${att.text}\n\n---\n${text}` : text;
    const messages = [{ role: 'system', content: systemPrompt() }].concat(history, [{ role: 'user', content: userContent }]);

    aborter = new AbortController();
    setBusy(true);
    try {
      const body = buildBody(messages, cfg.stream);
      const out = await postWithRetry(body, aborter.signal);
      if (out.res) await readStream(out.res, bot);
      else {
        const m = out.json.choices && out.json.choices[0] && out.json.choices[0].message || {};
        bot.content = m.content || out.json.choices && out.json.choices[0] && out.json.choices[0].text || '';
        bot.reasoning = m.reasoning_content || m.reasoning || '';
      }
      splitThink(bot);
    } catch (e) {
      if (e.name === 'AbortError') { bot.content += (bot.content ? '\n\n' : '') + LA.t('cp.interrupted'); }
      else { th.pop(); th.push({ role: 'assistant', error: true, content: explainError(e) }); }
    } finally {
      bot.pending = false;
      aborter = null;
      setBusy(false);
      renderThread();
    }
  }
  function stop() { if (aborter) aborter.abort(); }
  function setBusy(b) {
    els.orb.classList.toggle('busy', b);
    els.sendBtn.classList.toggle('stop', b);
    els.sendBtn.innerHTML = b ? SVG.stop : SVG.send;
    els.sendBtn.title = b ? LA.t('cp.stop') : LA.t('cp.send');
  }

  async function readStream(res, bot) {
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    let lastPaint = 0;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, idx).replace(/\r$/, '');
        buf = buf.slice(idx + 1);
        if (!line.startsWith('data:')) continue;
        const data = line.slice(5).trim();
        if (!data || data === '[DONE]') continue;
        try {
          const j = JSON.parse(data);
          if (j.error) throw new Error(j.error.message || JSON.stringify(j.error));
          const ch = j.choices && j.choices[0];
          if (!ch) continue;
          const d = ch.delta || ch.message || {};
          if (d.content) bot.content += d.content;
          else if (ch.text) bot.content += ch.text;
          if (d.reasoning_content) bot.reasoning += d.reasoning_content;
          else if (d.reasoning) bot.reasoning += typeof d.reasoning === 'string' ? d.reasoning : '';
        } catch (e) {
          if (/error/i.test(e.message)) throw e;
        }
      }
      const now = performance.now();
      if (now - lastPaint > 60) { lastPaint = now; paintPending(bot); }
    }
  }
  function splitThink(m) {
    const c = m.content || '';
    const open = c.indexOf('<think>');
    if (open >= 0) {
      const close = c.indexOf('</think>');
      if (close >= 0) { m.reasoning = (m.reasoning || '') + c.slice(open + 7, close); m.content = (c.slice(0, open) + c.slice(close + 8)).trim(); }
      else { m.reasoning = (m.reasoning || '') + c.slice(open + 7); m.content = c.slice(0, open); }
    } else if (c.includes('</think>')) {
      const close = c.indexOf('</think>');
      m.reasoning = (m.reasoning || '') + c.slice(0, close); m.content = c.slice(close + 8).trim();
    }
  }

  /* ============================================================ RENDER */
  function renderThread() {
    if (!els.msgs) return;
    const th = thread();
    els.msgs.innerHTML = '';
    if (!th.length) {
      const c = LA.getContext();
      const cv = h('canvas', { width: 120, height: 120 });
      els.msgs.appendChild(h('div', { class: 'cp-empty' }, cv,
        h('div', { class: 'big' }, LA.t('cp.askAnything')),
        h('div', null, c.type === 'chapter' ? LA.t('cp.reading', { t: c.title }) : LA.t('cp.seeing'))));
      emptyOrb(cv);
      return;
    }
    th.forEach((m) => els.msgs.appendChild(msgEl(m)));
    els.msgs.scrollTop = els.msgs.scrollHeight;
  }
  function msgEl(m) {
    if (m.role === 'user') {
      const el = h('div', { class: 'cp-msg user' });
      if (m.att) el.appendChild(h('span', { class: 'att' }, '📎 ' + m.att.label));
      el.appendChild(document.createTextNode(m.content));
      return el;
    }
    if (m.error) return h('div', { class: 'cp-msg err' }, h('b', null, LA.t('cp.cantAnswer')), h('div', { style: { whiteSpace: 'pre-wrap' } }, m.content), h('div', { class: 'acts' }, (() => { const b = h('button', null, LA.t('cp.configure')); b.addEventListener('click', () => showSettings(true)); return b; })(),
      cfg.provider === 'mlx' && LA.mlx && LA.mlx.open ? (() => { const b = h('button', null, LA.t('mlx.cpOpen')); b.addEventListener('click', () => LA.mlx.open()); return b; })() : null));
    const el = h('div', { class: 'cp-msg bot' }, h('div', { class: 'who' }, h('i'), 'Copilot'));
    fillBot(el, m);
    return el;
  }
  function fillBot(el, m) {
    const tmp = Object.assign({}, m); splitThink(tmp);
    Array.from(el.querySelectorAll('.md, details.think, .acts')).forEach((n) => n.remove());
    if (cfg.showThinking && tmp.reasoning && tmp.reasoning.trim()) {
      const d = h('details', { class: 'think' }, h('summary', null, m.pending && !tmp.content ? LA.t('cp.thinking') : LA.t('cp.reasoning')), h('div', { class: 'tx' }, tmp.reasoning.trim()));
      if (m.pending && !tmp.content) d.open = true;
      el.appendChild(d);
    }
    const md = h('div', { class: 'md', html: renderMarkdown(tmp.content || '') + (m.pending ? '<span class="cp-cursor"></span>' : '') });
    el.appendChild(md);
    if (!m.pending) {
      LA.renderMath && LA.renderMath(md);
      const copy = h('button', null, LA.t('code.copy'));
      copy.addEventListener('click', () => { navigator.clipboard && navigator.clipboard.writeText(tmp.content); copy.textContent = LA.t('code.copied'); });
      el.appendChild(h('div', { class: 'acts' }, copy));
    }
  }
  function paintPending(m) {
    const last = els.msgs.lastElementChild;
    if (!last || !last.classList.contains('bot')) return;
    fillBot(last, m);
    els.msgs.scrollTop = els.msgs.scrollHeight;
  }
  function emptyOrb(cv) {
    if (!LA.viz) return;
    const g = cv.getContext('2d');
    let t = 0, raf;
    const tick = () => {
      if (!document.body.contains(cv)) return;
      if (!LA.paused) t += 0.016;
      g.clearRect(0, 0, 120, 120);
      for (let i = 0; i < 3; i++) {
        const r = 22 + i * 12 + Math.sin(t * 2 + i) * 3;
        g.strokeStyle = ['rgba(63,244,155,.7)', 'rgba(190,142,253,.55)', 'rgba(252,147,214,.4)'][i];
        g.lineWidth = 2; g.beginPath(); g.ellipse(60, 60, r, r * (0.45 + 0.1 * i), t * (0.6 + i * 0.3), 0, Math.PI * 2); g.stroke();
      }
      LA.viz.glow(g, 60, 60, 26, '#3FF49B', 1);
      raf = requestAnimationFrame(tick);
    };
    tick();
  }

  /* ----------------------------------------------- markdown mínimo e seguro */
  function renderMarkdown(src) {
    const esc = LA.esc;
    const stash = [];
    const keep = (html) => `\u0000${stash.push(html) - 1}\u0000`;
    let s = String(src || '');
    // blocos de código
    s = s.replace(/```([\w+-]*)\n?([\s\S]*?)(```|$)/g, (_, lang, code) => keep(`<pre data-lang="${esc(lang)}"><code>${esc(code.replace(/\n$/, ''))}</code></pre>`));
    // matemática (protege do markdown; KaTeX renderiza depois)
    s = s.replace(/\$\$([\s\S]+?)\$\$/g, (m) => keep(esc(m)));
    s = s.replace(/\\\[([\s\S]+?)\\\]/g, (m) => keep(esc(m)));
    s = s.replace(/\\\(([\s\S]+?)\\\)/g, (m) => keep(esc(m)));
    // código inline
    s = s.replace(/`([^`\n]+)`/g, (_, c) => keep(`<code>${esc(c)}</code>`));
    s = esc(s);
    const inline = (t) => t
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>')
      .replace(/(^|\W)_([^_\n]+)_(?=\W|$)/g, '$1<em>$2</em>')
      .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+|#[^)\s]*)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    const lines = s.split('\n');
    const out = [];
    let i = 0;
    while (i < lines.length) {
      const L = lines[i];
      if (/^\s*$/.test(L)) { i++; continue; }
      let m;
      if ((m = L.match(/^(#{1,4})\s+(.*)$/))) { out.push(`<h${m[1].length + 1}>${inline(m[2])}</h${m[1].length + 1}>`); i++; continue; }
      if (/^\s*(---|\*\*\*)\s*$/.test(L)) { out.push('<hr>'); i++; continue; }
      if (/^\s*&gt;\s?/.test(L)) { const buf = []; while (i < lines.length && /^\s*&gt;\s?/.test(lines[i])) { buf.push(lines[i].replace(/^\s*&gt;\s?/, '')); i++; } out.push(`<blockquote>${inline(buf.join('<br>'))}</blockquote>`); continue; }
      if (/^\s*\|.*\|\s*$/.test(L) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
        const row = (r) => r.trim().replace(/^\||\|$/g, '').split('|').map((c) => inline(c.trim()));
        const hd = row(L); i += 2;
        const rows = [];
        while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) { rows.push(row(lines[i])); i++; }
        out.push(`<table><thead><tr>${hd.map((c) => `<th>${c}</th>`).join('')}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table>`);
        continue;
      }
      if (/^\s*([-*+]|\d+[.)])\s+/.test(L)) {
        const ordered = /^\s*\d+[.)]\s+/.test(L);
        const items = [];
        while (i < lines.length && /^\s*([-*+]|\d+[.)])\s+/.test(lines[i])) {
          let it = lines[i].replace(/^\s*([-*+]|\d+[.)])\s+/, ''); i++;
          while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([-*+]|\d+[.)])\s+/.test(lines[i])) { it += ' ' + lines[i].trim(); i++; }
          items.push(`<li>${inline(it)}</li>`);
        }
        out.push(ordered ? `<ol>${items.join('')}</ol>` : `<ul>${items.join('')}</ul>`);
        continue;
      }
      const para = [];
      while (i < lines.length && !/^\s*$/.test(lines[i]) && !/^(#{1,4})\s/.test(lines[i]) && !/^\s*([-*+]|\d+[.)])\s+/.test(lines[i]) && !/^\s*\|.*\|\s*$/.test(lines[i]) && !/^\s*&gt;/.test(lines[i])) { para.push(lines[i]); i++; }
      if (para.length) out.push(`<p>${inline(para.join('<br>'))}</p>`);
      else i++;
    }
    let html = out.join('');
    html = html.replace(/\u0000(\d+)\u0000/g, (_, k) => stash[+k]);
    return html;
  }
  LA.renderMarkdown = renderMarkdown;
})();
