/* ==========================================================================
   LATENT ATLAS — core.js
   Registro de capítulos, roteador, navegação, busca, glossário, progresso,
   quiz, catálogo e ciclo de vida das figuras.
   Scripts clássicos (sem módulos ES) para funcionar direto via file://.
   ========================================================================== */
(function () {
  'use strict';

  const LA = (window.LA = window.LA || {});
  LA.version = '1.0.0';
  LA.chapters = [];
  LA.byId = {};
  LA.content = LA.content || {};
  LA.glossary = LA.glossary || {};
  LA.catalog = LA.catalog || [];
  LA.parts = LA.parts || [];
  LA.loadErrors = [];
  LA.paused = false;

  /* ------------------------------------------------------------------ utils */
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  LA.$ = $; LA.$$ = $$;

  LA.h = function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      const v = attrs[k];
      if (v == null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'style' && typeof v === 'object') { for (const sk in v) { if (sk.startsWith('--')) el.style.setProperty(sk, v[sk]); else el.style[sk] = v[sk]; } }
      else if (k === 'html') el.innerHTML = v;
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      el.appendChild(typeof kid === 'string' || typeof kid === 'number' ? document.createTextNode(String(kid)) : kid);
    }
    return el;
  };
  const h = LA.h;

  LA.esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  LA.norm = (s) => String(s || '').toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');
  LA.stripHtml = (html) => {
    const d = document.createElement('div');
    d.innerHTML = html;
    d.querySelectorAll('script,style').forEach((n) => n.remove());
    return (d.textContent || '').replace(/\s+/g, ' ').trim();
  };

  /* ---------------------------------------------------------------- storage */
  LA.store = {
    get(k, def) {
      try { const v = localStorage.getItem('la:' + k); return v == null ? def : JSON.parse(v); } catch (e) { return def; }
    },
    set(k, v) {
      try { localStorage.setItem('la:' + k, JSON.stringify(v)); } catch (e) { /* storage indisponível */ }
    },
  };
  const progress = LA.store.get('progress', { done: {}, quiz: {}, last: null });
  LA.progress = progress;
  function saveProgress() { LA.store.set('progress', progress); }

  /* ------------------------------------------------------------------ theme */
  LA.theme = {
    listeners: [],
    get() { return document.documentElement.getAttribute('data-theme') || 'dark'; },
    set(t) {
      document.documentElement.setAttribute('data-theme', t);
      LA.store.set('theme', t);
      LA.theme.listeners.forEach((f) => { try { f(t); } catch (e) { console.error(e); } });
    },
    toggle() { LA.theme.set(LA.theme.get() === 'dark' ? 'light' : 'dark'); },
    onChange(f) { LA.theme.listeners.push(f); return () => { LA.theme.listeners = LA.theme.listeners.filter((x) => x !== f); }; },
  };
  document.documentElement.setAttribute('data-theme', LA.store.get('theme', 'dark'));

  /* ------------------------------------------------------ chapter registry */
  LA.chapter = function (def) {
    if (!def || !def.id) { console.error('LA.chapter sem id', def); return; }
    if (LA.byId[def.id]) { console.warn('capítulo duplicado', def.id); return; }
    def.recipes = def.recipes || [];
    def.goals = def.goals || [];
    def.prereqs = def.prereqs || [];
    def.quiz = def.quiz || [];
    def.summary = def.summary || [];
    LA.byId[def.id] = def;
    LA.chapters.push(def);
  };
  LA.chapterLoadError = function (id, err) {
    console.error('[LA] erro ao carregar capítulo', id, err);
    LA.loadErrors.push({ id, err: String(err && err.stack || err) });
  };

  function sortChapters() {
    LA.chapters.sort((a, b) => (a.order - b.order) || String(a.id).localeCompare(b.id));
    LA.chapters.forEach((c, i) => { c.index = i; c.num = typeof c.order === 'number' ? c.order : i; });
  }

  // partes: campos *_en quando o idioma é inglês
  function localizeParts() {
    LA.modules = LA.modules || [{ n: 1, title: '', short: '' }];
    LA.roadmap = LA.roadmap || [];
    if (LA.lang !== 'en') return;
    [].concat(LA.parts, LA.modules, LA.roadmap).forEach((p) => { if (p.title_en) p.title = p.title_en; if (p.short_en) p.short = p.short_en; if (p.desc_en) p.desc = p.desc_en; });
  }
  LA.moduleOf = (partN) => { const p = LA.partOf(partN); return LA.modules.find((m) => m.n === (p.module || 1)) || LA.modules[0]; };
  LA.partOf = (n) => LA.parts.find((p) => p.n === n) || { n, title: LA.t('ch.part') + ' ' + n, color: 'var(--viz-a)' };
  LA.partColor = (n) => (LA.partOf(n).color || 'var(--viz-a)');
  LA.recipeName = { glm: 'GLM-5.3', tf: 'GLM·TF', qwen: 'Qwen3.8', ds: 'DS-V4', all: LA.t('recipe.all') };
  LA.recipeBadge = (r) => `<span class="la-rb ${r}">${LA.recipeName[r] || r}</span>`;

  /* ------------------------------------------------------- cleanup (page) */
  let cleanups = [];
  LA.onCleanup = (fn) => { cleanups.push(fn); return fn; };
  function runCleanups() {
    const cs = cleanups; cleanups = [];
    cs.forEach((f) => { try { f(); } catch (e) { console.error(e); } });
  }

  /* ---------------------------------------------------------------- icons */
  const ICON = {
    search: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
    moon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z"/></svg>',
    sun: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
    pause: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>',
    play: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 5v14l12-7z"/></svg>',
    burger: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg>',
    home: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 11 12 4l9 7v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/></svg>',
    book: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z"/><path d="M4 19V5"/></svg>',
    grid: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg>',
    chev: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="m6 9 6 6 6-6"/></svg>',
  };
  LA.ICON = ICON;

  /* =================================================================== BOOT */
  LA.boot = function () {
    localizeParts();
    sortChapters();
    buildShell();
    buildSidebar();
    buildSearchIndex();
    setupTooltips();
    setupKeys();
    window.addEventListener('hashchange', route);
    route();
    if (LA.copilot && LA.copilot.init) { try { LA.copilot.init(); } catch (e) { console.error('[copilot]', e); } }
    document.dispatchEvent(new CustomEvent('la:ready'));
  };

  /* ---------------------------------------------------------------- shell */
  function buildShell() {
    const app = $('#la-app');
    app.innerHTML = '';
    const top = h('header', { class: 'la-top' },
      h('button', { class: 'la-burger', 'aria-label': LA.t('a11y.openTrail'), html: ICON.burger, onclick: () => document.body.classList.toggle('la-side-open') }),
      h('a', { class: 'la-brand', href: '#/' },
        h('div', { class: 'la-logo', 'aria-hidden': 'true' }, h('span'), h('span'), h('span')),
        h('div', { class: 'la-brand-text' },
          h('span', { class: 'la-brand-name' }, 'LATENT ATLAS'),
          h('span', { class: 'la-brand-sub' }, LA.t('brand.sub')))),
      h('div', { class: 'la-top-progress', title: LA.t('top.progress') },
        h('span', { id: 'la-prog-txt' }, '0%'), h('div', { class: 'bar' }, h('i', { id: 'la-prog-bar' }))),
      h('div', { class: 'la-top-actions' },
        h('button', { class: 'la-iconbtn', id: 'la-search-btn', title: LA.t('top.searchTitle'), html: ICON.search + '<span class="lbl">' + LA.t('top.search') + '</span><kbd>⌘K</kbd>', onclick: openSearch }),
        h('button', { class: 'la-iconbtn', id: 'la-pause-btn', title: LA.t('top.pause'), html: ICON.pause, onclick: togglePause }),
        LA.mlx && LA.mlx.topButton && LA.mlx.topButton(),
        h('button', { class: 'la-iconbtn', id: 'la-theme-btn', title: LA.t('top.theme'), html: LA.theme.get() === 'dark' ? ICON.sun : ICON.moon, onclick: () => { LA.theme.toggle(); } }),
        h('button', { class: 'la-iconbtn la-langbtn', id: 'la-lang-btn', title: LA.t('top.lang'), html: '<b>' + LA.t('top.langShort') + '</b>', onclick: () => LA.setLang && LA.setLang(LA.lang === 'en' ? 'pt' : 'en') })
      ));
    LA.theme.onChange((t) => { $('#la-theme-btn').innerHTML = t === 'dark' ? ICON.sun : ICON.moon; });
    const side = h('aside', { class: 'la-side', id: 'la-side', 'aria-label': LA.t('side.aria') });
    const main = h('main', { class: 'la-main', id: 'la-main' });
    const scrim = h('div', { class: 'la-scrim', onclick: () => document.body.classList.remove('la-side-open') });
    const readbar = h('div', { class: 'la-readbar', id: 'la-readbar' });
    app.append(top, readbar, side, scrim, main);

    // busca
    const search = h('div', { class: 'la-search', id: 'la-search', onclick: (e) => { if (e.target.id === 'la-search') closeSearch(); } },
      h('div', { class: 'la-search-box' },
        h('input', { id: 'la-search-in', type: 'search', placeholder: LA.t('search.ph'), autocomplete: 'off' }),
        h('div', { class: 'la-search-res', id: 'la-search-res' })));
    document.body.appendChild(search);
    $('#la-search-in').addEventListener('input', runSearch);
    $('#la-search-in').addEventListener('keydown', searchKeys);

    // tooltip
    document.body.appendChild(h('div', { class: 'la-tip', id: 'la-tip' }));

    window.addEventListener('scroll', onScroll, { passive: true });
  }

  function togglePause() {
    LA.paused = !LA.paused;
    document.body.classList.toggle('la-paused', LA.paused);
    $('#la-pause-btn').innerHTML = LA.paused ? ICON.play : ICON.pause;
  }
  LA.togglePause = togglePause;

  function onScroll() {
    const rb = $('#la-readbar');
    if (!rb) return;
    const d = document.documentElement;
    const max = d.scrollHeight - d.clientHeight;
    rb.style.width = (max > 0 && LA.route && LA.route.type === 'chapter' ? (d.scrollTop / max) * 100 : 0) + '%';
  }

  /* -------------------------------------------------------------- sidebar */
  function buildSidebar() {
    const side = $('#la-side');
    side.innerHTML = '';
    side.appendChild(h('a', { class: 'la-side-home', href: '#/', 'data-route': 'home', html: ICON.home + '<span>' + LA.t('side.home') + '</span>' }));
    const collapsed = LA.store.get('collapsed', {});
    let lastModule = null;
    LA.parts.forEach((p) => {
      const chs = LA.chapters.filter((c) => c.part === p.n);
      if (!chs.length) return;
      const mod = LA.moduleOf(p.n);
      if (mod && mod !== lastModule && LA.modules.length > 1) {
        lastModule = mod;
        side.appendChild(h('div', { class: 'la-side-module', style: { '--mc': mod.color } }, h('span', { class: 'mn' }, LA.t('mod.label') + ' ' + mod.n), h('span', { class: 'mt' }, mod.short || mod.title)));
      }
      const box = h('div', { class: 'la-side-part' + (collapsed[p.n] ? ' collapsed' : ''), style: { '--pc': p.color } });
      const hd = h('button', { class: 'la-side-part-h', onclick: () => { box.classList.toggle('collapsed'); collapsed[p.n] = box.classList.contains('collapsed'); LA.store.set('collapsed', collapsed); } },
        h('span', { class: 'pn' }, p.roman || String(p.n)), h('span', null, p.short || p.title), h('span', { class: 'chev', html: ICON.chev }));
      const ul = h('ul', { class: 'la-side-list' });
      chs.forEach((c) => {
        ul.appendChild(h('li', null, h('a', { href: '#/c/' + c.id, 'data-id': c.id },
          h('span', { class: 'num' }, String(c.num).padStart(2, '0')),
          h('span', null, c.navTitle || c.title),
          progress.done[c.id] ? h('span', { class: 'done' }, '✓') : null)));
      });
      box.append(hd, ul);
      side.appendChild(box);
    });
    const extra = h('div', { class: 'la-side-extra' },
      h('a', { class: 'la-side-home', href: '#/catalogo', 'data-route': 'catalogo', html: ICON.grid + '<span>' + LA.t('side.catalog') + '</span>' }),
      h('a', { class: 'la-side-home', href: '#/glossario', 'data-route': 'glossario', html: ICON.book + '<span>' + LA.t('side.glossary') + '</span>' }));
    side.appendChild(extra);
    if (LA.loadErrors.length) side.appendChild(h('div', { class: 'la-loaderr' }, LA.t('side.loadErr') + LA.loadErrors.map((e) => e.id).join(', ')));
    updateProgressBar();
  }

  function markSidebar() {
    $$('.la-side a').forEach((a) => a.classList.remove('active'));
    const r = LA.route || {};
    if (r.type === 'chapter') {
      const a = $(`.la-side a[data-id="${r.id}"]`);
      if (a) {
        a.classList.add('active');
        const part = a.closest('.la-side-part');
        if (part && part.classList.contains('collapsed')) part.classList.remove('collapsed');
        const side = $('#la-side');
        const ar = a.getBoundingClientRect();
        if (ar.top < 80 || ar.bottom > window.innerHeight - 40) a.scrollIntoView({ block: 'center' });
      }
    } else {
      const a = $(`.la-side a[data-route="${r.type}"]`);
      if (a) a.classList.add('active');
    }
  }

  function updateProgressBar() {
    const total = LA.chapters.length || 1;
    const done = LA.chapters.filter((c) => progress.done[c.id]).length;
    const pct = Math.round((done / total) * 100);
    const bar = $('#la-prog-bar'), txt = $('#la-prog-txt');
    if (bar) bar.style.width = pct + '%';
    if (txt) txt.textContent = LA.t('prog.txt', { d: done, t: total, p: pct });
  }

  /* ---------------------------------------------------------------- router */
  function parseHash() {
    const hs = decodeURIComponent(location.hash.replace(/^#\/?/, ''));
    const seg = hs.split('/').filter(Boolean);
    if (!seg.length) return { type: 'home' };
    if (seg[0] === 'c' && seg[1]) return { type: 'chapter', id: seg[1], anchor: seg[2] || null };
    if (seg[0] === 'catalogo') return { type: 'catalogo', q: seg[1] || '' };
    if (seg[0] === 'glossario') return { type: 'glossario', term: seg[1] || '' };
    return { type: 'home' };
  }

  function route() {
    const r = parseHash();
    const prev = LA.route;
    // mesma página, apenas âncora
    if (prev && prev.type === 'chapter' && r.type === 'chapter' && prev.id === r.id) {
      LA.route = r;
      if (r.anchor) scrollToAnchor(r.anchor);
      return;
    }
    runCleanups();
    LA.route = r;
    document.body.classList.remove('la-side-open');
    const main = $('#la-main');
    main.innerHTML = '';
    try {
      if (r.type === 'chapter' && LA.byId[r.id]) renderChapter(main, LA.byId[r.id], r.anchor);
      else if (r.type === 'catalogo') renderCatalog(main, r.q);
      else if (r.type === 'glossario') renderGlossary(main, r.term);
      else { LA.route = { type: 'home' }; renderHome(main); }
    } catch (e) {
      console.error(e);
      main.appendChild(h('div', { class: 'la-page' }, h('div', { class: 'la-loaderr' }, LA.t('err.render') + e.message)));
    }
    markSidebar();
    if (!(r.type === 'chapter' && r.anchor)) window.scrollTo({ top: 0, behavior: 'instant' in window ? 'instant' : 'auto' });
    onScroll();
    document.dispatchEvent(new CustomEvent('la:route', { detail: LA.route }));
  }
  LA.go = (hash) => { location.hash = hash; };

  function scrollToAnchor(id) {
    const el = document.getElementById(id);
    if (el) setTimeout(() => el.scrollIntoView({ behavior: 'smooth', block: 'start' }), 60);
  }

  /* ================================================================ HOME */
  function renderHome(main) {
    document.title = LA.t('home.docTitle');
    const page = h('div', { class: 'la-page' });
    const nFigs = LA.chapters.reduce((s, c) => s + ((LA.content[c.id] || '').match(/<figure class="la-fig/g) || []).length, 0);
    const hero = h('section', { class: 'la-home-hero' },
      h('canvas', { id: 'la-home-canvas' }),
      h('div', { class: 'la-home-hero-inner' },
        h('span', { class: 'la-eyebrow' }, h('i'), LA.t('home.eyebrow')),
        h('h1', { class: 'la-home-title' }, 'Latent', h('br'), 'Atlas'),
        h('p', { class: 'la-home-tag', html: LA.t('home.tag') }),
        h('div', { class: 'la-cta-row' },
          h('a', { class: 'la-btn la-btn-primary', href: '#/c/' + (resumeId() || (LA.chapters[0] && LA.chapters[0].id) || '') }, progress.last ? LA.t('home.continue') : LA.t('home.start')),
          h('a', { class: 'la-btn la-btn-ghost', href: '#/catalogo' }, LA.t('home.seeCatalog'))),
        h('div', { class: 'la-home-stats' },
          h('div', null, h('b', null, String(LA.chapters.length)), h('span', null, LA.t('home.st.chapters'))),
          h('div', null, h('b', null, String(nFigs)), h('span', null, LA.t('home.st.figs'))),
          h('div', null, h('b', null, String(LA.catalog.length)), h('span', null, LA.t('home.st.params'))),
          h('div', null, h('b', null, String(Object.keys(LA.glossary).length)), h('span', null, LA.t('home.st.terms'))))));
    page.appendChild(hero);

    // módulos
    if (LA.modules.length >= 1) {
      const mods = h('div', { class: 'la-modules' });
      LA.modules.forEach((m) => {
        const chs = LA.chapters.filter((c) => (LA.partOf(c.part).module || 1) === m.n);
        if (!chs.length) return;
        const done = chs.filter((c) => progress.done[c.id]).length;
        const parts = LA.parts.filter((p) => (p.module || 1) === m.n);
        mods.appendChild(h('a', { class: 'la-module-card', href: '#/c/' + chs[0].id, style: { '--mc': m.color } },
          h('span', { class: 'mk' }, LA.t('mod.label') + ' ' + m.n),
          h('h3', null, m.title), h('p', null, m.desc),
          h('div', { class: 'mmeta' }, `${chs.length} ${LA.t('home.st.chapters')} · ${parts.length} ${LA.t('mod.parts')} · ${LA.t('ch.part')} ${parts[0].roman}–${parts[parts.length - 1].roman}`),
          h('div', { class: 'meter' }, h('i', { style: { width: Math.round((done / chs.length) * 100) + '%' } })),
          h('span', { class: 'go' }, LA.t('mod.start'))));
      });
      LA.roadmap.forEach((r, i) => mods.appendChild(h('div', { class: 'la-module-card soon' },
        h('span', { class: 'mk' }, LA.t('mod.label') + ' ' + (LA.modules.length + i + 1) + ' · ' + LA.t('mod.soon')), h('h3', null, r.title), h('p', null, r.desc))));
      page.appendChild(h('section', { class: 'la-home-section la-reveal' }, h('h2', null, LA.t('mod.h2')), h('p', { class: 'lead', html: LA.t('mod.lead') }), mods));
    }

    // hardware: laboratório → produção
    page.appendChild(h('section', { class: 'la-home-section la-reveal' },
      h('h2', null, LA.t('home.hw.h2')),
      h('p', { class: 'lead', html: LA.t('home.hw.lead') }),
      h('div', { class: 'la-modules la-hw' }, LA.t('home.hw').map(([k, t, items], i) => h('div', { class: 'la-module-card', style: { '--mc': ['var(--viz-a)', 'var(--viz-b)', 'var(--viz-c)'][i] } },
        h('span', { class: 'mk' }, k), h('h3', null, t), h('ul', null, items.map((x) => h('li', null, x))))))));

    // receitas
    const rec = h('section', { class: 'la-home-section la-reveal' },
      h('h2', null, LA.t('home.rec.h2')),
      h('p', { class: 'lead', html: LA.t('home.rec.lead') }),
      h('div', { class: 'la-recipes' },
        recipeCard('glm', 'GLM-5.3-Flash'),
        recipeCard('tf', 'GLM-5.3-Flash · TensorFold'),
        recipeCard('qwen', 'Qwen3.8-Flash-Next'),
        recipeCard('ds', 'DeepSeek-V4-Flash')));
    page.appendChild(rec);

    // trilha
    const trail = h('div', { class: 'la-trail' });
    let lastMod = null;
    LA.parts.forEach((p) => {
      const chs = LA.chapters.filter((c) => c.part === p.n);
      if (!chs.length) return;
      const m = LA.moduleOf(p.n);
      if (LA.modules.length > 1 && m !== lastMod) { lastMod = m; trail.appendChild(h('div', { class: 'la-trail-mod', style: { '--mc': m.color } }, LA.t('mod.label') + ' ' + m.n + ' · ' + m.title)); }
      const done = chs.filter((c) => progress.done[c.id]).length;
      trail.appendChild(h('div', { class: 'la-trail-card la-reveal', style: { '--pc': p.color } },
        h('div', { class: 'pnum' }, p.roman || String(p.n)),
        h('h3', null, p.title),
        h('p', null, p.desc || ''),
        h('ol', null, chs.map((c) => h('li', null, h('a', { href: '#/c/' + c.id }, h('span', { class: 'n' }, String(c.num).padStart(2, '0')), h('span', null, c.navTitle || c.title), progress.done[c.id] ? h('span', { class: 'ok' }, '✓') : null)))),
        h('div', { class: 'meter' }, h('i', { style: { width: Math.round((done / chs.length) * 100) + '%' } }))));
    });
    page.appendChild(h('section', { class: 'la-home-section' },
      h('h2', { class: 'la-reveal' }, LA.t('home.trail.h2')),
      h('p', { class: 'lead la-reveal', html: LA.t('home.trail.lead') }),
      trail));

    // como usar
    page.appendChild(h('section', { class: 'la-home-section la-reveal' },
      h('h2', null, LA.t('home.how.h2')),
      h('div', { class: 'la-howto' },
        LA.t('home.how').map(([a, b]) => howto(a, b)))));

    // fontes / créditos
    page.appendChild(h('section', { class: 'la-home-section la-reveal' },
      h('h2', null, LA.t('home.src.h2')),
      h('p', { class: 'lead', html: LA.t('home.src.lead') }),
      h('div', { class: 'la-howto' },
        srcCard('GLM-5.3-Flash EXL3', 'https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks', 'glm'),
        srcCard('Qwen3.8-Flash-Next NVFP4', 'https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks', 'qwen'),
        srcCard('DeepSeek-V4-Flash DSpark', 'https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark', 'ds'))));

    main.appendChild(page);
    setupReveal(page);
    if (LA.viz && LA.viz.homeHero) LA.viz.homeHero($('#la-home-canvas'));
    $$('.la-recipe canvas', page).forEach((cv) => LA.viz && LA.viz.recipeGlyph && LA.viz.recipeGlyph(cv, cv.dataset.r));
  }
  function resumeId() { return progress.last && LA.byId[progress.last] ? progress.last : null; }
  function recipeCard(r, name) {
    const [sub, items] = LA.t('home.cards')[r];
    const color = { glm: 'var(--glm)', tf: 'var(--tf)', qwen: 'var(--qwen)', ds: 'var(--ds)' }[r];
    return h('div', { class: 'la-recipe', style: { '--rc': color } },
      h('canvas', { 'data-r': r === 'tf' ? 'glm' : r, width: 140, height: 140 }),
      h('div', { class: 'tag' }, ({ ds: 'DS', tf: 'GLM · TENSORFOLD' }[r] || r.toUpperCase()) + ' · ' + LA.t('home.recipeTag')),
      h('h3', null, name), h('div', { class: 'sub' }, sub),
      h('ul', null, items.map((i) => h('li', null, i))));
  }
  function howto(t, p) { return h('div', null, h('b', null, t), h('p', null, p)); }
  function srcCard(t, url, r) {
    return h('div', null, h('b', null, h('span', { html: LA.recipeBadge(r) }), ' ', t),
      h('p', null, h('a', { href: url, target: '_blank', rel: 'noopener' }, url.replace('https://github.com/', ''))));
  }

  /* ============================================================= CHAPTER */
  function renderChapter(main, c, anchor) {
    document.title = `${String(c.num).padStart(2, '0')} · ${c.title} — Latent Atlas`;
    progress.last = c.id; saveProgress();
    const part = LA.partOf(c.part);
    const pc = part.color;
    const page = h('div', { class: 'la-page la-chapter', style: { '--pc': pc }, 'data-id': c.id });

    const hero = h('header', { class: 'la-ch-hero' },
      h('canvas', { class: 'la-hero-bg' }),
      h('div', { class: 'la-ch-num' }, String(c.num).padStart(2, '0')),
      h('div', { class: 'la-ch-meta' },
        LA.modules.length > 1 ? h('span', { class: 'la-modchip', style: { '--mc': LA.moduleOf(c.part).color } }, `${LA.t('mod.label')} ${LA.moduleOf(c.part).n} · ${LA.moduleOf(c.part).short}`) : null,
        h('span', { class: 'part' }, `${LA.t('ch.part')} ${part.roman || part.n} · ${part.short || part.title}`),
        h('span', null, `${LA.t('ch.chapter')} ${String(c.num).padStart(2, '0')}`), h('span', { class: 'sep' }, '•'),
        h('span', null, `⏱ ${c.minutes || 20} min`),
        c.recipes.length ? h('span', { class: 'sep' }, '•') : null,
        h('span', { html: c.recipes.map(LA.recipeBadge).join(' ') })),
      h('h1', null, c.title),
      c.subtitle ? h('p', { class: 'la-sub' }, c.subtitle) : null,
      (c.goals.length || c.prereqs.length) ? h('div', { class: 'la-ch-hero-grid' },
        c.goals.length ? h('div', { class: 'la-goals' }, h('h4', null, LA.t('ch.learn')), h('ul', null, c.goals.map((g) => h('li', { html: g })))) : h('div'),
        h('div', { class: 'la-prereqs' }, h('h4', null, LA.t('ch.prereq')),
          c.prereqs.length ? h('div', { class: 'la-chips' }, c.prereqs.map((pid) => {
            const pch = LA.byId[pid];
            return pch ? h('a', { class: 'la-chip', href: '#/c/' + pid }, `${String(pch.num).padStart(2, '0')} · ${pch.navTitle || pch.title}`) : h('span', { class: 'la-chip' }, pid);
          })) : h('p', { style: { margin: 0, fontSize: '14px', color: 'var(--text-2)' } }, LA.t('ch.noPrereq')))) : null);
    page.appendChild(hero);

    const body = h('div', { class: 'la-ch-body' });
    const article = h('article', { class: 'la-article', id: 'la-article' });
    const html = LA.content[c.id];
    article.innerHTML = (c.__fallback ? '<div class="la-callout info"><b class="t">i18n</b><p>' + LA.t('ch.notTranslated') + '</p></div>' : '') + (html || '<p class="la-lead">' + LA.t('ch.wip') + '</p>');
    const toc = h('nav', { class: 'la-toc', 'aria-label': LA.t('ch.toc') });
    body.append(article, toc);
    page.appendChild(body);

    // resumo
    if (c.summary.length) page.appendChild(h('section', { class: 'la-summary la-reveal' }, h('h3', null, LA.t('ch.summary')), h('ul', null, c.summary.map((s) => h('li', { html: s })))));
    // quiz
    if (c.quiz.length) page.appendChild(buildQuiz(c));
    // navegação
    page.appendChild(buildChapterNav(c));

    main.appendChild(page);

    enhanceArticle(article, c);
    buildToc(article, toc);
    setupReveal(page);
    if (LA.viz && LA.viz.heroBg) LA.viz.heroBg($('canvas.la-hero-bg', hero), c.part);

    // monta as figuras do capítulo
    if (typeof c.mount === 'function') {
      try {
        const ret = c.mount(article, LA);
        if (typeof ret === 'function') LA.onCleanup(ret);
      } catch (e) {
        console.error('[LA] mount falhou em', c.id, e);
        article.insertBefore(h('div', { class: 'la-loaderr' }, LA.t('ch.mountErr') + e.message), article.firstChild);
      }
    }
    // elementos criados no mount também recebem o tratamento básico
    postMount(article);
    // math depois do mount (figuras podem injetar TeX)
    renderMath(page);
    if (anchor) scrollToAnchor(anchor);
  }
  LA.renderMath = renderMath;

  function enhanceArticle(article, c) {
    snapshotTags(article);
    // seções numeradas
    let n = 0;
    $$('h2', article).forEach((h2) => {
      n++;
      if (!h2.id) h2.id = 's' + n + '-' + LA.norm(h2.textContent).replace(/[^a-z0-9]+/g, '-').slice(0, 40);
      if (!h2.querySelector('.la-h-num')) h2.insertBefore(h('span', { class: 'la-h-num' }, String(n).padStart(2, '0')), h2.firstChild);
      h2.appendChild(h('a', { class: 'la-anchor', href: `#/c/${c.id}/${h2.id}`, title: LA.t('ch.secLink') }, '#'));
    });
    let n3 = 0;
    $$('h3', article).forEach((h3) => { n3++; if (!h3.id) h3.id = 'h3-' + n3 + '-' + LA.norm(h3.textContent).replace(/[^a-z0-9]+/g, '-').slice(0, 40); });

    // figuras: cabeçalho, número, legenda
    let f = 0;
    $$('figure.la-fig', article).forEach((fig) => {
      f++;
      fig.dataset.num = f;
      if (!fig.querySelector('.la-fig-head')) {
        const title = fig.dataset.title || LA.t('fig.default');
        const head = h('div', { class: 'la-fig-head' },
          h('span', { class: 'dot' }), h('span', { class: 't' }, title),
          h('span', { class: 'n' }, 'FIG. ' + String(c.num).padStart(2, '0') + '.' + f),
          h('button', { class: 'la-fig-btn', title: LA.t('fig.askTitle'), onclick: () => LA.copilot && LA.copilot.askAbout && LA.copilot.askAbout('figure', fig) }, LA.t('fig.ask')),
          h('button', { class: 'la-fig-btn', title: LA.t('fig.full'), onclick: () => { fig.classList.toggle('fullscreen'); window.dispatchEvent(new Event('resize')); } }, '⤢'));
        fig.insertBefore(head, fig.firstChild);
      }
      const cap = fig.querySelector('figcaption');
      if (cap && !cap.querySelector('.la-fig-label')) cap.insertBefore(h('b', { class: 'la-fig-label' }, `${LA.t('fig.label')} ${c.num}.${f}`), cap.firstChild);
    });

    // tabelas com rolagem
    $$('table', article).forEach((t) => {
      if (t.parentElement.classList.contains('la-table-wrap')) return;
      const w = h('div', { class: 'la-table-wrap' });
      t.parentNode.insertBefore(w, t); w.appendChild(t);
    });

    // blocos de código: highlight + copiar
    $$('pre', article).forEach((pre) => {
      const code = pre.querySelector('code');
      if (code && window.hljs) {
        const lang = pre.dataset.lang;
        if (lang && window.hljs.getLanguage && window.hljs.getLanguage(lang)) code.classList.add('language-' + lang);
        try { window.hljs.highlightElement(code); } catch (e) { /* ignore */ }
      }
      const btn = h('button', { class: 'la-copy' }, LA.t('code.copy'));
      btn.addEventListener('click', () => {
        const txt = (code || pre).innerText;
        const done = () => { btn.textContent = LA.t('code.copied'); setTimeout(() => (btn.textContent = LA.t('code.copy')), 1400); };
        if (navigator.clipboard) navigator.clipboard.writeText(txt).then(done, done); else done();
      });
      pre.appendChild(btn);
    });

    // badges de receita: <span class="la-rb glm"></span> vazio vira nome
    $$('.la-rb', article).forEach((b) => { if (!b.textContent.trim()) { const r = ['glm', 'tf', 'qwen', 'ds', 'all'].find((k) => b.classList.contains(k)); b.textContent = LA.recipeName[r] || r; } });

    // links internos [data-ch]
    $$('a[data-ch]', article).forEach((a) => { a.href = '#/c/' + a.dataset.ch + (a.dataset.sec ? '/' + a.dataset.sec : ''); });
  }

  /* Estudos de caso datados: o conteúdo GLM · TensorFold descreve a receita v1.3.2 (LA.snapshots.tf).
     Cada callout [data-recipe="tf"] ganha uma etiqueta e cada "Atualização/Update … TensorFold" uma nota,
     com link para a seção do cap. 49 que lista o que mudou depois. Blocos dentro dessa seção ficam sem etiqueta. */
  function snapshotTags(article) {
    const sn = LA.snapshots && LA.snapshots.tf;
    if (!sn || article.querySelector('.la-snap')) return;
    const href = '#/c/' + sn.ch + '/' + sn.sec;
    const since = article.querySelector('#' + sn.sec);
    const after = (el) => since && (since.compareDocumentPosition(el) & Node.DOCUMENT_POSITION_FOLLOWING);
    const vars = { v: sn.v, d: LA.lang === 'en' ? sn.date : sn.date.split('-').reverse().join('/'), e: sn.engine };
    $$('.la-callout[data-recipe="tf"] > .t', article).forEach((t) => {
      if (after(t)) return;
      t.appendChild(h('a', { class: 'la-snap', href, title: LA.t('snap.tip', vars) }, LA.t('snap.tag', vars)));
    });
    $$('h2', article).forEach((h2) => {
      if (after(h2) || h2.id === sn.sec || !/TensorFold/.test(h2.textContent) || !/^(Update|Atualização)\b/.test(h2.textContent.trim())) return;
      h2.insertAdjacentElement('afterend', h('p', { class: 'la-snapnote', html: LA.t('snap.note', vars).replace('{href}', href) }));
    });
  }

  function postMount(article) {
    $$('table', article).forEach((t) => {
      if (t.closest('.la-table-wrap') || t.closest('figure.la-fig')) return;
      const w = h('div', { class: 'la-table-wrap' });
      t.parentNode.insertBefore(w, t); w.appendChild(t);
    });
    $$('.la-rb', article).forEach((b) => { if (!b.textContent.trim()) { const r = ['glm', 'tf', 'qwen', 'ds', 'all'].find((k) => b.classList.contains(k)); b.textContent = LA.recipeName[r] || r; } });
  }
  LA.enhance = postMount;

  function buildToc(article, toc) {
    const hs = $$('h2, h3', article);
    if (!hs.length) { toc.remove(); return; }
    toc.appendChild(h('h4', null, LA.t('ch.toc')));
    const ol = h('ol');
    hs.forEach((x) => {
      const txt = x.cloneNode(true);
      txt.querySelectorAll('.la-h-num,.la-anchor').forEach((n) => n.remove());
      ol.appendChild(h('li', { class: x.tagName.toLowerCase() }, h('a', { href: `#/c/${LA.route.id}/${x.id}`, 'data-target': x.id }, txt.textContent.trim())));
    });
    toc.appendChild(ol);
    toc.appendChild(h('button', { class: 'la-b la-toc-ask', onclick: () => LA.copilot && LA.copilot.open && LA.copilot.open() }, LA.t('toc.ask')));
    // seção ativa
    const links = $$('a', ol);
    const io = new IntersectionObserver((ents) => {
      ents.forEach((e) => {
        if (e.isIntersecting) {
          links.forEach((l) => l.classList.toggle('active', l.dataset.target === e.target.id));
          LA.currentSection = e.target.id;
        }
      });
    }, { rootMargin: '-15% 0px -70% 0px' });
    hs.forEach((x) => io.observe(x));
    LA.onCleanup(() => io.disconnect());
  }

  function setupReveal(root) {
    const els = $$('.la-reveal', root);
    if (!('IntersectionObserver' in window)) { els.forEach((e) => e.classList.add('in')); return; }
    const io = new IntersectionObserver((ents) => ents.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { rootMargin: '0px 0px -8% 0px' });
    els.forEach((e) => io.observe(e));
    LA.onCleanup(() => io.disconnect());
  }
  LA.setupReveal = setupReveal;

  function renderMath(root) {
    if (typeof window.renderMathInElement !== 'function') return;
    try {
      window.renderMathInElement(root, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '\\[', right: '\\]', display: true },
          { left: '\\(', right: '\\)', display: false },
        ],
        throwOnError: false,
        ignoredClasses: ['la-nomath'],
      });
    } catch (e) { console.warn('katex', e); }
  }

  /* ------------------------------------------------------------------ quiz */
  function buildQuiz(c) {
    const st = progress.quiz[c.id] || { answers: {} };
    const box = h('section', { class: 'la-quiz la-reveal' });
    const score = h('span', { class: 'score' });
    box.appendChild(h('h3', null, LA.t('quiz.h'), score));
    const upd = () => {
      const ans = st.answers || {};
      const k = Object.keys(ans);
      const ok = k.filter((i) => ans[i] === c.quiz[i].a).length;
      score.textContent = k.length ? LA.t('quiz.correct', { ok, n: c.quiz.length }) : LA.t('quiz.count', { n: c.quiz.length });
    };
    c.quiz.forEach((q, i) => {
      const card = h('div', { class: 'la-q' });
      card.appendChild(h('div', { class: 'qt', html: `${i + 1}. ${q.q}` }));
      const opts = h('div', { class: 'opts' });
      const btns = (q.o || []).map((o, j) => {
        const b = h('button', { html: o });
        b.addEventListener('click', () => { answer(j); });
        opts.appendChild(b);
        return b;
      });
      const exp = h('div', { class: 'exp', html: q.e || '' });
      card.append(opts, exp);
      function answer(j, silent) {
        btns.forEach((b, k) => { b.disabled = true; if (k === q.a) b.classList.add('right'); else if (k === j) b.classList.add('wrong'); });
        card.classList.add('answered');
        if (!silent) { st.answers = st.answers || {}; st.answers[i] = j; progress.quiz[c.id] = st; saveProgress(); upd(); renderMath(card); }
      }
      if (st.answers && st.answers[i] != null) answer(st.answers[i], true);
      box.appendChild(card);
    });
    const reset = h('button', { class: 'la-b', onclick: () => { progress.quiz[c.id] = { answers: {} }; saveProgress(); const nb = buildQuiz(c); box.replaceWith(nb); nb.classList.add('in'); renderMath(nb); } }, LA.t('quiz.reset'));
    box.appendChild(reset);
    upd();
    return box;
  }

  function buildChapterNav(c) {
    const prev = LA.chapters[c.index - 1], next = LA.chapters[c.index + 1];
    const doneBtn = h('button', { class: 'la-b la-done' + (progress.done[c.id] ? ' is-done' : '') }, progress.done[c.id] ? LA.t('nav.done') : LA.t('nav.mark'));
    doneBtn.addEventListener('click', () => {
      progress.done[c.id] = !progress.done[c.id];
      if (!progress.done[c.id]) delete progress.done[c.id];
      saveProgress();
      doneBtn.classList.toggle('is-done', !!progress.done[c.id]);
      doneBtn.textContent = progress.done[c.id] ? LA.t('nav.done') : LA.t('nav.mark');
      buildSidebar(); markSidebar();
    });
    return h('footer', { class: 'la-ch-nav' },
      prev ? h('a', { class: 'nav prev', href: '#/c/' + prev.id }, h('span', null, LA.t('nav.prev')), h('b', null, `${String(prev.num).padStart(2, '0')} · ${prev.navTitle || prev.title}`)) : h('div'),
      h('div', { class: 'done-btn' }, doneBtn),
      next ? h('a', { class: 'nav next', href: '#/c/' + next.id }, h('span', null, LA.t('nav.next')), h('b', null, `${String(next.num).padStart(2, '0')} · ${next.navTitle || next.title}`)) : h('div'));
  }

  /* ============================================================ CATÁLOGO */
  LA.catLabels = LA.t('cat.labels');
  function renderCatalog(main, q0) {
    document.title = LA.t('cat.docTitle');
    const page = h('div', { class: 'la-page', style: { '--pc': 'var(--viz-c)' } });
    page.appendChild(h('div', { class: 'la-page-h' },
      h('span', { class: 'la-eyebrow' }, h('i'), LA.t('cat.eyebrow')),
      h('h1', null, LA.t('cat.h1')),
      h('p', { html: LA.t('cat.desc') })));
    const state = { q: q0 || '', r: 'all', kind: 'all', cat: 'all' };
    const input = h('input', { class: 'la-input', type: 'search', placeholder: LA.t('cat.ph'), value: state.q });
    const segR = segmented([['all', LA.t('cat.all')], ['glm', 'GLM'], ['tf', 'GLM·TensorFold'], ['qwen', 'Qwen'], ['ds', 'DeepSeek']], state.r, (v) => { state.r = v; draw(); });
    const segK = segmented([['all', LA.t('cat.everything')], ['param', LA.t('cat.params')], ['mod', LA.t('cat.mods')]], state.kind, (v) => { state.kind = v; draw(); });
    const cats = Array.from(new Set(LA.catalog.map((x) => x.cat || 'other')));
    const sel = h('select', { class: 'la-select' }, h('option', { value: 'all' }, LA.t('cat.allCats')), cats.sort().map((c) => h('option', { value: c }, LA.catLabels[c] || c)));
    sel.addEventListener('change', () => { state.cat = sel.value; draw(); });
    const cnt = h('span', { class: 'cnt' });
    input.addEventListener('input', () => { state.q = input.value; draw(); });
    page.appendChild(h('div', { class: 'la-filters' }, input, segR, segK, sel, cnt));
    const list = h('div', { class: 'la-cat-list' });
    page.appendChild(list);
    main.appendChild(page);
    function draw() {
      const terms = LA.norm(state.q).split(/\s+/).filter(Boolean);
      const items = LA.catalog.filter((x) => {
        if (state.r !== 'all' && !(x.recipes || []).includes(state.r)) return false;
        if (state.kind !== 'all' && x.kind !== state.kind) return false;
        if (state.cat !== 'all' && (x.cat || 'other') !== state.cat) return false;
        if (!terms.length) return true;
        const hay = LA.norm([x.name, x.what, x.why, x.value, x.where, x.concept, x.cat].join(' '));
        return terms.every((t) => hay.includes(t));
      });
      cnt.textContent = LA.t('cat.count', { a: items.length, b: LA.catalog.length });
      list.innerHTML = '';
      const frag = document.createDocumentFragment();
      items.slice(0, 600).forEach((x) => frag.appendChild(catItem(x)));
      list.appendChild(frag);
      if (items.length > 600) list.appendChild(h('p', { class: 'la-search-empty' }, LA.t('cat.more', { n: items.length })));
      if (!items.length) list.appendChild(h('p', { class: 'la-search-empty' }, LA.t('cat.none')));
    }
    draw();
  }
  function catItem(x) {
    const d = h('details', { class: 'la-cat-item' });
    d.appendChild(h('summary', null,
      h('div', { class: 'nm' }, h('span', { class: 'kind ' + (x.kind === 'mod' ? 'mod' : '') }, x.kind === 'mod' ? 'mod' : 'param'), x.name),
      h('div', { class: 'wt', html: x.what || '' }),
      h('div', { class: 'rbs', html: (x.recipes || []).map(LA.recipeBadge).join('') })));
    const dl = h('dl');
    const row = (k, v, isHtml) => { if (v) { dl.appendChild(h('dt', null, k)); dl.appendChild(h('dd', isHtml ? { html: v } : null, isHtml ? null : v)); } };
    row(LA.t('cat.f.value'), x.value ? `<code>${LA.esc(x.value)}</code>` : '', true);
    row(LA.t('cat.f.where'), x.where ? `<code>${LA.esc(x.where)}</code>` : '', true);
    row(LA.t('cat.f.cat'), LA.catLabels[x.cat] || x.cat);
    row(LA.t('cat.f.concept'), x.concept, true);
    row(LA.t('cat.f.why'), x.why, true);
    if (x.ch && x.ch.length) row(LA.t('cat.f.study'), x.ch.filter((id) => LA.byId[id]).map((id) => `<a href="#/c/${id}">${String(LA.byId[id].num).padStart(2, '0')} · ${LA.esc(LA.byId[id].navTitle || LA.byId[id].title)}</a>`).join(' · '), true);
    d.appendChild(h('div', { class: 'la-cat-body' }, dl));
    return d;
  }
  function segmented(opts, val, onChange) {
    const s = h('div', { class: 'la-seg' });
    opts.forEach(([v, l]) => {
      const b = h('button', { class: v === val ? 'on' : '' }, l);
      b.addEventListener('click', () => { $$('button', s).forEach((x) => x.classList.remove('on')); b.classList.add('on'); onChange(v); });
      s.appendChild(b);
    });
    return s;
  }

  /* ============================================================ GLOSSÁRIO */
  function renderGlossary(main, term) {
    document.title = LA.t('gl.docTitle');
    const page = h('div', { class: 'la-page', style: { '--pc': 'var(--viz-b)' } });
    page.appendChild(h('div', { class: 'la-page-h' },
      h('span', { class: 'la-eyebrow' }, h('i'), LA.t('gl.eyebrow')),
      h('h1', null, LA.t('gl.h1')),
      h('p', null, LA.t('gl.desc'))));
    const input = h('input', { class: 'la-input', type: 'search', placeholder: LA.t('gl.ph') });
    page.appendChild(h('div', { class: 'la-filters' }, input));
    const keys = Object.keys(LA.glossary).sort((a, b) => LA.norm(LA.glossary[a].t).localeCompare(LA.norm(LA.glossary[b].t)));
    const letters = h('div', { class: 'la-gloss-letters' });
    const grid = h('div', { class: 'la-gloss-grid' });
    page.append(letters, grid);
    main.appendChild(page);
    function draw() {
      const q = LA.norm(input.value);
      grid.innerHTML = ''; letters.innerHTML = '';
      let last = '';
      keys.forEach((k) => {
        const g = LA.glossary[k];
        if (q && !LA.norm(g.t + ' ' + g.d).includes(q)) return;
        const L = LA.norm(g.t).charAt(0).toUpperCase();
        if (L !== last) {
          last = L;
          grid.appendChild(h('div', { class: 'la-gloss-letter', id: 'gl-' + L }, L));
          letters.appendChild(h('a', { href: 'javascript:void 0', onclick: () => document.getElementById('gl-' + L).scrollIntoView({ behavior: 'smooth' }) }, L));
        }
        const ch = g.ch && LA.byId[g.ch];
        grid.appendChild(h('div', { class: 'la-gloss-item', id: 'term-' + k },
          h('h4', null, g.t), h('p', { html: g.d }),
          ch ? h('a', { class: 'go', href: '#/c/' + g.ch }, `→ ${String(ch.num).padStart(2, '0')} · ${ch.navTitle || ch.title}`) : null));
      });
    }
    input.addEventListener('input', draw);
    draw();
    renderMath(page);
    if (term) { const el = document.getElementById('term-' + term); if (el) setTimeout(() => { el.scrollIntoView({ block: 'center' }); el.style.borderColor = 'var(--mint)'; }, 50); }
  }

  /* ============================================================= TOOLTIPS */
  function setupTooltips() {
    const tip = $('#la-tip');
    let cur = null;
    document.addEventListener('mouseover', (e) => {
      const t = e.target.closest && e.target.closest('.la-term');
      if (!t || t === cur) return;
      const g = LA.glossary[t.dataset.term];
      if (!g) return;
      cur = t;
      tip.innerHTML = `<b>${LA.esc(g.t)}</b>${g.d}`;
      renderMath(tip);
      const r = t.getBoundingClientRect();
      tip.classList.add('on');
      const tw = tip.offsetWidth, th = tip.offsetHeight;
      let x = Math.min(window.innerWidth - tw - 12, Math.max(12, r.left + r.width / 2 - tw / 2));
      let y = r.top - th - 10;
      if (y < 70) y = r.bottom + 10;
      tip.style.left = x + 'px'; tip.style.top = y + 'px';
    });
    document.addEventListener('mouseout', (e) => {
      if (cur && (!e.relatedTarget || !cur.contains(e.relatedTarget))) { cur = null; tip.classList.remove('on'); }
    });
    document.addEventListener('click', (e) => {
      const t = e.target.closest && e.target.closest('.la-term');
      if (t && LA.glossary[t.dataset.term]) { tip.classList.remove('on'); LA.go('#/glossario/' + t.dataset.term); }
    });
  }

  /* ================================================================ BUSCA */
  let index = [];
  function buildSearchIndex() {
    index = [];
    LA.chapters.forEach((c) => {
      const pre = `${String(c.num).padStart(2, '0')} · `;
      index.push({ ty: LA.t('ty.chapter'), ti: pre + c.title, sn: c.subtitle || '', href: '#/c/' + c.id, hay: LA.norm(c.title + ' ' + (c.subtitle || '') + ' ' + (c.tags || []).join(' ')), w: 3 });
      const html = LA.content[c.id] || '';
      const d = document.createElement('div');
      d.innerHTML = html;
      d.querySelectorAll('script,style').forEach((n) => n.remove());
      let secId = '', secTitle = '', n = 0;
      d.querySelectorAll('h2, h3, p, li, td, figcaption').forEach((el) => {
        const txt = (el.textContent || '').replace(/\s+/g, ' ').trim();
        if (!txt) return;
        if (el.tagName === 'H2' || el.tagName === 'H3') {
          if (el.tagName === 'H2') n++;
          secId = el.id || (el.tagName === 'H2' ? 's' + n + '-' + LA.norm(txt).replace(/[^a-z0-9]+/g, '-').slice(0, 40) : secId);
          secTitle = txt;
          index.push({ ty: LA.t('ty.section'), ti: txt, sn: pre + c.title, href: `#/c/${c.id}/${el.tagName === 'H2' ? secId : ''}`, hay: LA.norm(txt), w: 2 });
        } else if (txt.length > 30) {
          index.push({ ty: LA.t('ty.excerpt'), ti: secTitle || c.title, sn: txt.slice(0, 220), href: `#/c/${c.id}/${secId}`, hay: LA.norm(txt), w: 1, full: txt });
        }
      });
    });
    Object.keys(LA.glossary).forEach((k) => { const g = LA.glossary[k]; index.push({ ty: LA.t('ty.glossary'), ti: g.t, sn: LA.stripHtml(g.d).slice(0, 160), href: '#/glossario/' + k, hay: LA.norm(g.t + ' ' + LA.stripHtml(g.d)), w: 2.5 }); });
    LA.catalog.forEach((x) => { index.push({ ty: x.kind === 'mod' ? LA.t('ty.mod') : LA.t('ty.param'), ti: x.name, sn: LA.stripHtml(x.what || '').slice(0, 160), href: '#/catalogo/' + encodeURIComponent(x.name), hay: LA.norm(x.name + ' ' + LA.stripHtml(x.what || '') + ' ' + (x.value || '')), w: 2.2 }); });
    LA.searchIndex = index;
  }
  let sel = 0, results = [];
  function openSearch() { $('#la-search').classList.add('on'); const i = $('#la-search-in'); i.value = ''; runSearch(); setTimeout(() => i.focus(), 20); }
  function closeSearch() { $('#la-search').classList.remove('on'); }
  LA.openSearch = openSearch;
  function runSearch() {
    const q = LA.norm($('#la-search-in').value.trim());
    const box = $('#la-search-res');
    box.innerHTML = '';
    if (!q) {
      results = LA.chapters.slice(0, 12).map((c) => ({ ty: LA.t('ty.chapter'), ti: `${String(c.num).padStart(2, '0')} · ${c.title}`, sn: c.subtitle || '', href: '#/c/' + c.id }));
    } else {
      const terms = q.split(/\s+/).filter(Boolean);
      results = index.map((it) => {
        if (!terms.every((t) => it.hay.includes(t))) return null;
        let s = it.w;
        if (LA.norm(it.ti).includes(q)) s += 3;
        if (LA.norm(it.ti).startsWith(q)) s += 2;
        return Object.assign({ s }, it);
      }).filter(Boolean).sort((a, b) => b.s - a.s).slice(0, 40);
      results.forEach((r) => {
        if (r.full) {
          const i = LA.norm(r.full).indexOf(terms[0]);
          r.sn = (i > 60 ? '…' : '') + r.full.slice(Math.max(0, i - 60), i + 160);
        }
      });
    }
    sel = 0;
    if (!results.length) { box.appendChild(h('div', { class: 'la-search-empty' }, LA.t('search.empty'))); return; }
    const hl = (s) => {
      let out = LA.esc(s);
      if (q) q.split(/\s+/).filter((t) => t.length > 1).forEach((t) => {
        const re = new RegExp('(' + t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'ig');
        out = out.replace(re, '<mark>$1</mark>');
      });
      return out;
    };
    results.forEach((r, i) => {
      const a = h('a', { href: r.href, class: i === 0 ? 'sel' : '', onclick: closeSearch },
        h('span', { class: 'ty' }, r.ty), h('span', { class: 'ti', html: hl(r.ti) }), r.sn ? h('span', { class: 'sn', html: hl(r.sn) }) : null);
      box.appendChild(a);
    });
  }
  function searchKeys(e) {
    const links = $$('#la-search-res a');
    if (e.key === 'ArrowDown') { sel = Math.min(links.length - 1, sel + 1); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { sel = Math.max(0, sel - 1); e.preventDefault(); }
    else if (e.key === 'Enter') { if (links[sel]) { location.hash = links[sel].getAttribute('href'); closeSearch(); } return; }
    else if (e.key === 'Escape') { closeSearch(); return; }
    else return;
    links.forEach((l, i) => l.classList.toggle('sel', i === sel));
    if (links[sel]) links[sel].scrollIntoView({ block: 'nearest' });
  }

  function setupKeys() {
    document.addEventListener('keydown', (e) => {
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === 'k') { e.preventDefault(); openSearch(); }
      else if (e.key === 'Escape') { closeSearch(); $$('.la-fig.fullscreen').forEach((f) => f.classList.remove('fullscreen')); }
      else if (!mod && !e.altKey && document.activeElement === document.body && LA.route && LA.route.type === 'chapter') {
        const c = LA.byId[LA.route.id];
        if (e.key === 'ArrowRight' && e.shiftKey && LA.chapters[c.index + 1]) LA.go('#/c/' + LA.chapters[c.index + 1].id);
        if (e.key === 'ArrowLeft' && e.shiftKey && LA.chapters[c.index - 1]) LA.go('#/c/' + LA.chapters[c.index - 1].id);
      }
    });
  }

  /* ====================================================== CONTEXTO (copilot) */
  LA.getContext = function () {
    const r = LA.route || { type: 'home' };
    const ctx = { type: r.type, title: LA.t('ctx.home'), part: '', section: '', sectionText: '', pageText: '', outline: '' };
    ctx.outline = LA.parts.map((p) => {
      const chs = LA.chapters.filter((c) => c.part === p.n);
      return chs.length ? `[${LA.t('mod.label')} ${LA.moduleOf(p.n).n}] ${LA.t('ch.part')} ${p.roman || p.n} — ${p.title}: ` + chs.map((c) => `${String(c.num).padStart(2, '0')} ${c.title}`).join('; ') : '';
    }).filter(Boolean).join('\n');
    if (r.type === 'chapter') {
      const c = LA.byId[r.id];
      const part = LA.partOf(c.part);
      ctx.title = `${LA.t('ch.chapter')} ${String(c.num).padStart(2, '0')} — ${c.title}`;
      ctx.subtitle = c.subtitle || '';
      ctx.part = `${LA.t('ch.part')} ${part.roman || part.n} — ${part.title}`;
      ctx.recipes = c.recipes.map((x) => LA.recipeName[x]).join(', ');
      const art = $('#la-article');
      if (art) {
        ctx.pageText = (art.innerText || '').replace(/\n{3,}/g, '\n\n');
        const secId = LA.currentSection;
        const sec = secId && document.getElementById(secId);
        if (sec) {
          ctx.section = sec.textContent.replace('#', '').trim();
          const container = sec.closest('section') || sec.parentElement;
          ctx.sectionText = container && container !== art ? container.innerText : '';
        }
      }
      ctx.summary = c.summary.map(LA.stripHtml).join(' | ');
    } else if (r.type === 'catalogo') {
      ctx.title = LA.t('cat.h1');
      ctx.pageText = ($('#la-main') || document.body).innerText.slice(0, 20000);
    } else if (r.type === 'glossario') {
      ctx.title = LA.t('gl.h1');
      ctx.pageText = ($('#la-main') || document.body).innerText.slice(0, 20000);
    } else {
      ctx.pageText = ($('#la-main') || document.body).innerText.slice(0, 8000);
    }
    return ctx;
  };
})();
