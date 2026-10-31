LA.chapter({
  id: 'api-tools-estruturado',
  part: 6,
  order: 34,
  title: 'The API: chat, Responses, tools and structured outputs',
  navTitle: 'API, tools & JSON',
  subtitle: 'How JSON becomes tokens and turns back into JSON: templates, separated reasoning, stop strings, tool calling, grammars that guarantee the shape of the output, Responses API state and the security of port 8888.',
  minutes: 55,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['OpenAI API', 'SSE', 'chat template', 'reasoning parser', 'tool calling', 'tool_choice', 'XGrammar', 'Responses API', 'api-key', 'TensorFold'],
  goals: [
    'Follow a request through the vLLM API: <b>template → tokens → engine → detokenizer → parsers → SSE</b>, and know what each field (<code>finish_reason</code>, <code>usage</code>, <code>cached_tokens</code>) means',
    'Configure <b>reasoning and tools</b>: <code>chat_template_kwargs</code>, <code>--reasoning-parser</code>, <code>--tool-call-parser</code>, <code>tool_choice</code>, and understand why stops and temperature need care at the <code>&lt;/think&gt;</code> boundary and on syntax tokens, comparing them with the choices of the TensorFold lane (whole calls, returned reasoning, sanitized history)',
    'Explain how a <b>grammar</b> (XGrammar) masks tokens at every step, and why speculation + grammar requires the termination and advance backports',
    'Choose between <b>stateless</b> and <b>stateful</b> Responses (LRU store) and set up the API\'s <b>security</b>: keys, unguarded routes, bind and secrets in logs',
    'Size an <b>agentic client</b> (contextWindow, maxTokens, subagents) against the recipe\'s limits',
  ],
  prereqs: ['geracao-amostragem', 'vllm-arquitetura'],
  summary: [
    'vLLM exposes an OpenAI-compatible server (<code>/v1/models</code>, <code>/v1/chat/completions</code>, <code>/v1/completions</code>, <code>/v1/responses</code>). Chat is <b>stateless</b>, and the prefix cache makes resending the history cheap. <code>--served-model-name</code>, <code>--host</code> and <code>--port 8888</code> define how the model shows up.',
    'Templates and <code>chat_template_kwargs</code> decide the text the model sees. Each family has its own names and values for thinking and effort (GLM low/high→otherwise Max, Qwen low/medium/xhigh, DeepSeek low/high/max). Anything that changes the start of the prompt costs prefix cache.',
    'The <b>reasoning parser</b> separates <code>reasoning</code> from <code>content</code> and marks the boundary at which stops, the grammar and tool calls take effect. Client stops stay <b>dormant</b> until <code>&lt;/think&gt;</code> (GLM and DeepSeek).',
    'Tool calling is text: the model writes markup (GLM XML, <code>qwen3_coder</code>, DSML) and the streaming parser, a state machine, converts it into <code>tool_calls</code>. On GLM, <code>tool_choice:"none"</code> masks the <code>&lt;tool_call&gt;</code> token via <code>bad_words</code>. DeepSeek\'s #191 prefers HTTP 500 to a wrong 200.',
    'Temperature applied to <b>syntax</b> tokens corrupts calls at T > 0, because the error multiplies along the call. A grammar (a bitmask of \\(\\lceil V/32 \\rceil\\) words per row) makes the structural positions deterministic.',
    'Grammar × speculation: batches of k+1 tokens run past the end of the grammar and the end of the reasoning. Backports #52805, #53046 and #44993 make the matcher stop at the terminal and advance through the right drafts.',
    'Stateful Responses keeps its state in process memory (an LRU of 256 on DeepSeek, lost on restart). The Bearer key only protects <code>/v1</code>, <code>/v2</code> and <code>/inference</code>: <code>/invocations</code>, <code>/tokenize</code>, <code>/metrics</code> and the reset routes stay open, so network-level control is mandatory.',
    'In the TensorFold lane <span class="la-rb tf"></span>, the same API changes at the edges: tool calls only go out <b>whole</b> (with an empty delta every 2 s), the server puts back the reasoning the client dropped, leaves a broken history call out of the prompt instead of returning 400, cuts the drafts the grammar rejects before verifying and answers <code>context_length_exceeded</code> with <code>param</code>. In exchange, there is no authentication, stops apply inside the thinking and <code>logprobs</code>/<code>n</code> are refused.',
  ],
  quiz: [
    { q: 'A harness sends <code>stop: ["Question:"]</code> to a reasoning model whose prompt ends in <code>&lt;think&gt;</code>. On stock vLLM, the response comes back with <code>content: null</code>. Why?', o: ['The model does not support stop strings', 'The stop was matched against the whole stream and fired inside the thinking, before <code>&lt;/think&gt;</code>', '<code>max_tokens</code> is too small', 'The reasoning parser erased the content'], a: 1, e: 'The thinking repeats the harness\'s markers. The suppression patch keeps the stops dormant until <code>&lt;/think&gt;</code> (and adjusts <code>stop_check_offset</code> when the marker lands in the middle of a speculative batch). With it, the measured case returned <code>17 + 25 = 42</code>.' },
    { q: 'Why does GLM enforce <code>tool_choice:"none"</code> with <code>bad_words=["&lt;tool_call&gt;"]</code> instead of using <code>--exclude-tools-when-tool-choice-none</code>?', o: ['Because bad_words is faster on the GPU', 'Because the upstream flag removes the tools block from the prompt and destroys the prefix cache\'s shared prefix', 'Because GLM does not support tool_choice', 'Because bad_words works with any number of tokens'], a: 1, e: 'Keeping the tools in the prompt preserves the common prefix across requests that declare the same tools, whether <code>tool_choice</code> is <code>none</code> or <code>auto</code> (the doc mentions a difference on the order of 168 vs 17 prompt tokens). The mask acts at every position, including the drafts, and only works well because <code>&lt;tool_call&gt;</code> is a single token (gate C of the preflight).' },
    { q: 'With the same model (FP8 weights on vLLM, ~2-bit GGUF experts on ds4), at temperature 1.0, DeepSeek on vLLM loses tool-calling points compared with the ds4 engine. What is the recipe\'s explanation?', o: ['The NVFP4 KV degrades the context', 'MTP changes the output', 'vLLM also samples the DSML syntax tokens at T=1.0, and a single wrong structural token turns the call into text', 'ds4 uses higher-precision weights'], a: 2, e: 'ds4 forces T=0 on the syntax and only samples the payload. It is not a weights effect: if the weights dominated, vLLM, with FP8, would beat the ~2-bit ds4. Since the chance of an entire call coming out right is the product of the chances of each structural token, small per-token losses turn into large per-call losses. Mitigations: T=0 in the benchmark, structural tags with a grammar, or a logits processor.' },
    { q: 'How many bytes does one row of the grammar bitmask take for the GLM vocabulary (154,880 tokens)?', o: ['154,880 B', '19,360 B', '4,840 B', '38,720 B'], a: 1, e: 'One bit per token, packed into 32-bit integers: ⌈154,880/32⌉ = 4,840 words × 4 B = 19,360 B per row (one row per sequence and per draft position).' },
    { q: 'The DeepSeek recipe needs to accept two API keys. What happens with <code>--api-key k1 --api-key k2</code>?', o: ['Both keys work', 'Only k2 works: the flag is an nargs list and the last occurrence overwrites the earlier one', 'The server does not start', 'Only k1 works'], a: 1, e: 'That is why the compose file builds a single <code>--api-key k1 k2 …</code> from <code>DSPARK_API_KEYS</code>. And remember: the key only protects <code>/v1</code>, <code>/v2</code> and <code>/inference</code>.' },
    { q: '<code>max_tokens</code> runs out in the middle of a tool call\'s arguments. With hotfix #55, what does the client receive?', o: ['finish_reason "tool_calls" with partial arguments', 'finish_reason "length", and tool calls whose arguments do not parse are discarded at the end', 'HTTP 400', 'finish_reason "stop"'], a: 1, e: 'Before, vLLM reported <code>tool_calls</code> with invalid JSON and the harness\'s next turn got HTTP 400. Fragments already sent in the stream cannot be undone (append-only), and the protection is the correct <code>finish_reason</code>.' },
    { q: 'In the GLM TensorFold lane, a streamed reply is cut by <code>max_tokens</code> in the middle of a tool call\'s arguments. What does the client receive?', o: ['The argument fragments already sent and <code>finish_reason: "tool_calls"</code>', 'No part of the call: it would only go out whole at <code>&lt;/tool_call&gt;</code>, and the reply ends with <code>length</code>', 'HTTP 400 with <code>context_length_exceeded</code>', 'The call completed by the server with a grammar'], a: 1, e: 'Patch 0053 holds each call until it closes and sends it in a single delta, with empty deltas every 2 s while it waits, for clients that drop idle connections. A call the reply ends inside never goes out. With fragment streaming (the vLLM design and patch 0036\'s), the client would already hold half the JSON, and only the <code>finish_reason</code> could be fixed. The context 400 happens before generating, when prompt + <code>max_tokens</code> do not fit the window, and the server does not complete arguments: the model writes them.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const $ = (s) => root.querySelector(s);

    /* ================= local utilities ================= */
    // JetBrains Mono has ligatures (<| becomes ◁): we turn them off in the chapter DOM and on the canvas.
    root.querySelectorAll('code, pre').forEach((el) => { el.style.fontVariantLigatures = 'none'; });
    const fig = (el, draw, o) => V.fig(el, (g, ...a) => { try { g.textRendering = 'optimizeSpeed'; } catch (err) { /* browser without support */ } draw(g, ...a); }, o);
    const FC = {};
    const font = (sz, f, wt) => { const k = sz + '|' + f + '|' + wt; return FC[k] || (FC[k] = V.font(sz, f, wt)); };
    const measure = (g, s, sz, f = 'mono', wt = 500) => { g.font = font(sz, f, wt); return g.measureText(String(s)).width; };
    const clip = (g, s, maxW, sz, f = 'mono', wt = 400) => {
      s = String(s);
      if (maxW <= 10) return '';
      if (measure(g, s, sz, f, wt) <= maxW) return s;
      let lo = 0, hi = s.length;
      while (lo < hi) { const m = (lo + hi + 1) >> 1; if (measure(g, s.slice(0, m) + '…', sz, f, wt) <= maxW) lo = m; else hi = m - 1; }
      return s.slice(0, lo) + '…';
    };
    function pill(g, s, x, y, o = {}) {
      const C = V.C();
      const sz = o.size || 12, f = o.font || 'mono', wt = o.weight || 500, px = o.px == null ? 7 : o.px, hh = o.h || sz + 10;
      const w = Math.max(o.minW || 0, measure(g, s, sz, f, wt) + px * 2);
      if (o.dry) return w;
      if (o.glow) V.glow(g, x + w / 2, y + hh / 2, Math.max(w, hh) * 0.75, o.glow, 0.45);
      if (o.dash) g.setLineDash(o.dash);
      V.box(g, x, y, w, hh, { r: o.r == null ? 6 : o.r, fill: o.fill || V.alpha(C.muted, 0.1), stroke: o.stroke, lw: o.lw || 1.2 });
      g.setLineDash([]);
      V.text(g, s, x + w / 2, y + hh / 2 + 0.5, { size: sz, font: f, weight: wt, color: o.color || C.ink, align: 'center', base: 'middle', alpha: o.alpha });
      if (o.strike) V.line(g, x + 4, y + hh / 2, x + w - 4, y + hh / 2, { color: o.strike, lw: 1.4 });
      return w;
    }
    function flow(g, items, x0, y0, maxW, o = {}) {
      const sz = o.size || 12, gap = o.gap == null ? 5 : o.gap, ph = o.h || sz + 10, rh = ph + (o.rowGap == null ? 6 : o.rowGap);
      let x = x0, y = y0;
      const rects = [];
      items.forEach((it) => {
        const w = pill(g, it.s, 0, 0, Object.assign({}, o, it, { dry: true }));
        if (x > x0 && x + w > x0 + maxW) { x = x0; y += rh; }
        rects.push({ x, y, w, h: ph });
        x += w + gap;
      });
      if (!o.dry) items.forEach((it, i) => pill(g, it.s, rects[i].x, rects[i].y, Object.assign({}, o, it)));
      return { rects, bottom: y + ph };
    }
    function wrapLines(g, s, maxW, sz, f = 'sans', wt = 400) {
      const words = String(s).split(' ');
      const lines = [];
      let cur = '';
      words.forEach((wd) => { const tst = cur ? cur + ' ' + wd : wd; if (cur && measure(g, tst, sz, f, wt) > maxW) { lines.push(cur); cur = wd; } else cur = tst; });
      if (cur) lines.push(cur);
      return lines;
    }
    function wrapChars(g, s, maxW, sz, f = 'mono', wt = 400) {
      const out = [];
      let cur = '';
      for (const ch of String(s)) { if (cur && measure(g, cur + ch, sz, f, wt) > maxW) { out.push(cur); cur = ch; } else cur += ch; }
      if (cur) out.push(cur);
      return out;
    }
    function textBlock(g, s, x, y, maxW, o = {}) {
      const sz = o.size || 12.5, lh = o.lh || Math.round(sz * 1.42);
      const L = wrapLines(g, s, maxW, sz, o.font || 'sans', o.weight || 400);
      L.forEach((l, i) => V.text(g, l, x, y + i * lh, Object.assign({ size: sz }, o)));
      return y + L.length * lh;
    }
    function fitH(fig, hd, hm, bp = 600) {
      const st = fig.querySelector('.la-fig-stage');
      const apply = () => { const w = fig.clientWidth || 800; const want = (w < bp ? hm : hd) + 'px'; if (st.style.height !== want) st.style.height = want; };
      apply();
      window.addEventListener('resize', apply);
      LA.onCleanup(() => window.removeEventListener('resize', apply));
    }
    function lock(g, x, y, s, col) { // drawn padlock (no emoji)
      g.strokeStyle = col; g.lineWidth = 1.6;
      g.beginPath(); g.arc(x, y - s * 0.25, s * 0.32, Math.PI, 0); g.stroke();
      V.box(g, x - s * 0.45, y - s * 0.25, s * 0.9, s * 0.7, { r: 2, fill: col });
    }
    function check(g, x, y, s, col) { g.strokeStyle = col; g.lineWidth = 2.2; g.beginPath(); g.moveTo(x - s * 0.5, y); g.lineTo(x - s * 0.1, y + s * 0.4); g.lineTo(x + s * 0.55, y - s * 0.45); g.stroke(); }
    function cross(g, x, y, s, col) { g.strokeStyle = col; g.lineWidth = 2.2; g.beginPath(); g.moveTo(x - s / 2, y - s / 2); g.lineTo(x + s / 2, y + s / 2); g.moveTo(x + s / 2, y - s / 2); g.lineTo(x - s / 2, y + s / 2); g.stroke(); }

    /* ================= Fig 1: journey of a request ================= */
    const f1 = $('#fig-jornada');
    fitH(f1, 380, 400);
    const J = [
      { t: 'The request arrives', d: '<code>POST /v1/chat/completions</code> with <code>model</code>, <code>messages</code> and <code>stream</code>. If there is an <code>--api-key</code> and the route is under <code>/v1</code>, <code>/v2</code> or <code>/inference</code>, the middleware requires <code>Authorization: Bearer</code>.' },
      { t: 'Template', d: 'The API server renders <code>messages</code> + <code>tools</code> into a single text (Jinja or, on DeepSeek, the checkpoint\'s Python encoder), applying <code>chat_template_kwargs</code> (thinking, effort).' },
      { t: 'Tokenization', d: 'The text becomes IDs. Markers such as <code>&lt;think&gt;</code> are a single ID (154841 on GLM). Ordinary text becomes BPE pieces (<code>hello</code> = 14978, <code>world</code> = 1879).' },
      { t: 'EngineCore', d: 'The request enters the scheduler queue. At every step the GPU returns logits, and the masks (<code>bad_words</code>, grammar) zero out the forbidden tokens <i>before</i> sampling.' },
      { t: 'Detokenization', d: 'The IDs turn back into text incrementally. EOS and stop strings are checked here, in a window that covers stops split across two tokens.' },
      { t: 'Parsers', d: 'The reasoning parser moves <code>&lt;think&gt;…&lt;/think&gt;</code> into <code>reasoning</code>, and the tool parser turns markup into <code>tool_calls</code>. Calls are only looked for in <code>content</code>.' },
      { t: 'Response', d: 'With <code>stream:true</code>, each piece goes out as an SSE event <code>data: {…}</code>. The last one carries <code>finish_reason</code> and, if requested, <code>usage</code>. Then comes <code>data: [DONE]</code>.' },
    ];
    const st1 = UI.stepper(f1, J, { autoplay: 6500 });
    const stNames = ['HTTP', 'Template', 'Tokens', 'Engine', 'Detok', 'Parsers', 'SSE'];
    function envelope(g, x, y, C) {
      V.box(g, x - 10, y - 7, 20, 14, { r: 2, fill: C.d, stroke: C.ink, lw: 1 });
      g.strokeStyle = C.bg; g.lineWidth = 1.3; g.beginPath(); g.moveTo(x - 9, y - 6); g.lineTo(x, y + 1); g.lineTo(x + 9, y - 6); g.stroke();
    }
    function detail1(g, k, x, y, W, H, t, e, C, narrow) {
      const sz = narrow ? 11 : 12.5;
      if (k === 0) {
        const L = ['POST /v1/chat/completions', 'Authorization: Bearer sk-…', '{ "model": "GLM-5.3-Flash-EXL3",', '  "messages": [{"role": "user",', '                "content": "hello world"}],', '  "stream": true }'];
        let budget = Math.floor(e * 70);
        const cw = narrow ? W : W * 0.62;
        L.forEach((l, i) => { const s = l.slice(0, Math.max(0, budget)); budget -= l.length; V.text(g, clip(g, s, cw, sz), x, y + 14 + i * (sz + 8), { size: sz, font: 'mono', color: i === 0 ? C.d : i === 1 ? C.b : C.ink }); });
        const gx = narrow ? x + W / 2 : x + W * 0.82, gy = narrow ? y + H - 34 : y + H / 2;
        const ok = e > 1.6;
        V.box(g, gx - 78, gy - 24, 156, 48, { r: 10, fill: V.alpha(ok ? C.a : C.muted, 0.14), stroke: ok ? C.a : C.muted });
        lock(g, gx - 56, gy + 2, 16, ok ? C.a : C.muted);
        V.text(g, 'Bearer guard', gx + 8, gy - 4, { size: 11.5, align: 'center', color: C.ink, weight: 600 });
        V.text(g, ok ? '/v1 → key ok' : 'checking…', gx + 8, gy + 12, { size: 11, font: 'mono', align: 'center', color: ok ? C.a : C.muted });
      } else if (k === 1) {
        const bx = x, by = y + 6, bwid = narrow ? W : Math.min(220, W * 0.34);
        V.box(g, bx, by, bwid, 40, { r: 12, fill: V.alpha(C.b, 0.12), stroke: V.alpha(C.b, 0.6) });
        V.text(g, 'user: "hello world"', bx + 12, by + 24, { size: sz, font: 'mono', color: C.ink });
        const items = [
          { s: '[gMASK]', fill: V.alpha(C.b, 0.22), stroke: C.b }, { s: '<sop>', fill: V.alpha(C.b, 0.22), stroke: C.b }, { s: '<|user|>', fill: V.alpha(C.b, 0.22), stroke: C.b },
          { s: 'hello world', fill: V.alpha(C.muted, 0.12) }, { s: '<|assistant|>', fill: V.alpha(C.b, 0.22), stroke: C.b }, { s: '<think>', fill: V.alpha(C.a, 0.25), stroke: C.a },
        ];
        const n = Math.min(items.length, 1 + Math.floor(e * 2.4));
        const fx = narrow ? x : bx + bwid + 46, fy = narrow ? by + 64 : by + 8;
        if (!narrow) V.arrow(g, bx + bwid + 8, by + 20, fx - 8, by + 20, { color: C.b, lw: 2 });
        else V.arrow(g, x + 30, by + 44, x + 30, fy - 6, { color: C.b, lw: 2 });
        flow(g, items.slice(0, n), fx, fy, x + W - fx, { size: sz });
        V.text(g, 'the template (Jinja) decides every marker and every line break', x, y + H - 4, { size: 11.5, color: C.muted });
      } else if (k === 2) {
        const tk = [['[gMASK]', 'special'], ['<sop>', 'special'], ['<|user|>', 'special'], ['hello', '14978'], [' world', '1879'], ['<|assistant|>', 'special'], ['<think>', '154841']];
        const items = tk.map(([s], i) => { const on = e > 0.25 + i * 0.28; return { s, fill: V.alpha(on ? C.f : C.muted, on ? 0.2 : 0.08), stroke: on ? C.f : undefined }; });
        const lay = flow(g, items, x, y + 8, W, { size: sz + 0.5, rowGap: 26, gap: 8 });
        lay.rects.forEach((r, i) => { const on = e > 0.25 + i * 0.28; V.text(g, on ? tk[i][1] : '·', r.x + r.w / 2, r.y + r.h + 14, { size: 11.5, font: 'mono', align: 'center', color: on ? C.f : C.muted, weight: 600 }); });
        textBlock(g, 'special markers become a single ID; the model never sees "role", only these numbers', x, Math.min(y + H - 4, lay.bottom + 44), W, { size: 11.5, color: C.muted });
      } else if (k === 3) {
        const qx = x, qy = y + 10;
        ['req A', 'req B', 'you'].forEach((s, i) => pill(g, s, qx + i * (narrow ? 50 : 62), qy, { size: 11, fill: V.alpha(i === 2 ? C.d : C.muted, 0.2), stroke: i === 2 ? C.d : undefined }));
        const gx = narrow ? x + W * 0.72 : x + 230, gy = qy + 12;
        const pulse = 0.5 + 0.5 * Math.sin(t * 6);
        V.glow(g, gx, gy, 30 + 6 * pulse, C.a, 0.5);
        V.box(g, gx - 44, gy - 16, 88, 32, { r: 8, fill: V.alpha(C.a, 0.22), stroke: C.a, label: 'GPU · forward', size: 11.5 });
        const bxs = narrow ? x : gx + 70, bys = narrow ? qy + 50 : y + 4, bw = narrow ? W : x + W - bxs, bh = narrow ? H - 96 : H - 44;
        const lg = [2.2, 3.4, 1.4, 2.9, 0.8, 1.9, 1.1, 2.5], lab = ['The', '<tool_call>', 'A', 'Sure', 'It', '42', '\\n', 'No'];
        const nb = lg.length, gw = bw / nb;
        const pick = Math.floor(t / 1.3) % 2 === 0 ? 3 : 0;
        lg.forEach((v, i) => {
          const bhh = (v / 3.6) * (bh - 30), bx = bxs + i * gw + 4, byy = bys + bh - 18 - bhh;
          const masked = i === 1;
          V.box(g, bx, byy, gw - 8, bhh, { r: 3, fill: masked ? V.alpha(C.red, 0.12) : V.alpha(i === pick ? C.c : C.a, i === pick ? 0.85 : 0.45), stroke: masked ? C.red : undefined });
          if (masked) cross(g, bx + (gw - 8) / 2, byy + bhh / 2, 10, C.red);
          V.text(g, clip(g, lab[i], gw - 2, 11), bx + (gw - 8) / 2, bys + bh - 4, { size: 11, font: 'mono', align: 'center', color: masked ? C.red : C.muted });
        });
        V.text(g, 'logits → mask → sampling', bxs, bys - (narrow ? 4 : -2) + (narrow ? 0 : 0), { size: 11, color: C.muted });
      } else if (k === 4) {
        const toks = ['17', ' +', ' 25', ' =', ' 42'];
        const n = Math.min(toks.length, Math.floor(e * 2.2) + 1);
        let px = x;
        const xs = [];
        toks.forEach((s, i) => { const w = pill(g, i < n ? s.trim() || '␣' : '·', px, y + 10, { size: sz + 2, minW: 34, fill: V.alpha(i < n ? C.e : C.muted, i < n ? 0.22 : 0.08), stroke: i === n - 1 ? C.e : undefined }); xs.push([px, px + w]); px += w + 8; });
        const txt = toks.slice(0, n).join('');
        V.text(g, 'text: "' + txt + '"', x, y + 78, { size: sz + 3, font: 'mono', color: C.ink, weight: 600 });
        const a0 = xs[Math.max(0, n - 2)][0], a1 = xs[n - 1][1];
        V.line(g, a0, y + 46, a1, y + 46, { color: C.c, lw: 2.5 });
        V.line(g, a0, y + 42, a0, y + 50, { color: C.c, lw: 2 }); V.line(g, a1, y + 42, a1, y + 50, { color: C.c, lw: 2 });
        V.text(g, 'stop window (new chars + the |s|−1 before them)', x, y + 102, { size: 11.5, color: C.c });
        textBlock(g, 'UTF-8 bytes only become a character once complete; the detokenizer keeps offsets between steps', x, y + 126, W, { size: 11.5, color: C.muted });
      } else if (k === 5) {
        const toks = ['<think>', 'Add', ' 17', ' and 25', '</think>', '42'];
        const u = (t * 0.45) % 1;
        const fx = x + 6, fy = y + H * 0.45;
        const splitX = narrow ? x + W * 0.35 : x + W * 0.38;
        V.line(g, fx, fy, splitX, fy, { color: C.muted, lw: 2 });
        const lanes = [['reasoning', C.b, '"Add 17 and 25"'], ['content', C.a, '"42"'], ['tool_calls', C.d, '[]']];
        lanes.forEach(([nm, col, val], i) => {
          const ly = y + 14 + i * (H - 28) / 2;
          const lab = clip(g, nm + ': ' + val, x + W - splitX - 40, 11.5, 'mono', 500);
          const pw5 = pill(g, lab, 0, 0, { size: 11.5, dry: true });
          const lx5 = narrow ? x + W - pw5 : x + W - Math.max(pw5, 244);
          V.arrow(g, splitX, fy, lx5 - 4, ly, { color: V.alpha(col, 0.8), lw: 2, curve: (i - 1) * -10 });
          pill(g, lab, lx5, ly - 11, { size: 11.5, fill: V.alpha(col, 0.2), stroke: col });
        });
        // conveyor: tokens in a row, no overlap, entering the splitter
        const tw5 = toks.map((s) => pill(g, s, 0, 0, { size: 11, dry: true }));
        const cum = [];
        tw5.reduce((acc, wv, i) => { cum[i] = acc; return acc + wv + 6; }, 0);
        const loopL = splitX - fx + cum[toks.length - 1] + tw5[toks.length - 1] + 40;
        const off = (t * 55) % loopL;
        void u;
        toks.forEach((s, i) => {
          const px2 = fx + off - cum[i] - tw5[i];
          if (px2 < fx - 2 || px2 + tw5[i] > splitX) return;
          const col = i === 4 ? C.d : i < 4 ? C.b : C.a;
          pill(g, s, px2, fy - 24, { size: 11, fill: V.alpha(col, 0.3), stroke: col });
        });
      } else {
        const L = ['data: {"delta":{"role":"assistant"}}', 'data: {"delta":{"reasoning":"Add 17 and 25"}}', 'data: {"delta":{"content":"42"}}', 'data: {"delta":{},"finish_reason":"stop"}', 'data: {"choices":[],"usage":{…}}', 'data: [DONE]'];
        const n = Math.min(L.length, Math.floor(e * 1.6) + 1);
        const lh = Math.min(sz + 12, (H - 4) / L.length);
        L.slice(0, n).forEach((l, i) => {
          const yy = y + 4 + i * lh;
          V.box(g, x, yy, W, lh - 4, { r: 6, fill: V.alpha(i === n - 1 ? C.d : C.muted, i === n - 1 ? 0.16 : 0.07) });
          V.text(g, clip(g, l, W - 16, sz), x + 8, yy + (lh - 4) / 2 + 1, { size: sz, font: 'mono', base: 'middle', color: i === 3 ? C.c : C.ink });
        });
      }
    }
    fig(f1, (g, w, h, t) => {
      const C = V.C();
      const narrow = w < 600;
      const cols = [C.d, C.b, C.f, C.a, C.e, C.c, C.d];
      const P = [];
      if (!narrow) for (let i = 0; i < 7; i++) P.push([28 + (w - 56) * (i + 0.5) / 7, 44]);
      else { for (let i = 0; i < 4; i++) P.push([16 + (w - 32) * (i + 0.5) / 4, 36]); for (let i = 0; i < 3; i++) P.push([16 + (w - 32) * (i + 1) / 4, 88]); }
      const bw = narrow ? Math.min(78, (w - 32) / 4 - 8) : Math.min(104, (w - 56) / 7 - 10), bh = 32;
      const k = st1.i, e = st1.elapsed();
      for (let i = 0; i < 6; i++) { const a = P[i], b = P[i + 1]; V.line(g, a[0], a[1], b[0], b[1], { color: V.alpha(C.muted, i < k ? 0.75 : 0.25), lw: i < k ? 2 : 1.2, dash: [4, 5], dashOffset: -t * 18 }); }
      P.forEach((p, i) => {
        const on = i === k, done = i < k;
        if (on) V.glow(g, p[0], p[1], 44, cols[i], 0.5);
        V.box(g, p[0] - bw / 2, p[1] - bh / 2, bw, bh, { r: 9, fill: V.alpha(cols[i], on ? 0.3 : done ? 0.14 : 0.06), stroke: on ? cols[i] : V.alpha(cols[i], done ? 0.6 : 0.3), lw: on ? 2 : 1 });
        V.text(g, stNames[i], p[0], p[1] + 1, { size: narrow ? 11 : 12, weight: on ? 700 : 500, align: 'center', base: 'middle', color: on || done ? C.ink : C.muted });
      });
      const u = V.ease.inOut(V.clamp(e / 0.9, 0, 1));
      const from = P[Math.max(0, k - 1)], to = P[k];
      const ex = V.lerp(from[0], to[0], k === 0 ? 1 : u), ey = V.lerp(from[1], to[1], k === 0 ? 1 : u) - bh / 2 - 9;
      envelope(g, ex, Math.max(8, ey + Math.sin(t * 4) * 1.5), C);
      const py = narrow ? 118 : 82, px = 12, pw = w - 24, ph = h - py - 10;
      V.box(g, px, py, pw, ph, { r: 12, fill: V.alpha(C.panel, 0.75), stroke: V.alpha(cols[k], 0.55) });
      V.text(g, (k + 1) + ' · ' + J[k].t.toUpperCase(), px + 14, py + 18, { size: 11, font: 'mono', color: cols[k], weight: 600 });
      detail1(g, k, px + 14, py + 26, pw - 28, ph - 34, t, e, C, narrow);
    });

    /* ================= Fig 2: SSE conveyor belt ================= */
    const f2 = $('#fig-sse');
    fitH(f2, 360, 420);
    const S2 = { sc: 'texto', usage: true, p55: true, t0: 0, reset: true, first: true };
    function chunks2() {
      const L = [{ k: 'role', lab: 'role', j: '{"delta":{"role":"assistant"}}' }];
      if (S2.sc === 'texto') {
        ['The', ' capital', ' is', ' Paris', '.'].forEach((s) => L.push({ k: 'c', v: s, lab: JSON.stringify(s), j: '{"delta":{"content":' + JSON.stringify(s) + '}}' }));
        L.push({ k: 'fin', v: 'stop', lab: 'stop', j: '{"delta":{},"finish_reason":"stop"}' });
      } else {
        const name = S2.sc === 'tool' ? 'get_weather' : 'write_file';
        L.push({ k: 'c', v: 'Let me check.', lab: '"Let me check."', j: '{"delta":{"content":"Let me check."}}' });
        L.push({ k: 'name', v: name, lab: name, j: '{"delta":{"tool_calls":[{"index":0,"id":"call_0","function":{"name":"' + name + '"}}]}}' });
        const frags = S2.sc === 'tool' ? ['{"city"', ': "Pa', 'ris"}'] : ['{"path": "/tmp/log.txt",', ' "content": "'];
        frags.forEach((s) => L.push({ k: 'arg', v: s, lab: s, j: '{"delta":{"tool_calls":[{"index":0,"function":{"arguments":' + JSON.stringify(s) + '}}]}}' }));
        const fin = S2.sc === 'tool' ? 'tool_calls' : (S2.p55 ? 'length' : 'tool_calls');
        L.push({ k: 'fin', v: fin, lab: fin, j: '{"delta":{},"finish_reason":"' + fin + '"}' });
      }
      if (S2.usage) L.push({ k: 'usage', lab: 'usage', j: '{"choices":[],"usage":{"prompt_tokens":812,"completion_tokens":57,"prompt_tokens_details":{"cached_tokens":768}}}' });
      L.push({ k: 'done', lab: '[DONE]', j: '[DONE]' });
      return L;
    }
    const c2 = UI.controls(f2);
    UI.seg(c2, { label: 'Scenario', options: [{ v: 'texto', label: 'text' }, { v: 'tool', label: 'natural tool call' }, { v: 'corte', label: 'truncated tool call' }], value: 'texto', onChange: (v) => { S2.sc = v; S2.reset = true; } });
    UI.toggle(c2, { label: 'include_usage', value: true, onChange: (v) => { S2.usage = v; S2.reset = true; } });
    UI.toggle(c2, { label: 'patch #55', value: true, onChange: (v) => { S2.p55 = v; S2.reset = true; } });
    UI.button(c2, '↻ resend', () => { S2.reset = true; });
    const rd2 = UI.readout(c2, 'finish_reason', '—');
    fig(f2, (g, w, h, t) => {
      const C = V.C();
      if (S2.reset) { S2.t0 = t - (S2.first ? 3.4 : 0); S2.reset = false; S2.first = false; }
      const narrow = w < 600;
      const L = chunks2();
      const gap = 0.62, travel = 1.5;
      const rel = t - S2.t0;
      const total = (L.length - 1) * gap + travel + 3.2;
      if (rel > total) S2.t0 = t;
      const sx = 12, sw = narrow ? 70 : 96, cxw = narrow ? 70 : 96, cx0 = w - 12 - cxw, by = 42;
      V.box(g, sx, by - 22, sw, 44, { r: 10, fill: V.alpha(C.b, 0.16), stroke: C.b });
      V.text(g, 'server', sx + sw / 2, by - 4, { size: 12, align: 'center', weight: 600 });
      V.text(g, 'vLLM', sx + sw / 2, by + 11, { size: 11, align: 'center', color: C.muted, font: 'mono' });
      V.box(g, cx0, by - 22, cxw, 44, { r: 10, fill: V.alpha(C.a, 0.16), stroke: C.a });
      V.text(g, 'client', cx0 + cxw / 2, by - 4, { size: 12, align: 'center', weight: 600 });
      V.text(g, 'SDK/agent', cx0 + cxw / 2, by + 11, { size: 11, align: 'center', color: C.muted, font: 'mono' });
      const bx0 = sx + sw + 8, bx1 = cx0 - 8;
      V.line(g, bx0, by + 18, bx1, by + 18, { color: V.alpha(C.muted, 0.5), lw: 2 });
      for (let x = bx0 + ((t * 40) % 18); x < bx1; x += 18) V.line(g, x, by + 18, x - 5, by + 24, { color: V.alpha(C.muted, 0.35), lw: 1.2 });
      const colOf = (c) => (c.k === 'c' ? C.a : c.k === 'name' || c.k === 'arg' ? C.d : c.k === 'fin' ? C.c : c.k === 'usage' ? C.b : C.muted);
      const st = { content: '', name: '', args: '', fin: '', usage: false, done: false };
      let last = null;
      const got = [];
      L.forEach((c, i) => {
        const p = (rel - i * gap) / travel;
        if (p >= 1) {
          last = c; got.push(c);
          if (c.k === 'c') st.content += c.v; else if (c.k === 'name') st.name = c.v; else if (c.k === 'arg') st.args += c.v; else if (c.k === 'fin') st.fin = c.v; else if (c.k === 'usage') st.usage = true; else if (c.k === 'done') st.done = true;
        } else if (p > 0) {
          const lab = clip(g, c.lab, narrow ? 64 : 104, 11);
          const ww = pill(g, lab, 0, 0, { size: 11, dry: true });
          const x = V.lerp(bx0 + ww / 2 + 2, bx1 - ww / 2 - 2, V.ease.inOut(p));
          pill(g, lab, x - ww / 2, by - 4, { size: 11, fill: V.alpha(colOf(c), 0.25), stroke: colOf(c) });
        }
      });
      rd2.set(st.fin || '—');
      const jy = by + 44;
      V.text(g, clip(g, last ? (last.k === 'done' ? 'data: [DONE]' : 'data: ' + last.j) : 'waiting for the first event…', w - 24, narrow ? 11 : 11.5), 12, jy, { size: narrow ? 11 : 11.5, font: 'mono', color: last ? colOf(last) : C.muted });
      const py = jy + 14, ph = h - py - 10;
      V.box(g, 12, py, w - 24, ph, { r: 12, fill: V.alpha(C.panel, 0.75), stroke: V.alpha(C.a, 0.35) });
      const fx = 26, lab = narrow ? 11 : 12;
      const logW = narrow ? 0 : Math.min(330, (w - 24) * 0.44);
      const colW = w - 52 - (logW ? logW + 16 : 0);
      if (!logW) {
        const maxL = Math.floor((ph - 190) / 15);
        if (maxL > 0) { V.text(g, 'RECEIVED EVENTS', fx, py + ph - 14 - maxL * 15, { size: 11, font: 'mono', color: C.muted }); got.slice(-maxL).forEach((c, i) => V.text(g, clip(g, c.k === 'done' ? 'data: [DONE]' : 'data: ' + c.j, w - 52, 11), fx, py + ph - 10 - (maxL - 1 - i) * 15, { size: 11, font: 'mono', color: colOf(c) })); }
      }
      if (logW) {
        const lx = w - 12 - logW;
        V.line(g, lx, py + 12, lx, py + ph - 12, { color: V.alpha(C.muted, 0.3), lw: 1 });
        V.text(g, 'RECEIVED EVENTS', lx + 14, py + 20, { size: 11, font: 'mono', color: C.muted });
        const maxL = Math.floor((ph - 36) / 16);
        got.slice(-maxL).forEach((c, i) => V.text(g, clip(g, c.k === 'done' ? 'data: [DONE]' : 'data: ' + c.j, logW - 22, 11), lx + 14, py + 40 + i * 16, { size: 11, font: 'mono', color: colOf(c) }));
      }
      let yy = py + 22;
      const row = (k, v, col) => { V.text(g, k, fx, yy, { size: 11, color: C.muted, font: 'mono' }); const lines = wrapLines(g, v, colW, lab, 'mono', 500); lines.slice(0, 2).forEach((l, i) => V.text(g, l, fx, yy + 15 + i * 15, { size: lab, font: 'mono', color: col || C.ink, weight: 500 })); yy += 22 + 15 * Math.min(2, lines.length); };
      row('content', st.content ? JSON.stringify(st.content) : 'null', C.a);
      if (S2.sc !== 'texto') {
        const cut = S2.sc === 'corte';
        let valid = true; try { JSON.parse(st.args); } catch (err) { valid = false; }
        row('tool_calls[0].function', st.name ? st.name + '(' + (st.args || '') + ')' : '—', cut && st.fin && !valid ? (S2.p55 ? C.muted : C.red) : C.d);
      }
      row('finish_reason · usage', (st.fin || '…') + '   ·   ' + (st.usage ? 'prompt 812 · output 57 · cached 768 (illustr.)' : S2.usage ? '…' : 'no include_usage: nothing in the stream'), st.fin === 'length' ? C.d : C.c);
      if (st.fin) {
        let msg, col;
        if (S2.sc === 'corte') {
          if (S2.p55) { msg = 'length: the client discards the incomplete call. Fragments already sent remain (append-only).'; col = C.a; }
          else { msg = 'tool_calls with invalid arguments: the harness resends them and the next turn gets HTTP 400.'; col = C.red; }
        } else if (S2.sc === 'tool') { msg = 'tool_calls: the client runs get_weather and replies with role "tool".'; col = C.a; }
        else { msg = 'stop: natural EOS. Complete answer.'; col = C.a; }
        textBlock(g, msg, fx, Math.min(h - 30, yy + 6), colW, { size: narrow ? 11.5 : 12.5, color: col, weight: 600 });
      }
    });

    /* ================= Fig 3: template renderer ================= */
    const f3 = $('#fig-template');
    fitH(f3, 380, 540);
    const S3 = { r: 'glm', think: true, eff: 'low', tools: false };
    function render3() {
      const C = V.C();
      const sp = (s) => ({ s, fill: V.alpha(C.b, 0.2), stroke: V.alpha(C.b, 0.8), k: 'sp' });
      const tx = (s) => ({ s, fill: V.alpha(C.muted, 0.12), k: 'tx' });
      const ef = (s) => ({ s, fill: V.alpha(C.d, 0.25), stroke: C.d, k: 'ef' });
      const tl = (s) => ({ s, fill: V.alpha(C.c, 0.2), stroke: C.c, k: 'tl' });
      const gn = (s) => ({ s, fill: V.alpha(C.a, 0.22), stroke: C.a, k: 'gn' });
      const e = S3.eff, th = S3.think;
      const R = { items: [], status: 200, err: '', eff: '', note: '', kw: '', div: -1, divChar: 0, ruler: '', foot: '' };
      const effKw = e === 'omit' ? '' : ', "reasoning_effort": "' + e + '"';
      if (S3.r === 'glm') {
        R.kw = 'chat_template_kwargs: {"enable_thinking": ' + th + effKw + '}';
        R.items.push(sp('[gMASK]'), sp('<sop>'));
        if (th) {
          const eff = e === 'low' ? 'Low' : e === 'high' ? 'High' : 'Max';
          R.eff = eff;
          R.items.push(sp('<|system|>'));
          R.div = R.items.length;
          R.items.push(ef('Reasoning Effort: ' + eff));
          R.divChar = 'Reasoning Effort: '.length; // prefix shared by every effort
          R.note = e === 'medium' ? 'medium → Max (outside low/high). The launcher refuses medium as a default.' : (e === 'low' || e === 'high') ? 'effort accepted: ' + eff : (e === 'omit' ? 'omitted → Max (template default)' : e + ' → Max (outside low/high)');
          R.ruler = 'the prompt diverges at character ~39, right after "Reasoning Effort: ". Each block\'s hash includes the previous one, so diverging in block 0 is a total prefix cache miss';
        } else { R.eff = '— (thinking off)'; R.note = 'thinking off: no effort line. Sending low here does not turn thinking back on.'; R.ruler = 'no directive: the effort does not change the prompt'; }
        if (S3.tools) R.items.push(sp('<|system|>'), tl('# Tools · get_weather(city) · <tool_call>…'));
        R.items.push(sp('<|system|>'), tx('You are an assistant.'), sp('<|user|>'), tx('What is 17+25?'), sp('<|assistant|>'), gn(th ? '<think>' : '<think></think>'));
      } else if (S3.r === 'qwen') {
        R.kw = 'chat_template_kwargs: {"enable_thinking": ' + th + effKw + '}';
        const valid = ['low', 'medium', 'xhigh'];
        R.items.push(sp('<|im_start|>system'));
        if (S3.tools) { R.items.push(tl('# Tools <tools>get_weather…</tools>')); R.foot = '* tools block position simplified: the order between the system text and # Tools depends on the checkpoint\'s chat_template.jinja'; }
        R.items.push(tx('You are an assistant.'), sp('<|im_end|>'), sp('<|im_start|>user'), tx('What is 17+25?'), sp('<|im_end|>'), sp('<|im_start|>assistant'), gn(th ? '<think>' : '<think></think>'));
        if (th) {
          if (e === 'omit' || valid.includes(e)) { R.eff = e === 'omit' ? 'xhigh (default)' : e; R.note = 'the template reads reasoning_effort = ' + R.eff; }
          else { R.status = 400; R.err = 'Unexpected reasoning effort ' + e + '. Supported types are xhigh (default), medium, and low.'; R.eff = '—'; }
          R.ruler = 'the directive\'s position in the prompt does not appear in the sources: measure it with /tokenize';
        } else {
          R.eff = '— (thinking off)';
          R.note = (e === 'omit' || valid.includes(e)) ? 'thinking off: the effort is not read' : 'thinking off: "' + e + '" is invalid but silently ignored (the check lives in the thinking branch)';
          R.ruler = 'without thinking: the effort does not enter the prompt';
        }
      } else {
        R.kw = 'chat_template_kwargs: {"thinking": ' + th + effKw + '}';
        R.items.push(sp('<｜begin▁of▁sentence｜>'));
        let m = 'low';
        if (e === 'max' || e === 'xhigh') m = 'max'; else if (e === 'high') m = 'high';
        if (th) {
          R.eff = m;
          if (m !== 'low') { R.div = R.items.length; R.items.push(ef('directive ' + m + ' · BOS+directive = ' + (m === 'max' ? 93 : 80) + ' tokens')); }
          else R.div = R.items.length;
          R.note = e === 'medium' ? 'medium → low (mapping patch; stock sent it to high)' : e === 'xhigh' ? 'xhigh → max' : e === 'omit' ? 'omitted → server default (DEFAULT_THINKING=low)' : 'effort ' + m + (m === 'low' ? ': no directive (0 tokens)' : '');
          R.ruler = 'the directive comes right after the BOS: different efforts diverge in block 0 (0 blocks in common; the optional #144 moves the directive to after the system)';
        } else { R.eff = '— (thinking off)'; R.note = 'thinking=false: the prompt ends in </think> (render without reasoning)'; R.ruler = 'no thinking, no directive'; }
        R.items.push(tx('You are an assistant.'));
        if (S3.tools) R.items.push(tl('## Tools · get_weather(city) · DSML')); // the encoder appends \n\n + tools after the system
        R.items.push(sp('<｜User｜>'), tx('What is 17+25?'), sp('<｜Assistant｜>'), gn(th ? '<think>' : '</think>'));
      }
      return R;
    }
    const c3 = UI.controls(f3);
    UI.seg(c3, { label: 'Recipe', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'glm', onChange: (v) => { S3.r = v; H3.redraw(); } });
    UI.toggle(c3, { label: 'thinking', value: true, onChange: (v) => { S3.think = v; H3.redraw(); } });
    UI.seg(c3, { label: 'Effort', options: [{ v: 'omit', label: '(omitted)' }, 'low', 'medium', 'high', 'xhigh', 'max'], value: 'low', onChange: (v) => { S3.eff = v; H3.redraw(); } });
    UI.toggle(c3, { label: 'declared tools', value: false, onChange: (v) => { S3.tools = v; H3.redraw(); } });
    const stg3 = f3.querySelector('.la-fig-stage');
    const H3 = fig(f3, (g, w, h, t) => {
      const C = V.C();
      const narrow = w < 600;
      const R = render3();
      const x0 = 16, W = w - 32;
      V.text(g, clip(g, R.kw, W, narrow ? 11 : 12), x0, 22, { size: narrow ? 11 : 12, font: 'mono', color: C.d });
      V.text(g, 'RENDERED PROMPT (what the model sees)', x0, 46, { size: 11, font: 'mono', color: C.muted });
      const lay = flow(g, R.items, x0, 56, W, { size: narrow ? 11 : 12.5, dry: true });
      // glow sweeping across the pills and the generation trigger pulsing
      R.items.forEach((it, i) => {
        const r = lay.rects[i];
        const sweep = Math.max(0, 1 - Math.abs(((t * 0.6) % 1.4) * R.items.length - i) / 1.5);
        const o = Object.assign({ size: narrow ? 11 : 12.5 }, it);
        if (it.k === 'gn') o.glow = V.alpha(C.a, 0.5 + 0.3 * Math.sin(t * 4));
        if (sweep > 0) o.lw = 1.2 + sweep * 1.5;
        pill(g, it.s, r.x, r.y, o);
      });
      let y = lay.bottom + 22;
      if (R.foot) y = textBlock(g, R.foot, x0, lay.bottom + 18, W, { size: 11, color: C.muted, lh: 14 }) + 8;
      const ok = R.status === 200;
      const noteS = ok ? R.note : R.err, noteF = ok ? 'sans' : 'mono', noteSz = narrow ? 11 : 12;
      const nLines = wrapLines(g, noteS, W - 24, noteSz, noteF, 400).length;
      const bh = Math.max(50, 33 + nLines * 15);
      V.box(g, x0, y, W, bh, { r: 10, fill: V.alpha(ok ? C.a : C.red, 0.12), stroke: ok ? V.alpha(C.a, 0.6) : C.red });
      V.text(g, ok ? 'HTTP 200 · effective effort: ' + R.eff : 'HTTP 400 · before the engine', x0 + 12, y + 18, { size: narrow ? 12 : 13, weight: 700, color: ok ? C.ink : C.red });
      textBlock(g, noteS, x0 + 12, y + 36, W - 24, { size: noteSz, color: ok ? C.muted : C.red, font: noteF, lh: 15 });
      y += bh + 20;
      let endY = y;
      if (narrow || y + 40 < h) {
        V.text(g, clip(g, 'PREFIX CACHE RULER (same request, different effort)', W, 11), x0, y, { size: 11, font: 'mono', color: C.muted });
        y += 10;
        const tot = R.items.reduce((s, it) => s + it.s.length, 0) || 1;
        let acc = 0, divX = null;
        R.items.forEach((it, i) => {
          const ww = (it.s.length / tot) * W;
          if (i === R.div && R.divChar > 0) {
            // the effort pill has a common prefix ("Reasoning Effort: ") and only the word diverges
            const cw = ww * (R.divChar / it.s.length);
            V.box(g, x0 + acc, y, Math.max(1, cw - 1), 14, { r: 2, fill: V.alpha(C.a, 0.55) });
            V.box(g, x0 + acc + cw, y, Math.max(1, ww - cw - 1), 14, { r: 2, fill: V.alpha(C.c, 0.45) });
            divX = x0 + acc + cw;
          } else {
            const common = R.div < 0 || i < R.div;
            if (i === R.div) divX = x0 + acc;
            V.box(g, x0 + acc, y, Math.max(1, ww - 1), 14, { r: 2, fill: common ? V.alpha(C.a, 0.55) : V.alpha(C.c, 0.45) });
          }
          acc += ww;
        });
        if (divX != null) { V.line(g, divX, y - 6, divX, y + 20, { color: C.c, lw: 2 }); V.text(g, 'diverges', divX + 4, y + 28, { size: 11, color: C.c, font: 'mono' }); }
        const ry2 = textBlock(g, R.ruler, x0, y + 46, W, { size: narrow ? 11 : 12, color: C.ink, lh: 16 });
        endY = ry2;
        const vals = [['GLM', 'low · high · (rest → Max)'], ['Qwen', 'low · medium · xhigh · (high/max → 400)'], ['DeepSeek', 'low · high · max · (medium → low, xhigh → max)']];
        const vy = ry2 + 14;
        if (narrow || vy + 54 < h - 6) {
          V.text(g, 'ACCEPTED EFFORT VALUES', x0, vy, { size: 11, font: 'mono', color: C.muted });
          vals.forEach(([nm, v], i) => {
            const on = (S3.r === 'glm' && i === 0) || (S3.r === 'qwen' && i === 1) || (S3.r === 'ds' && i === 2);
            const yy2 = narrow ? vy + 18 + i * 32 : vy + 18 + i * 17;
            V.text(g, nm, x0, yy2, { size: narrow ? 11 : 12, weight: on ? 700 : 400, color: on ? [C.glm, C.qwen, C.ds][i] : C.muted });
            V.text(g, clip(g, v, narrow ? W : W - 80, 11.5), narrow ? x0 : x0 + 76, narrow ? yy2 + 15 : yy2, { size: 11.5, font: 'mono', color: on ? C.ink : C.muted });
          });
          endY = narrow ? vy + 18 + 2 * 32 + 15 : vy + 18 + 2 * 17;
        }
      }
      // on phones, the height follows the content (no leftover space at the bottom)
      if (narrow) { const want = Math.round(endY + 16); if (Math.abs(parseFloat(stg3.style.height) - want) > 2) stg3.style.height = want + 'px'; }
    }, { static: false });

    /* ================= Fig 4: dormant stops ================= */
    const f4 = $('#fig-stops');
    fitH(f4, 330, 390);
    const T4 = [
      ["Let's", 'r'], [' add', 'r'], ['.', 'r'], [' Question', 'r'], [':', 'r'], [' 17', 'r'], ['+25', 'r'], ['?', 'r'], [' 42', 'r'], ['.', 'r'],
      ['</think>', 'm'], ['17', 'c'], [' +', 'c'], [' 25', 'c'], [' =', 'c'], [' 42', 'c'], ['\\n', 'c'], ['Question', 'c'], [':', 'c'],
    ];
    const S4 = { patch: false, chunk: 1, t0: 0, reset: true, first: true };
    function sim4(n) {
      // processes the first n tokens; returns the state and the stop index
      let rs = '', ct = '', inThink = true, stopAt = -1, markerAt = -1;
      const stop = 'Question:';
      for (let i = 0; i < n; i++) {
        const [s, z] = T4[i];
        const txt = s === '\\n' ? '\n' : s;
        if (z === 'm') { inThink = false; markerAt = i; continue; }
        if (inThink) rs += txt; else ct += txt;
        const hay = S4.patch ? (inThink ? '' : ct) : rs + ct;
        if (hay.includes(stop)) { stopAt = i; if (inThink) rs = rs.slice(0, rs.lastIndexOf(stop)); else ct = ct.slice(0, ct.lastIndexOf(stop)); break; }
      }
      return { rs, ct, stopAt, markerAt, inThink };
    }
    const c4 = UI.controls(f4);
    UI.toggle(c4, { label: 'patch suppress-stops', value: false, onChange: (v) => { S4.patch = v; S4.reset = true; } });
    UI.seg(c4, { label: 'Tokens per step', options: [{ v: 1, label: '1' }, { v: 4, label: 'k+1 = 4' }], value: 1, onChange: (v) => { S4.chunk = v; S4.reset = true; } });
    UI.button(c4, '↻ restart', () => { S4.reset = true; });
    const rd4 = UI.readout(c4, 'content', '—');
    fig(f4, (g, w, h, t) => {
      const C = V.C();
      if (S4.reset) { S4.t0 = t - (S4.first ? 1.6 : 0); S4.reset = false; S4.first = false; }
      const narrow = w < 600;
      const rel = t - S4.t0;
      const stepT = S4.chunk === 1 ? 0.5 : 1.35;
      const steps = Math.floor(rel / stepT) + 1;
      let n = Math.min(T4.length, steps * S4.chunk);
      // stop at the first step that contains the stop
      let sim = sim4(n);
      if (sim.stopAt >= 0) {
        const stopStep = Math.floor(sim.stopAt / S4.chunk);
        n = Math.min(T4.length, (stopStep + 1) * S4.chunk);
        sim = sim4(n);
      }
      const finished = sim.stopAt >= 0 || n >= T4.length;
      const endT = (Math.ceil(n / S4.chunk)) * stepT;
      if (finished && rel > endT + 3.2) S4.t0 = t;
      const x0 = 14, W = w - 28;
      const hdr4 = narrow ? 'prompt …<think> · stop ["Question:"]' : 'prompt: …<|assistant|><think>   ·   stop: ["Question:"]';
      V.text(g, clip(g, hdr4, W, narrow ? 11 : 11.5), x0, 20, { size: narrow ? 11 : 11.5, font: 'mono', color: C.muted });
      const items = T4.map(([s, z], i) => {
        const done = i < n;
        const col = z === 'r' ? C.b : z === 'm' ? C.d : C.a;
        return { s: s === '\\n' ? '↵' : s.replace(/^ /, '·'), fill: V.alpha(col, done ? 0.28 : 0.07), stroke: done ? V.alpha(col, 0.9) : V.alpha(col, 0.3), color: done ? C.ink : C.muted };
      });
      const lay = flow(g, items, x0, 46, W, { size: narrow ? 11 : 12.5, dry: true, rowGap: narrow ? 27 : 22, gap: 4 });
      // "Question:" stops (indices 3-4 and 17-18)
      const pairs = [[3, 4, true], [17, 18, false]];
      items.forEach((it, i) => pill(g, it.s, lay.rects[i].x, lay.rects[i].y, Object.assign({ size: narrow ? 11 : 12.5 }, it)));
      pairs.forEach(([a, b, inside]) => {
        if (n <= b) return;
        const ra = lay.rects[a], rb = lay.rects[b];
        const fired = sim.stopAt === b;
        const dormant = inside && S4.patch;
        const col = fired ? C.c : dormant ? C.muted : C.c;
        if (ra.y === rb.y) {
          g.setLineDash(dormant ? [3, 3] : []);
          V.box(g, ra.x - 3, ra.y - 3, rb.x + rb.w - ra.x + 6, ra.h + 6, { r: 8, stroke: col, lw: fired ? 2.4 : 1.6 });
          g.setLineDash([]);
        }
        if (fired) V.glow(g, rb.x + rb.w / 2, rb.y + rb.h / 2, 26, C.c, 0.6);
        const lx = ra.x, ly = ra.y - 6;
        V.text(g, dormant ? 'z z z  dormant' : fired ? 'STOP!' : 'active', lx, ly, { size: 11, font: 'mono', color: col, weight: 700, alpha: dormant ? 0.6 + 0.4 * Math.sin(t * 3) : 1 });
      });
      // detokenizer window and stop_check_offset
      const markerInBatch = S4.chunk === 4 && S4.patch && sim.markerAt >= 0 && Math.floor(sim.markerAt / 4) === Math.floor((n - 1) / 4);
      if (n > 0 && !finished) {
        const last = lay.rects[n - 1], first = lay.rects[Math.max(0, n - S4.chunk - 1)];
        const yb = last.y + last.h + 5;
        const sameRow = first.y === last.y;
        if (sameRow) V.line(g, first.x, yb, last.x + last.w, yb, { color: C.e, lw: 2.5 });
        else V.line(g, last.x, yb, last.x + last.w, yb, { color: C.e, lw: 2.5 });
        // with the marker in the batch, the "stop_check_offset" label takes the spot under </think>:
        // "window" moves to the start of the window, or disappears if there is no room
        let jx = last.x;
        if (markerInBatch) {
          const m = lay.rects[sim.markerAt];
          jx = sameRow && m.y === first.y && m.x - first.x > 60 ? first.x : null;
        }
        if (jx != null) V.text(g, 'window', jx, yb + 13, { size: 11, font: 'mono', color: C.e });
      }
      if (markerInBatch) {
        const m = lay.rects[sim.markerAt];
        const xo = m.x + m.w + 2;
        V.line(g, xo, m.y - 8, xo, m.y + m.h + 8, { color: C.d, lw: 2.2, dash: [4, 3] });
        const lw4 = measure(g, 'stop_check_offset', 11, 'mono', 600);
        V.text(g, 'stop_check_offset', Math.min(xo + 4, x0 + W - lw4), m.y + m.h + 20, { size: 11, font: 'mono', color: C.d, weight: 600 });
      }
      // output panel
      const py = Math.max(lay.bottom + 30, h - (narrow ? 136 : 112)), ph = h - py - 10;
      V.box(g, 12, py, w - 24, ph, { r: 12, fill: V.alpha(C.panel, 0.75), stroke: V.alpha(C.muted, 0.35) });
      const fx = 26;
      V.text(g, 'reasoning', fx, py + 20, { size: 11, font: 'mono', color: C.b });
      V.text(g, clip(g, JSON.stringify(sim.rs), w - 52 - 90, narrow ? 11 : 12), fx + 84, py + 20, { size: narrow ? 11 : 12, font: 'mono' });
      V.text(g, 'content', fx, py + 42, { size: 11, font: 'mono', color: C.a });
      const contentNull = sim.inThink && finished;
      const cStr = contentNull ? 'null' : JSON.stringify(sim.ct.replace(/\n+$/, ''));
      V.text(g, clip(g, cStr, w - 52 - 90, narrow ? 11 : 12), fx + 84, py + 42, { size: narrow ? 11 : 12, font: 'mono', color: contentNull ? C.red : C.ink, weight: 600 });
      rd4.set(finished ? (contentNull ? 'null' : '"' + sim.ct.trim() + '"') : '…');
      if (finished) {
        const msg = contentNull ? 'finish_reason: stop INSIDE the thinking: the answer never started (content null).' : 'finish_reason: stop after the answer. The harness stop fired where it should.';
        textBlock(g, msg, fx, py + 66, w - 52, { size: narrow ? 11.5 : 12.5, color: contentNull ? C.red : C.a, weight: 600, lh: 16 });
      } else V.text(g, 'generating… step ' + Math.ceil(n / S4.chunk), fx, py + 66, { size: 12, color: C.muted });
    });

    /* ================= Fig 5: parser state machine ================= */
    const f5 = $('#fig-parser');
    fitH(f5, 380, 480);
    const P5 = {
      glm: [['Let me', 'txt', 0], [' check.', 'txt', 0], ['<tool_call>', 'syn', 1], ['get_weather', 'name', 2], ['<arg_key>', 'syn', 3], ['city', 'key', 3], ['</arg_key>', 'syn', 3], ['<arg_value>', 'syn', 3], ['Paris', 'val', 3], ['</arg_value>', 'syn', 3], ['</tool_call>', 'syn', 4]],
      qwen: [['Let me', 'txt', 0], [' check.', 'txt', 0], ['<tool_call>', 'syn', 1], ['<function=', 'syn', 1], ['get_weather', 'name', 2], ['>', 'syn', 2], ['<parameter=', 'syn', 3], ['city', 'key', 3], ['>', 'syn', 3], ['Paris', 'val', 3], ['</parameter>', 'syn', 3], ['</function>', 'syn', 4], ['</tool_call>', 'syn', 4]],
      ds: [['Let me', 'txt', 0], [' check.', 'txt', 0], ['<｜DSML｜tool_calls>', 'syn', 1], ['<｜DSML｜invoke name="', 'syn', 1], ['get_weather', 'name', 2], ['">', 'syn', 2], ['<｜DSML｜parameter name="', 'syn', 3], ['city', 'key', 3], ['" string="true">', 'syn', 3], ['Paris', 'val', 3], ['</｜DSML｜parameter>', 'syn', 3], ['</｜DSML｜invoke>', 'syn', 4], ['</｜DSML｜tool_calls>', 'syn', 4]],
      dsx: [['Let me', 'txt', 0], [' check.', 'txt', 0], ['<｜DSML｜invoke name="', 'syn', 5], ['get_weather', 'name', 5], ['">', 'syn', 5], ['<｜DSML｜parameter name="', 'syn', 5], ['city', 'key', 5], ['" string="true">', 'syn', 5], ['Paris', 'val', 5], ['</｜DSML｜parameter>', 'syn', 5], ['</｜DSML｜invoke>', 'syn', 4]],
    };
    const S5 = { sc: 'glm', rec: false, t0: 0, reset: true, first: true };
    const c5 = UI.controls(f5);
    UI.seg(c5, { label: 'Format', options: [{ v: 'glm', label: 'GLM · glm47' }, { v: 'qwen', label: 'Qwen · qwen3_coder' }, { v: 'ds', label: 'DeepSeek · DSML' }, { v: 'dsx', label: 'DSML without wrapper' }], value: 'glm', onChange: (v) => { S5.sc = v; S5.reset = true; } });
    UI.toggle(c5, { label: 'DSML recovery', value: false, onChange: (v) => { S5.rec = v; S5.reset = true; } });
    UI.button(c5, '↻ reprocess', () => { S5.reset = true; });
    fig(f5, (g, w, h, t) => {
      const C = V.C();
      if (S5.reset) { S5.t0 = t - (S5.first ? 4.2 : 0); S5.reset = false; S5.first = false; }
      const narrow = w < 600;
      const toks = P5[S5.sc];
      const leak = S5.sc === 'dsx' && !S5.rec;
      const rel = t - S5.t0, per = 0.55;
      let n = Math.min(toks.length, Math.floor(rel / per) + 1);
      if (rel > toks.length * per + 3.4) S5.t0 = t;
      const x0 = 14, W = w - 28, sz = narrow ? 11 : 12;
      const clsCol = (c) => (c === 'txt' ? C.a : c === 'syn' ? C.b : C.d);
      const items = toks.map(([s, c], i) => ({ s, fill: V.alpha(clsCol(c), i < n ? 0.26 : 0.06), stroke: i < n ? clsCol(c) : V.alpha(clsCol(c), 0.3), color: i < n ? C.ink : C.muted, glow: i === n - 1 ? V.alpha(clsCol(c), 0.6) : undefined }));
      V.text(g, 'GENERATED TOKENS (syntax · payload · text)', x0, 18, { size: 11, font: 'mono', color: C.muted });
      const lay = flow(g, items, x0, 28, W, { size: sz, gap: 4 });
      // states
      const names = ['TEXT', 'OPENING', 'NAME', 'ARGUMENTS', 'CLOSING'];
      if (S5.sc === 'dsx' && S5.rec) names.push('PROVISIONAL');
      const cur = n > 0 ? toks[n - 1][2] : 0;
      const curSt = leak ? 0 : cur;
      const sy = lay.bottom + (narrow ? 26 : 40);
      const nn = names.length;
      // on phones: two rows in a "serpentine" (3 + 2/3), with wide nodes and an 11 px font
      const perR = narrow ? 3 : nn;
      const rowH5 = 46;
      const nodeW = Math.min(118, (W - (perR - 1) * (narrow ? 22 : 8)) / perR);
      const gapN = perR > 1 ? (W - nodeW * perR) / (perR - 1) : 0;
      const nodePos = (i) => {
        const r = Math.floor(i / perR), c = i % perR;
        const cc = r % 2 === 0 ? c : perR - 1 - c; // the bottom row runs right to left
        return [x0 + cc * (nodeW + gapN), sy + r * rowH5];
      };
      names.forEach((nm, i) => {
        const [nx, ny] = nodePos(i);
        const on = i === curSt;
        const prov = nm === 'PROVISIONAL';
        if (on) V.glow(g, nx + nodeW / 2, ny + 15, 36, prov ? C.d : C.b, 0.5);
        if (prov) g.setLineDash([4, 3]);
        V.box(g, nx, ny, nodeW, 30, { r: 15, fill: V.alpha(prov ? C.d : C.b, on ? 0.32 : 0.08), stroke: on ? (prov ? C.d : C.b) : V.alpha(C.muted, 0.4), lw: on ? 2 : 1 });
        g.setLineDash([]);
        V.text(g, clip(g, nm, nodeW - 6, 11, 'sans', 700), nx + nodeW / 2, ny + 16, { size: 11, align: 'center', base: 'middle', weight: 700, color: on ? C.ink : C.muted });
        if (i < nn - 1 && names[i + 1] !== 'PROVISIONAL') {
          const [qx, qy] = nodePos(i + 1);
          const col5 = { color: V.alpha(C.muted, 0.5), lw: 1.2, head: 5 };
          if (qy !== ny) V.arrow(g, nx + nodeW / 2, ny + 31, qx + nodeW / 2, qy - 1, col5);
          else if (qx > nx) V.arrow(g, nx + nodeW + 1, ny + 15, qx - 1, qy + 15, col5);
          else V.arrow(g, nx - 1, ny + 15, qx + nodeW + 1, qy + 15, col5);
        }
      });
      if (names[5] === 'PROVISIONAL') {
        // recovery enters the provisional state from TEXT and only commits at the closing
        const link = (i, j) => {
          const [ax, ay] = nodePos(i), [bx2, by2] = nodePos(j);
          const o5 = { color: V.alpha(C.d, 0.85), lw: 1.4, head: 5, dash: [4, 3] };
          if (ay === by2) {
            if (bx2 > ax) V.arrow(g, ax + nodeW + 1, ay + 15, bx2 - 1, by2 + 15, ay === sy && bx2 - ax > nodeW + gapN + 1 ? Object.assign({ curve: -26 }, o5) : o5);
            else V.arrow(g, ax - 1, ay + 15, bx2 + nodeW + 1, by2 + 15, o5);
          } else V.arrow(g, ax + nodeW / 2, ay + 31, bx2 + nodeW / 2, by2 - 1, o5);
        };
        if (narrow) link(0, 5);
        else { const [ax] = nodePos(0), [bx2] = nodePos(5); V.arrow(g, ax + nodeW / 2, sy - 1, bx2 + nodeW / 2, sy - 1, { color: V.alpha(C.d, 0.85), lw: 1.4, head: 5, dash: [4, 3], curve: -30 }); }
        link(5, 4);
      }
      const statesBottom = sy + (Math.ceil(nn / perR) - 1) * rowH5 + 30;
      // assembled output
      let content = '', name = '', args = '', committed = false, buf = '';
      const rec = S5.sc === 'dsx' && S5.rec;
      for (let i = 0; i < n; i++) {
        const [s, c, st] = toks[i];
        if (leak || c === 'txt') { content += s; continue; }
        if (rec && st === 5) buf += s;
        if (c === 'name') name = s;
        if (c === 'key') args += (args ? ', ' : '{') + '"' + s + '": ';
        if (c === 'val') args += '"' + s + '"';
        if (st === 4) committed = true;
      }
      if (args && committed) args += '}';
      const py = statesBottom + 18, ph = h - py - 10;
      V.box(g, 12, py, w - 24, ph, { r: 12, fill: V.alpha(C.panel, 0.75), stroke: V.alpha(leak ? C.red : C.muted, 0.45) });
      const fx = 26, lsz = narrow ? 11 : 12;
      V.text(g, 'content', fx, py + 20, { size: 11, font: 'mono', color: C.a });
      const cl = wrapChars(g, JSON.stringify(content), w - 52, lsz, 'mono', 400);
      cl.slice(0, 3).forEach((l, i) => V.text(g, l, fx, py + 38 + i * 15, { size: lsz, font: 'mono', color: leak && n > 2 ? C.red : C.ink }));
      let yy = py + 44 + Math.min(3, cl.length) * 15;
      V.text(g, 'tool_calls', fx, yy, { size: 11, font: 'mono', color: C.d });
      yy += 18;
      if (leak) {
        V.text(g, '[]', fx, yy, { size: lsz, font: 'mono', color: C.red });
        if (n > 3) textBlock(g, 'The invoke leaked as text: the parser never saw the wrapper, and a benchmark that only reads tool_calls counts "no call".', fx, yy + 20, w - 52, { size: narrow ? 11 : 12, color: C.red, weight: 600, lh: 15 });
      } else if (rec && !committed && buf) {
        g.setLineDash([4, 3]);
        V.box(g, fx - 4, yy - 12, Math.min(w - 44, measure(g, 'provisional buffer: ' + clip(g, buf, w - 200, lsz), lsz) + 12), 20, { r: 6, stroke: C.d });
        g.setLineDash([]);
        V.text(g, 'provisional buffer: ' + clip(g, buf, w - 200, lsz), fx + 2, yy + 2, { size: lsz, font: 'mono', color: C.d });
        textBlock(g, 'name "get_weather" ∈ declared tools ✓ · commits only at </｜DSML｜invoke>', fx, yy + 28, w - 52, { size: narrow ? 11 : 11.5, color: C.muted, lh: 15 });
      } else if (name) {
        const tl = wrapChars(g, '[{"type":"function","function":{"name":"' + name + '","arguments":' + JSON.stringify(args) + '}}]', w - 52, lsz, 'mono', committed ? 600 : 400);
        tl.slice(0, 4).forEach((l, i) => V.text(g, l, fx, yy + i * 15, { size: lsz, font: 'mono', color: committed ? C.d : C.ink, weight: committed ? 600 : 400 }));
        if (committed) textBlock(g, rec ? 'commit: recovered from DSML without a wrapper. Correct DSML stays byte-identical.' : 'finish_reason: "tool_calls". The client runs it and replies with role "tool".', fx, yy + 8 + Math.min(4, tl.length) * 15, w - 52, { size: narrow ? 11 : 12, color: C.a, weight: 600, lh: 16 });
      } else V.text(g, '[]', fx, yy, { size: lsz, font: 'mono', color: C.muted });
      if (S5.sc !== 'dsx' && S5.rec) V.text(g, 'recovery only acts on DSML without a wrapper', x0 + W, 18, { size: 11, align: 'right', color: C.muted });
    });

    /* ================= Fig 6: logits, mask and temperature ================= */
    const f6 = $('#fig-logits');
    fitH(f6, 380, 560);
    const S6 = { mode: 'none', T: 1.0, mask: false, D: 9, K: 60 };
    const c6 = UI.controls(f6);
    UI.seg(c6, { label: 'Mode', options: [{ v: 'none', label: 'tool_choice none (bad_words)' }, { v: 'temp', label: 'temperature × syntax' }], value: 'none', onChange: (v) => { S6.mode = v; vis6(); } });
    UI.slider(c6, { label: 'Temperature', min: 0.05, max: 1.5, step: 0.05, value: 1.0, fmt: (v) => v.toFixed(2), onInput: (v) => { S6.T = v; } });
    const tg6 = UI.toggle(c6, { label: 'bad_words = ["<tool_call>"]', value: false, onChange: (v) => { S6.mask = v; } });
    const sD = UI.slider(c6, { label: 'Margin Δ', min: 4, max: 14, step: 0.5, value: 9, fmt: (v) => v.toFixed(1), onInput: (v) => { S6.D = v; } });
    const sK = UI.slider(c6, { label: 'Syntax tokens K', min: 10, max: 200, step: 10, value: 60, fmt: (v) => String(v), onInput: (v) => { S6.K = v; } });
    const rd6 = UI.readout(c6, 'P(<tool_call>)', '—');
    function vis6() { const n = S6.mode === 'none'; tg6.el.style.display = n ? '' : 'none'; sD.el.style.display = n ? 'none' : ''; sK.el.style.display = n ? 'none' : ''; }
    vis6();
    const lab6 = ['<tool_call>', 'The', 'A', 'Sure', '\\n', 'No'], lg6 = [3.6, 2.9, 2.5, 1.8, 1.2, 0.9];
    const M6 = 20;
    const pOk = (T, D) => 1 / (1 + M6 * Math.exp(-D / Math.max(T, 1e-3)));
    const stg6 = f6.querySelector('.la-fig-stage');
    const msg6 = ['The drafted <tool_call> has p = 0 at the target and is rejected. The target samples another token (e.g. " 42"), at the cost of one acceptance.', 'Without the mask, the opener is accepted and the model opens a call. Under none the API suppresses tool_calls: out comes "complete call, no text".'];
    function bars6(g, C, p, labs, bx, by, bh, gw, wideFirst, colFn) {
      // thin 100% line: the space above the bars is scale, not leftover
      const top = by + bh - 24 - (bh - 40);
      V.line(g, bx, top, bx + gw * p.length - 4, top, { color: V.alpha(C.muted, 0.28), lw: 1, dash: [3, 4] });
      V.text(g, '100%', bx + gw * p.length - 4, top - 4, { size: 11, align: 'right', font: 'mono', color: V.alpha(C.muted, 0.8) });
      // with wideFirst, the 1st column (longest label, <tool_call>) is 1.5× wider, so it fits at 11 px
      const tot = gw * p.length, g0 = wideFirst ? tot / (p.length + 0.5) * 1.5 : gw, gr = wideFirst ? (tot - g0) / (p.length - 1) : gw;
      const colX = (i) => bx + (i === 0 ? 0 : g0 + (i - 1) * gr), colW = (i) => (i === 0 ? g0 : gr);
      p.forEach((v, i) => {
        const cw = colW(i);
        const hh = Math.max(2, v * (bh - 40)), xx = colX(i) + 4, yy = by + bh - 24 - hh;
        colFn(i, xx, yy, hh, v, cw);
        V.text(g, (v * 100).toFixed(v > 0.995 || v < 0.01 ? 2 : v < 0.1 ? 1 : 0) + '%', xx + (cw - 8) / 2, Math.min(yy - 4, by + bh - 28), { size: 11, align: 'center', font: 'mono', color: C.ink });
        V.text(g, clip(g, labs[i], cw + 2, 11), xx + (cw - 8) / 2, by + bh - 8, { size: 11, align: 'center', font: 'mono', color: i === 0 ? colFn.c0 : C.muted });
      });
    }
    fig(f6, (g, w, h, t) => {
      const C = V.C();
      const narrow = w < 600;
      const x0 = 16;
      const LW = narrow ? w - 32 : w * 0.5 - 24;
      const bx = x0, by = 34, bh = narrow ? 140 : h - 96;
      let want6 = 0;
      if (S6.mode === 'none') {
        const z = lg6.map((v, i) => (S6.mask && i === 0 ? -Infinity : v / S6.T));
        const mx = Math.max(...z.filter((v) => v > -Infinity));
        const ex = z.map((v) => (v === -Infinity ? 0 : Math.exp(v - mx)));
        const sm = ex.reduce((a, b) => a + b, 0);
        const p = ex.map((v) => v / sm);
        rd6.el.querySelector('span').textContent = 'P(<tool_call>)';
        rd6.set((p[0] * 100).toFixed(1) + '%');
        V.text(g, clip(g, 'PROBABILITIES AT THIS POSITION (T = ' + S6.T.toFixed(2) + ')', LW, 11), bx, 20, { size: 11, font: 'mono', color: C.muted });
        const nb = p.length, gw = LW / nb;
        const colFn = (i, xx, yy, hh, v, cw) => {
          const masked = S6.mask && i === 0;
          if (masked) {
            g.setLineDash([4, 3]);
            V.box(g, xx, by + 22, cw - 8, bh - 46, { r: 4, stroke: C.red });
            g.setLineDash([]);
            cross(g, xx + (cw - 8) / 2, by + bh / 2 - 6, 14, C.red);
            V.text(g, '−∞', xx + (cw - 8) / 2, by + bh / 2 + 20, { size: 12, align: 'center', color: C.red, font: 'mono', weight: 700 });
          } else V.box(g, xx, yy, cw - 8, hh, { r: 4, fill: i === 0 ? V.alpha(C.d, 0.8) : V.alpha(C.a, 0.6) });
        };
        colFn.c0 = C.d;
        bars6(g, C, p, lab6, bx, by, bh, gw, true, colFn);
        // speculative verification of the drafts
        const rx = narrow ? x0 : w * 0.5 + 8, ry = narrow ? by + bh + 46 : 46, RW = narrow ? w - 32 : w * 0.5 - 24;
        V.text(g, 'DRAFT VERIFICATION (k = 5)', rx, ry - 26, { size: 11, font: 'mono', color: C.muted });
        const drafts = ['The', ' value', ' is', '<tool_call>', 'lookup'];
        const cyc = ((t + 3.2) % 5.5) / 5.5;
        const scan = Math.floor(cyc * 7);
        let px = rx;
        const pw = [], pxs = [];
        drafts.forEach((s) => { pw.push(pill(g, s, 0, 0, { size: 11.5, dry: true })); });
        let rowY = ry;
        drafts.forEach((s, i) => {
          if (px + pw[i] > rx + RW) { px = rx; rowY += 34; }
          let fill = V.alpha(C.muted, 0.1), stroke = V.alpha(C.muted, 0.4), mark = null;
          if (i < scan) {
            const rej = S6.mask && i === 3, after = S6.mask && i > 3;
            if (rej) { fill = V.alpha(C.red, 0.18); stroke = C.red; mark = 'x'; }
            else if (after) { fill = V.alpha(C.muted, 0.05); stroke = V.alpha(C.muted, 0.3); mark = 'd'; }
            else { fill = V.alpha(C.a, 0.22); stroke = C.a; mark = 'v'; }
          }
          pill(g, s, px, rowY, { size: 11.5, fill, stroke, color: mark === 'd' ? C.muted : C.ink });
          if (mark === 'v') check(g, px + pw[i] / 2, rowY - 8, 8, C.a);
          if (mark === 'x') cross(g, px + pw[i] / 2, rowY - 8, 8, C.red);
          pxs.push([px, rowY]);
          px += pw[i] + 6;
        });
        // the token the target samples in place of the rejected opener, right below it
        if (scan >= 5 && S6.mask) {
          const [ox, oy] = pxs[3];
          V.arrow(g, ox + pw[3] / 2, oy + 23, ox + pw[3] / 2, oy + 31, { color: C.c, lw: 1.4, head: 4 });
          pill(g, ' 42', ox + pw[3] / 2 - 16, oy + 32, { size: 11, fill: V.alpha(C.c, 0.25), stroke: C.c });
        }
        const msgY = rowY + 70;
        const msz = narrow ? 11.5 : 12.5, mlh = 17;
        if (scan >= 5) textBlock(g, msg6[S6.mask ? 0 : 1], rx, msgY, RW, { size: msz, color: S6.mask ? C.a : C.red, weight: 600, lh: mlh });
        else V.text(g, 'verifying position ' + Math.min(5, scan + 1) + '…', rx, msgY, { size: 12, color: C.muted });
        // reserve the height of the longest message, so the layout does not jump on every cycle
        const maxL = Math.max(...msg6.map((m) => wrapLines(g, m, RW, msz, 'sans', 600).length));
        // mini-matrix z/T · m · p (the mask equation, cell by cell)
        const mY = msgY + (maxL - 1) * mlh + 22, cellW = Math.min(52, (RW - 40) / 6), cellH = 20;
        const mEnd = mY + 3 * (cellH + 3);
        if (narrow || mEnd + 16 < h - 18) {
          const rowsM = [['z/T', lg6.map((v) => (v / S6.T).toFixed(1))], ['m', lg6.map((v, i) => (S6.mask && i === 0 ? '0' : '1'))], ['p', p.map((v) => (v * 100).toFixed(0) + '%')]];
          rowsM.forEach(([nm, vals], r) => {
            const yy = mY + r * (cellH + 3);
            V.text(g, nm, rx, yy + cellH / 2 + 1, { size: 11, font: 'mono', base: 'middle', color: C.muted });
            vals.forEach((v, j) => {
              const xx = rx + 34 + j * (cellW + 3);
              const off = r === 1 && v === '0', hot = j === 0;
              V.box(g, xx, yy, cellW, cellH, { r: 4, fill: off ? V.alpha(C.red, 0.25) : V.alpha(hot ? C.d : C.a, r === 2 ? 0.08 + 0.5 * p[j] : 0.12) });
              V.text(g, v, xx + cellW / 2, yy + cellH / 2 + 1, { size: 11, font: 'mono', align: 'center', base: 'middle', color: off ? C.red : C.ink });
            });
          });
        }
        // on phones the final sentence goes right below the matrix; on desktop, at the bottom
        const fin6 = 'the mask applies at every position, including draft positions';
        let endY6;
        if (narrow) endY6 = textBlock(g, fin6, rx, mEnd + 18, RW, { size: 11, color: C.muted, lh: 15 }) - 15;
        else { endY6 = h - 12; V.text(g, clip(g, fin6, RW, 11, 'sans', 400), rx, endY6, { size: 11, color: C.muted }); }
        want6 = Math.round(endY6 + 14);
      } else {
        const po = pOk(S6.T, S6.D);
        const call = Math.pow(po, S6.K);
        rd6.el.querySelector('span').textContent = 'broken / 1000';
        rd6.set(String(Math.round((1 - call) * 1000)));
        V.text(g, clip(g, narrow ? 'SYNTAX after <｜DSML｜ (toy)' : 'SYNTAX POSITION: after <｜DSML｜ (toy model)', LW, 11), bx, 20, { size: 11, font: 'mono', color: C.muted });
        const alts = ['invoke', 'inv', 'tool', 'param', 'other 16'];
        const zc = S6.D / S6.T, zs = [0.3, 0.1, -0.1, 0];
        const ea = zs.map((z) => Math.exp(z / S6.T - zc));
        const denom = 1 + M6 * Math.exp(-zc) * 1 + 0; // visual approximation
        const pa = [1 / (1 + M6 * Math.exp(-zc))];
        const rest = 1 - pa[0];
        const shares = [ea[0], ea[1], ea[2]];
        const ssum = shares.reduce((a, b) => a + b, 0) + ea[3] * 16;
        shares.forEach((s) => pa.push(rest * s / ssum));
        pa.push(rest * (ea[3] * 16) / ssum);
        void denom;
        const nb = pa.length, gw = LW / nb;
        const colT = (i, xx, yy, hh, v, cw) => V.box(g, xx, yy, cw - 8, hh, { r: 4, fill: i === 0 ? V.alpha(C.a, 0.75) : V.alpha(C.red, 0.55) });
        colT.c0 = C.a;
        bars6(g, C, pa, alts, bx, by, bh, gw, false, colT);
        // P(call ok) × T chart
        const gx = narrow ? x0 + 34 : w * 0.5 + 44, gy = narrow ? by + bh + 30 : 40, GW = narrow ? w - 32 - 44 : w * 0.5 - 70, GH = narrow ? h - (by + bh + 30) - 60 : h - 100;
        V.text(g, clip(g, 'P(whole call well-formed) × T', w - (gx - 30) - 12, 11), gx - 30, gy - 12, { size: 11, font: 'mono', color: C.muted });
        V.line(g, gx, gy, gx, gy + GH, { color: C.muted, lw: 1 });
        V.line(g, gx, gy + GH, gx + GW, gy + GH, { color: C.muted, lw: 1 });
        [0, 0.5, 1].forEach((v) => V.text(g, v.toFixed(1), gx - 6, gy + GH - v * GH + 4, { size: 11, align: 'right', font: 'mono', color: C.muted }));
        [0, 0.5, 1, 1.5].forEach((v) => V.text(g, v.toFixed(1), gx + (v / 1.5) * GW, gy + GH + 15, { size: 11, align: 'center', font: 'mono', color: C.muted }));
        V.line(g, gx, gy + 3, gx + GW, gy + 3, { color: C.b, lw: 2, dash: [2, 4] });
        V.line(g, gx, gy, gx + GW, gy, { color: C.a, lw: 2, dash: [7, 4] });
        g.beginPath();
        for (let i = 0; i <= 80; i++) { const T = 0.02 + (i / 80) * 1.48; const y = gy + GH - Math.pow(pOk(T, S6.D), S6.K) * GH; const x = gx + (T / 1.5) * GW; if (i) g.lineTo(x, y); else g.moveTo(x, y); }
        g.strokeStyle = C.c; g.lineWidth = 2.5; g.stroke();
        const mxp = gx + (S6.T / 1.5) * GW, myp = gy + GH - call * GH;
        V.line(g, mxp, gy, mxp, gy + GH, { color: V.alpha(C.ink, 0.25), lw: 1, dash: [3, 3] });
        V.glow(g, mxp, myp, 14, C.c, 0.8); V.dot(g, mxp, myp, 4.5, C.c);
        const lx = gx + 8, ly = gy + GH - 40;
        V.text(g, '— vLLM: everything at T', lx, ly, { size: 11, color: C.c, weight: 600 });
        V.text(g, '- - ds4: syntax at T=0', lx, ly + 14, { size: 11, color: C.a, weight: 600 });
        V.text(g, '·· grammar: forced structure', lx, ly + 28, { size: 11, color: C.b, weight: 600 });
        V.text(g, clip(g, 'p_ok/token = ' + (po * 100).toFixed(3) + '%  →  call ok = ' + (call * 100).toFixed(1) + '%', w - 32, narrow ? 11 : 12, 'mono', 600), x0, h - 12, { size: narrow ? 11 : 12, font: 'mono', color: C.ink, weight: 600 });
        want6 = 520;
      }
      // on phones the height follows each mode's content (no empty bands)
      if (narrow && want6) { if (Math.abs(parseFloat(stg6.style.height) - want6) > 2) stg6.style.height = want6 + 'px'; }
    });

    /* ================= Fig 7: grammar step by step ================= */
    const f7 = $('#fig-gramatica');
    fitH(f7, 330, 410);
    const VOC = ['{', '{"', 'city', 'c', '":', '"', ' "', 'Par', 'is', 'Lima', '"}', '}', ',', '42', 'true', '<eos>'];
    const STR = VOC.filter((v) => v !== '":' && v !== '<eos>');
    const G7 = [
      { st: 0, allow: ['{', '{"'], pick: '{"' },
      { st: 1, allow: ['city', 'c'], pick: 'city' },
      { st: 2, allow: ['":', '"'], pick: '":' },
      { st: 3, allow: [' "', '"'], pick: ' "' },
      { st: 4, allow: STR, pick: 'Par' },
      { st: 4, allow: STR, pick: 'is' },
      { st: 4, allow: STR, pick: '"}' },
      { st: 5, allow: ['<eos>'], pick: '<eos>' },
    ];
    const stNames7 = ['object', 'key', 'close key', 'value', 'string', 'end ✓'];
    const r7 = V.rng(34);
    const LG7 = G7.map((s) => VOC.map((v) => (v === s.pick ? 3.2 : r7.normal() * 0.9 + 0.6)));
    const S7 = { T: 1.0 };
    const steps7 = [
      { t: 'Start of the object', d: 'The matcher is in the initial state of the schema <code>{"city": string}</code>. Only tokens that start an object get through: <code>{</code> and <code>{"</code>. The other 14 chips get bit 0 and logit \\(-\\infty\\).' },
      { t: 'The key', d: 'With <code>additionalProperties</code> closed, the only valid key is <code>city</code>. Even <code>c</code> (a prefix of city) would be accepted, but not <code>Lima</code>, <code>42</code> or <code>"</code> (an empty key).' },
      { t: 'Closing the key', d: 'After <code>{"city</code>, only a closing quote: <code>"</code> or the fused token <code>":</code>. A good tokenizer has many tokens like this, which cross grammar boundaries, and XGrammar handles them per state.' },
      { t: 'Opening the value', d: 'The schema asks for a string: <code>"</code> or <code> "</code> (with a space). Notice that sampling barely exists here, with two candidates and one of them dominant.' },
      { t: 'Inside the string', d: 'Inside a string almost anything goes: 14 of 16 chips. Only <code>":</code> (it would close the string and put a colon in a value) and EOS (incomplete object) stay out. Here the temperature matters again: this is <b>payload</b>.' },
      { t: 'Continuing the payload', d: 'Same state, same wide mask. The content is the model\'s decision, and the grammar only guarantees that the <i>shape</i> stays valid.' },
      { t: 'Closing everything', d: 'The token <code>"}</code> closes the string and the object at once. The matcher advances two “squares” of the grammar with a single token.' },
      { t: 'Terminal', d: 'Complete object: the only allowed output is EOS. The matcher <b>terminates</b>. Any token that arrives after that is an error, and that is exactly the bug the next section shows under speculation.' },
    ];
    const st7 = UI.stepper(f7, steps7, { autoplay: 4200 });
    const c7 = UI.controls(f7);
    UI.slider(c7, { label: 'Temperature', min: 0.1, max: 2, step: 0.1, value: 1, fmt: (v) => v.toFixed(1), onInput: (v) => { S7.T = v; } });
    const rd7 = UI.readout(c7, 'P(chosen)', '—');
    const rd7b = UI.readout(c7, 'allowed', '—');
    fig(f7, (g, w, h, t) => {
      const C = V.C();
      const narrow = w < 600;
      const k = st7.i, e = st7.elapsed();
      const S = G7[k];
      const x0 = 16, W = w - 32;
      const out = G7.slice(0, k).map((s) => s.pick).join('').replace('<eos>', '');
      V.text(g, 'OUTPUT', x0, 20, { size: 11, font: 'mono', color: C.muted });
      const ow = measure(g, out, narrow ? 13 : 15, 'mono', 600);
      const nextTok = S.pick === '<eos>' ? '' : S.pick;
      const blend = V.ease.out(V.clamp((e - 0.6) / 0.6, 0, 1));
      // empty output: the ∅ fades out when the first token arrives (it does not overlap it)
      if (out) V.text(g, out, x0 + 50, 21, { size: narrow ? 13 : 15, font: 'mono', weight: 600, color: C.ink });
      else if (!nextTok || blend < 1) V.text(g, '∅', x0 + 50, 21, { size: narrow ? 13 : 15, font: 'mono', weight: 600, color: C.muted, alpha: nextTok ? 1 - blend : 1 });
      if (nextTok && blend > 0) V.text(g, nextTok, x0 + 50 + ow, 21, { size: narrow ? 13 : 15, font: 'mono', weight: 700, color: C.c, alpha: blend });
      // states
      const sy = 40, nS = stNames7.length, sw = (W - 10) / nS;
      stNames7.forEach((nm, i) => {
        const cx = x0 + sw * (i + 0.5), on = i === S.st;
        if (i < nS - 1) V.line(g, cx + 8, sy + 10, cx + sw - 8, sy + 10, { color: V.alpha(C.muted, 0.4), lw: 1.4 });
        if (on) V.glow(g, cx, sy + 10, 22, C.b, 0.7);
        V.dot(g, cx, sy + 10, on ? 7 : 5, on ? C.b : V.alpha(C.muted, 0.5));
        // always 11 px; on phones, compound names wrap into two short lines
        const nl7 = narrow ? wrapLines(g, nm, sw - 2, 11, 'sans', on ? 700 : 400) : [nm];
        nl7.forEach((l, j) => V.text(g, l, cx, sy + 31 + j * 12, { size: 11, align: 'center', color: on ? C.ink : C.muted, weight: on ? 700 : 400 }));
      });
      // chips
      const cols = narrow ? 4 : 8, rows = Math.ceil(VOC.length / cols);
      const gy0 = sy + (narrow ? 56 : 50), gap = 8;
      const cw = (W - (cols - 1) * gap) / cols, chh = narrow ? 50 : 54;
      const allowed = VOC.map((v) => S.allow.includes(v));
      const z = VOC.map((v, i) => (allowed[i] ? LG7[k][i] / S7.T : -Infinity));
      const mx = Math.max(...z.filter((v) => v > -Infinity));
      const ex = z.map((v) => (v === -Infinity ? 0 : Math.exp(v - mx)));
      const sm = ex.reduce((a, b) => a + b, 0);
      const p = ex.map((v) => v / sm);
      const pi = VOC.indexOf(S.pick);
      rd7.set((p[pi] * 100).toFixed(1) + '%');
      rd7b.set(S.allow.length + ' / ' + VOC.length);
      const fade = V.clamp(e / 0.35, 0, 1);
      VOC.forEach((v, i) => {
        const r = Math.floor(i / cols), c = i % cols;
        const x = x0 + c * (cw + gap), y = gy0 + r * (chh + gap);
        const al = allowed[i], chosen = i === pi && e > 0.35;
        if (chosen) V.glow(g, x + cw / 2, y + chh / 2, cw * 0.6, C.c, 0.5);
        V.box(g, x, y, cw, chh, { r: 8, fill: al ? V.alpha(C.a, 0.08 + 0.14 * fade) : V.alpha(C.muted, 0.05), stroke: chosen ? C.c : al ? V.alpha(C.a, 0.4 + 0.5 * fade) : V.alpha(C.muted, 0.25), lw: chosen ? 2.6 : 1.2 });
        V.text(g, v, x + cw / 2, y + 18, { size: narrow ? 12 : 13, font: 'mono', align: 'center', color: al ? C.ink : C.muted, weight: al ? 600 : 400, alpha: al ? 1 : 0.55 });
        if (!al) V.line(g, x + 8, y + 18, x + cw - 8, y + 18, { color: V.alpha(C.muted, 0.6), lw: 1.2 });
        if (al) {
          const bw = (cw - 16) * p[i];
          V.box(g, x + 8, y + chh - 14, cw - 16, 6, { r: 3, fill: V.alpha(C.muted, 0.15) });
          V.box(g, x + 8, y + chh - 14, Math.max(2, bw), 6, { r: 3, fill: chosen ? C.c : C.a });
          V.text(g, (p[i] * 100).toFixed(p[i] > 0.995 ? 0 : 1) + '%', x + cw / 2, y + chh - 20, { size: 11, font: 'mono', align: 'center', color: C.muted });
        } else V.text(g, '−∞', x + cw / 2, y + chh - 12, { size: 11, font: 'mono', align: 'center', color: V.alpha(C.red, 0.8) });
      });
      // bitmask
      const by = gy0 + rows * (chh + gap) + 8;
      V.text(g, 'bitmask', x0, by + 12, { size: 11, font: 'mono', color: C.muted });
      const bx0 = x0 + 62, bcw = Math.min(22, (W - 62) / VOC.length - 2);
      VOC.forEach((v, i) => {
        const x = bx0 + i * (bcw + 2);
        V.box(g, x, by, bcw, 18, { r: 3, fill: allowed[i] ? V.alpha(C.a, 0.7) : V.alpha(C.muted, 0.12) });
        V.text(g, allowed[i] ? '1' : '0', x + bcw / 2, by + 10, { size: 11, font: 'mono', align: 'center', base: 'middle', color: allowed[i] ? C.bg : C.muted }); // C.bg contrasts with the mint in both themes
      });
      if (by + 42 < h) V.text(g, 'real: 1 bit × V tokens → 4·⌈V/32⌉ B per row (GLM 19,360 B)', x0, by + 38, { size: narrow ? 11 : 11.5, color: C.muted });
    });

    /* ================= Fig 8: grammar × speculation ================= */
    const f8 = $('#fig-spec-fsm');
    fitH(f8, 320, 370);
    const S8 = { sc: 'term', fix: false, t0: 0, reset: true, first: true };
    const c8 = UI.controls(f8);
    UI.seg(c8, { label: 'Scenario', options: [{ v: 'term', label: 'termination mid-batch' }, { v: 'think', label: '</think> mid-batch' }], value: 'term', onChange: (v) => { S8.sc = v; S8.reset = true; } });
    UI.toggle(c8, { label: 'with backport', value: false, onChange: (v) => { S8.fix = v; S8.reset = true; } });
    UI.button(c8, '↻ replay', () => { S8.reset = true; });
    fig(f8, (g, w, h, t) => {
      const C = V.C();
      if (S8.reset) { S8.t0 = t - (S8.first ? 5.4 : 0); S8.reset = false; S8.first = false; }
      const narrow = w < 600;
      const rel = t - S8.t0, per = 0.8;
      const term = S8.sc === 'term';
      const block = term ? ['Par', 'is', '"}', '<eos>', ' {', '"x'] : [' right', '.', '</think>', '{"', 'city', '":'];
      const n = Math.min(block.length, Math.floor(rel / per));
      const cycle = block.length * per + 4.5;
      if (rel > cycle) S8.t0 = t;
      const x0 = 16, W = w - 32;
      V.text(g, 'BATCH ACCEPTED BY THE VERIFIER (k+1 = 6)', x0, 18, { size: 11, font: 'mono', color: C.muted });
      const pws = block.map((s) => pill(g, s, 0, 0, { size: narrow ? 11 : 12.5, dry: true, minW: narrow ? 40 : 54 }));
      const totalW = pws.reduce((a, b) => a + b, 0) + 6 * (block.length - 1);
      let bx = x0 + Math.max(0, (W - totalW) / 2);
      const mY = narrow ? 130 : 120, mX = x0 + W / 2;
      // status of each token
      const status = block.map((s, i) => {
        if (term) {
          if (i <= 3) return 'ok';
          return S8.fix ? 'noop' : (i === 4 ? 'err' : 'stuck');
        }
        if (i < 2) return 'think';
        if (i === 2) return 'marker';
        return S8.fix ? 'ok' : 'skip';
      });
      block.forEach((s, i) => {
        const cardY = 28;
        const p = V.clamp((rel - (i + 1) * per + per) / (per * 0.7), 0, 1);
        const entered = i < n;
        const x = bx, ww = pws[i];
        let y = cardY, alpha = 1;
        if (i === n && p > 0 && p < 1) { y = V.lerp(cardY, mY - 12, V.ease.inOut(p)); }
        if (entered) alpha = 0.35;
        const stt = status[i];
        const col = stt === 'ok' ? C.a : stt === 'err' ? C.red : stt === 'stuck' ? C.red : stt === 'noop' ? C.muted : stt === 'think' ? C.b : stt === 'marker' ? C.d : C.c;
        pill(g, s, x, y, { size: narrow ? 11 : 12.5, minW: narrow ? 40 : 54, fill: V.alpha(col, entered ? 0.3 : 0.14), stroke: col, color: C.ink, alpha: entered ? 1 : 1 });
        if (entered) {
          const my = cardY + 34;
          if (stt === 'ok') check(g, x + ww / 2, my, 8, C.a);
          else if (stt === 'err' || stt === 'stuck') cross(g, x + ww / 2, my, 8, C.red);
          else if (stt === 'noop') V.text(g, 'no-op', x + ww / 2, my + 4, { size: 11, align: 'center', font: 'mono', color: C.muted });
          else if (stt === 'skip') V.text(g, 'out', x + ww / 2, my + 4, { size: 11, align: 'center', font: 'mono', color: C.c });
          else if (stt === 'think') V.text(g, 'free', x + ww / 2, my + 4, { size: 11, align: 'center', font: 'mono', color: C.b });
        }
        void alpha;
        bx += ww + 6;
      });
      // matcher
      let stName, stCol = C.b, err = false;
      if (term) {
        if (n <= 3) stName = n === 0 ? 'string' : n < 3 ? 'string' : 'end of object';
        else if (n === 4) stName = 'TERMINATED';
        else { if (S8.fix) stName = 'TERMINATED (ignores the rest)'; else { stName = 'OUT OF SYNC'; err = true; } }
        if (n >= 4) stCol = err ? C.red : C.a;
      } else {
        if (n < 3) stName = 'reasoning (no grammar)';
        else if (!S8.fix) { stName = 'object (did not advance)'; err = n > 3; }
        else stName = n >= 6 ? 'value (advanced 3 tokens)' : 'advancing…';
        if (n >= 3) stCol = err ? C.red : C.a;
      }
      const mw = Math.min(W, narrow ? W : 420);
      if (err) V.glow(g, mX, mY + 24, 70, C.red, 0.35 + 0.25 * Math.sin(t * 6));
      V.box(g, mX - mw / 2, mY, mw, 48, { r: 12, fill: V.alpha(stCol, 0.14), stroke: stCol, lw: 2 });
      V.text(g, 'XGrammar matcher', mX, mY + 17, { size: 11, align: 'center', color: C.muted, font: 'mono' });
      V.text(g, clip(g, 'state: ' + stName, mw - 16, narrow ? 12 : 13.5, 'sans', 700), mX, mY + 36, { size: narrow ? 12 : 13.5, align: 'center', weight: 700, color: err ? C.red : C.ink });
      // output and status
      const py = mY + 64, ph = h - py - 10;
      V.box(g, 12, py, w - 24, ph, { r: 12, fill: V.alpha(C.panel, 0.75), stroke: V.alpha(err ? C.red : C.muted, 0.4) });
      const fx = 26, done = n >= block.length;
      let outS, msg, mcol;
      if (term) {
        outS = '{"city": "' + block.slice(0, Math.min(n, 3)).join('');
        if (!done) { msg = 'the matcher advances token by token through the batch…'; mcol = C.muted; }
        else if (S8.fix) { msg = '#52805: accept_tokens stops at the terminal, counts it and ignores the rest. Request completed.'; mcol = C.a; }
        else { const secs = Math.min(1800, Math.round((rel - block.length * per) * 420)); msg = '"The matcher has terminated … but is trying to accept new token". sample_tokens waits… ' + secs + ' s / 1800 s (sped-up time, illustrative)'; mcol = C.red; }
      } else {
        if (!done) { outS = n >= 4 ? block.slice(3, n).join('') : '(thinking…)'; msg = '</think> arrived in the middle of the batch'; mcol = C.muted; }
        else if (S8.fix) { outS = '{"city":"Paris"}'; msg = '#44993 + #53046: the advance uses the step\'s new_token_ids, and the post-</think> drafts are validated against the new grammar.'; mcol = C.a; }
        else { outS = '{"city":"Paris{"city":"Paris"}'; msg = 'The tokens after </think> never entered the FSM: the grammar started over from scratch and duplicated the prefix (issue #24 in the DeepSeek recipe, fixed by the vLLM #44993 backport).'; mcol = C.red; }
      }
      V.text(g, 'output', fx, py + 20, { size: 11, font: 'mono', color: C.muted });
      V.text(g, clip(g, outS, w - 100, narrow ? 12 : 13.5), fx + 52, py + 21, { size: narrow ? 12 : 13.5, font: 'mono', weight: 600, color: done && !S8.fix && !term ? C.red : C.ink });
      textBlock(g, msg, fx, py + 46, w - 52, { size: narrow ? 11.5 : 12.5, color: mcol, weight: 600, lh: 17 });
    });

    /* ================= Fig 9: the Responses API LRU locker ================= */
    const f9 = $('#fig-store');
    fitH(f9, 340, 430);
    // The figure's own clock (the draw's t): it stops with the global ⏸, with the tab hidden and off screen,
    // so padlocks, falls and transitions do not "move on their own" while nobody is watching.
    const S9 = { mode: 'lru', cap: 4, items: [], next: 1, fall: [], evicted: 0, flash: '', flashT: -9, lost: false, arrow: null, auto: true, log: [], clock: 0, acc: 0 };
    const log9 = (s, k) => { S9.log.push({ s, k }); if (S9.log.length > 6) S9.log.shift(); };
    const rid = (n) => 'resp_' + String(n).padStart(2, '0');
    const now9 = () => S9.clock;
    function enforce9() {
      if (S9.mode !== 'lru') return;
      const tn = now9();
      while (S9.items.length > S9.cap) {
        let idx = -1;
        for (let i = S9.items.length - 1; i >= 0; i--) if (!(S9.items[i].pin > tn)) { idx = i; break; }
        if (idx < 0) break;
        const it = S9.items.splice(idx, 1)[0];
        if (it.cx != null) S9.fall.push({ id: it.id, x: it.cx, y: it.cy, t0: tn });
        S9.evicted++;
        log9('LRU eviction: ' + rid(it.id) + ' (least recent, unlocked)', 'ev');
      }
    }
    function nova9() { const id = S9.next++; S9.items.unshift({ id, pin: now9() + 1.6, cx: null, cy: null }); S9.lost = false; log9('POST /v1/responses store=true → ' + rid(id), 'new'); enforce9(); }
    function cont9() {
      if (!S9.items.length) { S9.flash = S9.lost ? '404: previous_response_id not found (state lost on restart)' : 'nothing to continue: create a response'; S9.flashT = now9(); if (S9.lost) log9('previous_response_id=' + rid(S9.next - 1) + ' → HTTP 404', 'err'); return; }
      const tn = now9();
      const prev = S9.items.splice(S9.items.length - 1, 1)[0];
      prev.pin = tn + 1.8;
      const nw = { id: S9.next++, pin: tn + 1.8, cx: null, cy: null };
      S9.items.unshift(nw, prev);
      S9.arrow = { from: nw.id, to: prev.id, t0: tn };
      log9('previous_response_id=' + rid(prev.id) + ' → ' + rid(nw.id) + ' (renews recency)', 'cont');
      enforce9();
    }
    function restart9() { S9.items = []; S9.fall = []; S9.lost = true; S9.flash = '⟳ restart: the store lives in process memory and was lost'; S9.flashT = now9(); log9('server restart: store wiped', 'err'); }
    const c9 = UI.controls(f9);
    UI.seg(c9, { label: 'Store', options: [{ v: 'lru', label: 'LRU backport' }, { v: 'stock', label: 'stock (unbounded)' }], value: 'lru', onChange: (v) => { S9.mode = v; enforce9(); } });
    UI.slider(c9, { label: 'Capacity', min: 2, max: 8, step: 1, value: 4, fmt: (v) => v + ' (real: 256)', onInput: (v) => { S9.cap = v; enforce9(); } });
    UI.button(c9, '+ new response', () => { S9.auto = false; nova9(); });
    UI.button(c9, '↪ continue the oldest', () => { S9.auto = false; cont9(); });
    UI.button(c9, '⟳ restart server', () => { S9.auto = false; restart9(); });
    UI.toggle(c9, { label: 'automatic traffic', value: true, onChange: (v) => { S9.auto = v; } });
    const rd9 = UI.readout(c9, 'entries · evictions', '0 · 0');
    const rng9 = V.rng(9);
    nova9(); nova9(); nova9();
    S9.items.forEach((it) => { it.pin = 0; }); // the locker starts with three responses that are already terminal
    // automatic traffic on the SAME clock as the padlocks (one request every 1.5 s of figure time),
    // so that the requests/expirations ratio does not depend on the frame rate
    function traffic9() {
      if (!S9.auto) return;
      if (S9.mode === 'stock' && S9.items.length > 20) { restart9(); return; }
      if (S9.items.length > 1 && rng9() < 0.42) cont9(); else nova9();
    }
    fig(f9, (g, w, h, t) => {
      const C = V.C();
      const fdt = V.clamp(t - S9.clock, 0, 0.1); // figure dt: 0 when paused
      S9.clock = t;
      S9.acc += fdt;
      if (S9.acc >= 1.5) { S9.acc -= 1.5; traffic9(); }
      enforce9(); // an expiring padlock releases the pending eviction
      const narrow = w < 600;
      const tn = S9.clock;
      const x0 = 16, W = w - 32;
      const lru = S9.mode === 'lru';
      V.text(g, clip(g, lru ? (narrow ? 'NEWEST ← LRU (cap ' + S9.cap + ') → OLDEST' : 'MOST RECENT  ←  STORE (LRU, cap = ' + S9.cap + ')  →  OLDEST') : 'STOCK STORE: nothing is evicted', W, 11), x0, 18, { size: 11, font: 'mono', color: C.muted });
      // grid: on phones, an LRU with more than 5 slots wraps into two rows
      const sh = 52, rowStep = sh + 26;
      let perRow, maxRows;
      if (lru) { const slots = S9.cap + 1; maxRows = narrow && slots > 5 ? 2 : 1; perRow = Math.ceil(slots / maxRows); }
      else { perRow = Math.max(4, Math.floor(W / (narrow ? 62 : 76))); maxRows = Math.max(1, Math.floor((h - 106) / rowStep)); }
      const sw = Math.min(narrow ? 76 : 96, (W - (perRow - 1) * 8) / perRow);
      const slotX = (i) => x0 + (i % perRow) * (sw + 8), slotY = (i) => 34 + Math.floor(i / perRow) * rowStep;
      const nSlots = perRow * maxRows;
      const hidden = S9.items.length > nSlots ? S9.items.length - (nSlots - 1) : 0;
      const shown = hidden ? S9.items.slice(0, nSlots - 1) : S9.items;
      // labels that fit in the box (≥ 11 px): "resp_12" or "#12"; "in progress" or "active"
      const idSz = narrow ? 11 : 12;
      const longId = measure(g, 'resp_00', idSz, 'mono', 600) <= sw - 8;
      const idLab = (id) => (longId ? rid(id) : '#' + String(id).padStart(2, '0'));
      const longAct = measure(g, 'in progress', 11, 'sans', 400) <= sw - 6;
      if (lru) for (let i = 0; i < S9.cap; i++) { g.setLineDash([4, 4]); V.box(g, slotX(i), slotY(i), sw, sh, { r: 10, stroke: V.alpha(C.muted, 0.4) }); g.setLineDash([]); }
      // evicted: they sink and vanish behind the live boxes
      S9.fall = S9.fall.filter((f) => tn - f.t0 < 1.0 && tn >= f.t0);
      S9.fall.forEach((f) => {
        const u = (tn - f.t0) / 1.0;
        const a = 0.8 * (1 - u), sc = 1 - 0.35 * u, fw = sw * sc, fh = sh * sc, fx2 = f.x + (sw - fw) / 2, y = f.y + 22 * V.ease.in(u) + (sh - fh) / 2;
        V.box(g, fx2, y, fw, fh, { r: 10, fill: V.alpha(C.red, 0.2 * a), stroke: V.alpha(C.red, a) });
        V.text(g, idLab(f.id), fx2 + fw / 2, y + fh * 0.42, { size: 11, font: 'mono', align: 'center', color: C.red, alpha: a });
        V.text(g, 'evicted', fx2 + fw / 2, y + fh * 0.8, { size: 11, align: 'center', color: C.red, alpha: a });
      });
      // positions (frame-rate-independent transition) and draw order: whoever moves the most goes on top
      const kk = 1 - Math.exp(-fdt * 12);
      const moves = shown.map((it, i) => {
        const tx = slotX(i), ty = slotY(i);
        if (it.cx == null) { it.cx = tx - 30; it.cy = ty; }
        it.cx = V.lerp(it.cx, tx, kk); it.cy = V.lerp(it.cy, ty, kk);
        return { it, i, d: Math.abs(it.cx - tx) + Math.abs(it.cy - ty) };
      });
      moves.slice().sort((a, b) => a.d - b.d).forEach(({ it, i }) => {
        const pinned = it.pin > tn;
        const over = lru && i >= S9.cap;
        V.box(g, it.cx, it.cy, sw, sh, { r: 10, fill: C.bg }); // opaque background: a box passing over hides the text of the one below
        V.box(g, it.cx, it.cy, sw, sh, { r: 10, fill: V.alpha(over ? C.d : C.a, pinned ? 0.3 : 0.16), stroke: over ? C.d : C.a, lw: pinned ? 2 : 1.2 });
        V.text(g, idLab(it.id), it.cx + sw / 2, it.cy + 22, { size: idSz, font: 'mono', align: 'center', weight: 600 });
        V.text(g, pinned ? (longAct ? 'in progress' : 'active') : 'terminal', it.cx + sw / 2, it.cy + 40, { size: 11, align: 'center', color: C.muted });
        if (pinned) lock(g, it.cx + sw - 12, it.cy + 3, 11, C.d); // padlock as a seal on the edge, without covering the id
      });
      if (hidden) {
        const hx = slotX(nSlots - 1), hy = slotY(nSlots - 1);
        g.setLineDash([4, 3]);
        V.box(g, hx, hy, sw, sh, { r: 10, fill: V.alpha(C.d, 0.12), stroke: C.d });
        g.setLineDash([]);
        V.text(g, '+' + hidden, hx + sw / 2, hy + 22, { size: idSz, font: 'mono', align: 'center', weight: 700, color: C.d });
        V.text(g, 'older', hx + sw / 2, hy + 40, { size: 11, align: 'center', color: C.muted });
      }
      if (S9.arrow && tn - S9.arrow.t0 < 1.8 && tn >= S9.arrow.t0) {
        const A = shown.find((x) => x.id === S9.arrow.from), B = shown.find((x) => x.id === S9.arrow.to);
        if (A && B && A.cx != null && B.cx != null) {
          V.arrow(g, A.cx + sw / 2, A.cy + sh + 2, B.cx + sw / 2, B.cy + sh + 2, { color: C.b, lw: 2, curve: -18 });
          const hw9 = measure(g, 'previous_response_id', 11, 'mono', 400) / 2 + 2;
          V.text(g, 'previous_response_id', V.clamp((A.cx + B.cx) / 2 + sw / 2, x0 + hw9, x0 + W - hw9), Math.max(A.cy, B.cy) + sh + 24, { size: 11, font: 'mono', align: 'center', color: C.b });
        }
      }
      // event log in the free space
      const rowsUsed = lru ? maxRows : Math.max(1, Math.ceil(Math.min(S9.items.length, nSlots) / perRow));
      const logY = 34 + rowsUsed * rowStep + 18;
      const logRows = Math.min(6, Math.floor((h - 104 - logY) / 16));
      if (logRows > 0) {
        V.text(g, 'EVENTS', x0, logY, { size: 11, font: 'mono', color: C.muted });
        S9.log.slice(-logRows).forEach((L, i) => V.text(g, clip(g, L.s, W, 11), x0, logY + 16 + i * 16, { size: 11, font: 'mono', color: L.k === 'ev' ? C.red : L.k === 'err' ? C.d : L.k === 'cont' ? C.b : C.a }));
      }
      // memory: the stock warning gets its own line, above the bar label
      const my = h - 58;
      if (!lru && S9.items.length > 12) V.text(g, clip(g, 'grows without bound (issue #62)', W, 11.5, 'sans', 600), x0, my - 26, { size: 11.5, color: C.red, weight: 600 });
      V.text(g, 'process memory (illustrative)', x0, my - 8, { size: 11, font: 'mono', color: C.muted });
      const frac = Math.min(1, S9.items.length / 26);
      V.box(g, x0, my, W, 14, { r: 7, fill: V.alpha(C.muted, 0.12) });
      V.box(g, x0, my, Math.max(6, W * frac), 14, { r: 7, fill: frac > 0.7 ? C.red : lru ? C.a : C.d });
      if (tn - S9.flashT < 2.6 && tn >= S9.flashT) V.text(g, clip(g, S9.flash, W, narrow ? 11 : 12.5, 'sans', 600), x0, h - 16, { size: narrow ? 11 : 12.5, color: S9.flash.startsWith('404') ? C.red : C.d, weight: 600 });
      else V.text(g, clip(g, lru ? (narrow ? 'active items are not evicted' : 'in-progress items are not evicted: the cap can be exceeded for a moment') : 'every response stays forever, until the restart', W, narrow ? 11 : 12, 'sans', 400), x0, h - 16, { size: narrow ? 11 : 12, color: C.muted });
      rd9.set(S9.items.length + ' · ' + S9.evicted);
    });

    /* ================= Fig 10: map of doors ================= */
    const f10 = $('#fig-rotas');
    fitH(f10, 400, 440);
    const S10 = { key: 'many', host: 'lan', extra: 'none', vis: 'nokey', redact: true };
    const R10 = [
      { p: '/v1/chat/completions', g: true }, { p: '/v1/responses', g: true }, { p: '/v1/models', g: true }, { p: '/v2/…', g: true }, { p: '/inference/…', g: true },
      { p: '/health' }, { p: '/metrics' }, { p: '/version' }, { p: '/tokenize' }, { p: '/detokenize' },
      { p: '/invocations', inf: true }, { p: '/generative_scoring', inf: true },
      { p: '/reset_prefix_cache', ex: 'reset' }, { p: '/reset_mm_cache', ex: 'reset' }, { p: '/reset_encoder_cache', ex: 'reset' },
      { p: '/sleep', ex: 'dev' }, { p: '/rpc', ex: 'dev' }, { p: '/rlhf', ex: 'dev' }, { p: '/server_info', ex: 'dev' },
    ];
    const c10 = UI.controls(f10);
    UI.seg(c10, { label: 'Key', options: [{ v: 'none', label: 'none' }, { v: 'glm', label: 'VLLM_API_KEY (GLM, env only)' }, { v: 'ds1', label: 'VLLM_API_KEY (DS, --api-key)' }, { v: 'many', label: 'DSPARK_API_KEYS (2)' }], value: 'many', onChange: (v) => { S10.key = v; } });
    UI.seg(c10, { label: 'Bind', options: [{ v: 'lan', label: '0.0.0.0' }, { v: 'local', label: '127.0.0.1' }], value: 'lan', onChange: (v) => { S10.host = v; } });
    UI.seg(c10, { label: 'Extra routes', options: [{ v: 'none', label: 'none' }, { v: 'reset', label: 'cache reset (GLM)' }, { v: 'dev', label: 'DEV_MODE' }], value: 'none', onChange: (v) => { S10.extra = v; } });
    UI.seg(c10, { label: 'Visitor', options: [{ v: 'nokey', label: 'LAN, no key' }, { v: 'key', label: 'LAN with key' }, { v: 'local', label: 'localhost, no key' }], value: 'nokey', onChange: (v) => { S10.vis = v; } });
    UI.toggle(c10, { label: 'log redaction (DS)', value: true, onChange: (v) => { S10.redact = v; } });
    const rd10 = UI.readout(c10, 'answering', '—');
    fig(f10, (g, w, h, t) => {
      const C = V.C();
      const narrow = w < 600;
      const routes = R10.filter((r) => !r.ex || (r.ex === 'reset' && S10.extra !== 'none') || (r.ex === 'dev' && S10.extra === 'dev'));
      const keysOn = S10.key !== 'none';
      const hasKey = S10.vis === 'key';
      const reach = !(S10.host === 'local' && S10.vis !== 'local');
      const stat = (r) => (!reach ? 'off' : r.g && keysOn && !hasKey ? '401' : '200');
      const x0 = 14, W = w - 28;
      // visitor
      const vx = narrow ? x0 + 20 : x0 + 34, vy = narrow ? 34 : 110;
      V.dot(g, vx, vy - 12, 9, S10.vis === 'key' ? C.a : S10.vis === 'local' ? C.b : C.d);
      V.box(g, vx - 13, vy, 26, 24, { r: 9, fill: S10.vis === 'key' ? C.a : S10.vis === 'local' ? C.b : C.d });
      V.text(g, S10.vis === 'key' ? 'LAN + Bearer' : S10.vis === 'local' ? 'localhost' : 'LAN', vx + (narrow ? 22 : 0), narrow ? vy - 8 : vy + 42, { size: 11, align: narrow ? 'left' : 'center', color: C.ink, weight: 600 });
      // wall/guard
      const gxw = narrow ? x0 : x0 + 90, gy0 = narrow ? 70 : 20;
      const doorsX = narrow ? x0 : gxw + 26, doorsW = narrow ? W : W - (gxw + 26 - x0);
      const cols = narrow ? 2 : 3;
      const dw = (doorsW - (cols - 1) * 8) / cols, dh = narrow ? 30 : 34;
      const rowsN = Math.ceil(routes.length / cols);
      const logH = narrow ? 92 : 70;
      const wantH = Math.max(narrow ? 0 : 200, gy0 + rowsN * (dh + 6) + 16 + logH + 8);
      const stg = f10.querySelector('.la-fig-stage');
      if (Math.abs(parseFloat(stg.style.height) - wantH) > 2) stg.style.height = Math.round(wantH) + 'px';
      const availH = h - gy0 - logH - 22;
      const dhh = Math.min(dh, (availH - (rowsN - 1) * 6) / rowsN);
      if (!narrow) {
        V.line(g, gxw, gy0, gxw, gy0 + rowsN * (dhh + 6), { color: V.alpha(C.muted, 0.5), lw: 3 });
        lock(g, gxw, gy0 + 30, 16, keysOn ? C.d : C.muted);
        V.text(g, keysOn ? 'Bearer' : 'no guard', gxw, gy0 + 54, { size: 11, align: 'center', color: keysOn ? C.d : C.muted, font: 'mono' });
      } else {
        lock(g, x0 + W - 12, 36, 14, keysOn ? C.d : C.muted);
        V.text(g, keysOn ? 'Bearer: only /v1 /v2 /inference' : 'no key: no guard at all', x0 + W - 26, 40, { size: 11, align: 'right', color: keysOn ? C.d : C.muted, font: 'mono' });
      }
      let open = 0;
      routes.forEach((r, i) => {
        const cx = doorsX + (i % cols) * (dw + 8), cy = gy0 + Math.floor(i / cols) * (dhh + 6);
        const s = stat(r);
        if (s === '200') open++;
        const col = s === 'off' ? C.muted : s === '401' ? C.d : r.inf || r.ex ? C.c : C.a;
        V.box(g, cx, cy, dw, dhh, { r: 7, fill: V.alpha(col, s === 'off' ? 0.06 : 0.16), stroke: V.alpha(col, s === 'off' ? 0.3 : 0.85) });
        const warn = (r.inf || r.ex) && s === '200';
        V.text(g, clip(g, (warn ? '⚠ ' : '') + r.p, dw - 44, narrow ? 11 : 11.5), cx + 8, cy + dhh / 2 + 1, { size: narrow ? 11 : 11.5, font: 'mono', base: 'middle', color: s === 'off' ? C.muted : C.ink });
        V.text(g, s === 'off' ? '—' : s, cx + dw - 8, cy + dhh / 2 + 1, { size: 11, font: 'mono', align: 'right', base: 'middle', weight: 700, color: col });
        if (s !== 'off' && !narrow) {
          const u = ((t * 0.5 + i * 0.13) % 1);
          const sx = vx + 14, sy = vy + 10, ex = s === '401' ? gxw - 4 : cx, ey = cy + dhh / 2;
          const px = V.lerp(sx, ex, u), py = V.lerp(sy, ey, u);
          V.dot(g, px, py, 2.2, V.alpha(col, 0.8));
        }
      });
      rd10.set(open + ' / ' + routes.length);
      // log
      const ly = Math.min(h - logH - 6, gy0 + rowsN * (dhh + 6) + 10);
      V.box(g, 12, ly, w - 24, logH, { r: 10, fill: V.alpha(C.panel, 0.8), stroke: V.alpha(C.muted, 0.35) });
      const nk = S10.key === 'many' ? 2 : 1;
      // GLM: the key arrives only via env (-e VLLM_API_KEY) and does not even appear in non-default args.
      // DS: VLLM_API_KEY / DSPARK_API_KEYS become --api-key in the argv; redaction only cleans the log line.
      const glmKey = S10.key === 'glm';
      const keyStr = S10.key === 'none' ? '(no api_key: API open)' : glmKey ? "(no 'api_key' here: the key came via env)" : S10.redact ? "'api_key': ['<redacted:" + nk + " value(s)>']" : "'api_key': [" + (nk === 2 ? "'sk-dspark-alice', 'sk-dspark-bob'" : "'sk-single-key'") + ']';
      const lsz = narrow ? 11 : 11.5;
      V.text(g, 'boot log · non-default args', 24, ly + 16, { size: 11, font: 'mono', color: C.muted });
      V.text(g, clip(g, keyStr, w - 48, lsz), 24, ly + 33, { size: lsz, font: 'mono', color: S10.key === 'none' ? C.d : glmKey || S10.redact ? C.a : C.red, weight: 600 });
      const argv = S10.key === 'none' ? 'bind ' + (S10.host === 'lan' ? '0.0.0.0: WARN "UNAUTHENTICATED API" in the launcher' : '127.0.0.1: only the head can reach it') : S10.key === 'many' ? 'DS: one --api-key sk-… sk-… in argv, visible in ps / docker inspect (redaction only hides the log)' : S10.key === 'ds1' ? 'DS: VLLM_API_KEY becomes --api-key sk-… in argv, visible in ps / docker inspect (log-only redaction)' : 'GLM: -e VLLM_API_KEY with no value → out of the argv, the log and the worker (nothing to redact)';
      const lines = wrapLines(g, argv, w - 48, lsz, 'mono', 400);
      lines.slice(0, narrow ? 3 : 2).forEach((l, i) => V.text(g, l, 24, ly + 51 + i * 14, { size: lsz, font: 'mono', color: C.muted }));
    });
    /* ================= Fig 11: whole call or nothing (TensorFold lane) ================= */
    const f11 = $('#fig-tf-calls');
    fitH(f11, 400, 575);
    const n1 = (v) => v.toFixed(1);
    const S11 = { strat: 'whole', sc: 'ok', t0: 0, reset: true, first: true };
    const TM11 = 14, IDLE11 = 5, KA11 = 2, CUT11 = 11;
    const THINK11 = 'The user wants to save the log. I will use write_file.';
    const TEXT11 = 'Saving the log.';
    const HEAD11 = '<tool_call>write_file<arg_key>path</arg_key><arg_value>/tmp/log.txt</arg_value><arg_key>content</arg_key><arg_value>';
    const L11 = { line: 'line', s: 's' };
    const c11 = UI.controls(f11);
    UI.seg(c11, { label: 'Strategy', options: [{ v: 'fim', label: 'v0.6.0: only at the end' }, { v: 'frag', label: '0036: fragments (≈ vLLM)' }, { v: 'whole', label: '0053: whole call' }], value: 'whole', onChange: (v) => { S11.strat = v; S11.reset = true; } });
    UI.seg(c11, { label: 'Scenario', options: [{ v: 'ok', label: 'complete call' }, { v: 'cut', label: 'cut at max_tokens' }], value: 'ok', onChange: (v) => { S11.sc = v; S11.reset = true; } });
    UI.button(c11, '↻ resend', () => { S11.reset = true; });
    const rd11a = UI.readout(c11, 'longest silence', '—');
    const rd11b = UI.readout(c11, 'finish_reason', '—');
    const lines11 = (s) => Math.max(1, Math.floor((Math.min(s, 13) - 4) / 0.12) + 1); // file lines written so far
    function plan11() {
      const cut = S11.sc === 'cut', end = cut ? CUT11 : 13.2, fin = cut ? CUT11 : 13.4;
      const E = [];
      for (let i = 1; i <= 8; i++) E.push({ t: i * 0.25, k: 'r' });
      [2.3, 2.55, 2.8].forEach((tt) => E.push({ t: tt, k: 'c' }));
      if (S11.strat === 'frag') {
        E.push({ t: 3.2, k: 'h' });
        for (let i = 0; 3.6 + 0.4 * i <= end - 0.15; i++) E.push({ t: 3.6 + 0.4 * i, k: 'a' });
        if (!cut) E.push({ t: 13.2, k: 'a', close: true });
      } else if (S11.strat === 'whole') {
        for (let tk = 2.8 + KA11; tk < end; tk += KA11) E.push({ t: tk, k: 'k' });
        if (!cut) E.push({ t: 13.2, k: 'W' });
      } else if (!cut) E.push({ t: 13.4, k: 'W' });
      E.push({ t: fin, k: 'f', v: cut ? 'length' : 'tool_calls' });
      E.sort((a, b) => a.t - b.t);
      let prev = 0, cutAt = null;
      for (const e of E) { if (e.t - prev > IDLE11) { cutAt = prev + IDLE11; break; } prev = e.t; }
      return { E, cutAt, end, cut };
    }
    function tail11(s, P) { // what the model just wrote
      if (s < 2) return '<think>' + THINK11.slice(0, Math.round(THINK11.length * s / 2));
      if (s < 2.8) return '</think>' + TEXT11.slice(0, Math.round(TEXT11.length * (s - 2) / 0.8));
      if (s < 4) return HEAD11.slice(0, Math.round(HEAD11.length * (s - 2.8) / 1.2));
      const k = lines11(Math.min(s, P.end));
      if (!P.cut && s >= 13) return L11.line + ' ' + k + ': ok</arg_value></tool_call>';
      return '<arg_value>' + L11.line + ' 1: ok↵…↵' + L11.line + ' ' + k + ': ok';
    }
    function args11(tl, closed) {
      if (tl < 3.6) return '';
      let s = '{"path": "/tmp/log.txt"';
      if (tl >= 4) { const k = lines11(tl); s += ', "content": "' + (k > 1 ? L11.line + ' 1: ok↵…↵' + L11.line + ' ' + k + ': ok' : L11.line + ' 1: ok'); }
      return closed ? s + '"}' : s;
    }
    const clipL = (g, s, maxW, sz) => { // clips on the left and keeps the end
      s = String(s);
      if (measure(g, s, sz, 'mono', 400) <= maxW) return s;
      let lo = 0, hi = s.length;
      while (lo < hi) { const m = (lo + hi) >> 1; if (measure(g, '…' + s.slice(m), sz, 'mono', 400) <= maxW) hi = m; else lo = m + 1; }
      return '…' + s.slice(lo);
    };
    fig(f11, (g, w, h, t) => {
      const C = V.C();
      if (S11.reset) { S11.t0 = t - (S11.first ? 9.3 : 0); S11.reset = false; S11.first = false; }
      const narrow = w < 600;
      let s = t - S11.t0;
      if (s > TM11 + 4.5) { S11.t0 = t; s = 0; }
      const now = Math.min(s, TM11);
      const P = plan11();
      const gone = P.cutAt != null && now >= P.cutAt;
      const horizon = gone ? P.cutAt : now;
      const got = P.E.filter((e) => e.t <= horizon + 1e-9);
      const fin = got.find((e) => e.k === 'f');
      const x0 = 14, x1 = w - 14, tx = (v) => x0 + (V.clamp(v, 0, TM11) / TM11) * (x1 - x0);
      const sz = narrow ? 11 : 11.5;
      // ---- what the model writes
      V.text(g, 'what the model writes', x0, 14, { size: sz, font: 'mono', color: C.muted });
      V.text(g, 't = ' + n1(now) + ' s (illustrative)', x1, 14, { size: sz, font: 'mono', color: C.muted, align: 'right' });
      const sy = 22, sh = narrow ? 24 : 26;
      const segs = [{ a: 0, b: 2, col: C.b, lab: '<think> thinks' }, { a: 2, b: 2.8, col: C.a, lab: 'text' }, { a: 2.8, b: P.cut ? CUT11 : 13.2, col: C.d, lab: narrow ? '<tool_call> write_file' : '<tool_call> write_file: the whole file goes in the arguments' }];
      segs.forEach((sg) => {
        const xa = tx(sg.a), xb = tx(sg.b) - 1, ww = Math.max(2, xb - xa);
        V.box(g, xa, sy, ww, sh, { r: 5, fill: V.alpha(sg.col, 0.1), stroke: V.alpha(sg.col, 0.6), lw: 1 });
        const xn = Math.min(xb, tx(now));
        if (xn > xa) { g.save(); V.rr(g, xa, sy, ww, sh, 5); g.clip(); g.fillStyle = V.alpha(sg.col, 0.34); g.fillRect(xa, sy, xn - xa, sh); g.restore(); }
        if (ww > 34) V.text(g, clip(g, sg.lab, ww - 10, sz), xa + 5, sy + sh / 2 + 1, { size: sz, font: 'mono', base: 'middle', color: C.ink });
      });
      if (P.cut) {
        const xc = tx(CUT11);
        V.line(g, xc, sy - 4, xc, sy + sh + 4, { color: C.red, lw: 2.4 });
        V.text(g, clip(g, 'max_tokens', x1 - xc - 6, sz), xc + 5, sy + sh / 2 + 1, { size: sz, font: 'mono', base: 'middle', color: C.red, weight: 600 });
      }
      V.text(g, clipL(g, tail11(now, P), x1 - x0, sz), x0, sy + sh + 17, { size: sz, font: 'mono', color: now < 2 ? C.b : now < 2.8 ? C.a : C.d });
      // ---- server
      const svY = sy + sh + 40;
      V.text(g, 'server', x0, svY, { size: sz, font: 'mono', color: C.muted });
      let stS, stC;
      if (now < 2) { stS = 'thinking → reasoning_content'; stC = C.b; } else if (now < 2.8) { stS = 'text → content'; stC = C.a; } else if (now < P.end) {
        const held = Math.round(HEAD11.length * V.clamp((now - 2.8) / 1.2, 0, 1) + (now >= 4 ? lines11(now) * 13 : 0));
        stS = S11.strat === 'fim' ? 'hides the call until the end (only the end parser reads it)' : S11.strat === 'frag' ? 'emits the arguments piece by piece' : 'holds the call: ' + held.toLocaleString('en-US') + ' characters';
        stC = C.d;
      } else if (!P.cut) { stS = S11.strat === 'whole' ? '</tool_call>: sends the whole call' : S11.strat === 'frag' ? '</tool_call>: last fragment' : '</tool_call>: the end parser builds the call'; stC = C.a; } else { stS = S11.strat === 'whole' ? 'the reply ended inside the call: nothing goes out' : 'the reply ended inside the call'; stC = S11.strat === 'whole' ? C.a : C.red; }
      const stT = clip(g, stS, x1 - x0 - 16, sz);
      const pw = pill(g, stT, 0, 0, { size: sz, dry: true });
      pill(g, stT, V.clamp(tx(now) - pw / 2, x0, x1 - pw), svY + 7, { size: sz, fill: V.alpha(stC, 0.16), stroke: stC });
      // ---- the SSE wire
      const wy = svY + 7 + sz + 10 + 26;
      V.text(g, 'SSE wire → client', x0, wy, { size: sz, font: 'mono', color: C.muted });
      const ay = wy + 36;
      V.line(g, tx(now), sy - 4, tx(now), ay + 24, { color: V.alpha(C.ink, 0.3), lw: 1, dash: [3, 3] });
      V.line(g, x0, ay, x1, ay, { color: V.alpha(C.muted, 0.55), lw: 1.5 });
      for (let k = 0; k <= TM11; k += 2) { const x = tx(k); V.line(g, x, ay + 3, x, ay + 7, { color: V.alpha(C.muted, 0.6), lw: 1 }); }
      const lastT = got.length ? got[got.length - 1].t : 0;
      if (!fin) {
        const xa = tx(lastT), xb = tx(horizon), sil = horizon - lastT;
        const col = sil >= IDLE11 ? C.red : sil > KA11 + 0.05 ? C.d : C.muted;
        if (xb - xa > 2) {
          V.box(g, xa, ay - 5, xb - xa, 10, { r: 4, fill: V.alpha(col, 0.25) });
          if (xb - xa > 84) V.text(g, 'silence ' + n1(sil) + ' ' + L11.s, (xa + xb) / 2, ay + 22, { size: 11, font: 'mono', align: 'center', color: col });
        }
      }
      const wT = (got.find((e) => e.k === 'W') || {}).t;
      let wBox = null; // where the whole-call pill sits: the {} labels under it are hidden
      if (wT != null) { const ww = pill(g, 'write_file{…}', 0, 0, { size: 11, dry: true }), wl = V.clamp(tx(wT) - ww / 2, x0, x1 - ww); wBox = [wl - 10, wl + ww + 10]; }
      got.forEach((e) => {
        const x = tx(e.t);
        if (e.k === 'r') V.dot(g, x, ay, 3, C.b);
        else if (e.k === 'c') V.dot(g, x, ay, 3.6, C.a);
        else if (e.k === 'a') V.box(g, x - 1.5, ay - 8, 3, 16, { r: 1.5, fill: C.d });
        else if (e.k === 'k') {
          g.beginPath(); g.arc(x, ay, 5, 0, Math.PI * 2); g.lineWidth = 1.6; g.strokeStyle = C.ink; g.stroke();
          if (!wBox || x < wBox[0] || x > wBox[1]) V.text(g, '{}', x, ay - 11, { size: 11, font: 'mono', align: 'center', color: C.muted });
        } else if (e.k === 'h' || e.k === 'W') {
          const lab = e.k === 'h' ? 'write_file' : 'write_file{…}';
          const ww = pill(g, lab, 0, 0, { size: 11, dry: true });
          V.line(g, x, ay - 9, x, ay + 9, { color: C.d, lw: 3 });
          pill(g, lab, V.clamp(x - ww / 2, x0, x1 - ww), ay - 34, { size: 11, fill: V.alpha(C.d, e.k === 'W' ? 0.4 : 0.2), stroke: C.d, glow: e.k === 'W' && now - e.t < 1.5 ? C.d : null });
        } else if (e.k === 'f') {
          const ww = pill(g, e.v, 0, 0, { size: 11, dry: true });
          V.line(g, x, ay - 7, x, ay + 7, { color: C.c, lw: 3 });
          pill(g, e.v, V.clamp(x - ww / 2, x0, x1 - ww), ay + 10, { size: 11, fill: V.alpha(C.c, 0.25), stroke: C.c });
        }
      });
      if (gone) {
        const xc = tx(P.cutAt);
        g.fillStyle = V.alpha(C.muted, 0.12); g.fillRect(xc, ay - 26, x1 - xc, 52);
        V.line(g, xc, ay - 28, xc, ay + 28, { color: C.red, lw: 2, dash: [4, 3] });
        V.text(g, clip(g, 'the client gives up', x1 - xc - 8, 11), xc + 6, ay + 22, { size: 11, font: 'mono', color: C.red, weight: 600 });
      }
      // ---- the client
      const py = ay + 40, ph = h - py - 8;
      V.box(g, 12, py, w - 24, ph, { r: 12, fill: V.alpha(C.panel, 0.75), stroke: V.alpha(gone ? C.red : C.a, 0.4) });
      const fx = 26, inl = !narrow, labW = 132;
      const meterW = narrow ? w - 52 : Math.min(250, (w - 24) * 0.34);
      const colW = narrow ? w - 52 : w - 52 - meterW - 24;
      const lab = narrow ? 11 : 12;
      V.text(g, 'the client has', fx, py + 18, { size: 11, font: 'mono', color: C.muted });
      let yy = py + 40;
      const row = (k, v, col, maxL, mark) => {
        const vw = inl ? colW - labW : colW;
        V.text(g, k, fx, yy, { size: 11, font: 'mono', color: C.muted });
        if (mark != null) {
          const mx = fx + measure(g, k, 11, 'mono', 400) + 12;
          if (mark) check(g, mx, yy - 4, 9, C.a); else cross(g, mx, yy - 4, 8, C.red);
          if (!inl) V.text(g, mark ? 'valid JSON' : 'incomplete JSON', mx + 10, yy, { size: 11, font: 'mono', color: mark ? C.a : C.red, weight: 600 });
        }
        let Ls = wrapChars(g, v, vw, lab, 'mono', 500);
        if (Ls.length > maxL) { Ls = Ls.slice(0, maxL); Ls[maxL - 1] = clip(g, Ls[maxL - 1] + '…', vw, lab); }
        const vx = inl ? fx + labW : fx, vy = inl ? yy : yy + 15;
        Ls.forEach((l, i) => V.text(g, l, vx, vy + i * 15, { size: lab, font: 'mono', weight: 500, color: col }));
        yy = vy + (Ls.length - 1) * 15 + (inl ? 20 : 23);
      };
      const nr = got.filter((e) => e.k === 'r').length, nc = got.filter((e) => e.k === 'c').length;
      let callS = '—', callC = C.muted, valid = null;
      if (S11.strat === 'frag') {
        const fr = got.filter((e) => e.k === 'a');
        if (got.some((e) => e.k === 'h')) { const lf = fr.length ? fr[fr.length - 1] : null; const closed = !!(lf && lf.close); callS = 'write_file(' + (lf ? args11(lf.t, closed) : '') + (closed ? ')' : ''); valid = lf ? closed : null; callC = C.d; }
      } else if (got.some((e) => e.k === 'W')) { callS = 'write_file(' + args11(13, true) + ')'; valid = true; callC = C.d; } else if (gone) callS = '(nothing arrived: connection dropped)';
      else if (fin && P.cut) { callS = '(none: a cut call is never sent)'; callC = C.a; } else if (now >= 2.8) callS = S11.strat === 'whole' ? '(held on the server: nothing yet)' : '(nothing yet)';
      row('reasoning_content', nr ? '"' + THINK11.slice(0, Math.round(THINK11.length * nr / 8)) + '"' : '—', C.b, 1);
      row('content', nc ? '"' + TEXT11.slice(0, Math.round(TEXT11.length * nc / 3)) + '"' : 'null', C.a, 1);
      row('tool_calls[0]', callS, callC, narrow ? 3 : 2, valid);
      row('finish_reason', fin ? fin.v : gone ? '(connection dropped)' : '…', fin ? (fin.v === 'length' ? C.d : C.c) : gone ? C.red : C.muted, 1);
      // silence meter
      const mx = narrow ? fx : w - 26 - meterW, my = narrow ? yy + 2 : py + 40;
      const sil = fin ? 0 : horizon - lastT, full = IDLE11 * 1.2;
      const mcol = sil >= IDLE11 ? C.red : sil > KA11 + 0.05 ? C.d : C.a;
      V.text(g, 'silence since the last delta', mx, my, { size: 11, font: 'mono', color: C.muted });
      const bx = mx, by = my + 9, bw = meterW, bh = 14;
      V.box(g, bx, by, bw, bh, { r: 7, fill: V.alpha(C.muted, 0.12), stroke: V.alpha(C.muted, 0.35), lw: 1 });
      if (sil > 0.02) V.box(g, bx, by, Math.max(8, bw * V.clamp(sil / full, 0, 1)), bh, { r: 7, fill: V.alpha(mcol, 0.6) });
      const xk = bx + bw * (KA11 / full), xi = bx + bw * (IDLE11 / full);
      V.line(g, xk, by - 3, xk, by + bh + 3, { color: C.ink, lw: 1, dash: [2, 2] });
      V.line(g, xi, by - 3, xi, by + bh + 3, { color: C.red, lw: 2 });
      V.text(g, '2 s', xk, by + bh + 14, { size: 11, font: 'mono', align: 'center', color: C.muted });
      V.text(g, 'timeout 5 s', Math.min(xi, bx + bw - 36), by + bh + 14, { size: 11, font: 'mono', align: 'center', color: C.red });
      V.text(g, fin ? 'stream closed' : 'now: ' + n1(Math.min(sil, IDLE11)) + ' s', bx, by + bh + 31, { size: 11.5, font: 'mono', color: fin ? C.a : mcol, weight: 600 });
      // message
      let msg, mc;
      if (gone) { msg = 'The client gave up after 5 s without a single byte. The server keeps generating, but nobody is listening.'; mc = C.red; } else if (fin) {
        if (S11.strat === 'frag') { if (P.cut) { msg = 'length, but the client already stored half a JSON. The stream is append-only: what was sent cannot be taken back.'; mc = C.red; } else { msg = 'tool_calls: the client joined the fragments (and watched the file grow) and runs write_file.'; mc = C.a; } } else if (S11.strat === 'whole') { msg = P.cut ? 'length, and no call went out: the client has nothing half-written to run.' : 'tool_calls: the call arrived whole, in a single delta, with valid JSON.'; mc = C.a; } else { msg = 'tool_calls: the call built by the end parser arrived all at once.'; mc = C.a; }
      } else if (now >= 2.8) {
        msg = S11.strat === 'whole' ? 'The call is held on the server. The empty delta every 2 s keeps the connection alive.' : S11.strat === 'frag' ? 'The arguments arrive piece by piece, as the model writes them.' : 'Nothing goes out while the call is being written, and the silence only grows.';
        mc = S11.strat === 'fim' ? C.d : C.ink;
      } else { msg = 'Thinking and text arrive token by token in all three strategies.'; mc = C.muted; }
      const msY = narrow ? by + bh + 52 : Math.max(yy + 4, by + bh + 52);
      const ML = wrapLines(g, msg, w - 52, narrow ? 12 : 12.5, 'sans', 600).slice(0, 3);
      ML.forEach((l, i) => V.text(g, l, fx, msY + i * 17, { size: narrow ? 12 : 12.5, weight: 600, color: mc }));
      let mg = 0, pv = 0;
      got.forEach((e) => { mg = Math.max(mg, e.t - pv); pv = e.t; });
      if (!fin) mg = Math.max(mg, horizon - pv);
      rd11a.set(n1(Math.min(mg, IDLE11)) + ' s', mg >= IDLE11 - 1e-6 ? 'bad' : mg > KA11 + 0.05 ? 'warn' : 'ok');
      rd11b.set(fin ? fin.v : gone ? 'connection dropped' : '…', fin ? (fin.v === 'length' ? 'warn' : 'ok') : gone ? 'bad' : undefined);
    });

    /* ================= Fig 12: the history that comes back (TensorFold lane) ================= */
    const f12 = $('#fig-tf-historia');
    fitH(f12, 420, 590);
    const S12 = { mode: 'think', keep: false, fix: false, t0: 0, reset: true, first: true };
    const c12 = UI.controls(f12);
    UI.seg(c12, { label: 'Case', options: [{ v: 'think', label: 'dropped reasoning' }, { v: 'args', label: 'corrupted arguments (#2)' }], value: 'think', onChange: (v) => { S12.mode = v; g12a.show(v === 'think'); g12b.show(v === 'args'); S12.reset = true; } });
    const g12a = UI.group(c12), g12b = UI.group(c12);
    UI.toggle(g12a.el, { label: 'KeptReasoning (0036)', value: false, onChange: (v) => { S12.keep = v; S12.reset = true; } });
    UI.toggle(g12b.el, { label: 'leave the call out (0051)', value: false, onChange: (v) => { S12.fix = v; S12.reset = true; } });
    g12b.show(false);
    UI.button(c12, '↻ replay', () => { S12.reset = true; });
    const rd12 = UI.readout(c12, 'result', '—');
    const T12 = { sys: 1200, user: 120, R1: 600, c1: 60, t1: 900, R2: 450, c2: 50, t2: 300, gen: 8 };
    function rows12(keep) {
      const slot = (id) => (keep ? { id, n: T12[id], kind: 'R', lab: '<think>' + id } : { id: 'E' + id, n: 2, kind: 'E', lab: '∅', min: 30 });
      const A = [{ id: 'sys', n: T12.sys, kind: 'S', lab: 'system + tools' }, { id: 'user', n: T12.user, kind: 'S', lab: 'user' }, slot('R1'), { id: 'c1', n: T12.c1, kind: 'C', lab: 'read_file' }, { id: 't1', n: T12.t1, kind: 'S', lab: 'tool' }];
      const row1 = A.concat([{ id: 'R2', n: T12.R2, kind: 'R', lab: '<think>R2', gen: true }, { id: 'c2', n: T12.c2, kind: 'C', lab: 'run_tests', gen: true }]);
      const row3 = A.concat([slot('R2'), { id: 'c2', n: T12.c2, kind: 'C', lab: 'run_tests' }, { id: 't2', n: T12.t2, kind: 'S', lab: 'tool' }, { id: 'gen', n: T12.gen, kind: 'G', lab: '▶', min: 22 }]);
      return { row1, row3 };
    }
    function scale12(row, W, minW) {
      let sc = W / row.reduce((a, b) => a + b.n, 0);
      for (let it = 0; it < 4; it++) {
        let fixed = 0, flex = 0;
        row.forEach((b) => { const m = b.min || minW; if (b.n * sc < m) fixed += m; else flex += b.n; });
        sc = Math.max(0.0001, (W - fixed) / Math.max(1, flex));
      }
      return sc;
    }
    const lay12 = (row, x0, sc, minW) => { let x = x0; return row.map((b) => { const bw = Math.max(b.min || minW, b.n * sc); const r = Object.assign({}, b, { x, w: bw }); x += bw + 2; return r; }); };
    function block12(g, b, y, hh, C, sz, alpha) {
      const col = b.kind === 'R' ? C.b : b.kind === 'C' ? C.d : b.kind === 'E' ? C.red : b.kind === 'G' ? C.a : C.muted;
      g.globalAlpha = alpha == null ? 1 : alpha;
      V.box(g, b.x, y, b.w, hh, { r: 5, fill: V.alpha(col, b.kind === 'S' ? 0.13 : 0.24), stroke: V.alpha(col, 0.8), lw: b.kind === 'E' ? 1.8 : 1 });
      if (b.w > 18) V.text(g, clip(g, b.lab, b.w - 6, sz), b.x + b.w / 2, y + hh / 2 + 1, { size: sz, font: 'mono', align: 'center', base: 'middle', color: b.kind === 'E' ? C.red : C.ink });
      g.globalAlpha = 1;
    }
    function result12(g, items, x, y, W, sz, lh) {
      items.forEach((it) => {
        if (it.ok === true) check(g, x + 6, y - 4, 9, it.col); else if (it.ok === false) cross(g, x + 6, y - 4, 8, it.col);
        const L = wrapLines(g, it.s, W - 22, sz, 'sans', 600);
        L.forEach((l, i) => V.text(g, l, x + 20, y + i * lh, { size: sz, weight: 600, color: it.col }));
        y += L.length * lh + 6;
      });
      return y;
    }
    function think12(g, w, h, s, C, narrow) {
      const x0 = 14, x1 = w - 14, W = x1 - x0, sz = 11, hh = narrow ? 26 : 28;
      const ref = rows12(true).row3;
      const sc = scale12(ref, W - 2 * (ref.length - 1), narrow ? 22 : 24);
      const { row1, row3 } = rows12(S12.keep);
      const r1 = lay12(row1, x0, sc, narrow ? 22 : 24), r3 = lay12(row3, x0, sc, narrow ? 22 : 24);
      let div = 0;
      while (div < r1.length && div < r3.length && r1[div].id === r3[div].id) div++;
      // row 1: the previous turn
      const y1 = 22;
      V.text(g, clip(g, narrow ? 'previous turn: prompt 2 + generated' : 'previous turn: step 2 prompt + what the model generated', W, 11.5), x0, 14, { size: 11.5, font: 'mono', color: C.muted });
      const rev = V.clamp(s / 1.0, 0, 1);
      r1.forEach((b, i) => block12(g, b, y1, hh, C, sz, V.clamp(rev * r1.length - i, 0, 1)));
      const gA = r1.find((b) => b.gen), gB = r1[r1.length - 1];
      if (gA && rev >= 1) {
        V.line(g, gA.x, y1 + hh + 6, gB.x + gB.w, y1 + hh + 6, { color: C.b, lw: 1.5 });
        V.text(g, 'generated in step 2', gB.x + gB.w, y1 + hh + 20, { size: 11, font: 'mono', color: C.b, align: 'right' });
      }
      // middle: the client and the LRU
      const mY = y1 + hh + 30;
      const cw = narrow ? W : W * 0.55, chh = narrow ? 112 : 122;
      const showC = V.clamp((s - 1.0) / 0.6, 0, 1);
      g.globalAlpha = showC;
      V.box(g, x0, mY, cw, chh, { r: 10, fill: V.alpha(C.panel, 0.8), stroke: V.alpha(C.muted, 0.4) });
      V.text(g, clip(g, narrow ? 'the client resends (no reasoning_content)' : 'the client resends the history (no reasoning_content)', cw - 24, 11), x0 + 12, mY + 17, { size: 11, font: 'mono', color: C.muted });
      const CL = [['user', '"test_log fails, fix it"', C.ink, null], ['assistant', '→ read_file', C.d, '—'], ['tool', 'contents of log.py', C.ink, null], ['assistant', '→ run_tests', C.d, '—'], ['tool', '"2 failures"', C.ink, null]];
      const lh = narrow ? 17 : 19;
      CL.forEach((r, i) => {
        const yy = mY + 37 + i * lh;
        V.text(g, r[0], x0 + 12, yy, { size: 11.5, font: 'mono', color: C.muted });
        V.text(g, clip(g, r[1], cw - 120 - (r[3] ? 90 : 0), 11.5), x0 + 92, yy, { size: 11.5, font: 'mono', color: r[2] });
        if (r[3]) V.text(g, 'reasoning: —', x0 + cw - 12, yy, { size: 11, font: 'mono', color: C.red, align: 'right' });
      });
      g.globalAlpha = 1;
      const lx = narrow ? x0 : x0 + cw + 14, ly = narrow ? mY + chh + 10 : mY, lw = narrow ? W : W - cw - 14, lhh = narrow ? 84 : chh;
      const lcol = S12.keep ? C.b : C.muted;
      g.globalAlpha = showC;
      V.box(g, lx, ly, lw, lhh, { r: 10, fill: V.alpha(lcol, S12.keep ? 0.1 : 0.05), stroke: V.alpha(lcol, S12.keep ? 0.7 : 0.35), lw: 1.4 });
      V.text(g, clip(g, 'KeptReasoning · LRU on the server', lw - 24, 11.5), lx + 12, ly + 17, { size: 11.5, font: 'mono', color: lcol, weight: 600 });
      const ent = [['call_7f3a…', 'R1'], ['call_c21d…', 'R2']];
      const entY = (i) => ly + (narrow ? 36 : 40) + i * (narrow ? 18 : 22);
      ent.forEach((e, i) => {
        const yy = entY(i);
        V.text(g, clip(g, e[0] + ' · sha256(history + calls)', lw - 70, 11), lx + 12, yy, { size: 11, font: 'mono', color: S12.keep ? C.ink : C.muted });
        pill(g, e[1], lx + lw - 48, yy - 13, { size: 11, fill: V.alpha(C.b, S12.keep ? 0.3 : 0.08), stroke: V.alpha(C.b, S12.keep ? 0.9 : 0.3) });
      });
      V.text(g, clip(g, S12.keep ? 'up to 1,024 replies · 32 MiB' : 'off: the template puts <think></think>', lw - 24, 11), lx + 12, ly + lhh - 12, { size: 11, font: 'mono', color: S12.keep ? C.muted : C.red });
      g.globalAlpha = 1;
      // row 3: the next prompt
      const y3 = narrow ? ly + lhh + 34 : mY + chh + 34;
      V.text(g, clip(g, 'step 3 prompt, built by the template', W, 11.5), x0, y3 - 8, { size: 11.5, font: 'mono', color: C.muted });
      const tb = (i) => 2.0 + i * 0.28;
      r3.forEach((b, i) => {
        const a = V.clamp((s - tb(i)) / 0.25, 0, 1);
        if (a <= 0) return;
        block12(g, b, y3, hh, C, sz, a);
        if (b.kind === 'E' && s - tb(i) < 0.9) V.glow(g, b.x + b.w / 2, y3 + hh / 2, 26, C.red, 0.7 * (1 - (s - tb(i)) / 0.9));
      });
      if (S12.keep) {
        r3.forEach((b, i) => {
          if (b.kind !== 'R') return;
          const k = b.id === 'R1' ? 0 : 1, u = (s - (tb(i) - 0.55)) / 0.55;
          if (u <= 0 || u >= 1) return;
          const e = V.ease.inOut(u), sx = lx + lw - 36, sy0 = entY(k) - 3, ex = b.x + b.w / 2, ey = y3 + hh / 2;
          const p = V.bez([sx, sy0], [sx, (sy0 + ey) / 2], [ex, (sy0 + ey) / 2], [ex, ey], e);
          pill(g, b.id, p[0] - 13, p[1] - 10, { size: 11, fill: V.alpha(C.b, 0.45), stroke: C.b, glow: C.b });
        });
      }
      // comparison sweep
      const scan = V.clamp((s - 4.8) / 1.8, 0, 1);
      if (scan > 0) {
        const xs = x0 + scan * W;
        r3.forEach((b, i) => {
          if (b.x > xs) return;
          const col = i < div ? C.a : S12.keep ? C.a : C.red;
          V.box(g, b.x, y3 + hh + 4, Math.min(b.w, xs - b.x), 4, { r: 2, fill: col });
          if (i < div && i < r1.length) V.box(g, r1[i].x, y1 + hh + 1, Math.min(r1[i].w, xs - r1[i].x), 3, { r: 1.5, fill: V.alpha(C.a, 0.8) });
        });
        if (scan < 1) [y1, y3].forEach((yr) => V.line(g, xs, yr - 5, xs, yr + hh + 9, { color: V.alpha(C.ink, 0.45), lw: 1.2, dash: [3, 3] }));
        if (!S12.keep && div < r3.length && xs >= r3[div].x) {
          const xd = r3[div].x - 1;
          [y1, y3].forEach((yr) => V.line(g, xd, yr - 6, xd, yr + hh + 6, { color: C.red, lw: 2.2, dash: [5, 3] }));
          V.text(g, 'diverges here', Math.min(xd + 6, x1 - 80), y3 + hh + 22, { size: 11, font: 'mono', color: C.red, weight: 600 });
        }
        if (S12.keep && scan >= 1) V.text(g, 'new', r3[div].x + 2, y3 + hh + 22, { size: 11, font: 'mono', color: C.a, weight: 600 });
      }
      const ry = y3 + hh + (narrow ? 44 : 46);
      if (s > 6.7) {
        const items = S12.keep ? [{ ok: true, col: C.a, s: 'The step 3 prompt continues the previous turn exactly as it was generated.' }, { ok: true, col: C.a, s: 'The reasoning comes back by call id (or by the SHA-256 signature of the conversation), and only into the conversation it was written for.' }]
          : [{ ok: false, col: C.red, s: 'The prompt stops matching what the model generated in step 2, and the prefix cache re-reads the rest.' }, { ok: false, col: C.red, s: '2 steps show up with <think></think>: a context the model never wrote.' }];
        result12(g, items, x0, ry, W, narrow ? 12 : 12.5, 17);
        rd12.set(S12.keep ? 'continues token by token' : 'diverges at step 2', S12.keep ? 'ok' : 'bad');
      } else {
        V.text(g, s < 4.7 ? 'building the prompt from the client history…' : 'comparing with the previous turn…', x0, ry, { size: 12, font: 'mono', color: C.muted });
        rd12.set('…');
      }
      // legend
      const lg = [[C.b, 'reasoning'], [C.d, 'call'], [C.red, '∅ = <think></think>'], [C.muted, 'system · user · tool']];
      const LR = [[]];
      let acc = 0;
      lg.forEach((it) => { const tw = measure(g, it[1], 11, 'mono', 400) + 26; if (acc + tw > W && LR[LR.length - 1].length) { LR.push([]); acc = 0; } LR[LR.length - 1].push([it[0], it[1], tw]); acc += tw; });
      LR.forEach((rw, j) => {
        let lxx = x0;
        const lyy = h - 12 - (LR.length - 1 - j) * 17;
        rw.forEach(([col, tt, tw]) => {
          V.box(g, lxx, lyy - 9, 11, 11, { r: 3, fill: V.alpha(col, 0.35), stroke: col, lw: 1 });
          V.text(g, tt, lxx + 16, lyy, { size: 11, font: 'mono', color: C.muted });
          lxx += tw;
        });
      });
    }
    function args12(g, w, h, s, C, narrow) {
      const x0 = 14, x1 = w - 14, W = x1 - x0;
      const hw = narrow ? W : W * 0.56;
      const HL = [['user', '"fix the test_log test"', 0], ['assistant', 'read_file {"path": "log.py"}', 1], ['tool', 'call_1 → contents of log.py', 0], ['assistant', 'write_file "{\\"path\\": \\"log.p', 2], ['tool', 'call_2 → "ok"', 3], ['assistant', 'run_tests {}', 1], ['tool', 'call_3 → "2 failures"', 0]];
      const lh = narrow ? 19 : 22, hy = 14;
      const hh = 28 + HL.length * lh + (S12.fix ? 22 : 8);
      V.box(g, x0, hy, hw, hh, { r: 10, fill: V.alpha(C.panel, 0.8), stroke: V.alpha(C.muted, 0.4) });
      V.text(g, clip(g, 'history the agent resends every turn', hw - 24, 11), x0 + 12, hy + 18, { size: 11, font: 'mono', color: C.muted });
      const omit = S12.fix && s > 0.6;
      HL.forEach((r, i) => {
        const yy = hy + 40 + i * lh;
        const bad = r[2] === 2, res = r[2] === 3, gone = omit && (bad || res);
        if (bad && !gone) V.box(g, x0 + 6, yy - 14, hw - 12, lh - 2, { r: 6, fill: V.alpha(C.red, 0.12), stroke: V.alpha(C.red, 0.6), lw: 1 });
        g.globalAlpha = gone ? 0.4 : 1;
        V.text(g, r[0], x0 + 12, yy, { size: 11.5, font: 'mono', color: C.muted });
        const tcol = bad ? C.red : r[0] === 'assistant' ? C.d : C.ink;
        const tw = hw - 104 - (bad && !narrow ? 24 : 0);
        V.text(g, clip(g, r[1], tw, 11.5), x0 + 92, yy, { size: 11.5, font: 'mono', color: tcol });
        if (gone) V.line(g, x0 + 90, yy - 4, x0 + 92 + Math.min(tw, measure(g, clip(g, r[1], tw, 11.5), 11.5, 'mono', 400)), yy - 4, { color: C.d, lw: 1.6 });
        g.globalAlpha = 1;
        if (r[2] === 1) check(g, x0 + hw - 18, yy - 4, 8, C.a);
        if (bad && !gone) cross(g, x0 + hw - 18, yy - 4, 7, C.red);
      });
      if (S12.fix) V.text(g, clip(g, 'log: call_2 + its result left out', hw - 24, 11), x0 + 12, hy + hh - 10, { size: 11, font: 'mono', color: C.d, weight: 600 });
      // turns
      const tx0 = narrow ? x0 : x0 + hw + 16, ty0 = narrow ? hy + hh + 16 : hy, tw = narrow ? W : W - hw - 16;
      V.text(g, 'next turns', tx0, ty0 + 14, { size: 11, font: 'mono', color: C.muted });
      const rowH = narrow ? 30 : 34;
      let ok = 0, shown = 0;
      for (let i = 0; i < 4; i++) {
        const st = 0.9 + i * 1.3, u = (s - st) / 0.9;
        if (u <= 0) continue;
        shown++;
        const yy = ty0 + 28 + i * rowH;
        V.text(g, 'turn ' + (i + 1), tx0, yy + 14, { size: 11.5, font: 'mono', color: C.ink });
        const pw = pill(g, S12.fix ? '200 · tool_calls' : 'HTTP 400', 0, 0, { size: 11, dry: true });
        const ax0 = tx0 + 68, ax1 = Math.min(tx0 + tw - (narrow ? 118 : 128), tx0 + tw - pw - 10);
        const e = V.clamp(u, 0, 1);
        if (S12.fix) {
          V.arrow(g, ax0, yy + 10, V.lerp(ax0, ax1, e), yy + 10, { color: C.a, lw: 1.6, head: 6 });
          if (e >= 1) { pill(g, '200 · tool_calls', ax1 + 8, yy, { size: 11, fill: V.alpha(C.a, 0.22), stroke: C.a }); ok++; }
        } else {
          const back = V.clamp((u - 0.5) * 2, 0, 1);
          const xh = e < 0.5 ? V.lerp(ax0, ax1, e * 2) : V.lerp(ax1, ax0 + 10, back);
          V.dot(g, xh, yy + 10, 4, e < 0.5 ? C.d : C.red);
          V.line(g, ax0, yy + 10, ax1, yy + 10, { color: V.alpha(C.muted, 0.4), lw: 1, dash: [3, 3] });
          if (e >= 0.5) pill(g, 'HTTP 400', ax1 + 8, yy, { size: 11, fill: V.alpha(C.red, 0.2), stroke: C.red });
        }
      }
      // conclusion
      const ry = narrow ? ty0 + 28 + 4 * rowH + 16 : hy + hh + 28;
      if (s > 6.2) {
        const items = S12.fix ? [{ ok: true, col: C.a, s: 'The broken call and its result (matched by id) are left out of the prompt; the other calls, the text and the reasoning stay.' }, { ok: true, col: C.a, s: 'The omission goes to the log and the conversation goes on: leaving it out only takes away a step the model could no longer read.' }]
          : [{ ok: false, col: C.red, s: 'The agent resends the same history every turn, and the 400 always comes back: the conversation is stuck.' }, { ok: null, col: C.muted, s: 'On the output, failing loudly protects whoever executes the call (#191). In the history, refusing kills the conversation.' }];
        result12(g, items, x0, ry, W, narrow ? 12 : 12.5, 17);
      }
      // the prompt the server builds
      const PB = [['user', 120, 'S'], ['read_file', 60, 'C'], ['tool', 700, 'S'], ['write_file', 90, 'X'], ['tool', 40, 'X'], ['run_tests', 40, 'C'], ['tool', 300, 'S'], ['▶', 8, 'G']];
      const py0 = h - (narrow ? 72 : 58), pbh = narrow ? 22 : 24;
      V.text(g, 'prompt the server builds this turn', x0, py0 - 8, { size: 11, font: 'mono', color: C.muted });
      const shrink = S12.fix ? V.ease.inOut(V.clamp((s - 0.6) / 0.8, 0, 1)) : 0;
      const totN = PB.reduce((a, b) => a + b[1], 0);
      let pxx = x0;
      const usable = W - 2 * (PB.length - 1) - 8 * 24;
      PB.forEach(([lb, n, k]) => {
        let bw = Math.max(24, (n / totN) * usable);
        if (k === 'X') bw *= 1 - shrink;
        if (bw < 1) return;
        const col = k === 'C' ? C.d : k === 'X' ? (S12.fix ? C.muted : C.red) : k === 'G' ? C.a : C.muted;
        V.box(g, pxx, py0, bw, pbh, { r: 5, fill: V.alpha(col, k === 'S' ? 0.13 : 0.24), stroke: V.alpha(col, 0.8), lw: 1 });
        if (bw > 20) V.text(g, clip(g, lb, bw - 6, 11), pxx + bw / 2, py0 + pbh / 2 + 1, { size: 11, font: 'mono', align: 'center', base: 'middle', color: k === 'X' && !S12.fix ? C.red : C.ink });
        pxx += bw + 2;
      });
      if (!S12.fix) {
        const pulse = 0.5 + 0.5 * Math.sin(s * 4);
        V.box(g, x0 - 2, py0 - 3, W + 4, pbh + 6, { r: 7, stroke: V.alpha(C.red, 0.5 + 0.4 * pulse), lw: 2 });
      }
      wrapLines(g, S12.fix ? 'the broken call and its result are gone; the model reads the rest' : 'refused with HTTP 400 before rendering: nothing reaches the model', W, 11.5, 'mono', 600).slice(0, 2).forEach((l, i) => V.text(g, l, x0, py0 + pbh + 17 + i * 15, { size: 11.5, font: 'mono', weight: 600, color: S12.fix ? C.a : C.red }));
      rd12.set('turns ok: ' + ok + '/4', shown < 4 ? undefined : ok === 4 ? 'ok' : 'bad');
    }
    fig(f12, (g, w, h, t) => {
      const C = V.C();
      if (S12.reset) { S12.t0 = t - (S12.first ? 7.6 : 0); S12.reset = false; S12.first = false; }
      const narrow = w < 600;
      let s = t - S12.t0;
      if (s > 11) { S12.t0 = t; s = 0; }
      if (S12.mode === 'think') think12(g, w, h, s, C, narrow); else args12(g, w, h, s, C, narrow);
    });
  },
});
