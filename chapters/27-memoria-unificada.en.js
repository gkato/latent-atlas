LA.chapter({
  id: 'memoria-unificada',
  part: 5,
  order: 27,
  title: 'Unified memory: CPU and GPU in the same RAM',
  navTitle: 'Unified memory',
  subtitle: 'On the GB10, the operating system, the page cache and the KV cache compete for the same 128 GB. Learn what that does to gpu_memory_utilization, to weight loading, to swap and to the safety of the host.',
  minutes: 58,
  recipes: ['glm', 'tf', 'qwen', 'ds'],
  tags: ['UMA', 'gpu_memory_utilization', 'page cache', 'MemAvailable', 'mmap', 'O_DIRECT', 'swap', 'cgroups', 'NFS', 'offload', 'TensorFold'],
  goals: [
    'Explain why, in unified memory, <b>MemFree ≠ MemAvailable</b> decides whether a model boots, and how page cache, <code>drop_caches</code> and <code>posix_fadvise</code> factor into the math',
    'Pick a <code>--gpu-memory-utilization</code> for the GB10 knowing that each 0.01 ≈ 1.2 GiB taken from the host, and read the values used by the three recipes (0.80–0.87)',
    'Compare the load paths (mmap, clone, O_DIRECT/InstantTensor) and the host defenses (swap, watermarks, cgroups, watchdogs) using the measured numbers',
    'Build the complete budget of a node (weights + KV + activations + graphs + outside the budget + OS) and derive the GMU the way Qwen’s <code>start-tp1.sh</code> does',
    'Compare vLLM’s budget (GMU, with a pinned or derived KV) with TensorFold’s (MemAvailable − reserve, a header-based estimate, refusal before loading) and say who absorbs the host’s variation in each',
  ],
  prereqs: ['gpu-por-dentro'],
  summary: [
    'On the GB10, CPU and GPU share ~121.69 GiB of LPDDR5X: page cache, processes, driver, weights and KV all come out of the same pool. Exhausting that pool <b>hangs the host</b>, with no OOM.',
    '<code>torch.cuda.mem_get_info()</code> returns the host’s <b>MemFree</b>, and page cache counts as used. InstantTensor aborted after an rsync (MemFree ~2 GiB, MemAvailable ~117 GiB) until the GLM patch sized it by MemAvailable.',
    '<code>--gpu-memory-utilization</code> on UMA: each 0.01 ≈ 1.2 GiB of host headroom. GLM 0.85 (KV pinned at 11 GiB), Qwen 0.80 (down from 0.835 after MemAvailable of 0.3–0.9 GiB), DeepSeek 0.835 (≈ +165K tokens per +0.01). CUDA graphs can sit outside the GMU (estimate turned off in DeepSeek; 0.54 GiB not subtracted in Qwen), and then they come out of the host headroom. The byte-sized KV suggestion in the log is for pinning the pool, never for growing it.',
    'Loading: mmap goes through the page cache; on the GLM kit’s 64 KiB kernel, the H2D copy from mmap hangs (hence <code>clone()</code>; on 4 KiB the patch is inert, check <code>getconf PAGESIZE</code>). O_DIRECT (InstantTensor) skips the cache: 164 GiB in 36 s at 4.82 GB/s, but it eats KV. <code>posix_fadvise(DONTNEED)</code> evicts the checkpoint without root.',
    'Swap and UVM: <code>swappiness=0</code> (it only postpones swapping), swap kept on and cycled before every start. The livelock leaves <code>/health</code> at 200, so detection relies on progress in <code>/metrics</code>. VM tunables helped in the TP1 exploration but were not enough (crash #3), and in dual the floor + scale cost 7–14% of decode; the TP1 that works runs with no sysctl changes.',
    'Where the weights live: offloading the PLE on UMA does not save physical memory, but with the table in evictable mmap and the host-side handshake (the GB10 has no stream memory ops), Qwen fit on a single Spark (~104 → ~77 GiB non-evictable). NFS over the CX7 with <code>nconnect=8</code> avoids duplicating 126–164 GiB per node, at the cost of streaming the checkpoint on every cold start; mounting over the management network is refused.',
    'TensorFold variant of the GLM recipe: no GMU. The budget is the MemAvailable at startup minus a reserve (14.5 GiB; default max(4 GiB, RAM/10)), the estimate comes from the weights’ headers, and anything that doesn’t fit is refused <b>before</b> loading, along with the largest window that does fit (<code>start.sh</code> restarts once). The pool, shared by the 4 requests, holds one full window plus what’s left (up to 12.5 GiB: 2,922,496 tokens), and the reserve covers the ~10 GiB peak beyond the estimate. On vLLM with a pinned KV, the host’s headroom absorbs the variation; on TensorFold, the capacity does.',
  ],
  quiz: [
    { q: 'After a 164 GiB rsync, the GB10 shows MemFree ≈ 2 GiB and MemAvailable ≈ 117 GiB. Why did stock InstantTensor abort?', o: ['Because the NVMe disk was full', 'Because on the GB10 mem_get_info() returns the host’s MemFree, and the (reclaimable) page cache counts as used', 'Because MemAvailable does not include the page cache', 'Because the 64 KiB kernel forbids O_DIRECT'], a: 1, e: 'On UMA, the "free GPU memory" is the host’s MemFree: 0.5 × MemFree ≈ 1.17 GiB, less than the ~1.18 GiB buffer, with 100+ GiB of reclaimable cache right next to it. The patch looks at MemAvailable.' },
    { q: 'On the GB10 (≈121.69 GiB per node), what does raising --gpu-memory-utilization from 0.835 to 0.845 do?', o: ['Nothing, because the OS uses a different memory', 'It gives vLLM ~1.2 GiB more (≈ +165K KV tokens in DeepSeek) and takes ~1.2 GiB away from the host', 'It increases memory bandwidth by 1%', 'It only changes the number of captured CUDA graphs'], a: 1, e: '0.01 × 121.69 ≈ 1.2 GiB, taken from the same RAM the operating system uses. Qwen dropped to 0.80 because at 0.835 MemAvailable sat at 0.3–0.9 GiB under soak.' },
    { q: 'In the Qwen boot at 0.835: budget 101.61 GiB, weights + non-torch 68.52, activation 1.07, KV 32.02 and 0.54 GiB of captured CUDA graphs. Where are the graphs?', o: ['Inside the KV', 'Outside the GMU: the math adds up without them (101.61 − 68.52 − 1.07 = 32.02), so they come out of the host headroom', 'Inside the GMU, subtracted from the activation', 'Nowhere: graphs take up no memory'], a: 1, e: 'With VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 the estimate comes out of the KV (inside the GMU); with no subtraction, as in Qwen, or with 0, as in DeepSeek, the graphs live outside the budget, and on UMA that comes out of the host.' },
    { q: 'During a UVM livelock on the GB10, which signal does NOT help detect it?', o: ['GPU at ~96% utilization drawing only 12–20 W', 'prompt+generation counters frozen in /metrics while requests are running', 'The /health endpoint', '"No available shared memory broadcast block found in 60 seconds" messages'], a: 2, e: '/health stays at 200: in V1 it only reads an error flag and never probes the workers. Reliable detection is the lack of progress in /metrics over several ticks.' },
    { q: 'What does the PLE patch on the GB10 replace, and why?', o: ['It replaces NVFP4 with BF16, because the GB10 has no FP4', 'It replaces the cuStreamWaitValue32 wait with a host-side handshake using a sequence number, because the GB10 reports CAN_USE_STREAM_MEM_OPS=0 and, inside a CUDA graph, nothing waits', 'It replaces ZMQ with NCCL, because ZMQ is slow', 'It replaces mmap with O_DIRECT reads, because mmap hangs'], a: 1, e: 'Without stream memory ops, the wait "returns success" and the captured graph waits for nothing: the worker spun forever. The patch synchronizes the copy_stream on the CPU, writes flag[0] = seq, and the model thread waits for flag ≥ seq.' },
    { q: 'Why does GLM use nconnect=8 and refuse to mount NFS over 10.0.0.x?', o: ['Because NFSv3 requires 8 connections', 'Because several parallel TCP connections fill the CX7 cable, and the management network would turn a 164 GiB load into an overnight job', 'Because 10.0.0.x is IPv6', 'Because nconnect turns off the worker’s page cache'], a: 1, e: 'A single TCP connection rarely fills 200 Gb/s: "nconnect=8 is what makes a cold load use the cable". The 10.0.0.x addresses are loopback/management aliases.' },
    { q: 'In the TensorFold variant of the GLM recipe, another process holds 10 GiB of (non-reclaimable) RAM before startup. What happens?', o: ['The engine starts with the same pool, and the host’s headroom under load drops by 10 GiB', 'The budget (MemAvailable − 14.5 GiB) shrinks: the token pool gets smaller and, if not even the 1M window fits, startup refuses before loading and start.sh restarts with the largest window that fits', 'docker run --memory limits the engine and prevents the conflict', 'Nothing changes: the kernel drops the other process’s memory as if it were page cache'], a: 1, e: 'TensorFold budgets by what is available at startup and leaves the reserve untouched, so what absorbs the variation is the capacity (a pool of ~2.1 to 2.9M tokens, then the window), not the host’s headroom. The first option describes vLLM with a pinned KV; on the GB10, Docker’s --memory doesn’t charge GPU allocations; and anonymous memory can’t be dropped like page cache.' },
  ],
  mount(root, LA) {
    const V = LA.viz, UI = LA.ui;
    const MT = 121.69; // GiB visible per node (GB10)

    /* ============================ helpers ============================ */
    const onFill = (C) => C.bg; // text on saturated fills: the theme background color
    const fmt1 = (x) => (Math.round(x * 10) / 10).toFixed(1);
    const fmt2 = (x) => x.toFixed(2);
    const fmt3 = (x) => x.toFixed(3);
    const fmtInt = (x) => Math.round(x).toLocaleString('en-US');
    function hatch(g, x, y, w, h, col, step = 6) {
      if (w <= 0.5 || h <= 0.5) return;
      g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
      g.strokeStyle = col; g.lineWidth = 1; g.beginPath();
      for (let k = -h; k < w; k += step) { g.moveTo(x + k, y + h); g.lineTo(x + k + h, y); }
      g.stroke(); g.restore();
    }
    /** stacked bar: parts = [{v, c, l, hatch, a}] */
    function stack(g, x, y, w, h, parts, total, o = {}) {
      const C = V.C();
      const r = o.r == null ? 8 : o.r;
      V.box(g, x, y, w, h, { r, fill: V.alpha(C.muted, 0.07), stroke: V.alpha(C.muted, 0.35), lw: 1 });
      g.save(); V.rr(g, x, y, w, h, r); g.clip();
      let cx = x; const out = [];
      parts.forEach((p) => {
        const sw = Math.max(0, (p.v / total) * w);
        if (sw > 0.2) {
          if (p.hatch) { g.fillStyle = V.alpha(p.c, 0.13); g.fillRect(cx, y, sw, h); hatch(g, cx, y, sw, h, V.alpha(p.c, 0.5)); }
          else { g.fillStyle = V.alpha(p.c, p.a == null ? 0.9 : p.a); g.fillRect(cx, y, sw, h); }
          g.fillStyle = V.alpha(C.bg, 0.9); g.fillRect(cx + sw - 1, y, 1, h);
        }
        out.push({ x: cx, w: sw, p });
        cx += sw;
      });
      g.restore();
      out.forEach(({ x: sx, w: sw, p }) => {
        if (!p.l) return;
        const size = o.size || 11;
        g.font = V.font(size, 'sans', 600);
        const tw = g.measureText(p.l).width;
        if (tw + 8 < sw) V.text(g, p.l, sx + sw / 2, y + h / 2 + 0.5, { size, weight: 600, align: 'center', base: 'middle', color: p.lc || (p.hatch ? C.ink : onFill(C)) });
      });
      return out;
    }
    function wrap(g, s, x, y, maxW, lh, o = {}) {
      g.font = V.font(o.size || 12, o.font || 'sans', o.weight || 400);
      const words = String(s).split(' ');
      let line = '', yy = y, n = 0;
      for (const wd of words) {
        const test = line ? line + ' ' + wd : wd;
        if (g.measureText(test).width > maxW && line) {
          V.text(g, line, x, yy, o); yy += lh; line = wd; n++;
          if (o.maxLines && n >= o.maxLines) return yy;
        } else line = test;
      }
      if (line) { V.text(g, line, x, yy, o); yy += lh; }
      return yy;
    }
    function card(g, x, y, w, h, title, sub, col, o = {}) {
      const C = V.C();
      V.box(g, x, y, w, h, { r: 10, fill: V.alpha(col, o.on ? 0.22 : 0.1), stroke: V.alpha(col, o.on ? 1 : 0.7), lw: o.on ? 2 : 1.3 });
      V.text(g, title, x + 10, y + (o.ty || 19), { size: o.ts || 13, weight: 700, color: C.ink, maxW: w - 16 });
      if (sub) V.text(g, sub, x + 10, y + (o.sy || 35), { size: o.ss || 11, color: C.muted, maxW: w - 16 });
    }
    function segLen(pts) { const s = []; let L = 0; for (let i = 1; i < pts.length; i++) { const d = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); s.push(d); L += d; } return { s, L }; }
    function along(pts, u) {
      const { s, L } = segLen(pts); let d = V.clamp(u, 0, 1) * L;
      for (let i = 0; i < s.length; i++) {
        if (d <= s[i] || i === s.length - 1) { const f = s[i] ? Math.min(1, d / s[i]) : 0; return [V.lerp(pts[i][0], pts[i + 1][0], f), V.lerp(pts[i][1], pts[i + 1][1], f), i]; }
        d -= s[i];
      }
      return [pts[0][0], pts[0][1], 0];
    }
    function poly(g, pts, o = {}) {
      g.beginPath(); g.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) g.lineTo(pts[i][0], pts[i][1]);
      g.strokeStyle = o.color || V.C().muted; g.lineWidth = o.lw || 2; g.lineJoin = 'round';
      if (o.dash) g.setLineDash(o.dash); if (o.off) g.lineDashOffset = o.off;
      g.stroke(); g.setLineDash([]); g.lineDashOffset = 0;
    }
    function sq(g, x, y, s, col) { V.rr(g, x - s / 2, y - s / 2, s, s, Math.min(3, s / 3)); g.fillStyle = col; g.fill(); }
    function fitStage(fig, mobileH) {
      const st = fig.querySelector('.la-fig-stage');
      if (!st) return;
      const w = st.getBoundingClientRect().width || root.getBoundingClientRect().width;
      if (w && w < 560) st.style.height = mobileH + 'px';
    }
    function badge(g, s, x, y, col, o = {}) {
      const C = V.C(); const size = o.size || 11;
      g.font = V.font(size, o.font || 'sans', 600);
      const tw = g.measureText(s).width;
      const bx = o.align === 'center' ? x - tw / 2 - 7 : o.align === 'right' ? x - tw - 14 : x;
      V.box(g, bx, y - size + 1, tw + 14, size + 9, { r: 7, fill: V.alpha(col, 0.16), stroke: V.alpha(col, 0.8), lw: 1 });
      V.text(g, s, bx + 7, y + 3, { size, weight: 600, color: o.color || C.ink, font: o.font || 'sans' });
      return tw + 14;
    }

    /* ================= Fig 1 — three memory architectures ================= */
    {
      const fig = root.querySelector('#fig-arquiteturas');
      fitStage(fig, 470);
      const S = { mode: 'uma' };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Architecture', options: [{ v: 'disc', label: 'Discrete GPU' }, { v: 'c2c', label: 'Coherent superchip' }, { v: 'uma', label: 'Unified (GB10 / Apple)' }], value: 'uma', onChange: (v) => { S.mode = v; } });
      const facts = {
        disc: ['Two physical memories, two independent budgets.', 'Every weight crosses PCIe: the CPU → GPU copy is mandatory.', 'Page cache and the OS never take a single byte from VRAM.'],
        c2c: ['Two physical memories, one coherent address space.', 'The GPU can read the CPU’s LPDDR5X over the link, but "far" is slower.', 'Non-uniform access (NUMA): there is still a "near" and a "far".'],
        uma: ['One physical pool: ~121.69 GiB visible per node on the GB10.', 'No PCIe, but a file can occupy RAM twice (page cache + tensor).', 'CPU and GPU compete for capacity and bandwidth (~273 GB/s theoretical).'],
      };
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(); const sm = w < 560;
        const pad = sm ? 10 : 20;
        const bw = (w - pad * 2) * (sm ? 0.42 : 0.34), bh = sm ? 48 : 56;
        const cpu = { x: pad, y: 14, w: bw, h: bh }, gpu = { x: w - pad - bw, y: 14, w: bw, h: bh };
        const factsH = sm ? 92 : 70;
        const poolY = cpu.y + bh + (sm ? 42 : 46), poolH = h - poolY - factsH - 8;
        const mode = S.mode;
        card(g, cpu.x, cpu.y, cpu.w, cpu.h, 'CPU', sm ? 'ARM cores' : 'ARM cores (Grace in the GB10)', C.b, { ts: sm ? 12 : 13, ss: sm ? 10 : 11, sy: sm ? 34 : 38 });
        card(g, gpu.x, gpu.y, gpu.w, gpu.h, 'GPU', sm ? 'SMs + tensor cores' : 'SMs + tensor cores (Blackwell)', C.a, { ts: sm ? 12 : 13, ss: sm ? 10 : 11, sy: sm ? 34 : 38 });
        const cy = cpu.y + bh / 2;
        const tenants = (x, y, w2, h2, list) => {
          // vertical list of tenants inside a pool
          const n = list.length, gh = Math.min(22, (h2 - 30) / n - 4);
          list.forEach((it, i) => {
            const yy = y + 28 + i * (gh + 4);
            V.box(g, x + 8, yy, w2 - 16, gh, { r: 5, fill: it.h ? V.alpha(it.c, 0.12) : V.alpha(it.c, 0.85) });
            if (it.h) hatch(g, x + 8, yy, w2 - 16, gh, V.alpha(it.c, 0.5));
            V.text(g, it.l, x + 14, yy + gh / 2 + 0.5, { size: 11, weight: 600, base: 'middle', color: it.h ? C.ink : onFill(C), maxW: w2 - 26 });
          });
        };
        const packets = (pts, n, col, speed = 0.16, size = 7, off = 0) => {
          for (let k = 0; k < n; k++) { const u = (k / n + t * speed + off) % 1; const p = along(pts, u); sq(g, p[0], p[1], size, col); }
        };
        if (mode === 'disc' || mode === 'c2c') {
          const ram = { x: cpu.x, y: poolY, w: bw, h: poolH }, vram = { x: gpu.x, y: poolY, w: bw, h: poolH };
          const c2c = mode === 'c2c';
          V.box(g, ram.x, ram.y, ram.w, ram.h, { r: 12, fill: V.alpha(C.b, 0.06), stroke: V.alpha(C.b, 0.6), lw: 1.2 });
          V.box(g, vram.x, vram.y, vram.w, vram.h, { r: 12, fill: V.alpha(C.a, 0.06), stroke: V.alpha(C.a, 0.6), lw: 1.2 });
          V.text(g, c2c ? 'CPU LPDDR5X' : 'Host RAM (DDR)', ram.x + 10, ram.y + 17, { size: 11.5, weight: 600, color: C.b, maxW: ram.w - 16 });
          V.text(g, c2c ? 'GPU HBM' : 'VRAM (HBM / GDDR)', vram.x + 10, vram.y + 17, { size: 11.5, weight: 600, color: C.a, maxW: vram.w - 16 });
          tenants(ram.x, ram.y, ram.w, ram.h, [{ l: 'OS + services', c: C.b }, { l: 'processes (Python)', c: C.e }, { l: 'page cache', c: C.d, h: true }]);
          tenants(vram.x, vram.y, vram.w, vram.h, [{ l: 'weights', c: C.a }, { l: 'KV cache', c: C.c }, { l: 'activations', c: C.f }]);
          V.line(g, cpu.x + bw / 2, cpu.y + bh, ram.x + bw / 2, ram.y, { color: V.alpha(C.b, 0.7), lw: 3 });
          V.line(g, gpu.x + bw / 2, gpu.y + bh, vram.x + bw / 2, vram.y, { color: V.alpha(C.a, 0.7), lw: 3 });
          // link CPU-GPU
          const lx1 = cpu.x + bw, lx2 = gpu.x;
          const ay = ram.y + ram.h * 0.66;
          if (c2c) {
            V.box(g, lx1 + 2, cy - 9, lx2 - lx1 - 4, 18, { r: 9, fill: V.alpha(C.d, 0.25), stroke: C.d, lw: 1.5 });
            V.line(g, lx1 + 8, cy, lx2 - 8, cy, { color: C.d, lw: 2, dash: [6, 6], dashOffset: -t * 30 });
            V.text(g, 'coherent NVLink-C2C', (lx1 + lx2) / 2, cy - 15, { size: 11, align: 'center', color: C.d, weight: 600, maxW: lx2 - lx1 });
            // the GPU reads the LPDDR5X directly
            V.arrow(g, vram.x + 6, ay, ram.x + ram.w - 6, ay, { color: V.alpha(C.a, 0.8), lw: 1.6, dash: [5, 4], curve: 16 });
            V.text(g, 'GPU reads memory "from afar"', w / 2, ay + 16, { size: 11, align: 'center', color: C.muted, maxW: lx2 - lx1 + 40 });
          } else {
            V.box(g, lx1 + 2, cy - 5, lx2 - lx1 - 4, 10, { r: 5, fill: V.alpha(C.muted, 0.2), stroke: V.alpha(C.muted, 0.7), lw: 1 });
            V.text(g, 'PCIe', (lx1 + lx2) / 2, cy - 11, { size: 11, align: 'center', color: C.muted, weight: 600 });
          }
          // disk
          const dk = { x: w / 2 - (sm ? 26 : 34), y: ram.y + ram.h - 34, w: sm ? 52 : 68, h: 30 };
          V.box(g, dk.x, dk.y, dk.w, dk.h, { r: 6, fill: V.alpha(C.muted, 0.14), stroke: C.muted, lw: 1, label: 'NVMe', size: 11 });
          const rowH = Math.min(22, (ram.h - 30) / 3 - 4);
          const pcY = ram.y + 28 + 2 * (rowH + 4) + rowH / 2;
          const inX = ram.x + ram.w - 18;
          const load = [[dk.x, dk.y + dk.h / 2], [inX, dk.y + dk.h / 2], [inX, pcY]];
          poly(g, load, { color: V.alpha(C.d, 0.6), lw: 1.5, dash: [4, 4], off: -t * 20 });
          if (!c2c) {
            const up = [[inX, pcY], [inX, cpu.y + bh + 2], [cpu.x + bw - 4, cy], [gpu.x + 4, cy], [gpu.x + 18, gpu.y + bh + 2], [gpu.x + 18, vram.y + 28 + rowH / 2]];
            poly(g, up, { color: V.alpha(C.d, 0.45), lw: 1.5, dash: [4, 4], off: -t * 20 });
            packets(load, 3, C.d, 0.2, sm ? 6 : 7);
            packets(up, 6, C.d, 0.12, sm ? 6 : 7);
            V.text(g, 'copy over PCIe', (lx1 + lx2) / 2, cy + 20, { size: 11, align: 'center', color: C.d, maxW: lx2 - lx1 });
          } else {
            packets(load, 3, C.d, 0.2, 6);
            packets([[ram.x + ram.w - 10, ay], [vram.x + 10, ay]], 5, C.a, 0.25, 6);
          }
        } else {
          // UMA
          const pool = { x: pad, y: poolY, w: w - 2 * pad, h: poolH };
          V.box(g, pool.x, pool.y, pool.w, pool.h, { r: 14, fill: V.alpha(C.a, 0.05), stroke: V.alpha(C.a, 0.7), lw: 1.5 });
          V.text(g, sm ? 'Unified LPDDR5X · 128 GB' : 'Unified LPDDR5X · 128 GB (≈ 121.69 GiB visible to CUDA)', pool.x + 12, pool.y + 18, { size: sm ? 11 : 12, weight: 700, color: C.ink, maxW: pool.w - 20 });
          // tenant bar (illustrative)
          const bx = pool.x + 12, bw2 = pool.w - 24, by = pool.y + 34, bh2 = Math.min(46, pool.h * 0.3);
          const parts = [
            { v: 6, c: C.b, l: 'OS' }, { v: 5, c: C.e, l: sm ? 'proc.' : 'processes' }, { v: 14, c: C.d, l: 'page cache', hatch: true },
            { v: 64, c: C.a, l: 'weights' }, { v: 26, c: C.c, l: 'KV' }, { v: 6.69, c: C.muted, l: '', hatch: true },
          ];
          const segs = stack(g, bx, by, bw2, bh2, parts, MT, { size: 11 });
          V.text(g, 'illustrative proportions', bx + bw2, by + bh2 + 13, { size: 11, align: 'right', color: C.muted });
          // CPU/GPU -> pool connections
          const cxm = cpu.x + bw / 2, gxm = gpu.x + bw / 2;
          V.line(g, cxm, cpu.y + bh, cxm, pool.y, { color: V.alpha(C.b, 0.8), lw: 4 });
          V.line(g, gxm, gpu.y + bh, gxm, pool.y, { color: V.alpha(C.a, 0.8), lw: 4 });
          V.text(g, 'same controller, same bandwidth', w / 2, pool.y - 12, { size: 11, align: 'center', color: C.muted, maxW: gxm - cxm - 10 });
          // disk and paths inside the RAM
          const dk = { x: pool.x + 12, y: pool.y + pool.h - 36, w: sm ? 54 : 70, h: 28 };
          V.box(g, dk.x, dk.y, dk.w, dk.h, { r: 6, fill: V.alpha(C.muted, 0.14), stroke: C.muted, lw: 1, label: 'NVMe', size: 11 });
          const pc = segs[2], pw = segs[3], kv = segs[4];
          const pcC = [pc.x + pc.w / 2, by + bh2 / 2], pwC = [pw.x + pw.w * 0.3, by + bh2 / 2];
          packets([[dk.x + dk.w, dk.y + dk.h / 2], [pcC[0], dk.y + dk.h / 2], pcC], 6, C.d, 0.16, 7);
          const midY = by + bh2 + (pool.h - bh2 - 34) * 0.45 + 10;
          const copyPath = [pcC, [pcC[0], midY], [pwC[0], midY], pwC];
          poly(g, copyPath, { color: V.alpha(C.d, 0.55), lw: 1.5, dash: [4, 4], off: -t * 20 });
          V.text(g, 'copy within the same RAM', (pcC[0] + pwC[0]) / 2, midY + 15, { size: 11, align: 'center', color: C.d, maxW: Math.max(90, pwC[0] - pcC[0] + 60) });
          packets(copyPath, 4, C.d, 0.12, 6, 0.5);
          // the GPU reads weights and KV
          packets([[pw.x + pw.w * 0.75, by], [gxm, gpu.y + bh + 4]], 5, C.a, 0.35, 6);
          packets([[kv.x + kv.w / 2, by], [gxm + 8, gpu.y + bh + 4]], 3, C.c, 0.3, 6, 0.3);
          packets([[segs[0].x + segs[0].w / 2, by], [cxm, cpu.y + bh + 4]], 3, C.b, 0.3, 6);
        }
        // facts
        const fy = h - factsH + 8;
        V.line(g, pad, fy - 8, w - pad, fy - 8, { color: V.alpha(C.muted, 0.25), lw: 1 });
        let yy = fy + 8;
        facts[mode].forEach((f, i) => { V.dot(g, pad + 5, yy - 4, 3, [C.a, C.d, C.c][i]); yy = wrap(g, f, pad + 16, yy, w - pad * 2 - 16, 15, { size: sm ? 11 : 12, color: C.ink }) + (sm ? 1 : 2); });
      });
    }

    /* ================= Fig 2 — /proc/meminfo live ================= */
    {
      const fig = root.querySelector('#fig-meminfo');
      fitStage(fig, 560);
      const CK0 = MT - 4.7 - 3 - 2.1;
      const S = { os: 4.7, ck: CK0, oth: 3, cu: 0, patched: true, msg: 'Initial state: right after the 164 GiB rsync to the worker (the real GLM case).' };
      const D = { os: 4.7, ck: CK0, oth: 3, cu: 0 };
      const ctl = UI.controls(fig);
      const act = (fn) => () => { fn(); };
      UI.button(ctl, '① 164 GiB rsync', act(() => { if (S.cu > 0) { S.cu = 0; } S.oth = Math.max(S.oth, 3); S.ck = Math.max(0, MT - S.os - S.oth - S.cu - 2.1); S.msg = 'The rsync read 164 GiB: the page cache kept what fit; MemFree dropped to ~2 GiB.'; }), { primary: true });
      UI.button(ctl, '② drop_caches (root)', act(() => { S.ck = 0; S.oth = 0.3; S.msg = 'sync; echo 1 > /proc/sys/vm/drop_caches: all the clean cache was dropped (needs root, and works only on the host).'; }));
      UI.button(ctl, '③ posix_fadvise on the checkpoint only', act(() => { S.ck = 0; S.msg = 'evict_page_cache.py: DONTNEED on the .safetensors files. Only the checkpoint pages leave; the rest of the cache stays. No root.'; }));
      UI.button(ctl, '④ vLLM allocates 0.85 × total', act(() => {
        const need = 0.85 * MT; const room = MT - S.os - 1.5 - need;
        let cache = S.ck + S.oth;
        if (cache > room) { let over = cache - room; const o1 = Math.min(S.oth, over); S.oth -= o1; over -= o1; S.ck = Math.max(0, S.ck - over); }
        S.cu = need; S.msg = 'The kernel drops clean cache on demand: the allocation fits. The problem was never a lack of memory.';
      }));
      UI.button(ctl, '↺ reset', act(() => { Object.assign(S, { ck: 0, oth: 3, cu: 0, msg: 'Freshly booted host: almost everything is free.' }); }));
      UI.toggle(ctl, { label: 'UMA cold-load patch (GLM)', value: true, onChange: (v) => { S.patched = v; } });
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(); const sm = w < 560; const pad = sm ? 10 : 18;
        ['os', 'ck', 'oth', 'cu'].forEach((k) => { D[k] += (S[k] - D[k]) * Math.min(1, dt * 3.5); });
        const free = Math.max(0, MT - D.os - D.ck - D.oth - D.cu);
        const cache = D.ck + D.oth;
        const avail = free + cache;
        V.text(g, 'UNIFIED POOL · 121.69 GiB', pad, 18, { size: 11, font: 'mono', color: C.muted });
        const by = 26, bh = sm ? 34 : 38, bw = w - 2 * pad;
        stack(g, pad, by, bw, bh, [
          { v: D.os, c: C.b, l: 'OS+proc.' }, { v: D.cu, c: C.a, l: 'CUDA (weights, KV)' },
          { v: D.ck, c: C.d, l: 'cache: checkpoint', hatch: true }, { v: D.oth, c: C.f, l: 'cache: other', hatch: true },
          { v: free, c: C.muted, l: 'free', hatch: false, a: 0.0, lc: C.muted },
        ], MT, { size: 11 });
        // legend
        let lx = pad, ly = by + bh + 16;
        [[C.b, 'OS + processes'], [C.a, 'CUDA allocations'], [C.d, 'page cache (checkpoint)'], [C.f, 'page cache (other)']].forEach(([c, l]) => {
          g.font = V.font(11, 'sans', 400); const tw = g.measureText(l).width;
          if (lx + tw + 22 > w - pad) { lx = pad; ly += 15; }
          V.box(g, lx, ly - 8, 10, 10, { r: 3, fill: c }); V.text(g, l, lx + 14, ly + 1, { size: 11, color: C.muted }); lx += tw + 28;
        });
        // terminal meminfo
        const ty = ly + 14, tw = sm ? bw : Math.min(300, bw * 0.44), th = 118;
        V.box(g, pad, ty, tw, th, { r: 10, fill: V.alpha(C.panel, 0.9), stroke: V.alpha(C.muted, 0.35), lw: 1 });
        V.text(g, '$ cat /proc/meminfo', pad + 10, ty + 18, { size: 11, font: 'mono', color: C.muted });
        const rows = [['MemTotal:', fmt2(MT) + ' GiB', C.ink], ['MemFree:', fmt2(free) + ' GiB', C.c], ['Cached:', fmt2(cache) + ' GiB', C.d], ['MemAvailable:', fmt2(avail) + ' GiB', C.a]];
        rows.forEach((r, i) => {
          V.text(g, r[0], pad + 10, ty + 38 + i * 17, { size: 11.5, font: 'mono', color: C.ink });
          V.text(g, r[1], pad + tw - 10, ty + 38 + i * 17, { size: 11.5, font: 'mono', color: r[2], align: 'right', weight: 600 });
        });
        V.text(g, 'mem_get_info() → ' + fmt2(free) + ' GiB (= MemFree)', pad + 10, ty + th - 10, { size: 11, font: 'mono', color: C.c, maxW: tw - 16 });
        // meters
        let gx = sm ? pad : pad + tw + 18, gy = sm ? ty + th + 14 : ty + 6, gw = sm ? bw : bw - tw - 18;
        const meter = (label, v, col, y) => {
          V.text(g, label, gx, y, { size: 11, color: C.muted, weight: 600 });
          V.box(g, gx, y + 6, gw, 14, { r: 7, fill: V.alpha(C.muted, 0.12) });
          V.box(g, gx, y + 6, Math.max(3, (v / MT) * gw), 14, { r: 7, fill: V.alpha(col, 0.85) });
          V.text(g, fmt1(v) + ' GiB', gx + gw, y, { size: 11, font: 'mono', color: col, align: 'right', weight: 600 });
        };
        meter('MemFree (what CUDA thinks it has)', free, C.c, gy + 10);
        meter('MemAvailable (what can be allocated)', avail, C.a, gy + 44);
        const diff = avail - free;
        V.text(g, 'difference = reclaimable cache: ' + fmt1(diff) + ' GiB', gx, gy + 88, { size: 11, color: C.d, maxW: gw });
        // verdicts
        const vy0 = sm ? gy + 112 : ty + th + 22;
        const loaded = D.cu > 1;
        const lines = [];
        const budgetStock = 0.5 * free;
        if (loaded) lines.push(['·', C.muted, 'InstantTensor: weights already loaded (the judges below apply to the boot).']);
        else if (budgetStock < 1.18) lines.push(['✕', C.red, 'Stock InstantTensor: 0.5 × MemFree = ' + fmt2(budgetStock) + ' GiB < 1.18 GiB buffer → RuntimeError: buffer_size exceeds device memory budget']);
        else if (budgetStock < 4) lines.push(['⚠', C.d, 'Stock InstantTensor: budget of ' + fmt2(budgetStock) + ' GiB → io_depth shrinks to ~' + Math.max(10, Math.round(512 * budgetStock / 4)) + ' (from 512); slow load']);
        else lines.push(['✓', C.a, 'Stock InstantTensor: budget of ' + fmt1(budgetStock) + ' GiB ≥ 4 GiB ring → full io_depth']);
        if (!loaded) {
          if (S.patched) lines.push([avail >= 7.4 ? '✓' : '✕', avail >= 7.4 ? C.a : C.red, 'With the patch: 4 GiB window + largest shard (~1.4) + 2 GiB = 7.4 GiB ≤ MemAvailable (' + fmt1(avail) + ') → ' + (avail >= 7.4 ? 'loads at ~4.8 GB/s' : 'does not fit')]);
          else lines.push(['·', C.muted, 'Patch off (GLM53_COLD_LOAD_UMA=0): only the stock judge decides.']);
        }
        const need = 0.85 * MT;
        if (!loaded) lines.push([avail >= need ? '✓' : '✕', avail >= need ? C.a : C.red, 'vLLM boot check (integrated GPU): MemAvailable ' + fmt1(avail) + ' ≥ 0.85 × total = ' + fmt1(need) + ' GiB?']);
        let yy = vy0;
        lines.forEach(([ic, col, s]) => {
          V.text(g, ic, pad, yy, { size: 14, weight: 700, color: col });
          yy = wrap(g, s, pad + 20, yy, w - 2 * pad - 20, 15, { size: sm ? 11 : 12, color: C.ink, maxLines: 3 }) + 4;
        });
        wrap(g, S.msg, pad, Math.min(h - 8, yy + 6), w - 2 * pad, 14, { size: 11, color: C.muted, maxLines: 2 });
      });
    }

    /* ================= Fig 3 — the GMU traffic light ================= */
    {
      const fig = root.querySelector('#fig-gmu');
      fitStage(fig, 560);
      const R = {
        qwen: { name: 'Qwen TP2 (KV FP8)', def: 0.80, W: 64.46, NT: 4.06, A: 1.07, O: 5.13, G: 0.54, tpg: 114060, model: { E: 12.89, S: 6 }, ticks: [[0.80, 'current'], [0.835, 'before'], [0.85, 'historical']], anchors: [{ u: 0.835, lo: 0.3, hi: 0.9, l: '6-worker soak: 0.3–0.9 GiB' }, { u: 0.80, lo: 4.5, hi: 7.8, l: '1 h soak (v0.30, BF16 KV): 4.5 / 7.8' }] },
        ds: { name: 'DeepSeek TP2', def: 0.835, W: 79.17, O: 5.40, tpg: 136821, ticks: [[0.78, 'capture OOM'], [0.80, 'conserv.'], [0.835, 'default'], [0.87, 'big prompt']], anchors: [{ u: 0.83, lo: 3.54, hi: 4.84, l: 'after boot 3.8–5.2 GB' }, { u: 0.83, lo: 2.14, hi: 2.14, l: 'adaptive chunk 2.3 GB' }, { u: 0.83, lo: 0.92, hi: 0.92, l: 'chunk 2048 ~1.0 GB' }] },
        glm: { name: 'GLM TP2 (fixed KV)', def: 0.85, W: 79.65, X: 3.29, KV: 11, tok: 1233779, ticks: [[0.75, 'TP4'], [0.80, 'TP3'], [0.85, 'TP2'], [0.865, 'long'], [0.87, 'old']], anchors: [{ u: 0.87, lo: 0, hi: 0, l: '0 GiB: a 256k prefill crashed the head' }] },
      };
      const S = { r: 'qwen', u: 0.80 };
      const ctl = UI.controls(fig);
      let sl;
      UI.seg(ctl, { label: 'Recipe', options: [{ v: 'glm', label: 'GLM' }, { v: 'qwen', label: 'Qwen' }, { v: 'ds', label: 'DeepSeek' }], value: 'qwen', onChange: (v) => { S.r = v; sl.set(R[v].def); } });
      sl = UI.slider(ctl, { label: 'GMU', min: 0.72, max: 0.92, step: 0.005, value: 0.80, fmt: (v) => fmt3(v), onInput: (v) => { S.u = v; upd(); } });
      const rBud = UI.readout(ctl, 'vLLM budget', '—');
      const rKV = UI.readout(ctl, 'KV', '—');
      const rDelta = UI.readout(ctl, 'Headroom vs. default', '—');
      function kvOf(r, u) { const d = R[r]; if (d.KV) return d.KV; return Math.max(0, u * MT - d.W - d.O); }
      function upd() {
        const d = R[S.r];
        rBud.set(fmt1(S.u * MT) + ' GiB');
        const kv = kvOf(S.r, S.u);
        rKV.set(d.KV ? '11 GiB fixed (≈1.23M tok)' : fmt1(kv) + ' GiB ≈ ' + (kv * d.tpg / 1e6).toFixed(2) + 'M tok');
        const dl = (d.def - S.u) * MT;
        rDelta.set((dl >= 0 ? '+' : '−') + fmt1(Math.abs(dl)) + ' GiB');
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(); const sm = w < 560; const pad = sm ? 10 : 18;
        const d = R[S.r]; const u = S.u;
        const bud = u * MT, out = MT - bud;
        V.text(g, d.name + ' · GMU ' + fmt3(u) + ' → ' + fmt1(bud) + ' GiB for vLLM', pad, 18, { size: sm ? 11.5 : 12.5, weight: 700, color: C.ink, maxW: w - 2 * pad });
        const by = 28, bh = sm ? 34 : 40, bw = w - 2 * pad;
        let parts;
        if (d.KV) parts = [{ v: d.W, c: C.a, l: 'weights 79.65' }, { v: d.X, c: C.e, l: sm ? '' : 'KDA 3.29' }, { v: d.KV, c: C.c, l: 'KV 11' }, { v: Math.max(0, bud - d.W - d.X - d.KV), c: C.b, l: 'rest' }, { v: out, c: C.muted, l: 'outside the budget', hatch: true }];
        else if (d.NT) { const kv = kvOf(S.r, u); parts = [{ v: d.W, c: C.a, l: 'weights ' + fmt1(d.W) }, { v: d.NT, c: C.e, l: sm ? '' : 'non-torch' }, { v: d.A, c: C.b, l: '' }, { v: kv, c: C.c, l: 'KV ' + fmt1(kv) }, { v: d.G, c: C.b, l: '', hatch: true }, { v: out - d.G, c: C.muted, l: 'outside the budget', hatch: true }]; }
        else { const kv = kvOf(S.r, u); parts = [{ v: d.W, c: C.a, l: 'weights ' + fmt1(d.W) }, { v: d.O, c: C.b, l: sm ? '' : 'rest' }, { v: kv, c: C.c, l: 'KV ' + fmt1(kv) }, { v: out, c: C.muted, l: 'outside the budget', hatch: true }]; }
        stack(g, pad, by, bw, bh, parts, MT, { size: 11 });
        const ux = pad + (u) * bw;
        V.line(g, ux, by - 5, ux, by + bh + 5, { color: C.ink, lw: 2 });
        V.text(g, 'u·M', ux, by + bh + 16, { size: 11, align: 'center', color: C.muted, font: 'mono' });
        V.text(g, sm ? 'outside: ' + fmt1(out) + ' GiB' : 'outside the budget: ' + fmt1(out) + ' GiB (OS, page cache, driver, workspaces)', pad, by + bh + 16, { size: 11, color: C.muted, maxW: Math.max(60, ux - pad - 24) });
        if (d.G) {
          const gx = ux + (d.G / MT) * bw / 2;
          if (sm || w - pad - ux < 180) V.text(g, 'graphs 0.54 GiB outside GMU', w - pad, by + bh + 30, { size: 11, color: C.b, weight: 600, align: 'right' });
          else { V.line(g, gx, by + bh + 1, ux + 14, by + bh + 11, { color: C.b, lw: 1 }); V.text(g, 'graphs 0.54 GiB outside GMU', ux + 16, by + bh + 16, { size: 11, color: C.b, weight: 600, maxW: w - pad - ux - 18 }); }
        }
        // chart
        const cx0 = pad + (sm ? 30 : 40), cx1 = w - pad - (sm ? 4 : 10), cy0 = by + bh + 44, cy1 = h - (sm ? 44 : 40);
        const xMin = 0.72, xMax = 0.92, yMax = 28;
        const X = (v) => cx0 + ((v - xMin) / (xMax - xMin)) * (cx1 - cx0);
        const Y = (v) => cy1 - (V.clamp(v, 0, yMax) / yMax) * (cy1 - cy0);
        // zones
        g.fillStyle = V.alpha(C.red, C.dark ? 0.16 : 0.1); g.fillRect(cx0, Y(2), cx1 - cx0, Y(0) - Y(2));
        g.fillStyle = V.alpha(C.d, C.dark ? 0.12 : 0.1); g.fillRect(cx0, Y(6), cx1 - cx0, Y(2) - Y(6));
        g.fillStyle = V.alpha(C.e, C.dark ? 0.07 : 0.06); g.fillRect(cx0, Y(yMax), cx1 - cx0, Y(6) - Y(yMax));
        const zx = sm ? cx0 + 5 : cx1 - 6, za = sm ? 'left' : 'right';
        V.text(g, sm ? 'hard-reset' : 'hard-reset zone', zx, Y(0) - 5, { size: 11, align: za, color: C.red, weight: 600 });
        V.text(g, sm ? '2–6 GiB' : '2–6 GiB: tight', zx, Y(2) - 5, { size: 11, align: za, color: C.d });
        V.text(g, sm ? '≥ 6 GiB' : '≥ 6 GiB: memwatch floor', zx, Y(6) - 5, { size: 11, align: za, color: C.e });
        // axes
        V.line(g, cx0, cy1, cx1, cy1, { color: V.alpha(C.muted, 0.6), lw: 1 });
        V.line(g, cx0, cy0, cx0, cy1, { color: V.alpha(C.muted, 0.6), lw: 1 });
        [0, 6, 12, 18, 24].forEach((v) => { V.text(g, v, cx0 - 5, Y(v) + 4, { size: 11, align: 'right', color: C.muted, font: 'mono' }); V.line(g, cx0, Y(v), cx1, Y(v), { color: V.alpha(C.muted, 0.1), lw: 1 }); });
        V.text(g, 'host headroom (GiB)', cx0 - (sm ? 26 : 34), cy0 - 8, { size: 11, color: C.muted });
        for (let v = 0.72; v <= 0.9201; v += 0.04) V.text(g, v.toFixed(2), X(v), cy1 + 14, { size: 11, align: 'center', color: C.muted, font: 'mono' });
        // recipe ticks
        d.ticks.forEach(([v, l], i) => {
          V.line(g, X(v), cy1, X(v), cy1 - 6, { color: C.ink, lw: 1.5 });
          V.text(g, l, X(v), cy1 + 27 + (i % 2) * 12, { size: 11, align: 'center', color: v === d.def ? C.ink : C.muted, weight: v === d.def ? 700 : 400 });
        });
        // raw ceiling (clipped to the chart area)
        g.save(); g.beginPath(); g.rect(cx0, cy0, cx1 - cx0, cy1 - cy0); g.clip();
        g.beginPath(); g.moveTo(X(xMin), cy1 - (MT * (1 - xMin) / yMax) * (cy1 - cy0)); g.lineTo(X(xMax), cy1 - (MT * (1 - xMax) / yMax) * (cy1 - cy0));
        g.strokeStyle = V.alpha(C.ink, 0.6); g.lineWidth = 1.5; g.setLineDash([5, 4]); g.stroke(); g.setLineDash([]);
        g.restore();
        V.text(g, 'raw ceiling M(1−u)', X(xMax) - 4, Y(MT * (1 - xMax)) + 16, { size: 11, color: C.ink, alpha: 0.8, align: 'right' });
        // model (Qwen)
        if (d.model) {
          g.beginPath(); let first = true;
          for (let v = xMin; v <= xMax + 1e-9; v += 0.005) { const y = MT * (1 - v) - d.model.E - d.model.S; if (y < 0) break; const px = X(v), py = Y(y); if (first) { g.moveTo(px, py); first = false; } else g.lineTo(px, py); }
          g.strokeStyle = C.qwen; g.lineWidth = 2.5; g.stroke();
          V.text(g, 'model: − 12.9 (driver) − 6 (OS)', X(0.745), Y(MT * (1 - 0.745) - 18.9) - 8, { size: 11, color: C.qwen, weight: 600 });
        }
        // measured anchors
        let lastY = -1e9;
        d.anchors.forEach((a, i) => {
          const px = X(a.u);
          if (a.hi > a.lo) { V.line(g, px, Y(a.lo), px, Y(a.hi), { color: C.c, lw: 4 }); }
          V.dot(g, px, Y((a.lo + a.hi) / 2), 5, C.c);
          V.dot(g, px, Y((a.lo + a.hi) / 2), 2, C.bg);
          let ly = Y((a.lo + a.hi) / 2) - 8;
          if (Math.abs(ly - lastY) < 13) ly = lastY - 13;
          lastY = ly;
          const right = px < (cx0 + cx1) / 2 + 40;
          V.text(g, a.l, px + (right ? 9 : -9), ly, { size: 11, color: C.c, weight: 600, align: right ? 'left' : 'right', maxW: right ? cx1 - px - 12 : px - cx0 - 12 });
        });
        // current marker
        const pulse = 0.5 + 0.5 * Math.sin(t * 4);
        V.line(g, X(u), cy0, X(u), cy1, { color: V.alpha(C.ink, 0.7), lw: 1.2, dash: [3, 3] });
        V.glow(g, X(u), Y(MT * (1 - u)), 10 + 4 * pulse, C.ink, 0.5);
        V.dot(g, X(u), Y(MT * (1 - u)), 3.5, C.ink);
        if (d.model) {
          const y = MT * (1 - u) - d.model.E - d.model.S;
          const col = y < 2 ? C.red : y < 6 ? C.d : C.e;
          V.glow(g, X(u), Y(Math.max(0, y)), 12 + 5 * pulse, col, 0.7);
          V.dot(g, X(u), Y(Math.max(0, y)), 4.5, col);
          const nearA = d.anchors.some((a) => Math.abs(a.u - u) < 0.008); // do not cover the measurement label
          V.text(g, (y < 0 ? '≤0' : fmt1(y)) + ' GiB', X(u) + (nearA ? 9 : -9), Y(Math.max(0, y)) + 4, { size: 11.5, weight: 700, color: col, align: nearA ? 'left' : 'right' });
        } else if (d.KV) {
          V.text(g, 'Fixed KV: the GMU only changes the headroom and the boot check', X(u) + (X(u) > (cx0 + cx1) / 2 ? -8 : 8), Y(8.5), { size: 11, color: C.muted, align: X(u) > (cx0 + cx1) / 2 ? 'right' : 'left', maxW: X(u) > (cx0 + cx1) / 2 ? X(u) - cx0 - 14 : cx1 - X(u) - 14 });
        } else {
          const kv = kvOf(S.r, u); const dk = kv - kvOf(S.r, d.def);
          const rt = X(u) > (cx0 + cx1) / 2;
          const msg = Math.abs(dk) < 0.05 ? 'each +0.01 ≈ +1.2 GiB of KV ≈ +165K tokens' : 'KV ' + (dk >= 0 ? '+' : '−') + fmt1(Math.abs(dk)) + ' GiB ≈ ' + (dk >= 0 ? '+' : '−') + fmtInt(Math.abs(dk) * d.tpg / 1000) + 'K tokens vs. 0.835';
          V.text(g, msg, X(u) + (rt ? -8 : 8), Y(8.5), { size: 11, color: C.c, align: rt ? 'right' : 'left', maxW: rt ? X(u) - cx0 - 14 : cx1 - X(u) - 14 });
        }
      });
    }

    /* ================= Fig 4 — allocator: fixed vs expandable segments ================= */
    {
      const fig = root.querySelector('#fig-alocador');
      fitStage(fig, 268);
      const EV = [
        { op: 'a', id: 'A', s: 6, t: 'allocate A (6)', d: 'A prefill chunk asks for 6 units of activation. On top, the allocator does a <code>cudaMalloc</code> just for it. Below, the single segment grows by 6.' },
        { op: 'a', id: 'b', s: 3, t: 'allocate b (3)', d: 'A small, long-lived tensor. On top, the segment is rounded up to 4, leaving 1 unit that nobody uses.' },
        { op: 'a', id: 'C', s: 6, t: 'allocate C (6)', d: 'Another large activation: one more segment on top, 6 more at the tail below.' },
        { op: 'a', id: 'd', s: 3, t: 'allocate d (3)', d: 'Another small, long-lived tensor, which is going to end up “in the way”.' },
        { op: 'a', id: 'E', s: 6, t: 'allocate E (6)', d: 'Third large activation. So far both lanes reserve almost the same amount.' },
        { op: 'f', id: 'A', t: 'free A', d: 'The step is over and A dies. The block <b>does not go back to the driver</b>: it stays in the cache (hatched), reserved but not allocated.' },
        { op: 'f', id: 'C', t: 'free C', d: 'C also becomes a cached hole. On top it is an entire empty segment; below, a hole between b and d.' },
        { op: 'f', id: 'E', t: 'free E', d: 'E is freed. Below, the hole left by E sits at the <b>tail</b> of the segment, and that is going to make a difference.' },
        { op: 'a', id: 'F', s: 9, t: 'allocate F (9)', d: 'A bigger prefill asks for 9. On top there are 20 free units in total, but no hole of 9: <b>new segment</b>. Below, F reuses the free tail (6) and only extends it by 3.' },
        { op: 'a', id: 'G', s: 5, t: 'allocate G (5)', d: 'A request for 5 fits in the best hole (best-fit) in both lanes, with 1 unit left over.' },
        { op: 'f', id: 'F', t: 'free F', d: 'F dies. On top, the 10-unit segment sits empty and stranded. Below, the tail has 9 free units again.' },
        { op: 'a', id: 'H', s: 12, t: 'allocate H (12)', d: 'The worst case: 12. On top no hole will do, and the total goes to <b>48 reserved for 23 allocated</b>. Below, a tail of 9 + an extension of 3: <b>30 reserved</b>. On UMA, the difference comes out of the host.' },
      ];
      function coalesce(bl) { for (let i = 0; i < bl.length - 1;) { if (!bl[i].id && !bl[i + 1].id) { bl[i].size += bl[i + 1].size; bl.splice(i + 1, 1); } else i++; } }
      function simFixed(n) {
        const segs = [];
        for (let i = 0; i <= n; i++) {
          const e = EV[i];
          if (e.op === 'a') {
            let best = null;
            segs.forEach((sg) => sg.blocks.forEach((b, bi) => { if (!b.id && b.size >= e.s && (!best || b.size < best.b.size)) best = { sg, b, bi }; }));
            let sg, bi;
            if (best) { sg = best.sg; bi = best.bi; } else { const size = Math.ceil(e.s / 2) * 2; sg = { size, blocks: [{ size, id: null }] }; segs.push(sg); bi = 0; }
            const b = sg.blocks[bi]; const rest = b.size - e.s; b.size = e.s; b.id = e.id; b.ev = i;
            if (rest > 0) sg.blocks.splice(bi + 1, 0, { size: rest, id: null });
          } else segs.forEach((sg) => { sg.blocks.forEach((b) => { if (b.id === e.id) { b.id = null; b.ev = i; } }); coalesce(sg.blocks); });
        }
        return segs;
      }
      function simExp(n) {
        const bl = []; let size = 0;
        for (let i = 0; i <= n; i++) {
          const e = EV[i];
          if (e.op === 'a') {
            let best = -1;
            bl.forEach((b, bi) => { if (!b.id && b.size >= e.s && (best < 0 || b.size < bl[best].size)) best = bi; });
            if (best >= 0) { const b = bl[best]; const rest = b.size - e.s; b.size = e.s; b.id = e.id; b.ev = i; if (rest > 0) bl.splice(best + 1, 0, { size: rest, id: null }); }
            else if (bl.length && !bl[bl.length - 1].id) { const last = bl[bl.length - 1]; size += e.s - last.size; last.size = e.s; last.id = e.id; last.ev = i; last.ext = true; }
            else { bl.push({ size: e.s, id: e.id, ev: i }); size += e.s; }
          } else { bl.forEach((b) => { if (b.id === e.id) { b.id = null; b.ev = i; } }); coalesce(bl); }
        }
        return { size, bl };
      }
      const st = UI.stepper(fig, EV.map((e) => ({ t: e.t, d: e.d })), { autoplay: 3200 });
      const colOf = (C, id) => ({ A: C.a, C: C.a, E: C.a, b: C.b, d: C.b, F: C.c, H: C.c, G: C.d }[id] || C.e);
      V.fig(fig, (g, w, h) => {
        const C = V.C(); const sm = w < 560; const pad = sm ? 10 : 18;
        const i = st.i; const el = V.clamp(st.elapsed() / 0.6, 0, 1); const grow = V.ease.out(el);
        const fx = simFixed(i), ex = simExp(i);
        const unitsMax = 50;
        const laneW = w - 2 * pad; const pu = (laneW - 6 * 5) / unitsMax;
        const laneH = sm ? 40 : 48;
        const drawBlock = (x, y, bw, b, isNew) => {
          if (b.id) {
            const ww = isNew ? bw * grow : bw;
            V.box(g, x + 1, y + 1, Math.max(0, ww - 2), laneH - 2, { r: 5, fill: V.alpha(colOf(C, b.id), 0.9) });
            if (bw > 14 && (!isNew || grow > 0.6)) V.text(g, b.id + (bw > 36 ? ' ' + b.size : ''), x + bw / 2, y + laneH / 2 + 1, { size: 12, weight: 700, align: 'center', base: 'middle', color: onFill(C) });
          } else {
            g.fillStyle = V.alpha(C.muted, 0.1); g.fillRect(x + 1, y + 1, bw - 2, laneH - 2);
            hatch(g, x + 1, y + 1, bw - 2, laneH - 2, V.alpha(C.muted, 0.55), 5);
            if (bw > 16) V.text(g, b.size, x + bw / 2, y + laneH / 2 + 1, { size: 11, align: 'center', base: 'middle', color: C.muted, font: 'mono' });
          }
          if (isNew) { g.strokeStyle = C.ink; g.lineWidth = 2; V.rr(g, x, y, bw, laneH, 6); g.stroke(); }
        };
        const stats = (res, alloc, y, col) => {
          const frag = res ? 1 - alloc / res : 0;
          V.text(g, 'reserved ' + res + ' · allocated ' + alloc + ' · idle ' + Math.round(frag * 100) + '%', w - pad, y, { size: sm ? 11 : 11.5, align: 'right', color: col, weight: 600, font: 'mono' });
        };
        // lane 1
        const y1 = sm ? 40 : 44;
        V.text(g, sm ? 'fixed segments (1 cudaMalloc each)' : 'fixed segments — each one is a separate cudaMalloc', pad, y1 - 12, { size: 11.5, weight: 700, color: C.ink, maxW: sm ? laneW * 0.55 : laneW * 0.6 });
        let x = pad; let res1 = 0, al1 = 0;
        fx.forEach((sg) => {
          res1 += sg.size;
          const sw = sg.size * pu;
          V.box(g, x, y1, sw, laneH, { r: 6, fill: 'transparent', stroke: V.alpha(C.ink, 0.45), lw: 1 });
          let bx = x; sg.blocks.forEach((b) => { const bw = b.size * pu; if (b.id) al1 += b.size; drawBlock(bx, y1, bw, b, b.ev === i && EV[i].op === 'a' && b.id === EV[i].id); bx += bw; });
          x += sw + 5;
        });
        const statsY1 = sm ? y1 + laneH + 14 : y1 - 12;
        stats(res1, al1, statsY1, C.c);
        // lane 2
        const y2 = y1 + laneH + (sm ? 60 : 64);
        V.text(g, sm ? 'expandable_segments:True' : 'expandable_segments:True — one segment that grows at the tail', pad, y2 - 12, { size: 11.5, weight: 700, color: C.ink, maxW: sm ? laneW * 0.55 : laneW * 0.6 });
        V.box(g, pad, y2, ex.size * pu, laneH, { r: 6, fill: 'transparent', stroke: V.alpha(C.ink, 0.45), lw: 1 });
        let bx = pad; let al2 = 0;
        ex.bl.forEach((b) => { const bw = b.size * pu; if (b.id) al2 += b.size; drawBlock(bx, y2, bw, b, b.ev === i && EV[i].op === 'a' && b.id === EV[i].id); bx += bw; });
        // reserved virtual space (dashed)
        V.line(g, pad + ex.size * pu + 2, y2 + laneH / 2, Math.min(w - pad, pad + ex.size * pu + 60), y2 + laneH / 2, { color: V.alpha(C.muted, 0.6), lw: 1, dash: [3, 4] });
        if (!sm) V.text(g, 'reserved virtual addresses', Math.min(w - pad, pad + ex.size * pu + 64), y2 + laneH / 2 + 4, { size: 11, color: C.muted });
        const statsY2 = sm ? y2 + laneH + 14 : y2 - 12;
        stats(ex.size, al2, statsY2, C.a);
        // comparison bars
        const cy = y2 + laneH + (sm ? 34 : 30);
        const cw = laneW;
        const bar = (lab, v, col, yy) => {
          V.text(g, lab, pad, yy + 10, { size: 11, color: C.muted });
          const bx0 = pad + (sm ? 70 : 90), avail = cw - (sm ? 70 : 90) - (sm ? 96 : 110);
          V.box(g, bx0, yy, avail * (v / 48), 14, { r: 5, fill: V.alpha(col, 0.8) });
          V.text(g, v + ' reserved', bx0 + avail * (v / 48) + 6, yy + 11, { size: 11, color: col, font: 'mono', weight: 600 });
        };
        if (cy + 40 < h) { bar('fixed', res1, C.c, cy); bar('expandable', ex.size, C.a, cy + 22); }
      });
    }

    /* ================= Fig 5 — load paths ================= */
    {
      const fig = root.querySelector('#fig-carga');
      fitStage(fig, 560);
      const S = { mode: 'odirect', p64: true, pc: 0, cu: 0 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Loader', options: [{ v: 'mmap', label: 'mmap (auto)' }, { v: 'clone', label: 'mmap + clone' }, { v: 'odirect', label: 'O_DIRECT (InstantTensor)' }], value: 'odirect', onChange: (v) => { S.mode = v; S.cu = 0; } });
      UI.toggle(ctl, { label: '64 KiB kernel (GLM kit)', value: true, onChange: (v) => { S.p64 = v; S.cu = 0; } });
      const txt = {
        mmap: ['mmap (standard safetensors): the shard enters the page cache through page faults and the GPU copies straight from the mapping.', 'On 4 KiB it works, but the checkpoint occupies RAM twice (cache + tensor) until the kernel evicts the cache.'],
        mmap64: ['mmap on the GLM kit’s 64 KiB kernel: cuMemcpyHtoDAsync hangs when the source is a file-backed mapping.', 'That is why GLM does clone() into anonymous memory when SC_PAGE_SIZE ≠ 4096. Qwen and DeepSeek use the stock mmap with no clone.'],
        clone: ['mmap + clone (GLM53_LOAD_CLONE=1): copy into anonymous memory before the GPU; consumed shards get POSIX_FADV_DONTNEED.', 'Works on 64 KiB, and the page cache rises and falls instead of filling up (read-ahead stays at 0 by default).'],
        odirect: ['O_DIRECT (InstantTensor): the disk writes straight into the 4 GiB pinned ring; the page cache is not touched.', 'Measured: 164 GiB in 36 s at 4.82 GB/s (NVMe ceiling: 4.9 GB/s per reader). Cost: ~4.4–5.6 GiB less KV.'],
      };
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(); const sm = w < 560; const pad = sm ? 10 : 16;
        const mode = S.mode; const stuck = mode === 'mmap' && S.p64;
        // layout
        let nv, pool, PC, AN, RB, CU;
        const textH = sm ? 96 : 64;
        if (!sm) {
          nv = { x: pad, y: (h - textH) / 2 - 44, w: 88, h: 88 };
          pool = { x: pad + 104, y: 12, w: w - pad - (pad + 104), h: h - textH - 20 };
          PC = { x: pool.x + 14, y: pool.y + 34, w: pool.w * 0.25, h: pool.h * 0.42 };
          AN = { x: pool.x + pool.w * 0.36, y: pool.y + 34, w: pool.w * 0.24, h: pool.h * 0.42 };
          RB = { cx: pool.x + pool.w * 0.2, cy: pool.y + pool.h * 0.77, r: Math.min(34, pool.h * 0.17) };
          CU = { x: pool.x + pool.w * 0.72, y: pool.y + 34, w: pool.w * 0.25, h: pool.h - 48 };
        } else {
          nv = { x: pad, y: 8, w: w - 2 * pad, h: 34 };
          pool = { x: pad, y: 56, w: w - 2 * pad, h: h - 56 - textH - 10 };
          PC = { x: pool.x + 22, y: pool.y + 30, w: pool.w * 0.4, h: pool.h * 0.3 };
          AN = { x: pool.x + pool.w * 0.55, y: pool.y + 30, w: pool.w * 0.41, h: pool.h * 0.3 };
          const rr = Math.min(26, pool.h * 0.11);
          RB = { cx: pool.x + 24 + rr, cy: PC.y + PC.h + 16 + rr, r: rr };
          CU = { x: pool.x + 10, y: RB.cy + rr + 16, w: pool.w - 20, h: 0 };
          CU.h = pool.y + pool.h - 10 - CU.y;
        }
        // meter targets
        const pcT = mode === 'mmap' ? (stuck ? 0.55 : 0.95) : mode === 'clone' ? 0.22 + 0.12 * Math.sin(t * 2.2) : 0.02;
        S.pc += (pcT - S.pc) * Math.min(1, dt * 1.5);
        if (!stuck) { S.cu += dt * (mode === 'odirect' ? 0.16 : mode === 'clone' ? 0.1 : 0.07); if (S.cu > 1.15) S.cu = 0; }
        // pool
        V.box(g, pool.x, pool.y, pool.w, pool.h, { r: 14, fill: V.alpha(C.a, 0.04), stroke: V.alpha(C.a, 0.55), lw: 1.3 });
        V.text(g, 'Unified RAM (121.69 GiB): everything below lives here', pool.x + 12, pool.y + 18, { size: sm ? 11 : 11.5, weight: 600, color: C.muted, maxW: pool.w - 20 });
        // NVMe
        V.box(g, nv.x, nv.y, nv.w, nv.h, { r: 8, fill: V.alpha(C.muted, 0.15), stroke: C.muted, lw: 1.2 });
        V.text(g, 'NVMe', nv.x + nv.w / 2, nv.y + (sm ? 15 : nv.h / 2 - 4), { size: 12, weight: 700, align: 'center', color: C.ink });
        V.text(g, '120 shards · 164 GiB', nv.x + nv.w / 2, nv.y + (sm ? 28 : nv.h / 2 + 12), { size: 11, align: 'center', color: C.muted, maxW: nv.w - 6 });
        // regions
        const region = (r, title, col, on, fill) => {
          V.box(g, r.x, r.y, r.w, r.h, { r: 10, fill: V.alpha(col, on ? 0.12 : 0.04), stroke: V.alpha(col, on ? 0.95 : 0.3), lw: on ? 1.8 : 1 });
          V.text(g, title, r.x + 8, r.y + 16, { size: sm ? 11 : 11.5, weight: 700, color: on ? C.ink : C.muted, maxW: r.w - 12 });
          if (fill != null) {
            const fh = (r.h - 30) * V.clamp(fill, 0, 1);
            g.save(); V.rr(g, r.x + 6, r.y + 24, r.w - 12, r.h - 30, 6); g.clip();
            g.fillStyle = V.alpha(col, 0.12); g.fillRect(r.x + 6, r.y + r.h - 6 - fh, r.w - 12, fh);
            hatch(g, r.x + 6, r.y + r.h - 6 - fh, r.w - 12, fh, V.alpha(col, 0.55), 6);
            g.restore();
          }
        };
        region(PC, 'page cache', C.d, mode !== 'odirect', S.pc);
        V.text(g, Math.round(S.pc * 100) + '%', PC.x + PC.w - 8, PC.y + 16, { size: 11, align: 'right', color: C.d, weight: 700, font: 'mono' });
        region(AN, 'anonymous memory', C.b, mode === 'clone', null);
        if (mode === 'clone') V.text(g, 'clone() of the tensor', AN.x + 8, AN.y + 34, { size: 11, color: C.muted, maxW: AN.w - 12 });
        region(CU, 'CUDA allocation (weights)', C.a, true, null);
        const cf = V.clamp(S.cu, 0, 1);
        const fh = (CU.h - 34) * cf;
        V.box(g, CU.x + 8, CU.y + CU.h - 8 - fh, CU.w - 16, Math.max(0, fh), { r: 6, fill: V.alpha(C.a, 0.75) });
        V.text(g, stuck ? 'hung' : Math.round(cf * 100) + '%', CU.x + CU.w / 2, CU.y + CU.h / 2 + 8, { size: 13, weight: 700, align: 'center', color: stuck ? C.red : C.ink });
        // pinned ring
        const onR = mode === 'odirect';
        g.beginPath(); g.arc(RB.cx, RB.cy, RB.r, 0, Math.PI * 2); g.strokeStyle = V.alpha(C.c, onR ? 0.9 : 0.25); g.lineWidth = onR ? 8 : 5; g.stroke();
        if (onR) for (let k = 0; k < 12; k++) { const a0 = (k / 12) * Math.PI * 2 - Math.PI / 2; const lit = (Math.floor(t * 9) - k + 120) % 12 < 5; g.beginPath(); g.arc(RB.cx, RB.cy, RB.r, a0 + 0.06, a0 + Math.PI / 6 - 0.06); g.strokeStyle = lit ? C.c : V.alpha(C.c, 0.3); g.lineWidth = 8; g.stroke(); }
        const rbW = sm ? pool.x + pool.w - (RB.cx + RB.r + 10) - 8 : CU.x - (RB.cx + RB.r + 10) - 8;
        if (!sm || onR) {
          V.text(g, '4 GiB pinned ring', RB.cx + RB.r + 10, RB.cy - 3, { size: 11, weight: 700, color: onR ? C.ink : C.muted, maxW: rbW });
          V.text(g, 'io_depth 256 × chunk 8 MiB', RB.cx + RB.r + 10, RB.cy + 12, { size: 11, color: C.muted, font: 'mono', maxW: rbW });
        }
        // paths
        const nvOut = sm ? [nv.x + nv.w * 0.3, nv.y + nv.h] : [nv.x + nv.w, nv.y + nv.h * 0.35];
        const nvOut2 = sm ? [pool.x + 8, nv.y + nv.h] : [nv.x + nv.w, nv.y + nv.h * 0.75];
        const pcC = [PC.x + PC.w / 2, PC.y + PC.h / 2], anC = [AN.x + AN.w / 2, AN.y + AN.h / 2];
        // on phones, the path enters the CUDA region on the right, below the title
        const cuC = sm ? [CU.x + CU.w * 0.8, CU.y + CU.h * 0.55] : [CU.x + CU.w * 0.5, CU.y + CU.h * 0.35];
        // mmap corridor: below the anonymous memory box (mmap does not go through it)
        const yc = sm ? PC.y + PC.h + 8 : AN.y + AN.h + 16;
        let path;
        if (mode === 'mmap') path = sm ? [nvOut, pcC, [pcC[0], yc], [cuC[0], yc], cuC] : [nvOut, pcC, [pcC[0], yc], [CU.x + 20, yc], cuC];
        else if (mode === 'clone') path = [nvOut, pcC, anC, cuC];
        else if (sm) path = [nvOut2, [pool.x + 8, RB.cy], [RB.cx, RB.cy], [RB.cx, CU.y - 8], [cuC[0], CU.y - 8], cuC];
        else { const yb = Math.min(RB.cy + RB.r + 14, CU.y + CU.h - 12); path = [nvOut2, [RB.cx, RB.cy], [RB.cx, yb], [CU.x + CU.w * 0.5, yb]]; } // below the ring labels
        poly(g, path, { color: V.alpha(C.ink, 0.35), lw: 2, dash: [5, 5], off: -t * 30 });
        const N = 10;
        for (let k = 0; k < N; k++) {
          let u = (k / N + t * 0.14) % 1;
          let p = along(path, u);
          if (stuck && p[2] >= 1) {
            // pile up before the H2D copy
            const q = [pcC[0] + 10 + (k % 5) * 9, yc - 4 + Math.floor(k / 5) * 9];
            sq(g, q[0], q[1], 7, V.alpha(C.d, 0.9));
            continue;
          }
          sq(g, p[0], p[1], 8, p[2] === 0 ? C.muted : mode === 'odirect' ? C.c : C.d);
        }
        if (stuck) {
          const mx = sm ? (pcC[0] + cuC[0]) / 2 : AN.x + AN.w / 2, my = yc;
          const bl = 0.6 + 0.4 * Math.sin(t * 6);
          V.glow(g, mx, my, 22, C.red, bl);
          V.text(g, '⛔', mx, my + 6, { size: 18, align: 'center', color: C.red });
          V.text(g, 'H2D hangs', mx, my + 24, { size: 11, weight: 700, align: 'center', color: C.red });
        }
        if (mode === 'clone') {
          const pulse = (t * 0.7) % 1;
          V.text(g, 'FADV_DONTNEED', PC.x + 8, PC.y + 36 + (1 - pulse) * 6, { size: 11, font: 'mono', color: C.d, alpha: 1 - pulse * 0.7, maxW: PC.w - 12 });
        }
        // text
        const key = stuck ? 'mmap64' : mode;
        let yy = h - textH + 14;
        txt[key].forEach((s, i) => { yy = wrap(g, s, pad, yy, w - 2 * pad, 15, { size: sm ? 11 : 12, color: i === 0 ? C.ink : C.muted, weight: i === 0 ? 600 : 400, maxLines: 3 }) + 2; });
      });
    }

    /* ================= Fig 6 — UVM livelock (stepper) ================= */
    {
      const fig = root.querySelector('#fig-livelock');
      fitStage(fig, 560);
      const steps = [
        { t: 'Days earlier: pages parked in swap', d: 'Under memory pressure, the kernel sent some of the vLLM process pages to swap. The service carries on as usual: the GPU works and the token counters go up.' },
        { t: 'swappiness=0 narrows the exit door', d: 'With <code>vm.swappiness=0</code> (the Spark default), the kernel only sends anonymous pages to swap as a last resort: <b>new</b> swap-outs become rare, not impossible. The ones already there stay there, and that is what lies in wait.' },
        { t: 'A ~236k-token prefill touches those pages', d: 'A long prompt touches the parked pages, and the page faults force the UVM driver to bring them back, right when memory is tight.' },
        { t: 'The driver starts spinning', d: '<code>nvidia-smi</code> shows <b>~96% utilization at 12–20 W</b>, the power draw of an idle GPU: it looks busy, but does nothing. <code>shm_broadcast</code> logs <i>No available shared memory broadcast block found in 60 seconds</i> every minute.' },
        { t: '/health lies; /metrics tells the truth', d: '<code>/health</code> stays at 200 because it only reads an error flag. The rule that works: running + waiting &gt; 0 <b>and</b> prompt + generation frozen for N = 10 ticks of 60 s.' },
        { t: 'Recovery without the server room (18 min)', d: 'Logs first, then <code>docker kill</code> (stop hangs), <code>swapoff -a &amp;&amp; swapon -a</code>, a trivial CUDA probe (<code>CUDA_OK</code>) and a restart. To prevent it: swappiness=0, keep swap on and cycle it before every start.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 5200 });
      const S = { tok: 1284000 };
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(); const sm = w < 560; const pad = sm ? 10 : 16;
        const i = st.i, el = st.elapsed();
        const frozen = i === 3 || i === 4;
        if (!frozen) S.tok += dt * (i === 5 ? 0 : 2400);
        const cols = sm ? 1 : 3;
        const pw = sm ? w - 2 * pad : (w - 2 * pad - 24) / 3;
        const ph = sm ? (h - 2 * pad - 20) / 3 : h - 2 * pad;
        const P = (k) => (sm ? { x: pad, y: pad + k * (ph + 10), w: pw, h: ph } : { x: pad + k * (pw + 12), y: pad, w: pw, h: ph });
        const panel = (p, title, col) => { V.box(g, p.x, p.y, p.w, p.h, { r: 12, fill: V.alpha(col, 0.05), stroke: V.alpha(col, 0.5), lw: 1.2 }); V.text(g, title, p.x + 10, p.y + 17, { size: 11.5, weight: 700, color: C.ink }); };
        // memory panel
        const pm = P(0); panel(pm, 'memory + swap', C.b);
        const inH = pm.h - 36;
        const ram = { x: pm.x + 10, y: pm.y + 26, w: pm.w - 20, h: inH * 0.56 };
        const sw = { x: pm.x + 10, y: ram.y + ram.h + inH * 0.14, w: pm.w - 20, h: inH * 0.3 };
        V.box(g, ram.x, ram.y, ram.w, ram.h, { r: 8, fill: V.alpha(C.a, 0.06), stroke: V.alpha(C.a, 0.5), lw: 1 });
        V.text(g, 'RAM (vLLM pages)', ram.x + 6, ram.y + 13, { size: 11, color: C.muted, font: 'mono', maxW: ram.w - 10 });
        V.box(g, sw.x, sw.y, sw.w, sw.h, { r: 10, fill: V.alpha(C.d, 0.08), stroke: V.alpha(C.d, 0.6), lw: 1 });
        V.text(g, 'swap (disk)', sw.x + 6, sw.y + 13, { size: 11, color: C.muted, font: 'mono' });
        const pcols = 8, rowsN = 2;
        const cs = Math.max(7, Math.min(18, Math.min((ram.w - 20) / pcols - 4, (ram.h - 30) / (rowsN + 1) - 4)));
        const cellX = (k) => ram.x + 10 + (k % pcols) * (cs + 4) + cs / 2;
        const cellY = (r) => ram.y + 22 + r * (cs + 4) + cs / 2;
        for (let k = 0; k < pcols * rowsN; k++) sq(g, cellX(k), cellY(Math.floor(k / pcols)), cs, V.alpha(C.a, 0.8));
        // 3 pages parked in swap
        const returned = i === 5 && el > 2.4;
        for (let k = 0; k < 3; k++) {
          let px = sw.x + 22 + k * (cs + 8) + cs / 2, py = sw.y + sw.h / 2 + 5;
          const moving = i === 2 || (i === 5 && el > 1);
          if (moving) { const f = V.clamp((el - (i === 5 ? 1 : 0.3) - k * 0.3) / 1.1, 0, 1); px = V.lerp(px, cellX(k), V.ease.inOut(f)); py = V.lerp(py, cellY(2), V.ease.inOut(f)); }
          if (i >= 3 && i <= 4) { px = cellX(k); py = cellY(2); }
          sq(g, px, py, cs, returned || (i === 5 && el > 1) ? C.e : C.d);
        }
        const gapY = (ram.y + ram.h + sw.y) / 2;
        if (i === 0) { V.text(g, 'parked for days', sw.x + sw.w - 8, sw.y + sw.h / 2 + 9, { size: 11, align: 'right', color: C.d, weight: 600 }); V.arrow(g, ram.x + ram.w * 0.7, ram.y + ram.h + 2, ram.x + ram.w * 0.7, sw.y - 2, { color: V.alpha(C.d, 0.8), lw: 1.5 }); }
        if (i === 1) { V.arrow(g, ram.x + ram.w * 0.7, ram.y + ram.h + 2, ram.x + ram.w * 0.7, sw.y - 2, { color: V.alpha(C.muted, 0.4), lw: 1.5 }); V.text(g, '✕', ram.x + ram.w * 0.7, gapY + 5, { size: 15, align: 'center', color: C.red, weight: 700 }); V.text(g, 'swappiness=0', ram.x + ram.w * 0.7 - 14, gapY + 4, { size: 11, font: 'mono', align: 'right', color: C.b, weight: 700 }); }
        if (i === 2) { V.arrow(g, ram.x + 30, sw.y - 2, ram.x + 30, ram.y + ram.h + 2, { color: C.c, lw: 1.8 }); V.text(g, 'page faults', ram.x + 40, gapY + 4, { size: 11, color: C.c, weight: 600 }); badge(g, 'prefill ~236k', ram.x + ram.w - 4, ram.y + ram.h - 8, C.c, { size: 11, align: 'right' }); }
        if (i === 3 || i === 4) { const a = t * 5; const cx = ram.x + ram.w - 26, cy = ram.y + ram.h - 20; g.beginPath(); g.arc(cx, cy, 11, a, a + Math.PI * 1.5); g.strokeStyle = C.red; g.lineWidth = 3; g.stroke(); V.text(g, 'UVM spins', cx - 18, cy + 4, { size: 11, align: 'right', color: C.red, weight: 700 }); }
        if (i === 5) V.text(g, 'swapoff -a && swapon -a', sw.x + sw.w - 8, sw.y + sw.h / 2 + 9, { size: 11, font: 'mono', align: 'right', color: C.e, weight: 700, maxW: sw.w * 0.6 });
        // GPU panel
        const pg = P(1); panel(pg, 'GPU (nvidia-smi)', C.a);
        const util = frozen ? 96 + Math.sin(t * 7) * 0.8 : i === 5 ? 3 : 55 + 30 * Math.sin(t * 1.7) + 8 * Math.sin(t * 5.3);
        // watts: livelock and truly idle sit in the 12–20 W band from the source; the "under load" value is illustrative
        const watt = frozen ? 16 + 3 * Math.sin(t * 3.1) : i === 5 ? 14 + 0.6 * Math.sin(t * 2) : 40 + 2 * Math.sin(t * 2);
        const gcx = pg.x + pg.w / 2, gcy = pg.y + pg.h * (sm ? 0.74 : 0.5), R = Math.min(pg.w * 0.34, pg.h * (sm ? 0.42 : 0.28));
        g.lineCap = 'round';
        g.beginPath(); g.arc(gcx, gcy, R, Math.PI, 2 * Math.PI); g.strokeStyle = V.alpha(C.muted, 0.2); g.lineWidth = 10; g.stroke();
        g.beginPath(); g.arc(gcx, gcy, R, Math.PI, Math.PI + Math.PI * util / 100); g.strokeStyle = frozen ? C.red : C.a; g.stroke();
        g.lineCap = 'butt';
        V.text(g, Math.round(util) + '%', gcx, gcy - 6, { size: sm ? 18 : 22, weight: 700, align: 'center', color: frozen ? C.red : C.ink, font: 'display' });
        V.text(g, 'utilization', gcx, gcy + 12, { size: 11, align: 'center', color: C.muted });
        const wy = sm ? pg.y + 34 : gcy + 36;
        V.text(g, fmt1(watt) + ' W', sm ? pg.x + pg.w - 12 : gcx, wy, { size: 14, weight: 700, align: sm ? 'right' : 'center', color: frozen ? C.d : C.ink, font: 'mono' });
        const wNote = frozen ? ['busy doing nothing', C.red] : i === 5 ? ['truly idle', C.e] : ['under load (illustrative)', C.muted];
        V.text(g, wNote[0], sm ? pg.x + pg.w - 12 : gcx, wy + 16, { size: 11, align: sm ? 'right' : 'center', color: wNote[1], weight: 600 });
        // wattage ruler with the reference idle band (12–20 W)
        {
          const WMAX = 60, bx0 = pg.x + (sm ? 12 : 16), bw0 = sm ? Math.min(96, gcx - R - 16 - bx0) : pg.w - 32, by0 = sm ? pg.y + pg.h - 30 : wy + 34;
          const WX = (v) => bx0 + (V.clamp(v, 0, WMAX) / WMAX) * bw0;
          V.box(g, bx0, by0, bw0, 8, { r: 4, fill: V.alpha(C.muted, 0.15) });
          g.fillStyle = V.alpha(C.d, 0.35); g.fillRect(WX(12), by0, WX(20) - WX(12), 8);
          V.line(g, WX(watt), by0 - 4, WX(watt), by0 + 12, { color: frozen ? C.red : C.ink, lw: 2.5 });
          V.text(g, sm ? 'idle 12–20 W' : 'idle band 12–20 W', bx0, by0 + 22, { size: 11, color: C.d, weight: 600, maxW: sm ? bw0 + 40 : bw0 * 0.62 });
          if (!sm) V.text(g, '0–60 W', bx0 + bw0, by0 + 22, { size: 11, color: C.muted, align: 'right', font: 'mono' });
        }
        if (i === 5 && el > 2.5) badge(g, 'CUDA_OK', sm ? pg.x + 12 : gcx, sm ? pg.y + 44 : wy + 82, C.e, { align: sm ? 'left' : 'center', font: 'mono', size: 11 });
        // API panel
        const pa = P(2); panel(pa, 'vLLM API', C.c);
        const hl = pa.y + 40;
        V.dot(g, pa.x + 20, hl, 7, i === 5 && el < 2 ? C.muted : C.e);
        V.text(g, 'GET /health → 200', pa.x + 34, hl + 4, { size: 11.5, font: 'mono', color: C.ink, maxW: pa.w - 40 });
        if (frozen) V.text(g, '(a lie)', pa.x + pa.w - 10, hl + 4, { size: 11, align: 'right', color: C.red, weight: 700 });
        V.text(g, 'prompt+generation_tokens_total', pa.x + 12, hl + 30, { size: 11, font: 'mono', color: C.muted, maxW: pa.w - 20 });
        V.text(g, fmtInt(S.tok), pa.x + 12, hl + 50, { size: 16, font: 'mono', weight: 700, color: frozen ? C.d : C.ink });
        if (frozen) V.text(g, 'frozen', pa.x + pa.w - 10, hl + 50, { size: 11, align: 'right', color: C.d, weight: 700 });
        const lg = pa.y + (sm ? 118 : 128);
        if (i === 3) wrap(g, 'shm_broadcast: No available shared memory broadcast block found in 60 seconds', pa.x + 12, lg, pa.w - 24, 13, { size: 11, font: 'mono', color: C.red, maxLines: 4 });
        if (i === 4) {
          const ticks = Math.min(10, Math.floor(el * 2.2));
          V.text(g, 'ticks without progress: ' + ticks + '/10', pa.x + 12, lg, { size: 11, color: ticks >= 10 ? C.red : C.ink, weight: 600 });
          for (let k = 0; k < 10; k++) V.box(g, pa.x + 12 + k * ((pa.w - 30) / 10), lg + 8, (pa.w - 30) / 10 - 3, 8, { r: 3, fill: k < ticks ? C.d : V.alpha(C.muted, 0.2) });
          if (ticks >= 10) V.text(g, '🚨 alert: livelock', pa.x + 12, lg + 34, { size: 11.5, color: C.red, weight: 700 });
        }
        if (i === 5) wrap(g, 'docker kill → swapoff -a && swapon -a → CUDA probe → restart', pa.x + 12, lg, pa.w - 24, 14, { size: 11, font: 'mono', color: C.e, maxLines: 4 });
        if (i <= 2) wrap(g, i === 2 ? 'the current step drags on... and on' : 'normal decode steps', pa.x + 12, lg, pa.w - 24, 14, { size: 11, color: C.muted });
      });
    }

    /* ================= Fig 7 — water tank ================= */
    {
      const fig = root.querySelector('#fig-caixa');
      fitStage(fig, 520);
      const PRE = {
        padrao: { mf: 45167 / 1048576, wsf: 10, swap: 0, name: 'Spark defaults' },
        tp1: { mf: 4, wsf: 300, swap: 16, name: 'HANDOFF tunables' },
      };
      const S = { pre: 'padrao', rate: 2, poll: 1, watch: true };
      const DEMAND = 24, KSW = 1.0, KILL_LAT = 1.5, FLOOR = 6, HOLD = 3.4;
      const wm = () => { const p = PRE[S.pre]; const d = Math.max(p.mf / 4, MT * p.wsf / 10000); return { min: p.mf, low: p.mf + d, high: p.mf + 2 * d, d }; };
      /** simulates a whole episode (fixed dt) and returns the trajectory */
      function simulate() {
        const p = PRE[S.pre], W = wm(), dt = 0.02;
        let t = 0, L = 20, R = 1.5 + p.swap, used = 0, out = null, killAt = null, next = 0, rec = false;
        const hist = [], polls = [];
        while (!out && t < 20) {
          const open = t > 1.2; // the process keeps allocating until the kill completes
          if (open) {
            const a = Math.min(S.rate * dt, DEMAND - used);
            if (L - a <= W.min && W.min >= 1) {
              const r = Math.min(a, R); R -= r; used += r; L = Math.max(W.min, L - (a - r));
              if (R <= 0.001 && a - r > 0) out = 'erro';
            } else { L -= a; used += a; }
            if (used >= DEMAND - 1e-6 && !out) out = 'coube';
          }
          if (L < W.low && R > 0) rec = true;
          if (L >= W.high) rec = false;
          if (rec && R > 0) { const r = Math.min(KSW * dt, R); L += r; R -= r; }
          if (S.watch && t >= next - 1e-9) { polls.push(t); next += S.poll; if (L < FLOOR && killAt === null) killAt = t + KILL_LAT; }
          if (killAt !== null && t >= killAt && !out) out = 'kill';
          if (L <= 0.03 && !out) { L = 0; out = 'trava'; }
          hist.push([t, L, R, rec && !out, open && !out]);
          t += dt;
        }
        const tEnd = t;
        // recovery (or freeze) after the outcome
        for (let s = 0; s < HOLD; s += 0.05) { if (out !== 'trava') L = Math.min(20, L + 0.05 * 6); hist.push([tEnd + s, L, R, false, false]); }
        return { hist, polls, killAt, out, tEnd, total: tEnd + HOLD };
      }
      let SIM = simulate();
      const E = { t: 0 };
      const reset = () => { SIM = simulate(); E.t = 0; };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'VM tunables', options: [{ v: 'padrao', label: 'defaults (44 MB · 10 · swap 0)' }, { v: 'tp1', label: 'HANDOFF §6 (4 GiB · 300 · 30)' }], value: 'padrao', onChange: (v) => { S.pre = v; reset(); } });
      UI.slider(ctl, { label: 'UVM allocation', min: 0.5, max: 6, step: 0.5, value: 2, fmt: (v) => fmt1(v) + ' GiB/s', onInput: (v) => { S.rate = v; reset(); } });
      UI.slider(ctl, { label: 'Watchdog interval', min: 1, max: 5, step: 1, value: 1, fmt: (v) => v + ' s', onInput: (v) => { S.poll = v; reset(); } });
      UI.toggle(ctl, { label: 'memwatch.sh on (6 GiB floor)', value: true, onChange: (v) => { S.watch = v; reset(); } });
      const at = (tt) => { const hs = SIM.hist; if (!hs.length) return [0, 20, 0, false, false]; let lo = 0, hi = hs.length - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (hs[m][0] <= tt) lo = m; else hi = m - 1; } return hs[lo]; };
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(); const sm = w < 560; const pad = sm ? 10 : 16;
        const W = wm();
        if (!LA.paused) { E.t += dt; if (E.t > SIM.total) E.t = 0; }
        const cur = at(E.t);
        const lvl = cur[1], R = cur[2], recOn = cur[3], open = cur[4];
        const done = E.t >= SIM.tEnd;
        // layout
        const tankW = sm ? 76 : 120, tx = pad, ty = 30, th = h - ty - (sm ? 96 : 84);
        const yMax = 22;
        const Y = (v) => ty + th - (V.clamp(v, 0, yMax) / yMax) * th;
        const zc = lvl < 2 ? C.red : lvl < FLOOR ? C.d : C.e;
        V.text(g, 'available memory', tx, ty - 10, { size: 11, weight: 700, color: C.ink });
        V.box(g, tx, ty, tankW, th, { r: 10, fill: V.alpha(C.muted, 0.06), stroke: V.alpha(C.muted, 0.6), lw: 1.4 });
        g.save(); V.rr(g, tx, ty, tankW, th, 10); g.clip();
        g.beginPath(); g.moveTo(tx, Y(lvl));
        const frozen = done && SIM.out === 'trava';
        for (let x = 0; x <= tankW; x += 4) g.lineTo(tx + x, Y(lvl) + Math.sin(x / 9 + t * 4) * (frozen ? 0 : 2));
        g.lineTo(tx + tankW, ty + th); g.lineTo(tx, ty + th); g.closePath();
        g.fillStyle = V.alpha(zc, 0.42); g.fill();
        g.restore();
        V.text(g, fmt1(lvl) + ' GiB', tx + tankW / 2, Math.max(ty + 14, Y(lvl) - 6), { size: 12, weight: 700, align: 'center', color: C.ink });
        // watermarks (labels stacked from bottom to top)
        const labX = tx + tankW + 6;
        const labs = [[W.min, 'min ' + (W.min < 1 ? (W.min * 1024).toFixed(0) + ' MiB' : fmt1(W.min) + ' GiB'), C.red], [W.low, 'low ' + fmt2(W.low), C.d], [W.high, 'high ' + fmt2(W.high), C.e]];
        if (S.watch) labs.push([FLOOR, sm ? 'watch 6' : 'watchdog 6', C.c, true]);
        labs.sort((p1, p2) => p1[0] - p2[0]);
        let prevY = 1e9;
        labs.forEach(([v, l, c, dash]) => {
          if (dash) V.line(g, tx - 4, Y(v), tx + tankW + 4, Y(v), { color: c, lw: 1.5, dash: [4, 3] });
          else V.line(g, tx, Y(v), tx + tankW, Y(v), { color: c, lw: 1.5 });
          let ly = Math.min(Y(v) + 4, ty + th); if (prevY - ly < 12) ly = prevY - 12; prevY = ly;
          V.text(g, l, labX, ly, { size: 11, color: c, font: 'mono', weight: 600 });
        });
        // UVM drain and reserve
        const dxp = tx + tankW * 0.7, dyp = ty + th;
        if (open) { for (let k = 0; k < 5; k++) { const u = (t * 2 + k / 5) % 1; sq(g, dxp, dyp + 5 + u * 12, 5, C.a); } }
        V.text(g, 'UVM allocates ' + fmt1(S.rate) + ' GiB/s', tx, dyp + 32, { size: 11, color: open ? C.a : C.muted, weight: 600, maxW: sm ? w - 2 * pad : tankW + 80 });
        V.text(g, (recOn ? '⟳ kswapd reclaims · ' : 'reclaimable reserve: ') + 'cache+swap ' + fmt1(R) + ' GiB', tx, dyp + 48, { size: 11, color: recOn ? C.d : C.muted, maxW: sm ? w - 2 * pad : tankW + 150 });
        // time chart
        const gx0 = tx + tankW + (sm ? 70 : 100), gx1 = w - pad, gy0 = ty + 14, gy1 = ty + th;
        if (gx1 - gx0 > 80) {
          const tMax = Math.max(12, SIM.total);
          const X = (s) => gx0 + (V.clamp(s, 0, tMax) / tMax) * (gx1 - gx0);
          V.line(g, gx0, gy1, gx1, gy1, { color: V.alpha(C.muted, 0.6), lw: 1 });
          V.line(g, gx0, gy0, gx0, gy1, { color: V.alpha(C.muted, 0.6), lw: 1 });
          g.fillStyle = V.alpha(C.red, C.dark ? 0.14 : 0.08); g.fillRect(gx0, Y(2), gx1 - gx0, Y(0) - Y(2));
          [[W.min, C.red], [W.low, C.d]].forEach(([v, c]) => V.line(g, gx0, Y(v), gx1, Y(v), { color: V.alpha(c, 0.55), lw: 1, dash: [2, 3] }));
          if (S.watch) V.line(g, gx0, Y(FLOOR), gx1, Y(FLOOR), { color: V.alpha(C.c, 0.6), lw: 1, dash: [4, 3] });
          for (let s = 0; s <= tMax; s += 5) V.text(g, s + ' s', X(s), gy1 + 13, { size: 11, align: 'center', color: C.muted, font: 'mono' });
          // full trajectory (ghost) and the stretch already played
          const hs = SIM.hist;
          g.beginPath(); hs.forEach(([s, v], k) => { if (k % 3) return; const px = X(s), py = Y(v); if (k) g.lineTo(px, py); else g.moveTo(px, py); }); g.strokeStyle = V.alpha(C.ink, 0.22); g.lineWidth = 2; g.stroke();
          g.beginPath(); let started = false; for (let k = 0; k < hs.length && hs[k][0] <= E.t; k += 2) { const px = X(hs[k][0]), py = Y(hs[k][1]); if (started) g.lineTo(px, py); else { g.moveTo(px, py); started = true; } } g.strokeStyle = C.ink; g.lineWidth = 2.2; g.stroke();
          V.dot(g, X(E.t), Y(lvl), 4, zc);
          SIM.polls.forEach((s) => V.line(g, X(s), gy1, X(s), gy1 - 6, { color: C.c, lw: 1.5 }));
          if (SIM.killAt !== null) { V.line(g, X(SIM.killAt), gy0, X(SIM.killAt), gy1, { color: V.alpha(C.c, 0.8), lw: 1.5, dash: [3, 3] }); { const kr = X(SIM.killAt) + 64 > gx1; V.text(g, 'docker kill', X(SIM.killAt) + (kr ? -4 : 4), gy0 + 34, { size: 11, color: C.c, weight: 600, align: kr ? 'right' : 'left' }); } }
          if (!sm) V.text(g, 'MemAvailable over time · ticks = watchdog readings', gx1, gy1 + 28, { size: 11, color: C.muted, align: 'right', maxW: gx1 - gx0 - 6 });
          // outcome (always visible; highlighted when playback gets there)
          const map = { trava: ['✕ host hung: no OOM, no logs', C.red], kill: [sm ? '✓ watchdog killed the container' : '✓ the watchdog killed the container in time', C.e], erro: ['✓ readable ValueError, host alive', C.d], coube: ['✓ it fit, hovering at the floor', C.e] };
          const [s, c] = map[SIM.out] || ['…', C.muted];
          const bx = gx0 + 12, by = gy0 - 22;
          g.font = V.font(12.5, 'sans', 700); const tw = Math.min(g.measureText(s).width, gx1 - gx0 - 30);
          V.box(g, bx - 8, by - 15, tw + 16, 24, { r: 8, fill: V.alpha(c, done ? 0.22 : 0.08), stroke: V.alpha(c, done ? 1 : 0.5), lw: 1.5 });
          V.text(g, s, bx, by + 1, { size: 12.5, weight: 700, color: c, maxW: gx1 - gx0 - 30, alpha: done ? 1 : 0.7 });
        }
        // footer
        const foot = S.pre === 'tp1' ? 'It helps, but it is not enough (crash #3 happened with them). CUDA sees less (112 → 100.68 GiB) and, in dual, floor + scale cost 7–14% of decode.' : 'Defaults: the reserve barely exists, and kswapd only wakes up with ~0.16 GiB free (~0.1%).';
        wrap(g, foot, pad, h - 20, w - 2 * pad, 13, { size: 11, color: C.muted, maxLines: 2 });
      });
    }

    /* ================= Fig 8 — the PLE handshake ================= */
    {
      const fig = root.querySelector('#fig-ple');
      fitStage(fig, 374);
      const S = { stock: false };
      const ctl = UI.controls(fig);
      UI.toggle(ctl, { label: 'stock mode (cuStreamWaitValue32 on the GB10)', value: false, onChange: (v) => { S.stock = v; } });
      const steps = [
        { t: 'Request: seq = 42 over ZMQ', d: 'The model thread (layer 1) puts <code>input_ids</code>, <code>query_start_loc</code> and <code>ngram_context</code> into shared, <b>pinned</b> CPU buffers (<code>cuMemHostRegister</code>) and sends the request with <code>seq = 42</code> over a ZMQ PUSH socket (msgpack).' },
        { t: 'Hash: 16 ids per token', d: 'For each token, the offload process computes 8 bigram ids and 8 trigram ids: <code>id = (mixed mod p_h) + offset_h</code>, with one prime \\(p_h\\) per head. The addressing is deterministic, which is why the lookup can happen off the GPU.' },
        { t: 'Gather from the mmap table', d: '<code>index_select</code> pulls the 16 rows of 90 B from the packed table <code>[~320M × 90] uint8</code>, mapped with <code>np.memmap</code>. These are file pages: evictable, outside the anonymous RSS.' },
        { t: 'Async H2D copy (CUDA IPC)', d: 'The result goes to the rank’s GPU buffer through an asynchronous copy on <code>copy_stream</code>. The buffer was opened through a <b>CUDA IPC</b> handle (one per TP rank): 16 × 90 = 1440 B per token.' },
        { t: 'Synchronize and raise the flag', d: 'Only after <code>copy_stream.synchronize()</code> on all ranks does the worker write <code>flag[0] = 42</code> into the int64 <code>done_flag</code> in shared memory. In stock mode, <code>cuStreamWriteValue32</code> would go here.' },
        { t: 'Host-side wait: flag ≥ seq', d: '<code>_wait_done(42)</code>: 2000 busy spins, then <code>sleep(50 µs)</code> per iteration until <code>flag ≥ 42</code>, with a 300 s timeout (<code>VLLM_PLE_OFFLOAD_STEP_TIMEOUT</code>). The monotonic counter means the flag never has to be reset, and resetting would open up a race.' },
        { t: 'Dequant and continue layer 1', d: 'The GPU dequantizes NVFP4 (E2M1 code × block FP8 scale × global scale) to BF16 and injects the result into the residual stream. Since the PLE sits at layer 1, the lost CPU/GPU overlap is negligible.' },
      ];
      const st = UI.stepper(fig, steps, { autoplay: 5000 });
      const rng = V.rng(27);
      const table = Array.from({ length: 9 }, () => Array.from({ length: 10 }, () => rng.int(256)));
      const rowIds = ['8,104,377', '8,104,378', '12,950,211', '19,600,042', '27,331,905', '40,118,760', '52,777,013', '61,002,488', '…'];
      const hitRows = [1, 3, 4, 6];
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(); const sm = w < 560; const pad = sm ? 10 : 16;
        const i = st.i, el = st.elapsed(), f = V.clamp(el / 1.6, 0, 1), fe = V.ease.inOut(f);
        const stock = S.stock;
        const bw = sm ? (w - 2 * pad - 10) / 2 : w * 0.3, bh = sm ? 66 : 72;
        const gpu = { x: pad, y: 12, w: bw, h: bh }, cpu = { x: w - pad - bw, y: 12, w: bw, h: bh };
        card(g, gpu.x, gpu.y, gpu.w, gpu.h, sm ? 'GPU worker' : 'GPU worker · model thread', sm ? 'layer 1' : 'layer 1 requests the PLE', C.a, { on: i === 0 || i >= 5, ts: sm ? 12 : 13, ss: 10.5 });
        card(g, cpu.x, cpu.y, cpu.w, cpu.h, sm ? 'Offload (CPU)' : 'Offload process (CPU)', sm ? 'model on meta' : 'PleOffloadWorker · model on the meta device', C.b, { on: i >= 1 && i <= 4, ts: sm ? 12 : 13, ss: 10.5 });
        // ZMQ channel
        const zy = gpu.y + 26;
        const mx0 = gpu.x + gpu.w, mx1 = cpu.x;
        if (!sm) {
          V.arrow(g, mx0 + 6, zy, mx1 - 6, zy, { color: V.alpha(C.muted, 0.6), lw: 1.5 });
          V.text(g, 'ZMQ PUSH · msgpack', (mx0 + mx1) / 2, zy - 7, { size: 11, align: 'center', color: C.muted, maxW: mx1 - mx0 - 10 });
        }
        if (i === 0) {
          const p0 = sm ? [gpu.x + gpu.w / 2, gpu.y + gpu.h] : [mx0 + 6, zy], p1 = sm ? [cpu.x + cpu.w / 2, cpu.y + cpu.h] : [mx1 - 6, zy];
          const path = sm ? [p0, [p0[0], p0[1] + 12], [p1[0], p1[1] + 12], p1] : [p0, p1];
          const p = along(path, fe);
          badge(g, 'seq=42', p[0], p[1] + 4, C.d, { align: 'center', font: 'mono', size: 11 });
        }
        // flag
        const fy = sm ? gpu.y + gpu.h + 34 : gpu.y + gpu.h - 10;
        const fxc = w / 2;
        const flagV = (i >= 5 || (i === 4 && fe > 0.6)) && !stock ? 42 : 41;
        const fbw = sm ? 150 : Math.min(190, mx1 - mx0 - 20);
        V.box(g, fxc - fbw / 2, fy - 16, fbw, 30, { r: 8, fill: V.alpha(C.d, i === 4 || i === 5 ? 0.22 : 0.08), stroke: V.alpha(C.d, 0.8), lw: 1.3 });
        V.text(g, 'done_flag = ' + flagV, fxc, fy + 4, { size: 12, font: 'mono', weight: 700, align: 'center', color: C.ink });
        V.text(g, 'int64 · shared memory', fxc, fy + 30, { size: 11, align: 'center', color: C.muted });
        // mmap table (right, bottom)
        const tby = sm ? fy + 60 : gpu.y + gpu.h + 58;
        const cellW = sm ? Math.min(14, (w / 2 - pad - 44) / 10) : Math.min(20, (w * 0.42 - 70) / 10);
        const tbx = w - pad - 10 * (cellW + 2);
        V.text(g, sm ? 'mmap table [rows×90 B]' : 'packed table (mmap, evictable) · [~320M × 90] uint8', w - pad, tby - 10, { size: 11, align: 'right', color: C.d, weight: 600, maxW: sm ? w / 2 : w * 0.5 });
        table.forEach((row, ri) => {
          const yy = tby + ri * (cellW + 2);
          const hit = hitRows.includes(ri) && i >= 2;
          const last = ri === table.length - 1;
          if (!sm && !last) V.text(g, rowIds[ri], tbx - 6, yy + cellW / 2 + 4, { size: 11, align: 'right', color: hit ? C.ink : C.muted, font: 'mono' });
          if (last) { V.text(g, '⋮', tbx + 5 * (cellW + 2), yy + cellW, { size: 12, align: 'center', color: C.muted }); return; }
          row.forEach((v, ci) => {
            const isScale = ci >= 9;
            const col = isScale ? C.c : C.d;
            V.box(g, tbx + ci * (cellW + 2), yy, cellW, cellW, { r: 3, fill: V.alpha(col, hit ? 0.85 : 0.18 + (v / 255) * 0.25) });
          });
          if (hit && i === 2) { g.strokeStyle = C.ink; g.lineWidth = 1.5; V.rr(g, tbx - 2, yy - 2, 10 * (cellW + 2) + 2, cellW + 4, 4); g.stroke(); }
        });
        if (!sm) V.text(g, 'E2M1 codes (80 B) · FP8 scales (10 B)', w - pad, tby + table.length * (cellW + 2) + 12, { size: 11, align: 'right', color: C.muted });
        // hashing (step 1)
        if (i === 1) {
          const hx = sm ? w / 2 + 4 : cpu.x + 8, hy = cpu.y + cpu.h + 18;
          const ids = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8'];
          const cs = sm ? 9 : 11;
          ids.forEach((s, k) => { const on = k < Math.floor(fe * 16) + 1; sq(g, hx + (k % 8) * (cs + 3) + cs / 2, hy + Math.floor(k / 8) * (cs + 3) + cs / 2, cs, on ? (k < 8 ? C.b : C.f) : V.alpha(C.muted, 0.25)); });
          V.text(g, '8 bigram + 8 trigram', hx, hy + 2 * (cs + 3) + 12, { size: 11, color: C.muted });
        }
        // GPU buffer (left, bottom)
        const gbx = pad, gby = tby, gcs = sm ? Math.min(12, (w / 2 - pad - 20) / 10) : Math.min(16, (w * 0.4) / 12);
        V.text(g, sm ? 'IPC buffer (GPU)' : 'IPC buffer on the GPU · 16 × 90 B = 1440 B/token', gbx, gby - 10, { size: 11, color: C.a, weight: 600, maxW: sm ? w / 2 - pad : w * 0.45 });
        const filled = i > 3 ? 16 : i === 3 ? Math.floor(fe * 16) : 0;
        const rowsShown = 8;
        for (let r = 0; r < rowsShown; r++) for (let c = 0; c < 10; c++) {
          const idx = r * 2 + (c >= 5 ? 1 : 0);
          const on = idx < filled;
          const stale = stock && i >= 5;
          const dq = i === 6 && !stock;
          const col = dq ? V.cmap(0.35 + 0.6 * Math.abs(Math.sin(r * 1.3 + c * 0.7 + t)), 'seq') : stale ? V.alpha(C.red, 0.35) : on ? V.alpha(c % 5 === 4 ? C.c : C.d, 0.85) : V.alpha(C.muted, 0.14);
          V.box(g, gbx + c * (gcs + 2) + (c >= 5 ? 6 : 0), gby + r * (gcs + 2), gcs, gcs, { r: 3, fill: col });
        }
        // on phones, the long texts go below the table and the buffer, at full width
        const belowY = sm ? Math.max(gby + rowsShown * (gcs + 2), tby + (table.length - 1) * (cellW + 2) + cellW) + 20 : gby + rowsShown * (gcs + 2) + 18;
        const belowW = sm ? w - 2 * pad : w * 0.45;
        if (i === 6 && !stock) V.text(g, '→ BF16 into the residual', gbx, gby + rowsShown * (gcs + 2) + 16, { size: 11, color: C.a, weight: 700 });
        if (i === 6 && stock) wrap(g, 'stock: the result never arrived; the worker stays stuck spinning', gbx, belowY, belowW, 14, { size: 11, color: C.red, weight: 600, maxLines: 4 });
        // H2D copy (step 3)
        if (i === 3) {
          const a = [tbx, tby + 3 * (cellW + 2)], b = [gbx + 10 * (gcs + 2) + 10, gby + 2 * (gcs + 2)];
          const mid = [(a[0] + b[0]) / 2, Math.max(a[1], b[1]) + (sm ? 40 : 60)];
          const path = [a, mid, b];
          poly(g, path, { color: V.alpha(C.a, 0.5), lw: 2, dash: [5, 4], off: -t * 30 });
          for (let k = 0; k < 6; k++) { const p = along(path, (k / 6 + t * 0.5) % 1); sq(g, p[0], p[1], 7, C.d); }
          V.text(g, 'H2D async · copy_stream', mid[0], mid[1] + 14, { size: 11, align: 'center', color: C.a, weight: 600 });
        }
        // step 4: sync
        if (i === 4) {
          V.text(g, stock ? 'cuStreamWriteValue32 (stream mem op)' : 'copy_stream.synchronize() ✓', fxc, fy - 24, { size: 11, font: 'mono', align: 'center', color: stock ? C.red : C.e, weight: 700, maxW: w - 2 * pad });
        }
        // step 5: wait (text below the GPU buffer)
        if (i === 5) {
          const sx = gbx, sy = belowY;
          const mw = belowW;
          if (stock) {
            const a = t * 6; g.beginPath(); g.arc(sx + 9, sy - 4, 8, a, a + Math.PI * 1.5); g.strokeStyle = C.red; g.lineWidth = 3; g.stroke();
            wrap(g, 'CAN_USE_STREAM_MEM_OPS = 0: the wait "returns success", but inside the graph nothing waits. The GPU reads a stale buffer and the worker spins forever.', sx + 24, sy, mw - 24, 14, { size: 11, color: C.red, weight: 600, maxLines: 7 });
          } else {
            const spins = Math.min(2000, Math.floor(el * 1400));
            const done = el > 2;
            const a = t * 6;
            if (!done) { g.beginPath(); g.arc(sx + 9, sy - 4, 8, a, a + Math.PI * 1.5); g.strokeStyle = C.a; g.lineWidth = 3; g.stroke(); }
            V.text(g, done ? '✓ flag ≥ 42: layer 1 continues' : spins < 2000 ? '_wait_done(42) · spins ' + spins + '/2000' : '_wait_done(42) · sleep(50 µs)…', sx + (done ? 0 : 24), sy, { size: 11.5, font: 'mono', color: done ? C.e : C.ink, weight: 700, maxW: mw });
          }
        }
      });
    }

    /* ================= Fig 9 — rsync vs NFS ================= */
    {
      const fig = root.querySelector('#fig-nfs');
      fitStage(fig, 560);
      const SIZE = { glm: 164, qwen: 126, ds: 157 };
      const S = { mode: 'cx8', r: 'glm', wpc: 0 };
      const ctl = UI.controls(fig);
      UI.seg(ctl, { label: 'Distribution', options: [{ v: 'rsync', label: 'rsync' }, { v: 'mgmt', label: 'NFS · management' }, { v: 'cx1', label: 'NFS · CX7 ×1' }, { v: 'cx8', label: 'NFS · CX7 nconnect=8' }], value: 'cx8', onChange: (v) => { S.mode = v; } });
      UI.seg(ctl, { label: 'Recipe', options: [{ v: 'glm', label: 'GLM 164 GiB' }, { v: 'qwen', label: 'Qwen ~126 GiB' }, { v: 'ds', label: 'DS ~157 GiB' }], value: 'glm', onChange: (v) => { S.r = v; } });
      const rDisk = UI.readout(ctl, 'Disk on the worker', '—');
      const rHead = UI.readout(ctl, 'Head must stay up?', '—');
      const rWire = UI.readout(ctl, 'Over the cable per cold start', '—');
      V.fig(fig, (g, w, h, t, dt) => {
        const C = V.C(); const sm = w < 560; const pad = sm ? 10 : 18;
        const m = S.mode, sz = SIZE[S.r];
        const nfs = m !== 'rsync';
        rDisk.set(nfs ? '0 GiB' : sz + ' GiB (2nd copy)');
        rHead.set(nfs ? 'yes, always' : 'no');
        rWire.set(nfs ? '~' + sz + ' GiB' : '0 (copied 1×)');
        const footH = sm ? 92 : 64;
        // layout: side by side (desktop) or stacked (phone)
        let head, wk, ih;
        if (!sm) {
          const bw = w * 0.28, bh = h - 14 - footH - 8;
          head = { x: pad, y: 14, w: bw, h: bh }; wk = { x: w - pad - bw, y: 14, w: bw, h: bh };
          ih = Math.min(44, (bh - 56) / 3 - 6);
        } else {
          const bh = 150;
          head = { x: pad, y: 10, w: w - 2 * pad, h: bh }; wk = { x: pad, y: h - footH - 8 - bh, w: w - 2 * pad, h: bh };
          ih = 30;
        }
        card(g, head.x, head.y, head.w, head.h, 'Head (rank 0)', sm ? '' : 'exports the HF cache', C.b, { ts: sm ? 12 : 13 });
        card(g, wk.x, wk.y, wk.w, wk.h, 'Worker (rank 1)', sm ? '' : nfs ? 'mounts the cache (ro)' : 'has its own copy', C.a, { ts: sm ? 12 : 13 });
        const inner = (b, y, hh, title, sub, col, on = true) => {
          V.box(g, b.x + 8, y, b.w - 16, hh, { r: 8, fill: V.alpha(col, on ? 0.14 : 0.04), stroke: V.alpha(col, on ? 0.8 : 0.3), lw: 1 });
          if (sm) {
            V.text(g, title, b.x + 14, y + hh / 2 + 4, { size: 11, weight: 700, color: on ? C.ink : C.muted, maxW: (b.w - 26) * 0.46 });
            if (sub) V.text(g, sub, b.x + b.w - 14, y + hh / 2 + 4, { size: 11, color: C.muted, align: 'right', maxW: (b.w - 26) * 0.52 });
          } else {
            V.text(g, title, b.x + 14, y + 15, { size: 11, weight: 700, color: on ? C.ink : C.muted, maxW: b.w - 26 });
            if (sub && hh > 30) V.text(g, sub, b.x + 14, y + 30, { size: 11, color: C.muted, maxW: b.w - 26 });
          }
        };
        const iy = (b) => b.y + (sm ? 28 : 44);
        inner(head, iy(head), ih, 'NVMe · HF cache', sz + ' GiB · snapshot', C.d);
        inner(head, iy(head) + ih + 8, ih, 'nfsd container', 'rpc.nfsd 8 · NFSv4 only', C.b, nfs);
        inner(head, iy(head) + 2 * (ih + 8), ih, 'export ro,fsid=0', 'pseudoroot = $HF_CACHE_DIR', C.b, nfs);
        if (nfs) {
          inner(wk, iy(wk), ih, 'NFS 4.2 volume (ro)', '/root/.cache/huggingface', C.a);
          inner(wk, iy(wk) + ih + 8, ih, 'local JIT caches', 'writable (Triton, FlashInfer…)', C.e);
        } else {
          inner(wk, iy(wk), ih, 'local NVMe', sz + ' GiB · 2nd copy', C.d);
          inner(wk, iy(wk) + ih + 8, ih, 'local JIT caches', 'writable', C.e);
        }
        // worker page cache
        const pcT = nfs && m !== 'mgmt' ? 0.8 : m === 'rsync' ? 0.35 : 0.1;
        S.wpc += (pcT - S.wpc) * Math.min(1, dt);
        const py = iy(wk) + 2 * (ih + 8);
        V.box(g, wk.x + 8, py, wk.w - 16, ih, { r: 8, fill: V.alpha(C.d, 0.05), stroke: V.alpha(C.d, 0.5), lw: 1 });
        V.box(g, wk.x + 8, py, (wk.w - 16) * S.wpc, ih, { r: 8, fill: V.alpha(C.d, 0.25) });
        inner(wk, py, ih, 'worker page cache', nfs ? 'NFS fills the cache too' : 'local read', C.d, false);
        // links: coordinate along the link (u) and lateral offset (o)
        const cablH = sm ? 40 : 46;
        let mg, cb;
        if (!sm) {
          const lx0 = head.x + head.w, lx1 = wk.x;
          mg = { a: [lx0, head.y + head.h * 0.2], b: [lx1, head.y + head.h * 0.2] };
          cb = { a: [lx0, head.y + head.h * 0.62], b: [lx1, head.y + head.h * 0.62] };
        } else {
          const ly0 = head.y + head.h, ly1 = wk.y;
          mg = { a: [pad + (w - 2 * pad) * 0.2, ly0], b: [pad + (w - 2 * pad) * 0.2, ly1] };
          cb = { a: [pad + (w - 2 * pad) * 0.62, ly0], b: [pad + (w - 2 * pad) * 0.62, ly1] };
        }
        const P = (L, u, o = 0) => (sm ? [L.a[0] + o, V.lerp(L.a[1] + 6, L.b[1] - 6, u)] : [V.lerp(L.a[0] + 8, L.b[0] - 8, u), L.a[1] + o]);
        V.line(g, mg.a[0], mg.a[1], mg.b[0], mg.b[1], { color: V.alpha(C.muted, 0.7), lw: 2 });
        if (sm) V.text(g, 'mgmt 10.0.0.x', mg.a[0] + 8, (mg.a[1] + mg.b[1]) / 2 - 4, { size: 11, color: C.muted });
        else V.text(g, 'management network · 10.0.0.x', (mg.a[0] + mg.b[0]) / 2, mg.a[1] - 7, { size: 11, align: 'center', color: C.muted });
        if (sm) V.box(g, cb.a[0] - cablH / 2, cb.a[1], cablH, cb.b[1] - cb.a[1], { r: cablH / 2, fill: V.alpha(C.a, 0.08), stroke: V.alpha(C.a, 0.7), lw: 1.5 });
        else V.box(g, cb.a[0], cb.a[1] - cablH / 2, cb.b[0] - cb.a[0], cablH, { r: cablH / 2, fill: V.alpha(C.a, 0.08), stroke: V.alpha(C.a, 0.7), lw: 1.5 });
        if (sm) V.text(g, 'CX7 · 200 Gb/s', cb.a[0] + cablH / 2 + 8, (cb.a[1] + cb.b[1]) / 2 - 4, { size: 11, color: C.a, weight: 600 });
        else V.text(g, 'ConnectX-7 · RoCE 200 Gb/s · 10.0.22.x', (cb.a[0] + cb.b[0]) / 2, cb.a[1] - cablH / 2 - 7, { size: 11, align: 'center', color: C.a, weight: 600, maxW: cb.b[0] - cb.a[0] - 6 });
        const sideX = sm ? cb.a[0] + cablH / 2 + 8 : null;
        const note = (s, dy, col = C.muted) => {
          if (sm) V.text(g, s, sideX, (cb.a[1] + cb.b[1]) / 2 + 10 + dy, { size: 11, color: col, maxW: w - pad - sideX });
          else V.text(g, s, (cb.a[0] + cb.b[0]) / 2, cb.a[1] + cablH / 2 + 14 + dy, { size: 11, align: 'center', color: col, maxW: cb.b[0] - cb.a[0] - 6 });
        };
        const lanes = m === 'cx8' ? 8 : 1;
        if (m === 'mgmt') {
          for (let k = 0; k < 3; k++) { const q = P(mg, (k / 3 + t * 0.05) % 1); sq(g, q[0], q[1], 6, C.muted); }
          const mx = sm ? mg.a[0] : (mg.a[0] + mg.b[0]) / 2, my = sm ? (mg.a[1] + mg.b[1]) / 2 + 18 : mg.a[1] + 22;
          V.glow(g, mx, my - 5, 18, C.red, 0.5 + 0.3 * Math.sin(t * 5));
          V.text(g, '✕ refused', mx + (sm ? 8 : 0), my, { size: 12, weight: 700, align: sm ? 'left' : 'center', color: C.red });
          if (!sm) wrap(g, '"turns a 164 GiB load into an overnight job"', mg.a[0] + 8, my + 17, mg.b[0] - mg.a[0] - 16, 14, { size: 11, color: C.muted, maxLines: 3 });
        } else if (m === 'rsync') {
          for (let k = 0; k < 4; k++) { const q = P(cb, (k / 4 + t * 0.12) % 1); sq(g, q[0], q[1], 7, V.alpha(C.d, 0.5)); }
          note('one-time copy before the 1st boot', 0);
        } else {
          for (let l = 0; l < lanes; l++) {
            const o = lanes === 1 ? 0 : -cablH / 2 + 5 + (l + 0.5) * ((cablH - 10) / lanes);
            const n = lanes === 1 ? 3 : 5;
            const sp = lanes === 1 ? 0.2 : 0.16 + ((l * 7) % 5) * 0.025;
            for (let k = 0; k < n; k++) {
              const q = P(cb, (k / n + t * sp + ((l * 0.618) % 1)) % 1, o);
              const big = lanes === 1;
              if (sm) V.box(g, q[0] - (big ? 3.5 : 1.6), q[1] - 5, big ? 7 : 3.2, big ? 12 : 10, { r: 2, fill: C.a });
              else V.box(g, q[0] - 6, q[1] - (big ? 3.5 : 1.6), big ? 14 : 12, big ? 7 : 3.2, { r: 2, fill: C.a });
            }
          }
          const util = m === 'cx8' ? 0.85 : 0.22;
          if (sm) {
            const ux = sideX, uw = Math.min(110, w - pad - sideX), uy = (cb.a[1] + cb.b[1]) / 2 + 4;
            V.box(g, ux, uy, uw, 7, { r: 3.5, fill: V.alpha(C.muted, 0.15) }); V.box(g, ux, uy, uw * util, 7, { r: 3.5, fill: C.a });
            note(lanes === 1 ? '1 connection: idle' : '8 connections: full', 13);
          } else {
            const ux = cb.a[0] + 8, uw = cb.b[0] - cb.a[0] - 16, uy = cb.a[1] + cablH / 2 + 12;
            V.box(g, ux, uy, uw, 8, { r: 4, fill: V.alpha(C.muted, 0.15) }); V.box(g, ux, uy, uw * util, 8, { r: 4, fill: C.a });
            note((lanes === 1 ? '1 TCP connection: idle cable' : '8 TCP connections: full cable') + ' (illustrative)', 24);
          }
        }
        // footer
        const foot = { rsync: 'rsync: decoupled nodes and a shorter cold start; it costs a second checkpoint on the worker disk.', mgmt: 'nfs-share.sh refuses 10.0.0.* and 127.*: they are loopback/management aliases, not the data path.', cx1: 'NFS over the CX7 with one connection: correct, but a single TCP connection rarely fills 200 Gb/s.', cx8: 'nconnect=8, 1 MiB rsize/wsize, hard, timeo=600: "nconnect=8 is what makes a cold load use the cable".' }[m];
        wrap(g, foot, pad, h - footH + 16, w - 2 * pad, 15, { size: sm ? 11 : 12, color: C.ink, maxLines: sm ? 4 : 2 });
        if (!sm) V.text(g, 'mount options: nfsvers=4.2,ro,nconnect=8,rsize=1048576,wsize=1048576,hard,timeo=600', pad, h - 8, { size: 11, font: 'mono', color: C.muted, maxW: w - 2 * pad });
      });
    }

    /* ================= Fig 10 — complete per-node budget ================= */
    {
      const fig = root.querySelector('#fig-orcamento');
      fitStage(fig, 620);
      const S = { L: 65536, kvT: 8, slack: 10, mtp: false, ple: true, avail: 110 };
      const ctl = UI.controls(fig);
      let H;
      const re = () => H && H.redraw();
      UI.slider(ctl, { label: 'TP1 · context', min: 16384, max: 262144, step: 16384, value: 65536, fmt: (v) => Math.round(v / 1024) + 'k', onInput: (v) => { S.L = v; re(); } });
      UI.slider(ctl, { label: 'KV_TARGET_GIB', min: 4, max: 24, step: 1, value: 8, fmt: (v) => v + ' GiB', onInput: (v) => { S.kvT = v; re(); } });
      UI.slider(ctl, { label: 'HOST_SLACK_GIB', min: 2, max: 12, step: 1, value: 10, fmt: (v) => v + ' GiB', onInput: (v) => { S.slack = v; re(); } });
      UI.slider(ctl, { label: 'MemAvailable', min: 80, max: 118, step: 1, value: 110, fmt: (v) => v + ' GiB', onInput: (v) => { S.avail = v; re(); } });
      UI.toggle(ctl, { label: 'MTP (+1.49 GiB)', value: false, onChange: (v) => { S.mtp = v; re(); } });
      UI.toggle(ctl, { label: 'PLE_OFFLOAD (mmap table)', value: true, onChange: (v) => { S.ple = v; re(); } });
      function tp1() {
        const P = 98.66, PLE = 26.82, O = 5.6, MTPG = S.mtp ? 1.49 : 0, KVB = 29482;
        const w = P - (S.ple ? PLE : 0);
        const kvNeed = (S.L * KVB) / 2 ** 30;
        const kv = Math.max(kvNeed, S.kvT);
        const budget = w + O + MTPG + kv;
        const gmu = budget / MT;
        const cap = Math.floor(budget + S.slack);
        const maxCap = Math.floor(MT - 16);
        const tok = Math.floor((kv * 2 ** 30) / KVB);
        return { w, kvNeed, kv, budget, gmu, cap, maxCap, tok, O, MTPG, PLE };
      }
      H = V.fig(fig, (g, w, h) => {
        const C = V.C(); const sm = w < 560; const pad = sm ? 10 : 18;
        const labW = sm ? 0 : 150;
        const bx = pad + labW, bw = w - pad - bx;
        const rowH = sm ? 30 : 34, gap = sm ? 30 : 23;
        let y = 14;
        // legend
        const leg = [[C.a, 'weights'], [C.e, sm ? 'fixed extra (KDA BF16 / non-torch)' : 'fixed extra: KDA BF16 copy (GLM) · non-torch (Qwen)'], [C.b, 'activation / rest'], [C.b, 'CUDA graphs outside GMU', true], [C.c, 'KV'], [C.f, 'MTP'], [C.d, 'cgroup slack', true], [C.muted, 'outside / OS / cache', true]];
        let lx = pad, ly = y + 6;
        leg.forEach(([c, l, hz]) => {
          g.font = V.font(11, 'sans', 400); const tw = g.measureText(l).width;
          if (lx + tw + 22 > w - pad) { lx = pad; ly += 17; }
          if (hz) { V.box(g, lx, ly - 8, 11, 11, { r: 3, fill: V.alpha(c, 0.15), stroke: V.alpha(c, 0.6), lw: 1 }); hatch(g, lx, ly - 8, 11, 11, V.alpha(c, 0.7), 4); }
          else V.box(g, lx, ly - 8, 11, 11, { r: 3, fill: c });
          V.text(g, l, lx + 15, ly + 1.5, { size: 11, color: C.muted }); lx += tw + 28;
        });
        y = ly + 16;
        /** notes: [[start GiB, text, color, 'left'|'right-edge']] — labels for narrow segments, right below the bar */
        const row = (label, sub, parts, marks, notes) => {
          if (sm) { V.text(g, label, pad, y + 10, { size: 11.5, weight: 700, color: C.ink, maxW: w * 0.5 }); if (sub) V.text(g, sub, w - pad, y + 10, { size: 11, align: 'right', color: C.muted, maxW: w * 0.48 }); y += 16; }
          else { V.text(g, label, pad, y + rowH / 2 - 2, { size: 12, weight: 700, color: C.ink, maxW: labW - 10 }); if (sub) V.text(g, sub, pad, y + rowH / 2 + 13, { size: 11, color: C.muted, maxW: labW - 10 }); }
          stack(g, bx, y, bw, rowH, parts, MT, { size: 11 });
          const mk = (marks || []).slice().sort((p1, p2) => p1[0] - p2[0]); mk.forEach(([v, l, c], k) => { const mx = bx + (Math.min(v, MT) / MT) * bw; V.line(g, mx, y - 4, mx, y + rowH + 4, { color: c, lw: 2 }); if (l) { const left = mk.length > 1 && k === 0; V.text(g, l, mx + (left ? -3 : 3), y + rowH + 14, { size: 11, align: left ? 'right' : 'left', color: c, weight: 600 }); } });
          (notes || []).forEach(([v, l, c, al]) => {
            const nx = al === 'right-edge' ? w - pad : bx + (v / MT) * bw;
            V.line(g, al === 'right-edge' ? bx + (v / MT) * bw + 1 : nx + 1, y + rowH, al === 'right-edge' ? bx + (v / MT) * bw + 1 : nx + 1, y + rowH + 4, { color: c, lw: 1.5 });
            V.text(g, l, al === 'right-edge' ? nx : nx + 1, y + rowH + 14, { size: 11, align: al === 'right-edge' ? 'right' : 'left', color: c, weight: 600 });
          });
          y += rowH + gap;
        };
        // GLM: rest = 0.85·M − 79.65 − 3.29 − 11 (computed difference)
        row('GLM · TP2', 'GMU 0.85 · fixed KV', [{ v: 79.65, c: C.a, l: 'weights 79.65' }, { v: 3.29, c: C.e, l: '' }, { v: 11, c: C.c, l: 'KV 11' }, { v: 0.85 * MT - 93.94, c: C.b, l: 'rest*' }, { v: MT * 0.15, c: C.muted, l: 'outside 18.3', hatch: true }], [[0.85 * MT, '', C.ink]],
          [[79.65, sm ? 'KDA 3.29' : 'KDA BF16 copy 3.29', C.e]]);
        // Qwen (README): 64.46 + 4.06 + 1.07 + 32.02 = 101.61 = 0.835·M; the 0.54 of graphs sits after the u·M line
        row('Qwen · TP2', 'GMU 0.835 · measured', [{ v: 64.46, c: C.a, l: 'weights 64.46' }, { v: 4.06, c: C.e, l: '' }, { v: 1.07, c: C.b, l: '' }, { v: 32.02, c: C.c, l: 'KV 32.02' }, { v: 0.54, c: C.b, l: '', hatch: true }, { v: MT - 102.15, c: C.muted, l: 'outside 19.5', hatch: true }], [[0.835 * MT, '', C.ink]],
          [[64.46, sm ? 'non-torch' : 'non-torch 4.06 + activ. 1.07', C.e], [101.61, sm ? 'graphs out' : 'graphs 0.54 outside GMU', C.b, 'right-edge']]);
        // DeepSeek: weights and KV from the 0.83 boot (fable5-1); rest = difference up to 0.835·M
        row('DeepSeek · TP2', sm ? 'weights/KV: 0.83 boot' : 'weights/KV from the 0.83 boot', [{ v: 79.17, c: C.a, l: 'weights 79.17' }, { v: 0.835 * MT - 96.21, c: C.b, l: 'rest*' }, { v: 17.04, c: C.c, l: 'KV 17.04' }, { v: MT - 0.835 * MT, c: C.muted, l: 'outside 20.1', hatch: true }], [[0.835 * MT, '', C.ink]]);
        const r = tp1();
        const capBad = r.cap > r.maxCap, availBad = S.avail < r.cap + 4;
        if (S.ple) {
          row('Qwen · TP1 (calc.)', 'derived GMU ' + fmt3(r.gmu), [{ v: r.w, c: C.a, l: 'weights ' + fmt1(r.w) }, { v: r.O, c: C.b, l: '' }, { v: r.MTPG, c: C.f, l: '' }, { v: r.kv, c: C.c, l: 'KV ' + fmt1(r.kv) }, { v: Math.min(S.slack, MT - r.budget), c: C.d, l: 'slack', hatch: true }, { v: Math.max(0, MT - r.budget - S.slack), c: C.muted, l: 'OS + PLE mmap', hatch: true }],
            [[r.maxCap, (sm ? 'max ' : 'ceiling ') + r.maxCap, C.red], [Math.min(MT, r.cap), 'cap ' + r.cap, capBad ? C.red : C.ink]]);
        } else {
          row('Qwen · TP1 (calc.)', 'no PLE offload', [{ v: 98.66, c: C.a, l: 'weights 98.66 through UVM' }, { v: MT - 98.66, c: C.red, l: 'refused', hatch: true }], []);
        }
        // guards
        const gy0 = y + (sm ? 0 : 4);
        V.line(g, pad, gy0 - 8, w - pad, gy0 - 8, { color: V.alpha(C.muted, 0.25), lw: 1 });
        V.text(g, 'start-tp1.sh guards', pad, gy0 + 8, { size: 12, weight: 700, color: C.ink });
        const items = [];
        if (!S.ple) items.push([false, 'PLE_OFFLOAD=false: "98.6 GiB of weights through UVM hung the host last session". Refused.']);
        else {
          items.push([true, 'Budget B = ' + fmt2(r.w) + ' + 5.6 + ' + fmt2(r.MTPG) + ' + max(' + fmt2(r.kvNeed) + ', ' + S.kvT + ') = ' + fmt2(r.budget) + ' GiB → GMU ' + fmt3(r.gmu)]);
          items.push([r.kv >= r.kvNeed, 'Expected KV ' + fmt2(r.kv) + ' GiB (~' + fmtInt(r.tok / 1000) + 'k tokens, script estimate) ≥ ' + fmt2(r.kvNeed) + ' GiB for ' + Math.round(S.L / 1024) + 'k tokens']);
          items.push([!capBad, 'cgroup cap = int(B + slack) = ' + r.cap + ' GiB ≤ int(MemTotal − 16) = ' + r.maxCap + ' GiB' + (capBad ? ': refused ("line between a killed container and a hung host")' : '')]);
          items.push([!availBad, 'MemAvailable now ' + S.avail + ' GiB ≥ cap + 4 = ' + (r.cap + 4) + ' GiB' + (availBad ? ': refused ("something else is holding memory")' : '')]);
        }
        let yy = gy0 + 28;
        items.forEach(([ok, s]) => {
          V.text(g, ok ? '✓' : '✕', pad, yy, { size: 14, weight: 700, color: ok ? C.e : C.red });
          yy = wrap(g, s, pad + 20, yy, w - 2 * pad - 20, 15, { size: sm ? 11 : 12, color: C.ink, maxLines: 3 }) + 3;
        });
        if (yy + 14 < h) V.text(g, '* rest of the budget = computed difference (interpretation)', pad, Math.min(h - 6, yy + 8), { size: 11, color: C.muted });
      }, { static: true });
    }
    /* ===== TensorFold startup model (illustrative, calibrated on the recipe's published numbers) =====
       88.09 GiB estimate at 1,048,576 tokens; 12.5 GiB pool ceiling → 2,922,496 tokens; peak ~10 GiB beyond the estimate */
    const GI = 2 ** 30;
    const TFM = { SO: 5.5, E1M: 88.09, PT: 6356, L1M: 1048576, PCAP: 12.5, ST: 1.40625, DPK: 10 };
    TFM.WFIX = TFM.E1M - (TFM.L1M * TFM.PT) / GI; // ≈ 81.88 GiB: weights + states and scratch that don't grow with L
    TFM.est = (L) => TFM.WFIX + (L * TFM.PT) / GI;
    TFM.fit = (B) => (B < TFM.WFIX ? 0 : Math.min(TFM.L1M, Math.floor(((B - TFM.WFIX) * GI) / TFM.PT)));
    TFM.pool = (L, extra) => L + (extra > TFM.ST ? Math.floor(((extra - TFM.ST) * GI) / TFM.PT / 2048) * 2048 : 0);
    const tfCol = () => getComputedStyle(document.documentElement).getPropertyValue('--tf').trim() || V.C().d;
    const fmtM = (n) => (n / 1e6).toFixed(2) + 'M';
    const fmtK = (n) => (n >= 1e6 ? (n / 1e6).toFixed(2) + 'M' : Math.round(n / 1e3) + 'k');
    function bracket(g, x1, x2, y, col, label, o = {}) {
      const C = V.C();
      g.save(); g.globalAlpha = o.alpha == null ? 1 : o.alpha;
      g.strokeStyle = col; g.lineWidth = 1.6; if (o.dash) g.setLineDash(o.dash);
      g.beginPath(); g.moveTo(x1, y - 5); g.lineTo(x1, y); g.lineTo(x2, y); g.lineTo(x2, y - 5); g.stroke(); g.setLineDash([]);
      if (label) {
        g.font = V.font(11, 'sans', 600); const tw = g.measureText(label).width;
        const right = o.align === 'right' || (x1 + (x2 - x1) / 2 + tw / 2 > (o.maxX || 1e9));
        const lx = right ? x2 : Math.max(x1, (x1 + x2) / 2 - tw / 2);
        V.text(g, label, lx, y + 13, { size: 11, weight: 600, color: o.lc || col, align: right ? 'right' : 'left' });
      }
      g.restore();
      void C;
    }

    /* ================= Fig 11 — TensorFold's startup ================= */
    {
      const fig = root.querySelector('#fig-tf-partida');
      fitStage(fig, 620);
      const RS = 40; // GiB of page cache from an rsync (illustrative)
      const S = { x: 0, rs: false, R: 14.5, t0: -1e9, req: true };
      let P = null;
      function plan() {
        const A = MT - TFM.SO - S.x;
        const free = A - (S.rs ? RS : 0);
        const B = A - S.R;
        const steps = []; let lo = 0, hi = TFM.WFIX > B ? 0 : TFM.L1M;
        while (lo < hi) { const mid = (lo + hi + 1) >> 1; const ok = TFM.est(mid) <= B; steps.push([lo, hi, mid, ok]); if (ok) lo = mid; else hi = mid - 1; }
        const Lfit = lo, fits = Lfit >= TFM.L1M, refit = !fits && Lfit > 0;
        const L = fits ? TFM.L1M : Lfit;
        const extra = L > 0 ? V.clamp(B - TFM.est(L), 0, TFM.PCAP) : 0;
        return { A, free, B, steps, Lfit, fits, refit, L, extra, tok: L > 0 ? TFM.pool(L, extra) : 0, Bfree: free - S.R };
      }
      const ctl = UI.controls(fig);
      UI.slider(ctl, { label: 'Other tenants at startup', min: 0, max: 24, step: 0.5, value: 0, fmt: (v) => fmt1(v) + ' GiB', onInput: (v) => { S.x = v; upd(); } });
      UI.slider(ctl, { label: 'Reserve R', min: 2, max: 24, step: 0.5, value: 14.5, fmt: (v) => fmt1(v) + ' GiB', onInput: (v) => { S.R = v; upd(); } });
      UI.toggle(ctl, { label: 'page cache from an rsync (40 GiB)', value: false, onChange: (v) => { S.rs = v; upd(); } });
      UI.button(ctl, '▶ start', () => { S.req = true; }, { primary: true });
      const rB = UI.readout(ctl, 'Budget B', '—');
      const rL = UI.readout(ctl, 'Window', '—');
      const rP = UI.readout(ctl, 'Pool', '—');
      function upd() {
        P = plan(); S.t0 = -1e9; S.req = false;
        rB.set(fmt1(P.B) + ' GiB');
        rL.set(P.L ? fmtInt(P.L) : 'refused', P.fits ? 'ok' : P.L ? 'warn' : 'bad');
        rP.set(P.tok ? fmtM(P.tok) + ' tok' : '—', P.tok ? (P.extra >= TFM.PCAP ? 'ok' : 'warn') : 'bad');
      }
      upd(); S.req = true;
      const T = () => [1.1, 1.1, 1.3, Math.max(0.4, P.steps.length * 0.13), 1.7, P.refit ? 1.7 : 0, P.L ? 1.5 : 0];
      function phaseAt(el) {
        const D = T(); let acc = 0;
        for (let i = 0; i < D.length; i++) { if (el < acc + D[i]) return { i, f: (el - acc) / D[i] }; acc += D[i]; }
        return { i: 7, f: 1 };
      }
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), TF = tfCol(), sm = w < 560, pad = sm ? 10 : 18;
        if (S.req) { S.t0 = t; S.req = false; }
        const { i: pi, f } = phaseAt(t - S.t0);
        const ef = V.ease.out(V.clamp(f, 0, 1));
        const bw = w - 2 * pad, X = (v) => pad + (V.clamp(v, 0, MT) / MT) * bw;
        const a0 = TFM.SO + S.x;
        // ---- phase title
        const nPool = P.refit ? '⑦' : '⑥';
        const verdict = P.fits ? ['⑤ it fits: only now do the weights load', 'E(1,048,576) ≤ B. Both ranks agree on the window, and loading begins.']
          : P.refit ? ['⑤ it doesn’t fit: refuses without loading anything', 'The requested window exceeds the budget. The log reports the largest one that fits; no weights or KV were allocated.']
            : ['⑤ refused: not even the smallest window fits', 'Weights and fixed states already exceed the budget. Free some memory or lower the reserve (carefully).'];
        const PH = [
          ['① read MemAvailable', 'Integrated GPU: the budget starts from MemAvailable in /proc/meminfo, not from CUDA’s free figure (which is MemFree).'],
          ['② set the reserve R aside', 'R stays untouched: OS, peaks and whatever the estimate doesn’t count (CUDA context, NCCL, graphs, workspaces).'],
          ['③ estimate from the headers', 'Shapes and dtypes from the safetensors + a bytes-per-token formula. Not a single weight byte has been read yet.'],
          ['④ binary search for the window', 'The largest L with E(L) ≤ B. Each rank runs its own; both keep the smaller result.'],
          verdict,
          ['⑥ start.sh restarts once', 'It reads “largest fitting prompt-plus-reply window: N tokens” and starts again with --context N.'],
          [nPool + ' what’s left becomes pool', 'The shared pool grows into the rest of B, up to 12.5 GiB per Spark beyond the window.'],
          [P.L ? (P.refit ? '✓ up, with a smaller window' : '✓ up and serving') : '✕ didn’t start', P.L ? 'Press ▶ start to replay the phases, or play with the controls.' : 'Nothing was allocated: the machine is untouched. Press ▶ start to replay the phases.'],
        ];
        const [tt, td] = PH[pi];
        V.text(g, tt, pad, 18, { size: 13.5, weight: 700, color: pi === 7 ? (P.L ? C.e : C.red) : C.ink, maxW: bw });
        let y = wrap(g, td, pad, 36, bw, 14, { size: 11.5, color: C.muted, maxLines: 2 }) + 10;
        // ---- row 1: the host
        V.text(g, 'HOST AT STARTUP · 121.69 GiB', pad, y, { size: 11, font: 'mono', color: C.muted });
        const bh = sm ? 26 : 30, by1 = y + 8;
        stack(g, pad, by1, bw, bh, [{ v: TFM.SO, c: C.b, l: 'OS' }, { v: S.x, c: C.f, l: 'others' }, { v: S.rs ? RS : 0, c: C.muted, l: 'page cache (rsync)', hatch: true }, { v: Math.max(0, P.free), c: C.muted, a: 0, l: 'free', lc: C.muted }], MT, { size: 11 });
        let by = by1 + bh + 8;
        const aA = pi === 0 ? ef : 1;
        bracket(g, X(a0), X(MT), by, C.a, 'MemAvailable = ' + fmt1(P.A) + ' GiB', { alpha: aA, maxX: w - pad });
        if (S.rs) { by += 22; bracket(g, X(a0 + RS), X(MT), by, C.c, 'MemFree = ' + fmt1(P.free) + ' (CUDA’s “free”)', { alpha: aA, dash: [4, 3], maxX: w - pad }); }
        // ---- row 2: budget
        y = by + 32;
        V.text(g, 'BUDGET B = MemAvailable − R', pad, y, { size: 11, font: 'mono', color: C.muted });
        const by2 = y + 8;
        V.box(g, pad, by2, bw, bh, { r: 8, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.25), lw: 1 });
        const rW = pi >= 1 ? (pi === 1 ? ef : 1) * S.R : 0;
        const xb0 = X(a0), xb1 = X(MT - rW);
        V.box(g, xb0, by2, Math.max(0, xb1 - xb0), bh, { r: 6, fill: V.alpha(C.a, 0.16), stroke: C.a, lw: 1.5 });
        const bLab = pi >= 1 ? 'B = ' + fmt1(P.A - rW) + ' GiB' : 'MemAvailable ' + fmt1(P.A) + ' GiB';
        V.text(g, bLab, (xb0 + xb1) / 2, by2 + bh / 2 + 4, { size: 11.5, weight: 700, color: C.ink, align: 'center', maxW: Math.max(40, xb1 - xb0 - 8) });
        if (rW > 0.05) {
          const xr = X(MT - rW);
          g.fillStyle = V.alpha(C.ink, 0.06); g.fillRect(xr, by2, X(MT) - xr, bh); hatch(g, xr, by2, X(MT) - xr, bh, V.alpha(C.ink, 0.45));
          const glow = pi === 7 ? 0.5 + 0.5 * Math.sin(t * 3) : 1;
          V.box(g, xr, by2, X(MT) - xr, bh, { r: 4, stroke: V.alpha(C.ink, 0.35 + 0.45 * glow), lw: 1.5 });
          V.text(g, 'R ' + fmt1(rW) + (sm ? '' : ' (untouched)'), X(MT), by2 + bh + 13, { size: 11, weight: 700, color: C.ink, align: 'right' });
        }
        // ---- row 3: estimate + pool
        y = by2 + bh + 30;
        V.text(g, sm ? 'ESTIMATE E(L) + POOL' : 'ESTIMATE E(L), FROM THE HEADERS ONLY + POOL', pad, y, { size: 11, font: 'mono', color: C.muted });
        const by3 = y + 8;
        V.box(g, pad, by3, bw, bh, { r: 8, fill: V.alpha(C.muted, 0.05), stroke: V.alpha(C.muted, 0.25), lw: 1 });
        const wf = pi < 2 ? 0 : pi === 2 ? ef * TFM.WFIX : TFM.WFIX;
        let Lc = 0, kStep = -1;
        if (pi === 3 && P.steps.length) { kStep = Math.min(P.steps.length - 1, Math.floor(f * P.steps.length)); Lc = P.steps[kStep][2]; }
        else if (pi === 4) Lc = P.L ? TFM.L1M : 0; // the request: 1M (red if it exceeds B)
        else if (pi === 5) Lc = V.lerp(TFM.L1M, P.L, ef);
        else if (pi >= 6) Lc = P.L;
        const cw = (Lc * TFM.PT) / GI;
        const ex = pi === 6 ? ef * P.extra : pi === 7 ? P.extra : 0;
        const xs = X(a0), xW = X(a0 + wf), xC = X(a0 + wf + cw), xP = X(a0 + wf + cw + ex), xB = X(a0 + P.B);
        if (wf > 0) {
          V.box(g, xs, by3, xW - xs, bh, { r: 4, fill: V.alpha(TF, 0.85) });
          if (xW - xs > 120) V.text(g, 'weights + fixed ' + fmt1(wf), (xs + xW) / 2, by3 + bh / 2 + 4, { size: 11, weight: 600, color: onFill(C), align: 'center' });
        }
        if (cw > 0) { g.fillStyle = V.alpha(C.c, 0.85); g.fillRect(xW, by3, xC - xW, bh); }
        if (ex > 0.01) {
          g.fillStyle = V.alpha(C.e, 0.75); g.fillRect(xC, by3, xP - xC, bh);
          const sh = (t * 0.6) % 1; g.fillStyle = V.alpha(C.bg, 0.25); g.fillRect(xC + sh * (xP - xC), by3, Math.min(6, xP - xC), bh);
          if (xP - xC > 34) V.text(g, 'pool', (xC + xP) / 2, by3 + bh / 2 + 4, { size: 11, weight: 600, color: onFill(C), align: 'center' });
        }
        // headers flying into the estimate (phase ③)
        if (pi === 2) for (let k = 0; k < 6; k++) { const u = V.clamp(f * 1.6 - k * 0.1, 0, 1); if (u <= 0 || u >= 1) continue; const px = V.lerp(pad + 4, xW - 6, u), py = by3 - 10 - 14 * Math.sin(Math.PI * u); V.box(g, px - 7, py - 5, 14, 10, { r: 2, fill: V.alpha(C.ink, 0.12), stroke: V.alpha(C.ink, 0.6), lw: 1 }); }
        // excess in red
        const tot = a0 + wf + cw + ex;
        if (wf > 0 && tot > a0 + P.B + 0.01) {
          const x0 = Math.max(xB, xs), x1 = X(tot);
          g.fillStyle = V.alpha(C.red, 0.18); g.fillRect(x0, by3 - 3, x1 - x0, bh + 6); hatch(g, x0, by3 - 3, x1 - x0, bh + 6, V.alpha(C.red, 0.8), 5);
        }
        if (pi >= 1) {
          V.line(g, xB, by3 - 6, xB, by3 + bh + 6, { color: C.ink, lw: 1.5, dash: [4, 3] });
          V.text(g, 'B', xB + 4, by3 + bh + 14, { size: 11, weight: 700, color: C.ink });
        }
        if (cw > 0 && pi >= 3) V.text(g, 'window ' + fmtK(Lc), xW + 2, by3 + bh + 14, { size: 11, color: C.c, weight: 600, align: xW > w * 0.6 ? 'right' : 'left' });
        // ---- row 4: binary search
        y = by3 + bh + 34;
        V.text(g, 'BINARY SEARCH FOR THE WINDOW L (tokens)', pad, y, { size: 11, font: 'mono', color: C.muted });
        const ay = y + 32, LX = (L) => pad + (L / TFM.L1M) * bw;
        V.line(g, pad, ay, w - pad, ay, { color: V.alpha(C.muted, 0.6), lw: 1.5 });
        [0, 262144, 524288, 786432, 1048576].forEach((v, k) => {
          V.line(g, LX(v), ay - 4, LX(v), ay + 4, { color: V.alpha(C.muted, 0.7), lw: 1 });
          V.text(g, ['0', '262k', '524k', '786k', '1.05M'][k], LX(v), ay + 17, { size: 11, color: C.muted, align: k === 0 ? 'left' : k === 4 ? 'right' : 'center', font: 'mono' });
        });
        if (pi === 3 && kStep >= 0) {
          const [lo, hi, mid, ok] = P.steps[kStep];
          g.fillStyle = V.alpha(C.ink, 0.08); g.fillRect(LX(lo), ay - 9, Math.max(2, LX(hi) - LX(lo)), 18);
          g.fillStyle = V.alpha(C.e, 0.25); g.fillRect(pad, ay - 3, LX(lo) - pad, 6);
          V.dot(g, LX(mid), ay, 6, ok ? C.e : C.red);
          const msg = 'step ' + (kStep + 1) + '/' + P.steps.length + ': L = ' + fmtInt(mid) + (ok ? ' fits ✓' : ' doesn’t fit ✕');
          V.text(g, msg, V.clamp(LX(mid), pad + 90, w - pad - 90), ay - 14, { size: 11, weight: 600, color: ok ? C.e : C.red, align: 'center' });
        } else if (pi >= 4) {
          g.fillStyle = V.alpha(C.e, 0.35); g.fillRect(pad, ay - 3, LX(P.Lfit) - pad, 6);
          if (P.Lfit > 0) { V.dot(g, LX(P.Lfit), ay, 6, C.e); V.text(g, 'L* = ' + fmtInt(P.Lfit), V.clamp(LX(P.Lfit), pad + 70, w - pad - 70), ay - 14, { size: 11, weight: 700, color: C.e, align: 'center' }); }
          else V.text(g, 'L* = 0: nothing fits', pad + 4, ay - 14, { size: 11, weight: 700, color: C.red });
        }
        // ---- verdict
        let yy = ay + 38;
        const lines = [];
        if (pi >= 1) lines.push(['·', C.ink, 'B = A₀ − R = ' + fmt1(P.A) + ' − ' + fmt1(S.R) + ' = ' + fmt1(P.B) + ' GiB']);
        if (pi >= 4) {
          if (P.fits) lines.push(['✓', C.e, 'E(1,048,576) = ' + fmt2(TFM.E1M) + ' GiB ≤ B: a 1,048,576-token window, and only now do the weights load']);
          else if (P.refit) lines.push(['✕', C.red, 'E(1,048,576) = ' + fmt2(TFM.E1M) + ' GiB > B: refuses without loading anything; largest window that fits: ' + fmtInt(P.Lfit) + ' tokens']);
          else lines.push(['✕', C.red, 'weights + fixed = ' + fmt1(TFM.WFIX) + ' GiB > B: refuses without loading anything']);
        }
        if (pi >= 5 && P.refit) lines.push(['↻', C.d, 'start.sh restarts once with --context ' + P.Lfit]);
        if (pi >= 6 && P.L) lines.push(['✓', C.e, 'pool: ' + fmtInt(P.L) + ' + ' + fmtInt(P.tok - P.L) + ' = ' + fmtInt(P.tok) + ' tokens (' + fmt1(P.extra) + ' GiB beyond the window' + (P.extra >= TFM.PCAP ? ', the ceiling' : '') + ')']);
        if (S.rs) {
          const Lf = TFM.fit(P.Bfree);
          lines.push([Lf >= TFM.L1M ? '·' : '✕', C.c, 'If the judge were MemFree: B = ' + fmt1(P.free) + ' − ' + fmt1(S.R) + ' = ' + fmt1(P.Bfree) + ' GiB → ' + (Lf >= TFM.L1M ? 'it would also fit' : Lf > 0 ? 'it would cut the window to ' + fmtInt(Lf) : 'it would refuse even the smallest window')]);
        }
        if (P.A < 110) lines.push(['⚠', C.d, 'start.sh warns: only ' + fmt1(P.A) + ' GiB available (the default wants ~110 GiB)']);
        // ---- phase strip (bottom)
        const chips = [['① read', 0], ['② reserve', 1], ['③ estimate', 2], ['④ search', 3], ['⑤ decide', 4]];
        if (P.refit) chips.push(['⑥ restart', 5]);
        if (P.L) chips.push([(P.refit ? '⑦' : '⑥') + ' pool', 6]);
        g.font = V.font(11, 'sans', 600);
        const rows = [[]]; let rw = 0;
        chips.forEach((c) => { const cwd = g.measureText(c[0]).width + 18; if (rw + cwd > bw && rows[rows.length - 1].length) { rows.push([]); rw = 0; } rows[rows.length - 1].push([c, cwd]); rw += cwd + 6; });
        const stripTop = h - 10 - rows.length * 24;
        rows.forEach((row, ri) => {
          let cx = pad; const cyy = stripTop + ri * 24;
          row.forEach(([[lab, ph], cwd]) => {
            const on = ph === pi, past = ph < pi;
            V.box(g, cx, cyy, cwd, 19, { r: 9, fill: on ? V.alpha(C.a, 0.25) : past ? V.alpha(C.a, 0.08) : 'transparent', stroke: on ? C.a : V.alpha(C.muted, past ? 0.6 : 0.35), lw: on ? 1.6 : 1 });
            V.text(g, lab, cx + cwd / 2, cyy + 13.5, { size: 11, weight: 600, color: on ? C.ink : past ? C.ink : C.muted, align: 'center' });
            cx += cwd + 6;
          });
        });
        lines.forEach(([ic, col, s]) => {
          if (yy > stripTop - 10) return;
          V.text(g, ic, pad, yy, { size: 13, weight: 700, color: col });
          yy = wrap(g, s, pad + 18, yy, bw - 18, 14, { size: sm ? 11 : 11.5, color: C.ink, maxLines: 2 }) + 3;
        });
      });
    }

    /* ================= Fig 12 — same host, two counters ================= */
    {
      const fig = root.querySelector('#fig-tf-vs-gmu');
      fitStage(fig, 720);
      const A0 = MT - TFM.SO, U = 0.85 * MT, PRE = U + 2, VUSE = A0 - 6.2, VTOK = 1233779, XMAX = 24;
      const S = { x: 0, R: 14.5, sweep: false, st0: 0, req: false };
      const vl = (x) => { const A = A0 - x; if (A < PRE) return { ok: false, A, tok: 0 }; return { ok: true, A, tok: VTOK, head: A - VUSE }; };
      const tfm = (x, R) => {
        const A = A0 - x, B = A - R, L = TFM.fit(B);
        if (!L) return { ok: false, A, B, tok: 0 };
        const extra = V.clamp(B - TFM.est(L), 0, TFM.PCAP);
        return { ok: true, A, B, L, extra, E: TFM.est(L), tok: TFM.pool(L, extra), head: A - TFM.est(L) - extra - TFM.DPK };
      };
      let cur = null, curR = null;
      const curves = () => { if (curR !== S.R) { cur = []; for (let k = 0; k <= XMAX * 10; k++) { const x = k / 10; cur.push([x, vl(x), tfm(x, S.R)]); } curR = S.R; } return cur; };
      const ctl = UI.controls(fig);
      const slX = UI.slider(ctl, { label: 'Other tenants at startup', min: 0, max: XMAX, step: 0.5, value: 0, fmt: (v) => fmt1(v) + ' GiB', onInput: (v) => { S.x = v; S.sweep = false; upd(); } });
      UI.slider(ctl, { label: 'TensorFold reserve R', min: 4, max: 24, step: 0.5, value: 14.5, fmt: (v) => fmt1(v) + ' GiB', onInput: (v) => { S.R = v; upd(); } });
      UI.button(ctl, '▶ sweep', () => { S.req = true; }, { primary: true });
      const rV = UI.readout(ctl, 'vLLM (GLM)', '—');
      const rT = UI.readout(ctl, 'TensorFold', '—');
      const zone = (hd) => (hd < 0 ? 'bad' : hd < 2 ? 'warn' : 'ok');
      function upd() {
        const v = vl(S.x), f = tfm(S.x, S.R);
        rV.set(v.ok ? fmtM(v.tok) + ' tok · headroom ' + fmt1(v.head) + ' GiB' : 'preflight refuses', v.ok ? zone(v.head) : 'bad');
        rT.set(f.ok ? fmtM(f.tok) + ' tok · headroom ' + fmt1(f.head) + ' GiB' : 'refused at startup', f.ok ? zone(f.head) : 'bad');
      }
      upd();
      V.fig(fig, (g, w, h, t) => {
        const C = V.C(), TF = tfCol(), sm = w < 560, pad = sm ? 10 : 18;
        if (S.req) { S.req = false; S.sweep = true; S.st0 = t; }
        if (S.sweep) {
          const el = t - S.st0, P = 10;
          if (el >= P) { S.sweep = false; S.x = 0; } else S.x = (XMAX * (1 - Math.cos((2 * Math.PI * el) / P))) / 2;
          slX.set(Math.round(S.x * 2) / 2, true); upd();
        }
        const v = vl(S.x), f = tfm(S.x, S.R);
        const bw = w - 2 * pad, X = (q) => pad + (V.clamp(q, 0, MT) / MT) * bw;
        const bh = sm ? 24 : 28;
        let y = 16;
        V.text(g, 'SAME HOST · 121.69 GiB · other tenants: ' + fmt1(S.x) + ' GiB', pad, y, { size: 11, font: 'mono', color: C.muted, maxW: bw });
        const zc = (hd) => (hd < 0 ? C.red : hd < 2 ? C.d : C.e);
        const lane = (yy, title, col, parts, ok, head, fail, extra) => {
          V.text(g, title, pad, yy, { size: sm ? 11.5 : 12.5, weight: 700, color: col, maxW: bw * (sm ? 1 : 0.62) });
          const st = ok ? (head < 0 ? 'short ' + fmt1(-head) + ' GiB: freezes' : 'headroom ' + fmt1(head) + ' GiB') : fail;
          if (!sm) V.text(g, st, w - pad, yy, { size: 12, weight: 700, color: ok ? zc(head) : C.red, align: 'right' });
          const by0 = yy + (sm ? 22 : 8);
          if (sm) V.text(g, st, pad, yy + 15, { size: 11, weight: 700, color: ok ? zc(head) : C.red });
          stack(g, pad, by0, bw, bh, parts, MT, { size: 11 });
          if (ok && head < 0) {
            const x0 = X(MT + head);
            const pulse = 0.55 + 0.45 * Math.sin(t * 5);
            g.fillStyle = V.alpha(C.red, 0.2 * pulse + 0.1); g.fillRect(x0, by0 - 3, X(MT) - x0, bh + 6); hatch(g, x0, by0 - 3, X(MT) - x0, bh + 6, V.alpha(C.red, 0.9), 5);
          }
          if (extra) extra(by0);
          return by0 + bh;
        };
        // vLLM
        const pv = [{ v: TFM.SO, c: C.muted, a: 0.5, l: 'OS', lc: C.ink }, { v: S.x, c: C.f, l: 'others' }];
        if (v.ok) pv.push({ v: U, c: C.glm, l: sm ? '0.85·M' : 'budget 0.85 × total = ' + fmt1(U) }, { v: Math.max(0, VUSE - U + Math.min(0, v.head)), c: C.glm, l: '', hatch: true }, { v: Math.max(0, v.head), c: zc(v.head), a: 0.35, l: 'headroom', lc: C.ink });
        let yb = lane(y + 20, 'vLLM · GMU 0.85 · KV pinned at 11 GiB', C.glm, pv, v.ok, v.head, 'preflight refuses: MemAvailable ' + fmt1(v.A) + ' < ' + fmt1(PRE), (by0) => {
          if (v.ok) V.text(g, sm ? 'outside the GMU' : 'outside the GMU ≈ ' + fmt1(VUSE - U) + ' (calibrated)', X(TFM.SO + S.x + U), by0 + bh + 13, { size: 11, color: C.glm, weight: 600, align: X(TFM.SO + S.x + U) > w * 0.62 ? 'right' : 'left' });
        });
        // TensorFold
        const pt = [{ v: TFM.SO, c: C.muted, a: 0.5, l: 'OS', lc: C.ink }, { v: S.x, c: C.f, l: 'others' }];
        if (f.ok) pt.push({ v: f.E, c: TF, l: sm ? 'E(L)' : 'estimate ' + fmt1(f.E) }, { v: f.extra, c: TF, a: 0.5, l: 'pool', lc: C.ink }, { v: Math.max(0, TFM.DPK + Math.min(0, f.head)), c: TF, l: 'peak +10', hatch: true }, { v: Math.max(0, f.head), c: zc(f.head), a: 0.35, l: 'headroom', lc: C.ink });
        yb = lane(yb + (sm ? 26 : 30), 'TensorFold · MemAvailable − R · R = ' + fmt1(S.R) + ' GiB', TF, pt, f.ok, f.head, 'refused at startup (nothing loaded)', (by0) => {
          bracket(g, X(MT - S.R), X(MT), by0 + bh + 6, C.ink, 'reserve R', { align: 'right' });
        });
        // ---- legend + charts
        let lx = pad; const ly = yb + (sm ? 24 : 22);
        [[C.glm, 'vLLM (GLM)'], [TF, 'TensorFold']].forEach(([c, l]) => { V.line(g, lx, ly - 4, lx + 18, ly - 4, { color: c, lw: 3 }); V.text(g, l, lx + 24, ly, { size: 11, color: C.ink, weight: 600 }); g.font = V.font(11, 'sans', 600); lx += 24 + g.measureText(l).width + 18; });
        const cy = ly + (sm ? 24 : 26);
        const gw = sm ? bw : (bw - 24) / 2, gh = sm ? (h - cy - 16) / 2 - 10 : h - cy - 6;
        const chart = (x0, y0, cw, ch, title, yMin, yMax, ticks, fy, key, zones, events) => {
          const px0 = x0 + 36, px1 = x0 + cw - 4, py0 = y0 + 12, py1 = y0 + ch - 30;
          const PX = (q) => px0 + (q / XMAX) * (px1 - px0), PY = (q) => py1 - ((V.clamp(q, yMin, yMax) - yMin) / (yMax - yMin)) * (py1 - py0);
          V.text(g, title, x0, y0, { size: 11.5, weight: 700, color: C.ink, maxW: cw });
          (zones || []).forEach(([a, b, c, l, lc]) => { g.fillStyle = c; g.fillRect(px0, PY(b), px1 - px0, PY(a) - PY(b)); if (l) V.text(g, l, px1 - 4, PY(a) - 4, { size: 11, weight: 600, color: lc, align: 'right' }); });
          ticks.forEach((q) => { V.line(g, px0, PY(q), px1, PY(q), { color: V.alpha(C.muted, q === 0 ? 0.5 : 0.12), lw: 1 }); V.text(g, fy(q), px0 - 5, PY(q) + 4, { size: 11, color: C.muted, align: 'right', font: 'mono' }); });
          [0, 6, 12, 18, 24].forEach((q) => V.text(g, String(q), PX(q), py1 + 14, { size: 11, color: C.muted, align: 'center', font: 'mono' }));
          V.text(g, 'other tenants (GiB)', (px0 + px1) / 2, py1 + 27, { size: 11, color: C.muted, align: 'center' });
          (events || []).forEach(([q, l, c], k) => { if (q == null || q > XMAX) return; V.line(g, PX(q), py0, PX(q), py1, { color: V.alpha(c, 0.55), lw: 1, dash: [3, 3] }); V.text(g, l, PX(q) + (PX(q) > px1 - 70 ? -3 : 3), py0 + 10 + k * 13, { size: 11, color: c, align: PX(q) > px1 - 70 ? 'right' : 'left' }); });
          const pts = curves();
          [[1, C.glm], [2, TF]].forEach(([ix, col]) => {
            g.beginPath(); let on = false;
            pts.forEach((p) => { const val = key(p[ix]); if (val == null) { on = false; return; } const xx = PX(p[0]), yy2 = PY(val); if (!on) { g.moveTo(xx, yy2); on = true; } else g.lineTo(xx, yy2); });
            g.strokeStyle = col; g.lineWidth = 2.5; g.lineJoin = 'round'; g.stroke();
          });
          V.line(g, PX(S.x), py0, PX(S.x), py1, { color: V.alpha(C.ink, 0.5), lw: 1.2 });
          [[v, C.glm], [f, TF]].forEach(([r, col]) => { const val = key(r); if (val == null) return; V.glow(g, PX(S.x), PY(val), 9, col, 0.5); V.dot(g, PX(S.x), PY(val), 4, col); });
        };
        const xPre = A0 - PRE, xRefit = A0 - S.R - TFM.E1M, xRef = A0 - S.R - TFM.WFIX;
        const capT = 'KV capacity (millions of tokens)';
        chart(pad, cy, gw, gh, capT, 0, 3.2, [0, 1, 2, 3], (q) => String(q), (r) => r.tok / 1e6, null,
          [[xPre, 'preflight', C.glm], [xRefit, 'window < 1M', TF], [xRef, 'refused', C.red]]);
        const hx = sm ? pad : pad + gw + 24, hy = sm ? cy + gh + 20 : cy;
        chart(hx, hy, gw, gh, 'Lowest host headroom under load (GiB)', -8, 16, [-5, 0, 5, 10, 15], (q) => String(q), (r) => (r.ok ? r.head : null),
          [[-8, 0, V.alpha(C.red, C.dark ? 0.16 : 0.1), 'freezes', C.red], [0, 2, V.alpha(C.d, C.dark ? 0.14 : 0.1), 'tight', C.d]], null);
      });
    }
  },
});
