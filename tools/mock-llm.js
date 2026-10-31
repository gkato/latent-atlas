#!/usr/bin/env node
/* Servidor falso no contrato OpenAI (para testar o copilot sem modelo): node tools/mock-llm.js [porta] */
'use strict';
const http = require('http');
const PORT = +process.argv[2] || 18080;
const cors = { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': '*', 'Access-Control-Allow-Headers': '*' };
http.createServer((req, res) => {
  if (req.method === 'OPTIONS') { res.writeHead(204, cors); return res.end(); }
  if (req.url.endsWith('/models')) { res.writeHead(200, { ...cors, 'Content-Type': 'application/json' }); return res.end(JSON.stringify({ object: 'list', data: [{ id: 'mock-tutor' }, { id: 'mock-2' }] })); }
  let body = '';
  req.on('data', (c) => (body += c));
  req.on('end', () => {
    let j = {}; try { j = JSON.parse(body); } catch (e) {}
    const sys = (j.messages || [])[0]?.content || '';
    const user = (j.messages || []).slice(-1)[0]?.content || '';
    const page = (sys.match(/## Página atual\n([^\n]+)/) || [])[1] || '?';
    const answer = `**Contexto recebido:** ${page}\n\nVocê perguntou: _${user.slice(-80)}_\n\n- item 1 com \`código\`\n- item 2\n\nFórmula: \\( a^2 + b^2 = c^2 \\)\n\n\`\`\`bash\nvllm serve modelo --tensor-parallel-size 2\n\`\`\`\n\n| a | b |\n|---|---|\n| 1 | 2 |`;
    if (!j.stream) { res.writeHead(200, { ...cors, 'Content-Type': 'application/json' }); return res.end(JSON.stringify({ model: 'mock-tutor', choices: [{ message: { role: 'assistant', content: answer } }] })); }
    res.writeHead(200, { ...cors, 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' });
    const chunks = ['Pensando no contexto… ', 'ok.'].map((r) => ({ choices: [{ delta: { reasoning: r } }] }))
      .concat(answer.match(/[\s\S]{1,12}/g).map((t) => ({ choices: [{ delta: { content: t } }] })));
    let i = 0;
    const tick = () => {
      if (i < chunks.length) { res.write('data: ' + JSON.stringify(chunks[i++]) + '\n\n'); setTimeout(tick, 15); }
      else { res.write('data: [DONE]\n\n'); res.end(); }
    };
    tick();
  });
}).listen(PORT, '127.0.0.1', () => console.log('mock LLM em http://127.0.0.1:' + PORT + '/v1'));
