/* Glossary (English): LA.glossary[key] = { t: 'Term', d: 'definition (HTML)', ch: 'chapter-id' } */
window.LA = window.LA || {};
LA.glossary = Object.assign(LA.glossary || {}, {
  'pesos': { t: 'Weights (parameters)', d: 'The numbers learned during training that define the model’s behavior. A frontier LLM has hundreds of billions of them.', ch: 'mapa-da-jornada' },
  'arquitetura': { t: 'Architecture', d: 'The mathematical design of the model: which layers exist, how tensors flow between them and which operations are applied.', ch: 'mapa-da-jornada' },
  'vllm': { t: 'vLLM', d: 'Open-source inference engine for LLMs, with PagedAttention, continuous batching and an OpenAI-compatible API.', ch: 'mapa-da-jornada' },
  'kv-cache': { t: 'KV cache', d: 'Memory where the model keeps the attention keys (K) and values (V) of every token already processed, so they are not recomputed for each new token.', ch: 'mapa-da-jornada' },
  'decode': { t: 'Decode', d: 'The generation phase in which the model produces one token at a time, reusing the KV cache. It is limited by memory bandwidth.', ch: 'mapa-da-jornada' },
});
