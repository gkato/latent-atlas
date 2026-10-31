/* Glossário: LA.glossary[chave] = { t: 'Termo', d: 'definição (HTML)', ch: 'id-do-capitulo' } */
window.LA = window.LA || {};
LA.glossary = Object.assign(LA.glossary || {}, {
  'pesos': { t: 'Pesos (parâmetros)', d: 'Os números aprendidos no treino que definem o comportamento do modelo. Um LLM de fronteira tem centenas de bilhões deles.', ch: 'mapa-da-jornada' },
  'arquitetura': { t: 'Arquitetura', d: 'O desenho matemático do modelo: quais camadas existem, como os tensores fluem entre elas e quais operações são aplicadas.', ch: 'mapa-da-jornada' },
  'vllm': { t: 'vLLM', d: 'Motor de inferência open-source para LLMs, com PagedAttention, batching contínuo e API compatível com OpenAI.', ch: 'mapa-da-jornada' },
  'kv-cache': { t: 'KV cache', d: 'Memória onde o modelo guarda as chaves (K) e valores (V) da atenção de todos os tokens já processados, para não recalculá-los a cada novo token.', ch: 'mapa-da-jornada' },
  'decode': { t: 'Decode', d: 'Fase da geração em que o modelo produz um token por vez, reaproveitando o KV cache. É limitada pela banda de memória.', ch: 'mapa-da-jornada' },
});
