#!/usr/bin/env python3
"""Gera docs/APANHADO.md (pt) e docs/APANHADO.en.md (en) a partir de data/catalog[.en].js.
Uso: python3 tools/gen-apanhado.py
"""
import collections, glob, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

LABELS = {
    'pt': {'launch': 'Lançamento / topologia', 'vllm-engine': 'Motor de inferência', 'scheduling': 'Scheduler / batching', 'kv-cache': 'KV cache', 'prefix-cache': 'Prefix caching', 'spec-decode': 'Decodificação especulativa', 'quantization': 'Quantização', 'model-arch': 'Arquitetura do modelo', 'attention': 'Atenção / kernels de atenção', 'moe': 'MoE', 'distributed': 'Distribuído / rede', 'memory': 'Memória / carga de pesos', 'kernels': 'Kernels / CUDA', 'serving-api': 'API / serviço', 'sampling': 'Amostragem / geração', 'reasoning-tools': 'Raciocínio / tools', 'multimodal': 'Multimodal', 'alignment': 'Abliteração', 'ops': 'Operação / Docker', 'bench': 'Benchmark / qualidade', 'other': 'Outros'},
    'en': {'launch': 'Launch / topology', 'vllm-engine': 'Inference engine', 'scheduling': 'Scheduler / batching', 'kv-cache': 'KV cache', 'prefix-cache': 'Prefix caching', 'spec-decode': 'Speculative decoding', 'quantization': 'Quantization', 'model-arch': 'Model architecture', 'attention': 'Attention / kernels', 'moe': 'MoE', 'distributed': 'Distributed / network', 'memory': 'Memory / weight loading', 'kernels': 'Kernels / CUDA', 'serving-api': 'API / serving', 'sampling': 'Sampling / generation', 'reasoning-tools': 'Reasoning / tools', 'multimodal': 'Multimodal', 'alignment': 'Abliteration', 'ops': 'Operations / Docker', 'bench': 'Benchmark / quality', 'other': 'Other'},
}
RNAME = {'glm': 'GLM-5.3-Flash EXL3 (vLLM)', 'tf': 'GLM-5.3-Flash EXL3 (TensorFold)', 'qwen': 'Qwen3.8-Flash-Next NVFP4', 'ds': 'DeepSeek-V4-Flash DSpark'}
T = {
    'pt': dict(title='# Apanhado — tudo o que as receitas modificam e parametrizam', recipe='Receita', params='Parâmetros', mods='Modificações',
               mtitle='Modificações (patches, overlays, kernels, hotfixes, transformações de checkpoint)', ptitle='Parâmetros (env vars, flags, configs, perfis)',
               cols='| Nome | Valor | O que faz | Por que importa | Estude em |',
               intro='Gerado a partir do catálogo da plataforma ({n} itens). Cada item aponta o capítulo do **Latent Atlas** que explica o conceito por trás dele. A versão navegável (busca e filtros) está em `index.html#/catalogo`; os inventários completos, com arquivo:linha e números medidos, estão em `docs/pesquisa/`.'),
    'en': dict(title='# Inventory — everything the recipes modify and parameterize', recipe='Recipe', params='Parameters', mods='Modifications',
               mtitle='Modifications (patches, overlays, kernels, hotfixes, checkpoint transforms)', ptitle='Parameters (env vars, flags, configs, profiles)',
               cols='| Name | Value | What it does | Why it matters | Study in |',
               intro='Generated from the platform catalog ({n} items). Each item points to the **Latent Atlas** chapter that explains the concept behind it. The browsable version (search and filters) is at `index.html?lang=en#/catalogo`; the full research inventories, with file:line references and measured numbers, are in `docs/pesquisa/` (Portuguese).'),
}


def nav_titles(lang):
    out = {}
    pat = '*.en.js' if lang == 'en' else '*.js'
    for f in glob.glob('chapters/' + pat):
        if lang == 'pt' and f.endswith('.en.js'):
            continue
        s = open(f, encoding='utf-8').read()
        m = re.search(r"navTitle:\s*(['\"`])(.*?)\1", s)
        i = re.search(r"id:\s*['\"]([^'\"]+)['\"]", s)
        n = re.match(r'chapters/(\d+)-', f)
        if m and i and n:
            out[i.group(1)] = (int(n.group(1)), m.group(2))
    return out


def strip(s):
    return re.sub(r'<[^>]+>', '', s or '').replace('|', '\\|').replace('\n', ' ').strip()


for lang, src, dst in (('pt', 'data/catalog.js', 'docs/APANHADO.md'), ('en', 'data/catalog.en.js', 'docs/APANHADO.en.md')):
    s = open(src, encoding='utf-8').read()
    items = json.loads(s[s.index('LA.catalog = ') + len('LA.catalog = '):s.rindex(';')])
    nav, L, t, lab = nav_titles(lang), [], T[lang], LABELS[lang]
    L += [t['title'], '', t['intro'].format(n=len(items)), '']
    cnt = collections.Counter((r, x['kind']) for x in items for r in x.get('recipes', []))
    order = [r for r in ('glm', 'tf', 'qwen', 'ds') if cnt[(r, 'param')] or cnt[(r, 'mod')]]
    L += [f"| {t['recipe']} | {t['params']} | {t['mods']} |", '|---|---:|---:|']
    L += [f"| {RNAME[r]} | {cnt[(r, 'param')]} | {cnt[(r, 'mod')]} |" for r in order] + ['']
    for r in order:
        L += [f'## {RNAME[r]}', '']
        for kind, title in (('mod', t['mtitle']), ('param', t['ptitle'])):
            L += [f'### {title}', '']
            by = collections.defaultdict(list)
            for x in items:
                if r in x.get('recipes', []) and x['kind'] == kind:
                    by[x.get('cat') or 'other'].append(x)
            for cat in sorted(by, key=lambda c: lab.get(c, c)):
                L += [f'#### {lab.get(cat, cat)}', '', t['cols'], '|---|---|---|---|---|']
                for x in sorted(by[cat], key=lambda x: x['name'].lower()):
                    ch = ', '.join(f'{nav[c][0]:02d} {nav[c][1]}' for c in (x.get('ch') or [])[:2] if c in nav)
                    L.append(f"| `{strip(x['name'])}` | {strip(x.get('value'))[:180]} | {strip(x.get('what'))[:280]} | {strip(x.get('why'))[:340]} | {ch} |")
                L.append('')
    open(dst, 'w', encoding='utf-8').write('\n'.join(L))
    print(f'{dst}: {len(items)} itens, {len(L)} linhas')
