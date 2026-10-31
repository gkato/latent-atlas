#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Latent Atlas — servidor local da plataforma + helper do copilot MLX.

Serve o site estático (substitui o `python3 -m http.server` do serve.sh) e expõe uma API JSON local em /api/
que baixa modelos do Hugging Face para o cache padrão do HF, instala o mlx-lm e sobe/derruba o mlx_lm.server.
Só stdlib, Python >= 3.9 (o python3 que vem no macOS).

O mlx_lm.server não checa Origin/Host nem o corpo dos pedidos (qualquer site poderia usá-lo às cegas ou trocar o
modelo pelo campo `model`). Por isso ele nunca fica exposto direto: roda numa porta interna aleatória, só aceita
pedidos com um segredo por execução (header X-Atlas-Key, aplicado por um bootstrap) e morre junto com o helper
(watchdog no stdin). A porta pública (p.ex. :8080, a que o copilot usa) é um proxy deste processo que aplica
Host/Origin/Content-Type, fixa o modelo gerenciado e limita max_tokens a 32768.

Uso:
  python3 tools/atlas_server.py [--port 8765] [--host 127.0.0.1] [--root <dir>] [--open] [--mlx-port 8080] [--quiet]

Variáveis de ambiente (testes / uso avançado):
  HF_ENDPOINT     hub alternativo (padrão https://huggingface.co)
  HF_HUB_CACHE    cache do HF (padrão ~/.cache/huggingface/hub; também HUGGINGFACE_HUB_CACHE e HF_HOME)
  HF_TOKEN        token para modelos gated — só é enviado ao host do HF_ENDPOINT
  LA_MLX_SERVER   caminho explícito do mlx_lm.server
  UV_TOOL_DIR / UV_TOOL_BIN_DIR  repassados ao `uv tool install` (UV_TOOL_BIN_DIR também entra na busca)
"""

import argparse
import atexit
import collections
import errno
import fnmatch
import functools
import hashlib
import http.client
import http.server
import json
import math
import os
import platform
import re
import secrets
import shutil
import signal
import socket
import stat
import string
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request

APP = 'latent-atlas'
VERSION = '1.0'
HELPER_API = 1
USER_AGENT = 'latent-atlas-helper/1.0'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# exatamente os padrões que o mlx-lm baixa (mlx_lm.utils.DEFAULT_ALLOW_PATTERNS; fnmatch, `*` também casa `/`)
ALLOW_PATTERNS = ['*.json', 'model*.safetensors', '*.py', 'tokenizer.model', '*.tiktoken', 'tiktoken.model',
                  '*.txt', '*.jsonl', '*.jinja']
REPO_ID_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}')
COMMIT_RE = re.compile(r'[0-9A-Za-z]{6,64}')
ETAG_RE = re.compile(r'[0-9a-f]{40}|[0-9a-f]{64}')
LOOPBACK_HOSTS = ('127.0.0.1', 'localhost', '::1')

KIB = 1024
GIB = 1 << 30
CHUNK = 1 << 20              # 1 MiB por leitura
MAX_BODY = 64 * KIB          # corpo máximo de um POST na API
LOG_LINES = 600              # tamanho de cada ring buffer de log
LOG_BYTES = 1 << 20          # teto de caracteres guardados por ring buffer
LOG_LINE_MAX = 2000          # caracteres por linha guardada (o resto vira '…')
LOG_READ_MAX = 4096          # bytes lidos por linha de um pipe (o resto da linha é descartado)
DOWNLOAD_WORKERS = 3         # arquivos em paralelo (maiores primeiro)
READ_TIMEOUT = 30            # timeout por leitura de socket no download
FILE_ATTEMPTS = 3            # tentativas por arquivo (com resume)
LOCAL_SCAN_TTL = 5.0         # reescaneia o cache no máximo a cada 5 s
HEALTH_TIMEOUT = 90.0        # /health do mlx_lm.server
HEALTH_503_LIMIT = 3         # /health 503 seguidos = a thread de geração morreu (falha ao carregar)
WARMUP_TIMEOUT = 900.0       # 1ª geração (é aqui que os pesos carregam)
STOP_GRACE = 6.0             # SIGTERM → espera → SIGKILL
TAIL_LINES = 15              # linhas de log do servidor no status (starting/error)
MAX_GEN_TOKENS = 32768        # teto de max_tokens por pedido (o mesmo da validação do start)
HTTP_TIMEOUT = 30            # timeout de socket por conexão (cliente ocioso ou lento)
MAX_CONNECTIONS = 128        # conexões atendidas ao mesmo tempo por servidor HTTP (as extras são fechadas)
PROXY_MAX_BODY = 8 << 20     # corpo máximo de um pedido de chat via proxy
UPSTREAM_TIMEOUT = WARMUP_TIMEOUT  # espera por bytes do mlx_lm.server (carga + prompt longo)
GATE_HEADER = 'X-Atlas-Key'  # segredo por execução exigido pelo mlx_lm.server interno
COMPLETION_PATHS = ('/v1/chat/completions', '/chat/completions', '/v1/completions')
O_NOFOLLOW = getattr(os, 'O_NOFOLLOW', 0)

_INSTALL_HINT = ['curl -LsSf https://astral.sh/uv/install.sh | sh', 'brew install mlx-lm']
# linhas de acesso do próprio /health (o helper consulta a cada 0,5 s) não entram no log
_HEALTH_LOG_RE = re.compile(r'"(?:GET|HEAD) /health HTTP/1\.[01]" \d{3}')
_FATAL_LOG_RE = re.compile(r'generation thread died:?\s*(.*)$')
_ERROR_LOG_RE = re.compile(r'(?:\b[A-Za-z_]*(?:Error|Exception)\b:\s*\S|\s-\sERROR\s-\s)')


# ----------------------------------------------------------------------------------------------------------------
# erros
# ----------------------------------------------------------------------------------------------------------------

class ApiError(Exception):
    """Erro devolvido pela API como {"error": code, "message": ..., ...extra} com status HTTP."""

    def __init__(self, status, code, message, **extra):
        Exception.__init__(self, message)
        self.status = status
        self.code = code
        self.message = message
        self.extra = extra

    def payload(self):
        out = {'error': self.code, 'message': self.message}
        out.update(self.extra)
        return out


class DownloadFailed(Exception):
    """Falha definitiva de um download (vira download.state = error)."""

    def __init__(self, code, message):
        Exception.__init__(self, message)
        self.code = code
        self.message = message


class _Retryable(Exception):
    """Falha transitória de um arquivo (rede/timeout/corpo curto) — tenta de novo com resume."""


class _Canceled(Exception):
    """Download cancelado (ou abortado porque outro arquivo falhou)."""


# ----------------------------------------------------------------------------------------------------------------
# validação e utilitários
# ----------------------------------------------------------------------------------------------------------------

def valid_repo_id(repo_id):
    """org/nome no formato do Hugging Face — sem `--` (separador do cache) e sem `..`."""
    return (isinstance(repo_id, str) and REPO_ID_RE.fullmatch(repo_id) is not None
            and '--' not in repo_id and '..' not in repo_id)


def allowed_file(name):
    return any(fnmatch.fnmatchcase(name, pat) for pat in ALLOW_PATTERNS)


def safe_rel_path(name):
    """Nome de arquivo vindo da API do HF → True só se for um caminho relativo inofensivo."""
    if not isinstance(name, str) or not name or len(name) > 1024:
        return False
    if '\x00' in name or '\\' in name or name.startswith('/'):
        return False
    return all(part not in ('', '.', '..') for part in name.split('/'))


def repo_folder_name(repo_id):
    org, name = repo_id.split('/', 1)
    return 'models--%s--%s' % (org, name)


def is_loopback_host(host):
    return host in LOOPBACK_HOSTS


def allowed_hosts(port):
    return ('127.0.0.1:%d' % port, 'localhost:%d' % port, '[::1]:%d' % port)


def allowed_origins(port):
    return tuple('http://' + h for h in allowed_hosts(port))


def default_cache_dir(env):
    for key in ('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE'):
        if env.get(key):
            return os.path.abspath(os.path.expanduser(env[key]))
    if env.get('HF_HOME'):
        return os.path.join(os.path.abspath(os.path.expanduser(env['HF_HOME'])), 'hub')
    return os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')


def default_token(env):
    for key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'):
        value = (env.get(key) or '').strip()
        if value:
            return value
    home = os.path.expanduser(env.get('HF_HOME') or os.path.join('~', '.cache', 'huggingface'))
    try:
        with open(os.path.join(home, 'token'), encoding='utf-8') as fh:
            return fh.read().strip() or None
    except (OSError, UnicodeDecodeError):
        return None


def default_endpoint(env):
    return (env.get('HF_ENDPOINT') or 'https://huggingface.co').rstrip('/')


def disk_free(path):
    """Espaço livre no volume de `path` (sobe até o primeiro ancestral que existe)."""
    p = os.path.abspath(path)
    while not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return None


def port_listening(port, host='127.0.0.1', timeout=0.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def human_bytes(n):
    if n is None:
        return '?'
    if n >= 1e9:
        return '%.1f GB' % (n / 1e9)
    if n >= 1e6:
        return '%.1f MB' % (n / 1e6)
    if n >= 1e3:
        return '%.1f kB' % (n / 1e3)
    return '%d B' % n


def clip_line(text, limit=LOG_LINE_MAX):
    return text if len(text) <= limit else text[:limit] + '…'


class LogRing(object):
    """Ring buffer de linhas com teto de linhas E de caracteres; linhas longas são truncadas.
    (O mlx_lm.server registra o corpo cru de pedidos inválidos: sem teto, qualquer pedido enorme viraria memória.)"""

    def __init__(self, max_lines=LOG_LINES, max_chars=LOG_BYTES):
        self.max_lines = max_lines
        self.max_chars = max_chars
        self._lines = collections.deque()
        self._chars = 0

    def append(self, line):
        line = clip_line(line)
        self._lines.append(line)
        self._chars += len(line)
        while self._lines and (len(self._lines) > self.max_lines or self._chars > self.max_chars):
            self._chars -= len(self._lines.popleft())

    def clear(self):
        self._lines.clear()
        self._chars = 0

    def __iter__(self):
        return iter(list(self._lines))

    def __len__(self):
        return len(self._lines)


def bounded_lines(stream, limit=LOG_READ_MAX):
    """Linhas (bytes) de um pipe sem nunca ler mais que `limit` bytes por linha: o resto de uma linha longa é
    descartado e o pedaço guardado ganha um '…'."""
    skipping = False
    while True:
        raw = stream.readline(limit)
        if not raw:
            return
        complete = raw.endswith(b'\n')
        if skipping:
            skipping = not complete
            continue
        if not complete and len(raw) >= limit:
            skipping = True
            yield raw + '…'.encode('utf-8')
        else:
            yield raw


def free_local_port():
    """Porta livre em 127.0.0.1 (para o mlx_lm.server interno)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]
    finally:
        s.close()


def regular_size(path):
    """Tamanho se `path` for um arquivo comum (sem seguir symlink); None se não existir ou não for arquivo comum."""
    try:
        st = os.lstat(path)
    except OSError:
        return None
    return st.st_size if stat.S_ISREG(st.st_mode) else None


def _run_text(args, timeout=5):
    try:
        out = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             timeout=timeout, check=False)
        return out.stdout.decode('utf-8', 'replace').strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _sysctl(key):
    exe = '/usr/sbin/sysctl' if os.path.exists('/usr/sbin/sysctl') else 'sysctl'
    return _run_text([exe, '-n', key])


def detect_mac():
    """Plataforma: só Macs Apple Silicon rodam MLX."""
    system = platform.system()
    info = {'os': system, 'arm64': False, 'chip': None, 'macos': None, 'ram_bytes': None, 'disk_free_bytes': None}
    if system == 'Darwin':
        # hw.optional.arm64 também acerta quando o python roda sob Rosetta
        info['arm64'] = platform.machine() == 'arm64' or _sysctl('hw.optional.arm64') == '1'
        info['chip'] = _sysctl('machdep.cpu.brand_string')
        try:
            info['ram_bytes'] = int(_sysctl('hw.memsize') or 0) or None
        except ValueError:
            info['ram_bytes'] = None
        info['macos'] = platform.mac_ver()[0] or _run_text(['sw_vers', '-productVersion'])
    else:
        try:
            info['ram_bytes'] = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        except (ValueError, OSError, AttributeError):
            info['ram_bytes'] = None
    return info


# ----------------------------------------------------------------------------------------------------------------
# HTTP de saída (Hugging Face e o mlx_lm.server local)
# ----------------------------------------------------------------------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Não segue redirects sozinho: o laço manual decide headers (Range sim, Authorization só no mesmo host)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_HUB_OPENER = urllib.request.build_opener(_NoRedirect)
# chamadas para 127.0.0.1 nunca passam por proxy de ambiente
_LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)


def _origin_of(url):
    parts = urllib.parse.urlsplit(url)
    scheme = parts.scheme.lower()
    try:
        port = parts.port
    except ValueError:
        port = None
    return scheme, (parts.hostname or '').lower(), port or (443 if scheme == 'https' else 80)


class HubClient(object):
    """Cliente mínimo do Hugging Face (ou de um HF_ENDPOINT falso nos testes)."""

    def __init__(self, endpoint, token=None):
        self.endpoint = endpoint.rstrip('/')
        self.token = token or None
        self._origin = _origin_of(self.endpoint)

    def open(self, url, headers=None, timeout=READ_TIMEOUT, max_redirects=10):
        current = url
        for _ in range(max_redirects + 1):
            req = urllib.request.Request(current, headers=dict(headers or {}))
            req.add_header('User-Agent', USER_AGENT)
            if self.token and _origin_of(current) == self._origin:
                # unredirected: mesmo que algum handler siga um redirect, o token não vai junto
                req.add_unredirected_header('Authorization', 'Bearer ' + self.token)
            try:
                return _HUB_OPENER.open(req, timeout=timeout)
            except urllib.error.HTTPError as e:
                if e.code not in (301, 302, 303, 307, 308):
                    raise
                location = e.headers.get('Location') if e.headers else None
                try:
                    e.close()
                except Exception:
                    pass
                if not location:
                    raise
                # mesmo cuidado do urllib: só escapa espaços/não-ASCII, mantém %xx intactos
                nxt = urllib.parse.quote(urllib.parse.urljoin(current, location), safe=string.punctuation)
                if urllib.parse.urlsplit(nxt).scheme not in ('http', 'https'):
                    raise urllib.error.URLError('refusing redirect to %r' % nxt)
                current = nxt
        raise urllib.error.URLError('too many redirects')

    def get_json(self, url, timeout=20, limit=32 * 1024 * 1024):
        with self.open(url, timeout=timeout) as resp:
            raw = resp.read(limit + 1)
        if len(raw) > limit:
            raise ValueError('response too large')
        return json.loads(raw.decode('utf-8'))

    def model_api(self, repo_id):
        return self.get_json('%s/api/models/%s/revision/main?blobs=true' % (self.endpoint, repo_id))

    def file_url(self, repo_id, commit, name):
        return '%s/%s/resolve/%s/%s' % (self.endpoint, repo_id, commit, urllib.parse.quote(name, safe='/'))


def hub_error(exc, repo_id):
    """Converte erros do hub em ApiError (404 not_found | 502 hf_unreachable)."""
    if isinstance(exc, ApiError):
        return exc
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in (401, 403, 404):
            return ApiError(404, 'not_found',
                            'Model %s not found on Hugging Face (it may be private or gated: those need an HF token '
                            'with access).' % repo_id, hf_status=exc.code)
        return ApiError(502, 'hf_unreachable', 'Hugging Face answered HTTP %d.' % exc.code, hf_status=exc.code)
    if isinstance(exc, (ValueError, UnicodeDecodeError)):
        return ApiError(502, 'hf_unreachable', 'Hugging Face sent an invalid answer: %s' % exc)
    reason = getattr(exc, 'reason', None) or exc
    return ApiError(502, 'hf_unreachable', 'Could not reach Hugging Face: %s' % reason)


def local_request(url, body=None, timeout=2.0, headers=None):
    """GET/POST JSON em 127.0.0.1 → (status, dict|None). HTTPError 4xx/5xx sobe para quem chamou."""
    data = None
    headers = dict(headers or {})
    headers['User-Agent'] = USER_AGENT
    if body is not None:
        data = json.dumps(body).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(url, data=data, headers=headers)
    with _LOCAL_OPENER.open(req, timeout=timeout) as resp:
        raw = resp.read(8 * 1024 * 1024)
        status = resp.status
    try:
        return status, json.loads(raw.decode('utf-8'))
    except (ValueError, UnicodeDecodeError):
        return status, None


# ----------------------------------------------------------------------------------------------------------------
# metadados do hub
# ----------------------------------------------------------------------------------------------------------------

def parse_siblings(api, strict):
    """Arquivos do repo filtrados pelos padrões do mlx-lm → [{name, size, etag, lfs, sha256}].
    strict=True (download): nomes perigosos ou metadados faltando derrubam tudo; strict=False (info): ignora."""
    out = []
    for sib in api.get('siblings') or []:
        if not isinstance(sib, dict):
            continue
        name = sib.get('rfilename')
        if not isinstance(name, str) or not allowed_file(name):
            continue
        if not safe_rel_path(name):
            if strict:
                raise ApiError(502, 'unsafe_path', 'Refusing unsafe file name from the hub: %r' % name)
            continue
        lfs = sib.get('lfs') if isinstance(sib.get('lfs'), dict) else None
        size = lfs.get('size') if lfs and lfs.get('size') is not None else sib.get('size')
        etag = lfs.get('sha256') if lfs else sib.get('blobId')
        ok_size = isinstance(size, int) and not isinstance(size, bool) and size >= 0
        ok_etag = isinstance(etag, str) and ETAG_RE.fullmatch(etag) is not None
        if lfs and ok_etag and len(etag) != 64:
            ok_etag = False
        if not (ok_size and ok_etag):
            if strict:
                raise ApiError(502, 'bad_metadata', 'The hub did not return size/hash for %s.' % name)
            out.append({'name': name, 'size': size if ok_size else 0, 'etag': None, 'lfs': bool(lfs),
                        'sha256': None})
            continue
        out.append({'name': name, 'size': size, 'etag': etag, 'lfs': bool(lfs),
                    'sha256': etag if lfs else None})
    return out


_MOE_KEYS = ('num_experts', 'num_local_experts', 'n_routed_experts', 'moe_intermediate_size')


def config_facts(cfg):
    """quant_bits e moe a partir do config (topo ou text_config)."""
    cfg = cfg if isinstance(cfg, dict) else {}
    bits = None
    for key in ('quantization', 'quantization_config'):
        q = cfg.get(key)
        if isinstance(q, dict) and isinstance(q.get('bits'), (int, float)) and not isinstance(q.get('bits'), bool):
            bits = q['bits']
            break
    moe = False
    for scope in (cfg, cfg.get('text_config')):
        if isinstance(scope, dict) and any(scope.get(k) for k in _MOE_KEYS):
            moe = True
    return bits, moe


# ----------------------------------------------------------------------------------------------------------------
# cache local do HF
# ----------------------------------------------------------------------------------------------------------------

def snapshot_complete(snap):
    """Snapshot utilizável pelo mlx-lm: config, pesos, tokenizer e todos os symlinks resolvendo."""
    if not os.path.isdir(snap):
        return False
    for dirpath, dirnames, filenames in os.walk(snap):
        for n in filenames + dirnames:
            p = os.path.join(dirpath, n)
            if os.path.islink(p) and not os.path.exists(p):
                return False
    if not os.path.isfile(os.path.join(snap, 'config.json')):
        return False
    index = os.path.join(snap, 'model.safetensors.index.json')
    if os.path.isfile(index):
        try:
            with open(index, encoding='utf-8') as fh:
                weight_map = (json.load(fh) or {}).get('weight_map') or {}
        except (OSError, ValueError, AttributeError):
            return False
        shards = set(v for v in weight_map.values() if isinstance(v, str)) if isinstance(weight_map, dict) else set()
        if not shards:
            return False
        for shard in shards:
            if not safe_rel_path(shard) or not os.path.isfile(os.path.join(snap, shard)):
                return False
    else:
        try:
            names = os.listdir(snap)
        except OSError:
            return False
        if not any(fnmatch.fnmatchcase(n, 'model*.safetensors') and os.path.isfile(os.path.join(snap, n))
                   for n in names):
            return False
    try:
        names = os.listdir(snap)
    except OSError:
        return False
    tokenizer = any(os.path.isfile(os.path.join(snap, n)) for n in ('tokenizer.json', 'tokenizer.model',
                                                                     'tiktoken.model'))
    tokenizer = tokenizer or any(n.endswith('.tiktoken') and os.path.isfile(os.path.join(snap, n)) for n in names)
    return tokenizer


def scan_repo(repo_dir):
    """Um models--org--nome → {bytes, complete, commit, mtime}."""
    total = 0
    mtime = 0.0
    incomplete = False
    try:
        for entry in os.scandir(os.path.join(repo_dir, 'blobs')):
            if entry.is_file(follow_symlinks=False):
                st = entry.stat(follow_symlinks=False)
                total += st.st_size
                mtime = max(mtime, st.st_mtime)
                if entry.name.endswith('.incomplete'):
                    incomplete = True
    except OSError:
        pass
    commit = None
    ref = os.path.join(repo_dir, 'refs', 'main')
    try:
        with open(ref, encoding='utf-8') as fh:
            commit = fh.read().strip() or None
        mtime = max(mtime, os.path.getmtime(ref))
    except (OSError, UnicodeDecodeError):
        commit = None
    complete = False
    if commit and COMMIT_RE.fullmatch(commit) and not incomplete:
        complete = snapshot_complete(os.path.join(repo_dir, 'snapshots', commit))
    return {'bytes': total, 'complete': complete, 'commit': commit if commit and COMMIT_RE.fullmatch(commit) else None,
            'mtime': int(mtime)}


def scan_cache(cache_dir):
    out = []
    try:
        entries = sorted(os.scandir(cache_dir), key=lambda e: e.name)
    except OSError:
        return out
    for entry in entries:
        if not entry.name.startswith('models--') or not entry.is_dir(follow_symlinks=False):
            continue
        rest = entry.name[len('models--'):]
        if '--' not in rest:
            continue
        org, name = rest.split('--', 1)
        repo_id = org + '/' + name
        if not valid_repo_id(repo_id):
            continue
        info = scan_repo(entry.path)
        if info['bytes'] <= 0 and not info['commit']:
            continue
        out.append({'id': repo_id, 'bytes': info['bytes'], 'complete': info['complete'], 'commit': info['commit'],
                    'mtime': info['mtime']})
    return out


# ----------------------------------------------------------------------------------------------------------------
# detecção do mlx-lm
# ----------------------------------------------------------------------------------------------------------------

_DETECT_SCRIPT = r'''
import ast, json, os, pkgutil
out = {"version": None, "mlx_version": None, "model_types": [], "remap": {}, "error": None}
try:
    import importlib.metadata as md
    import importlib.util
    for key, dist in (("version", "mlx-lm"), ("mlx_version", "mlx")):
        try:
            out[key] = md.version(dist)
        except Exception:
            pass
    spec = importlib.util.find_spec("mlx_lm")
    if spec is None or not spec.origin:
        out["error"] = "mlx_lm is not importable"
    else:
        base = os.path.dirname(spec.origin)
        out["model_types"] = sorted(set(m.name for m in pkgutil.iter_modules([os.path.join(base, "models")])
                                        if not m.name.startswith("_")))
        try:
            with open(os.path.join(base, "utils.py"), encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
            for node in tree.body:
                targets = node.targets if isinstance(node, ast.Assign) else (
                    [node.target] if isinstance(node, ast.AnnAssign) else [])
                if any(isinstance(t, ast.Name) and t.id == "MODEL_REMAPPING" for t in targets) and node.value:
                    out["remap"] = {str(k): str(v) for k, v in ast.literal_eval(node.value).items()}
        except Exception as e:
            out["error"] = "remap: %s" % e
except Exception as e:
    out["error"] = str(e)
print(json.dumps(out))
'''


def search_path(env):
    parts = [p for p in (env.get('PATH') or '').split(os.pathsep) if p]
    for extra in (env.get('UV_TOOL_BIN_DIR'), os.path.join(os.path.expanduser('~'), '.local', 'bin'),
                  '/opt/homebrew/bin', '/usr/local/bin'):
        if extra and extra not in parts:
            parts.append(extra)
    return os.pathsep.join(parts)


def find_mlx_server(env):
    explicit = env.get('LA_MLX_SERVER')
    if explicit:
        p = os.path.abspath(os.path.expanduser(explicit))
        return p if os.path.isfile(p) and os.access(p, os.X_OK) else None
    return shutil.which('mlx_lm.server', path=search_path(env))


def find_installers(env):
    sp = search_path(env)
    return {'uv': shutil.which('uv', path=sp), 'brew': shutil.which('brew', path=sp)}


def find_python_for(bin_path):
    """Python do entry point: trampolim do `uv tool` (#!/bin/sh + '''exec' '/py' …) ou shebang direto."""
    real = os.path.realpath(bin_path)
    try:
        with open(real, 'rb') as fh:
            head = fh.read(4096)
    except OSError:
        return None
    lines = head.split(b'\n')
    if lines and lines[0].startswith(b'#!'):
        interp = lines[0][2:].strip().split()
        if interp:
            exe = interp[0].decode('utf-8', 'replace')
            base = os.path.basename(exe)
            if base in ('sh', 'bash', 'zsh', 'dash') and len(lines) > 1:
                m = re.match(rb"^'''exec' '([^']+)'", lines[1])
                if m:
                    cand = m.group(1).decode('utf-8', 'replace')
                    if os.path.isfile(cand) and os.access(cand, os.X_OK):
                        return cand
            elif base == 'env':
                args = [a.decode('utf-8', 'replace') for a in interp[1:] if not a.startswith(b'-')]
                if args:
                    found = shutil.which(args[0])
                    if found:
                        return found
            elif 'python' in base and os.path.isfile(exe) and os.access(exe, os.X_OK):
                return exe
    for name in ('python3', 'python'):
        cand = os.path.join(os.path.dirname(real), name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def probe_mlx(env):
    """→ {installed, server_bin, python, version, mlx_version, model_types, remap}."""
    out = {'installed': False, 'server_bin': None, 'python': None, 'version': None, 'mlx_version': None,
           'model_types': [], 'remap': {}}
    server_bin = find_mlx_server(env)
    if not server_bin:
        return out
    out['installed'] = True
    out['server_bin'] = server_bin
    py = find_python_for(server_bin)
    out['python'] = py
    if not py:
        return out
    try:
        res = subprocess.run([py, '-c', _DETECT_SCRIPT], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, timeout=60, check=False,
                             cwd=os.path.dirname(os.path.realpath(server_bin)), env=dict(env))
        lines = [ln for ln in res.stdout.decode('utf-8', 'replace').splitlines() if ln.strip()]
        data = json.loads(lines[-1]) if lines else {}
    except (OSError, subprocess.SubprocessError, ValueError):
        data = {}
    if isinstance(data, dict):
        out['version'] = data.get('version') if isinstance(data.get('version'), str) else None
        out['mlx_version'] = data.get('mlx_version') if isinstance(data.get('mlx_version'), str) else None
        if isinstance(data.get('model_types'), list):
            out['model_types'] = [t for t in data['model_types'] if isinstance(t, str)]
        if isinstance(data.get('remap'), dict):
            out['remap'] = {str(k): str(v) for k, v in data['remap'].items()}
    return out


# ----------------------------------------------------------------------------------------------------------------
# mlx_lm.server protegido: bootstrap (segredo + watchdog) e proxy na porta pública
# ----------------------------------------------------------------------------------------------------------------

# Roda no python do mlx-lm: `python -c BOOTSTRAP <entry point> --model …`. Lê o segredo na 1ª linha do stdin,
# faz todo pedido HTTP sem `X-Atlas-Key: <segredo>` receber 403 antes de chegar ao mlx_lm.server (patch genérico em
# http.server, que é o que o mlx_lm.server usa) e encerra o processo quando o stdin fecha — ou seja, quando o helper
# morre, até por SIGKILL. Depois executa o entry point (trampolim do uv, script do pip/brew ou o servidor falso dos
# testes); se ele não for Python, importa mlx_lm.server direto.
# O stdin é lido com os.read cru no fd 0: uma thread daemon presa em sys.stdin.buffer segura o lock do
# BufferedReader e o Python aborta (SIGABRT) no fim do processo. O script fica só em ASCII (vai no argv).
_BOOTSTRAP = r'''
import sys
sys.path[:] = [p for p in sys.path if p not in ("", ".")]
import hmac, os, runpy, signal, threading, time
import http.server as _hs


def _read_key():
    buf = b""
    while not buf.endswith(b"\n") and len(buf) < 256:
        c = os.read(0, 1)
        if not c:
            break
        buf += c
    return buf.strip()


_key = _read_key()


def _watchdog():
    try:
        while os.read(0, 1024):
            pass
    except Exception:
        pass
    try:
        os.killpg(os.getpgrp(), signal.SIGTERM)
    except Exception:
        pass
    time.sleep(5)
    os._exit(1)


if _key:
    _parse = _hs.BaseHTTPRequestHandler.parse_request

    def _gated(self):
        if not _parse(self):
            return False
        got = (self.headers.get("X-Atlas-Key") or "").encode("latin-1", "replace")
        if hmac.compare_digest(got, _key):
            return True
        body = b'{"error": "forbidden"}'
        try:
            self.send_response_only(403)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except OSError:
            pass
        self.close_connection = True
        return False

    _hs.BaseHTTPRequestHandler.parse_request = _gated

threading.Thread(target=_watchdog, name="atlas-watchdog", daemon=True).start()
_target = sys.argv[1]
sys.argv = [_target] + sys.argv[2:]
try:
    with open(_target, "rb") as _fh:
        compile(_fh.read(), _target, "exec")
    _python_entry = True
except (SyntaxError, ValueError):
    _python_entry = False
if not _python_entry:
    from mlx_lm.server import main
    sys.exit(main())
runpy.run_path(_target, run_name="__main__")
'''


class BoundedThreadingServer(http.server.ThreadingHTTPServer):
    """ThreadingHTTPServer com teto de conexões simultâneas: acima dele a conexão nova é fechada na hora
    (com o timeout por socket dos handlers, ninguém segura threads para sempre)."""
    daemon_threads = True
    allow_reuse_address = True
    max_connections = MAX_CONNECTIONS

    def __init__(self, *args, **kwargs):
        self._slots = threading.BoundedSemaphore(self.max_connections)
        http.server.ThreadingHTTPServer.__init__(self, *args, **kwargs)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            http.server.ThreadingHTTPServer.process_request(self, request, client_address)
        except Exception:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            http.server.ThreadingHTTPServer.process_request_thread(self, request, client_address)
        finally:
            self._slots.release()

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, socket.timeout)) or getattr(self, 'quiet', True):
            return
        http.server.ThreadingHTTPServer.handle_error(self, request, client_address)


_SAFE_HEADER_LIST = re.compile(r'[A-Za-z0-9_\-, ]{0,1000}')


class MlxGateway(object):
    """O mlx_lm.server gerenciado visto de fora: porta interna + segredo + modelo + limite de tokens.
    Também é o proxy que escuta na porta pública (p.ex. :8080) — `listen()` abre a porta, `close()` fecha."""

    def __init__(self, port, upstream_port, key, model_id, max_tokens, helper_port):
        self.port = port
        self.upstream_port = upstream_port
        self.key = key
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.helper_port = helper_port
        self.created = int(_now())
        self.httpd = None
        self._thread = None

    # ------------------------------------------------------------------ porta pública

    def listen(self):
        self.httpd = _GatewayHTTPServer(('127.0.0.1', self.port), GatewayHandler, self)
        self._thread = threading.Thread(target=self.httpd.serve_forever, kwargs={'poll_interval': 0.25},
                                        name='mlx-proxy-%d' % self.port, daemon=True)
        self._thread.start()
        return self

    def close(self):
        httpd, self.httpd = self.httpd, None
        if httpd is None:
            return
        try:
            httpd.shutdown()
        except Exception:
            pass
        try:
            httpd.server_close()
        except Exception:
            pass

    # ------------------------------------------------------------------ pedidos

    def upstream_headers(self):
        return {GATE_HEADER: self.key}

    def models_payload(self):
        # só o modelo gerenciado (o /v1/models do mlx_lm lista o cache inteiro do HF)
        return {'object': 'list', 'data': [{'id': self.model_id, 'object': 'model', 'created': self.created,
                                            'owned_by': 'mlx-lm'}]}

    def sanitize(self, body):
        """Corpo de um pedido de geração: fixa o modelo gerenciado (sem trocar/recarregar pesos por `model`,
        `adapters` ou `draft_model`) e limita max_tokens ao teto da API (MAX_GEN_TOKENS). O --max-tokens do start
        é só o padrão de quem não manda o campo: o copilot pode pedir mais nas configurações dele."""
        body['model'] = self.model_id
        body.pop('adapters', None)
        body.pop('draft_model', None)
        cap = max(self.max_tokens, MAX_GEN_TOKENS)
        for key in ('max_tokens', 'max_completion_tokens'):
            value = body.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > cap:
                body[key] = cap
        return body

    def relay(self, handler, method, path, data, cors_origin):
        """Encaminha ao mlx_lm.server interno e devolve a resposta ao cliente (inclusive SSE, pedaço a pedaço)."""
        headers = self.upstream_headers()
        headers['User-Agent'] = USER_AGENT
        if data is not None:
            headers['Content-Type'] = 'application/json'
        conn = http.client.HTTPConnection('127.0.0.1', self.upstream_port, timeout=UPSTREAM_TIMEOUT)
        try:
            conn.request(method, path, body=data, headers=headers)
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as e:
            conn.close()
            raise ApiError(502, 'upstream_unavailable', 'mlx_lm.server is not answering (it may still be loading, '
                                                        'or it stopped): %s' % (e.__class__.__name__,))
        try:
            handler.send_response(resp.status)
            for name in ('Content-Type', 'Content-Length', 'Cache-Control'):
                value = resp.getheader(name)
                if value:
                    handler.send_header(name, value)
            handler.send_header('X-Content-Type-Options', 'nosniff')
            if cors_origin:
                handler.send_header('Access-Control-Allow-Origin', cors_origin)
                handler.send_header('Vary', 'Origin')
            handler.close_connection = True
            handler.end_headers()
            if method == 'HEAD':
                return
            while True:
                chunk = resp.read1(64 * KIB)
                if not chunk:
                    break
                handler.wfile.write(chunk)
        except (OSError, http.client.HTTPException):
            handler.close_connection = True   # cliente foi embora ou o servidor caiu no meio do stream
        finally:
            try:
                resp.close()
            finally:
                conn.close()


def read_json_body(handler, limit):
    """Corpo JSON (objeto) de um pedido, com Content-Length obrigatório e teto de tamanho."""
    if handler.headers.get('Transfer-Encoding'):
        raise ApiError(411, 'length_required', 'Send a Content-Length (chunked bodies are not accepted).')
    raw_len = handler.headers.get('Content-Length')
    try:
        n = int(raw_len) if raw_len else 0
    except ValueError:
        raise ApiError(400, 'bad_request', 'Invalid Content-Length.')
    if n < 0:
        raise ApiError(400, 'bad_request', 'Invalid Content-Length.')
    if n > limit:
        raise ApiError(413, 'too_large', 'Request body is larger than %d KiB.' % (limit // KIB))
    raw = handler.rfile.read(n) if n else b''
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw.decode('utf-8'))
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise ApiError(400, 'bad_json', 'The body is not valid JSON.')
    if not isinstance(data, dict):
        raise ApiError(400, 'bad_json', 'The body must be a JSON object.')
    return data


def json_content_type(handler):
    return (handler.headers.get('Content-Type') or '').split(';')[0].strip().lower() == 'application/json'


class GatewayHandler(http.server.BaseHTTPRequestHandler):
    """Porta pública do servidor MLX. Aceita: clientes sem Origin (curl, scripts, SDKs) e páginas do próprio
    Latent Atlas; recusa outros sites (Origin), DNS rebinding (Host) e POSTs "simples" que não são JSON."""
    server_version = 'LatentAtlas-MLX/' + VERSION
    sys_version = ''
    timeout = HTTP_TIMEOUT

    def log_message(self, fmt, *args):
        pass

    def _send_json(self, status, payload, cors_origin=None):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if cors_origin:
            self.send_header('Access-Control-Allow-Origin', cors_origin)
            self.send_header('Vary', 'Origin')
        if status == 405:
            self.send_header('Allow', 'GET, POST, OPTIONS')
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def _check(self):
        """Host e Origin → a Origin permitida (para o CORS) ou None se o pedido não tem Origin."""
        gw = self.server.gateway
        host = (self.headers.get('Host') or '').strip().lower()
        if host not in allowed_hosts(gw.port):
            raise ApiError(403, 'forbidden_host', 'Unexpected Host header.')
        origin = self.headers.get('Origin')
        if origin is not None and origin not in allowed_origins(gw.helper_port):
            raise ApiError(403, 'forbidden_origin', 'Cross-origin requests are not allowed.')
        return origin

    def _run(self, fn):
        origin = None
        try:
            origin = self._check()
            fn(origin)
        except ApiError as e:
            try:
                self._send_json(e.status, e.payload(), origin)
            except OSError:
                self.close_connection = True
        except (OSError, http.client.HTTPException):
            self.close_connection = True

    def do_OPTIONS(self):
        def preflight(origin):
            if origin is None or self.headers.get('Access-Control-Request-Method') not in ('GET', 'POST'):
                raise ApiError(403, 'forbidden', 'Unexpected preflight.')
            self.send_response(204)
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Vary', 'Origin')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST')
            wanted = self.headers.get('Access-Control-Request-Headers') or ''
            if wanted and _SAFE_HEADER_LIST.fullmatch(wanted):
                self.send_header('Access-Control-Allow-Headers', wanted)
            self.send_header('Access-Control-Max-Age', '600')
            self.send_header('Content-Length', '0')
            self.end_headers()
        self._run(preflight)

    def do_GET(self):
        self._run(lambda origin: self._dispatch('GET', origin))

    def do_POST(self):
        self._run(lambda origin: self._dispatch('POST', origin))

    def do_HEAD(self):
        self._run(lambda origin: self._dispatch('HEAD', origin))

    def do_PUT(self):
        self.do_HEAD()

    def do_DELETE(self):
        self.do_HEAD()

    def do_PATCH(self):
        self.do_HEAD()

    def _dispatch(self, method, origin):
        gw = self.server.gateway
        path = urllib.parse.urlsplit(self.path).path
        if path == '/v1/models' or path.startswith('/v1/models/'):
            if method != 'GET':
                raise ApiError(405, 'method_not_allowed', 'Method not allowed.')
            return self._send_json(200, gw.models_payload(), origin)
        if path == '/health':
            if method != 'GET':
                raise ApiError(405, 'method_not_allowed', 'Method not allowed.')
            return gw.relay(self, 'GET', '/health', None, origin)
        if path in COMPLETION_PATHS:
            if method != 'POST':
                raise ApiError(405, 'method_not_allowed', 'Method not allowed.')
            if not json_content_type(self):
                raise ApiError(415, 'unsupported_media_type', 'POST requests must be application/json.')
            body = gw.sanitize(read_json_body(self, PROXY_MAX_BODY))
            return gw.relay(self, 'POST', path, json.dumps(body).encode('utf-8'), origin)
        raise ApiError(404, 'not_found', 'Unknown endpoint.')


class _GatewayHTTPServer(BoundedThreadingServer):
    quiet = True

    def __init__(self, address, handler, gateway):
        self.gateway = gateway
        BoundedThreadingServer.__init__(self, address, handler)


# ----------------------------------------------------------------------------------------------------------------
# o helper: estado + download + instalação + mlx_lm.server
# ----------------------------------------------------------------------------------------------------------------

def _now():
    return time.time()


class _DlCtx(object):
    def __init__(self, repo_id, commit, repo_dir, cancel):
        self.repo_id = repo_id
        self.commit = commit
        self.repo_dir = repo_dir
        self.blobs = os.path.join(repo_dir, 'blobs')
        self.snap = os.path.join(repo_dir, 'snapshots', commit)
        self.cancel = cancel
        self.fail = threading.Event()
        self.real_repo = None
        self.active = []

    def stopped(self):
        return self.cancel.is_set() or self.fail.is_set()


class Helper(object):
    """Estado e operações do helper MLX. Thread-safe; tudo que muda o status incrementa `seq`."""

    def __init__(self, port=0, mlx_port=8080, cache_dir=None, endpoint=None, token=None, env=None):
        self.env = dict(os.environ if env is None else env)
        self.port = port                       # porta do próprio helper (definida depois do bind)
        self.cache_dir = os.path.abspath(cache_dir or default_cache_dir(self.env))
        self.endpoint = (endpoint or default_endpoint(self.env)).rstrip('/')
        self.token = token if token is not None else default_token(self.env)
        self.hub = HubClient(self.endpoint, self.token)
        self.lock = threading.RLock()          # protege estado/seq
        self._ctl = threading.RLock()          # serializa start/stop/delete do servidor
        self.seq = 1
        self.logs = dict((k, LogRing()) for k in ('server', 'install', 'download'))
        self.mac = detect_mac()
        self.mlx = {'installed': False, 'server_bin': None, 'python': None, 'version': None, 'mlx_version': None,
                    'model_types': [], 'remap': {}, 'detecting': False}
        self.install = {'state': 'idle', 'method': None, 'error': None, 'error_code': None, 'started': 0,
                        'finished': 0}
        self.download = self._idle_download()
        self.server = {'state': 'stopped', 'id': None, 'port': int(mlx_port), 'pid': None,
                       'url': 'http://127.0.0.1:%d/v1' % int(mlx_port), 'started': 0, 'ready_at': 0,
                       'load_s': None, 'tok_s': None, 'error': None, 'error_code': None}
        self._local = None
        self._local_at = 0.0
        self._dl_thread = None
        self._dl_cancel = None
        self._proc = None
        self._gateway = None                   # MlxGateway do servidor gerenciado (proxy na porta pública)
        self._fatal = None                     # (gen, motivo) quando o log diz que a carga do modelo falhou
        self._gen = 0
        self._stopping = False
        self._install_proc = None
        self._install_thread = None
        self._detect_thread = None
        self._closed = False

    # -------------------------------------------------------------------------------------------- básico

    @staticmethod
    def _idle_download():
        return {'state': 'idle', 'id': None, 'total_bytes': 0, 'done_bytes': 0, 'speed_bps': 0, 'eta_s': None,
                'files_total': 0, 'files_done': 0, 'current_file': None, 'error': None, 'error_code': None,
                'started': 0, 'finished': 0, 'canceling': False}

    def _bump(self):
        with self.lock:
            self.seq += 1

    def _log(self, kind, line, bump=False):
        with self.lock:
            self.logs[kind].append(line)
            if bump:
                self.seq += 1

    def _logt(self, kind, line):
        self._log(kind, '[%s] %s' % (time.strftime('%H:%M:%S'), line))

    def repo_dir(self, repo_id):
        return os.path.join(self.cache_dir, repo_folder_name(repo_id))

    def same_repo(self, a, b):
        """Mesmo repo? Ids do HF não diferenciam maiúsculas, e no APFS (sem distinção de caixa) ids com caixa
        diferente caem na mesma pasta — compara as duas coisas."""
        if not a or not b:
            return False
        if a.lower() == b.lower():
            return True
        try:
            return os.path.samefile(self.repo_dir(a), self.repo_dir(b))
        except (OSError, ValueError):
            return False

    def _check_open(self):
        """Chamado com self.lock: depois do shutdown nada novo (processo, thread) é iniciado."""
        if self._closed:
            raise ApiError(503, 'shutting_down', 'The platform server is shutting down.')

    def local_models(self, force=False):
        with self.lock:
            fresh = self._local is not None and (_now() - self._local_at) < LOCAL_SCAN_TTL
            if fresh and not force:
                return [dict(m) for m in self._local]
        models = scan_cache(self.cache_dir)
        with self.lock:
            if models != self._local:
                self.seq += 1
            self._local = models
            self._local_at = _now()
            return [dict(m) for m in models]

    def local_entry(self, repo_id, force=False):
        for m in self.local_models(force=force):
            if m['id'] == repo_id:
                return m
        return None

    def status(self):
        local = self.local_models()
        with self.lock:
            mac = dict(self.mac)
            mac['disk_free_bytes'] = disk_free(self.cache_dir)
            mlx = dict(self.mlx)
            mlx['model_types'] = list(mlx['model_types'])
            mlx['remap'] = dict(mlx['remap'])
            mlx['installers'] = find_installers(self.env)
            mlx['install'] = dict(self.install)
            server = dict(self.server)
            server['log_tail'] = (list(self.logs['server'])[-TAIL_LINES:]
                                  if server['state'] in ('starting', 'error') else [])
            return {'seq': self.seq, 'mac': mac, 'mlx': mlx,
                    'hf': {'cache_dir': self.cache_dir, 'token': bool(self.token), 'endpoint': self.endpoint},
                    'download': dict(self.download), 'server': server, 'local_models': local}

    def logs_for(self, kind, n):
        with self.lock:
            lines = list(self.logs[kind])
        return lines[-n:] if n else []

    # -------------------------------------------------------------------------------------------- mlx-lm

    def detect_mlx(self):
        with self.lock:
            self.mlx['detecting'] = True
            self.seq += 1
        try:
            info = probe_mlx(self.env)
        except Exception as e:  # detecção nunca derruba o helper
            info = {'installed': False, 'server_bin': None, 'python': None, 'version': None, 'mlx_version': None,
                    'model_types': [], 'remap': {}}
            self._logt('install', 'mlx-lm detection failed: %s' % e)
        with self.lock:
            self.mlx.update(info)
            self.mlx['detecting'] = False
            self.seq += 1
        return info

    def start_detection(self):
        t = threading.Thread(target=self.detect_mlx, name='mlx-detect', daemon=True)
        self._detect_thread = t
        t.start()
        return t

    def _require_platform(self):
        if self.mac.get('os') != 'Darwin' or not self.mac.get('arm64'):
            raise ApiError(409, 'unsupported_platform',
                           'MLX only runs on Apple Silicon Macs (arm64). Use another copilot provider.')

    def start_install(self, upgrade=False):
        self._require_platform()
        with self.lock:
            self._check_open()
            if self.install['state'] == 'running':
                raise ApiError(409, 'busy', 'mlx-lm is already being installed.')
            inst = find_installers(self.env)
            if inst['uv']:
                method = 'uv'
                cmd = [inst['uv'], 'tool', 'install', '--python', '3.12'] + (['--upgrade'] if upgrade else [])
                cmd.append('mlx-lm')
            elif inst['brew']:
                method = 'brew'
                cmd = [inst['brew'], 'upgrade' if upgrade else 'install', 'mlx-lm']
            else:
                raise ApiError(409, 'no_installer', 'Neither uv nor Homebrew was found to install mlx-lm.',
                               hint=list(_INSTALL_HINT))
            self.install = {'state': 'running', 'method': method, 'error': None, 'error_code': None,
                            'started': _now(), 'finished': 0}
            self.logs['install'].clear()
            self.seq += 1
            t = threading.Thread(target=self._run_install, args=(cmd,), name='mlx-install', daemon=True)
            self._install_thread = t
        t.start()
        return self.status()

    def _run_install(self, cmd):
        self._logt('install', '$ ' + ' '.join(cmd))
        code = None
        err = None
        try:
            env = dict(self.env)
            env.setdefault('NONINTERACTIVE', '1')
            env.setdefault('HOMEBREW_NO_ENV_HINTS', '1')
            # Popen sob o lock: o shutdown (que lê _install_proc sob o mesmo lock) ou vê o processo e o mata,
            # ou já marcou _closed e nada é iniciado — nunca sobra um uv/brew órfão
            with self.lock:
                if self._closed:
                    raise OSError(errno.ECANCELED, 'the platform server is shutting down')
                proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, env=env, cwd=os.path.expanduser('~'),
                                        start_new_session=True)
                self._install_proc = proc
            for raw in bounded_lines(proc.stdout):
                for piece in raw.decode('utf-8', 'replace').replace('\r', '\n').split('\n'):
                    if piece.strip():
                        self._log('install', piece.rstrip())
            proc.stdout.close()
            code = proc.wait()
        except OSError as e:
            err = str(e)
        with self.lock:
            self._install_proc = None
        if err is None and code != 0:
            last = [ln for ln in self.logs_for('install', 5) if ln.strip()]
            err = 'Installer exited with code %s%s' % (code, (': ' + last[-1]) if last else '')
        self._logt('install', 'done' if err is None else 'failed: %s' % err)
        self.detect_mlx()
        with self.lock:
            self.install['finished'] = _now()
            if err is None and self.mlx['installed']:
                self.install['state'] = 'done'
            else:
                self.install['state'] = 'error'
                self.install['error'] = err or 'mlx_lm.server is still not found after the install.'
                self.install['error_code'] = 'install_failed' if err else 'not_found_after_install'
            self.seq += 1

    # -------------------------------------------------------------------------------------------- info

    def model_card(self, repo_id):
        if not valid_repo_id(repo_id):
            raise ApiError(400, 'bad_id', 'Invalid Hugging Face model id (expected org/name).')
        try:
            api = self.hub.model_api(repo_id)
        except Exception as e:
            raise hub_error(e, repo_id)
        if not isinstance(api, dict):
            raise ApiError(502, 'hf_unreachable', 'Hugging Face sent an unexpected answer.')
        files = parse_siblings(api, strict=False)
        sha = api.get('sha') if isinstance(api.get('sha'), str) else None
        cfg = api.get('config') if isinstance(api.get('config'), dict) else {}
        # o config.json de verdade (pequeno) traz quantização e MoE que a API às vezes omite
        if sha and COMMIT_RE.fullmatch(sha) and any(f['name'] == 'config.json' for f in files):
            try:
                full = self.hub.get_json(self.hub.file_url(repo_id, sha, 'config.json'), timeout=15,
                                         limit=4 * 1024 * 1024)
                if isinstance(full, dict):
                    merged = dict(cfg)
                    merged.update(full)
                    cfg = merged
            except Exception:
                pass
        model_type = cfg.get('model_type') if isinstance(cfg.get('model_type'), str) else None
        with self.lock:
            installed = self.mlx['installed']
            types = list(self.mlx['model_types'])
            remap = dict(self.mlx['remap'])
        mapped = remap.get(model_type, model_type) if model_type else None
        supported = None
        if installed and types:
            supported = bool(mapped and mapped in types)
        bits, moe = config_facts(cfg)
        st = api.get('safetensors') if isinstance(api.get('safetensors'), dict) else {}
        local = self.local_entry(repo_id)
        return {
            'id': api.get('id') if isinstance(api.get('id'), str) else repo_id,
            'sha': sha,
            'files': [{'name': f['name'], 'size': f['size']} for f in files],
            'download_bytes': sum(f['size'] for f in files),
            'model_type': model_type,
            'mapped_type': mapped,
            'supported': supported,
            'library': api.get('library_name'),
            'pipeline_tag': api.get('pipeline_tag'),
            'tags': [t for t in (api.get('tags') or []) if isinstance(t, str)],
            'gated': api.get('gated', False),
            'params': st.get('total') if isinstance(st.get('total'), int) else None,
            'quant_bits': bits,
            'moe': moe,
            'downloads': api.get('downloads'),
            'likes': api.get('likes'),
            'last_modified': api.get('lastModified'),
            'local': {'bytes_present': local['bytes'] if local else 0,
                      'complete': bool(local and local['complete'])},
        }

    # -------------------------------------------------------------------------------------------- download

    def start_download(self, repo_id):
        if not valid_repo_id(repo_id):
            raise ApiError(400, 'bad_id', 'Invalid Hugging Face model id (expected org/name).')
        with self.lock:
            self._check_open()
            if self.download['state'] == 'running':
                if self.same_repo(self.download['id'], repo_id):
                    return 200, self.status()
                raise ApiError(409, 'busy', 'Another model is downloading (%s).' % self.download['id'],
                               id=self.download['id'])
        try:
            api = self.hub.model_api(repo_id)
        except Exception as e:
            raise hub_error(e, repo_id)
        if not isinstance(api, dict):
            raise ApiError(502, 'hf_unreachable', 'Hugging Face sent an unexpected answer.')
        commit = api.get('sha')
        if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
            raise ApiError(502, 'bad_metadata', 'The hub did not return a valid commit for %s.' % repo_id)
        files = parse_siblings(api, strict=True)
        if not files:
            raise ApiError(409, 'no_files', 'No files in %s match what mlx-lm downloads (*.safetensors, *.json, …).'
                           % repo_id)
        blobs = collections.OrderedDict()
        for f in files:
            b = blobs.setdefault(f['etag'], {'etag': f['etag'], 'size': f['size'], 'lfs': f['lfs'],
                                             'sha256': f['sha256'], 'names': []})
            if b['size'] != f['size']:
                raise ApiError(502, 'bad_metadata', 'Inconsistent sizes for blob %s.' % f['etag'])
            b['names'].append(f['name'])
        repo_dir = self.repo_dir(repo_id)
        blobs_dir = os.path.join(repo_dir, 'blobs')
        total = sum(b['size'] for b in blobs.values())
        present = sum(self._present_bytes(blobs_dir, b) for b in blobs.values())
        free = disk_free(self.cache_dir)
        need = (total - present) + GIB
        if free is not None and free < need:
            raise ApiError(409, 'no_space', 'Not enough free disk space for %s.' % repo_id,
                           need_bytes=need, free_bytes=free)
        with self.lock:
            self._check_open()
            if self.download['state'] == 'running':
                raise ApiError(409, 'busy', 'Another model is downloading (%s).' % self.download['id'],
                               id=self.download['id'])
            cancel = threading.Event()
            self._dl_cancel = cancel
            self.download = self._idle_download()
            self.download.update({'state': 'running', 'id': repo_id, 'total_bytes': total, 'done_bytes': present,
                                  'files_total': len(files), 'started': _now()})
            self.logs['download'].clear()
            self.seq += 1
            ctx = _DlCtx(repo_id, commit, repo_dir, cancel)
            t = threading.Thread(target=self._run_download, args=(ctx, list(blobs.values())), name='mlx-download',
                                 daemon=True)
            self._dl_thread = t
        t.start()
        return 202, self.status()

    @staticmethod
    def _present_bytes(blobs_dir, b):
        blob = os.path.join(blobs_dir, b['etag'])
        if regular_size(blob) == b['size']:
            return b['size']
        part = regular_size(blob + '.incomplete')
        return min(part, b['size']) if part else 0

    def cancel_download(self, wait=4.0):
        with self.lock:
            running = self.download['state'] == 'running'
            cancel = self._dl_cancel
            t = self._dl_thread
            if running and cancel is not None:
                cancel.set()
                self.download['canceling'] = True
                self.seq += 1
        if running and t is not None and wait:
            t.join(wait)
        return self.status()

    def _progress(self, n):
        with self.lock:
            self.download['done_bytes'] = max(0, self.download['done_bytes'] + n)
            self.seq += 1

    def _set_current(self, ctx):
        with self.lock:
            self.download['current_file'] = ctx.active[0] if ctx.active else None
            self.seq += 1

    def _run_download(self, ctx, blobs):
        state, code, message = 'done', None, None
        try:
            self._prepare_repo_dirs(ctx)
            for b in blobs:
                for name in b['names']:
                    self._check_inside(ctx, os.path.join(ctx.snap, name))
            self._logt('download', '%s @ %s · %d files · %s' % (ctx.repo_id, ctx.commit[:12],
                                                                sum(len(b['names']) for b in blobs),
                                                                human_bytes(sum(b['size'] for b in blobs))))
            queue = sorted(blobs, key=lambda b: -b['size'])
            qlock = threading.Lock()
            errors = []

            def worker():
                while not ctx.stopped():
                    with qlock:
                        if not queue:
                            return
                        item = queue.pop(0)
                    try:
                        self._fetch_blob(ctx, item)
                    except _Canceled:
                        return
                    except DownloadFailed as e:
                        errors.append(e)
                        ctx.fail.set()
                        return
                    except Exception as e:
                        errors.append(DownloadFailed('internal', 'Unexpected error: %s' % e))
                        ctx.fail.set()
                        return

            workers = [threading.Thread(target=worker, name='mlx-dl-%d' % i, daemon=True)
                       for i in range(max(1, min(DOWNLOAD_WORKERS, len(queue))))]
            for w in workers:
                w.start()
            last_t = _now()
            with self.lock:
                last_b = self.download['done_bytes']
            speed = 0.0
            while any(w.is_alive() for w in workers):
                for w in workers:
                    w.join(0.25)
                    if w.is_alive():
                        break
                now = _now()
                if now - last_t >= 1.0:
                    with self.lock:
                        done = self.download['done_bytes']
                        inst = max(0.0, (done - last_b) / (now - last_t))
                        speed = inst if speed <= 0 else 0.6 * speed + 0.4 * inst
                        self.download['speed_bps'] = int(speed)
                        remaining = max(0, self.download['total_bytes'] - done)
                        self.download['eta_s'] = int(math.ceil(remaining / speed)) if speed > 1 else None
                        self.seq += 1
                    last_t, last_b = now, done
            if errors:
                raise errors[0]
            if ctx.cancel.is_set():
                raise _Canceled()
            # refs/main só no fim: antes disso o snapshot não "existe" para o huggingface_hub / mlx-lm
            refs = os.path.join(ctx.repo_dir, 'refs')
            tmp = os.path.join(refs, '.main.tmp')
            if os.path.lexists(tmp):
                os.remove(tmp)
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | O_NOFOLLOW, 0o644)
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                fh.write(ctx.commit)
            os.replace(tmp, os.path.join(refs, 'main'))
            self._logt('download', 'complete · refs/main → %s' % ctx.commit)
        except _Canceled:
            state, code, message = 'canceled', None, None
            self._logt('download', 'canceled (partial files kept, the next download resumes)')
        except DownloadFailed as e:
            state, code, message = 'error', e.code, e.message
            self._logt('download', 'error [%s]: %s' % (e.code, e.message))
        except Exception as e:
            state, code, message = 'error', 'internal', 'Unexpected error: %s' % e
            self._logt('download', 'error: %s' % e)
        self.local_models(force=True)
        with self.lock:
            self.download.update({'state': state, 'error': message, 'error_code': code, 'finished': _now(),
                                  'speed_bps': 0, 'eta_s': None, 'current_file': None, 'canceling': False})
            if state == 'done':
                self.download['done_bytes'] = self.download['total_bytes']
                self.download['files_done'] = self.download['files_total']
            self.seq += 1

    @staticmethod
    def _safe_mkdir(path):
        """Cria uma pasta (um nível) ou aceita uma pasta de verdade que já existe — nunca um symlink."""
        try:
            os.mkdir(path)
            return
        except FileExistsError:
            pass
        if os.path.islink(path) or not os.path.isdir(path):
            raise DownloadFailed('unsafe_path', 'Refusing to use %s: it is a symlink or not a folder.' % path)

    def _prepare_repo_dirs(self, ctx):
        """models--org--nome/{blobs,refs,snapshots/<commit>} — confere cada nível ANTES de criar o próximo, para
        nunca criar pastas através de um symlink que aponte para fora do cache."""
        os.makedirs(self.cache_dir, exist_ok=True)
        self._safe_mkdir(ctx.repo_dir)
        ctx.real_repo = os.path.realpath(ctx.repo_dir)
        if os.path.dirname(ctx.real_repo) != os.path.realpath(self.cache_dir):
            raise DownloadFailed('unsafe_path', 'The model folder resolves outside the HF cache.')
        for d in (ctx.blobs, os.path.join(ctx.repo_dir, 'refs'), os.path.join(ctx.repo_dir, 'snapshots'), ctx.snap):
            self._safe_mkdir(d)

    def _check_inside(self, ctx, path):
        parent = os.path.realpath(os.path.dirname(path))
        if parent != ctx.real_repo and not parent.startswith(ctx.real_repo + os.sep):
            raise DownloadFailed('unsafe_path', 'Refusing to write outside the model folder: %s' % path)

    def _fetch_blob(self, ctx, b):
        blob = os.path.join(ctx.blobs, b['etag'])
        part = blob + '.incomplete'
        self._check_inside(ctx, blob)
        with self.lock:
            ctx.active.append(b['names'][0])
        self._set_current(ctx)
        try:
            if regular_size(blob) != b['size']:
                if os.path.lexists(blob):
                    os.remove(blob)
                attempt = 0
                while True:
                    if ctx.stopped():
                        raise _Canceled()
                    try:
                        self._stream(ctx, b, part)
                        break
                    except _Retryable as e:
                        attempt += 1
                        self._logt('download', 'retry %d/%d %s: %s' % (attempt, FILE_ATTEMPTS - 1, b['names'][0], e))
                        if attempt >= FILE_ATTEMPTS:
                            raise DownloadFailed('network', 'Failed to download %s after %d attempts: %s'
                                                 % (b['names'][0], FILE_ATTEMPTS, e))
                        if ctx.cancel.wait(min(2.0 * attempt, 5.0)) or ctx.fail.is_set():
                            raise _Canceled()
                os.replace(part, blob)
            for name in b['names']:
                self._link(ctx, blob, name)
            with self.lock:
                self.download['files_done'] += len(b['names'])
                self.seq += 1
        finally:
            with self.lock:
                if b['names'][0] in ctx.active:
                    ctx.active.remove(b['names'][0])
            self._set_current(ctx)

    def _link(self, ctx, blob, name):
        link = os.path.join(ctx.snap, name)
        cur = ctx.snap
        for part in name.split('/')[:-1]:   # subpastas (p.ex. sub/x.json), uma a uma, sem atravessar symlinks
            cur = os.path.join(cur, part)
            self._safe_mkdir(cur)
        self._check_inside(ctx, link)
        rel = os.path.relpath(blob, os.path.dirname(link))
        if os.path.lexists(link):
            if os.path.isdir(link) and not os.path.islink(link):
                raise DownloadFailed('unsafe_path', 'A folder is in the way of %s.' % name)
            os.remove(link)
        os.symlink(rel, link)

    def _new_hasher(self, b):
        if b['lfs']:
            return hashlib.sha256()
        # arquivos normais: etag = oid do blob git = sha1("blob <tamanho>\0" + conteúdo)
        h = hashlib.sha1()
        h.update(b'blob %d\x00' % b['size'])
        return h

    def _hash_file(self, ctx, h, path):
        with os.fdopen(os.open(path, os.O_RDONLY | O_NOFOLLOW), 'rb') as fh:
            while True:
                if ctx.stopped():
                    raise _Canceled()
                chunk = fh.read(CHUNK)
                if not chunk:
                    break
                h.update(chunk)

    def _verify(self, b, h, part):
        if h.hexdigest() != b['etag']:
            try:
                os.remove(part)
            except OSError:
                pass
            self._progress(-b['size'])
            raise DownloadFailed('sha_mismatch', 'Checksum mismatch for %s (the partial file was deleted).'
                                 % b['names'][0])

    def _stream(self, ctx, b, part):
        size = b['size']
        if os.path.lexists(part) and regular_size(part) is None:
            # um .incomplete que é symlink (ou outra coisa) nunca é seguido: some com ele e recomeça
            if os.path.isdir(part) and not os.path.islink(part):
                raise DownloadFailed('unsafe_path', 'A folder is in the way of %s.' % part)
            os.remove(part)
        have = regular_size(part) or 0
        if have > size:
            os.remove(part)
            self._progress(-min(have, size))
            have = 0
        h = self._new_hasher(b)
        if have:
            self._hash_file(ctx, h, part)
        headers = {'Range': 'bytes=%d-' % have} if have else {}
        url = self.hub.file_url(ctx.repo_id, ctx.commit, b['names'][0])
        if have == size and size > 0:
            # já está tudo no .incomplete (p.ex. cancelado logo antes do rename)
            self._verify(b, h, part)
            return
        if have == 0:
            self._logt('download', 'GET %s (%s)' % (b['names'][0], human_bytes(size)))
        else:
            self._logt('download', 'resume %s at %s of %s' % (b['names'][0], human_bytes(have), human_bytes(size)))
        try:
            resp = self.hub.open(url, headers=headers, timeout=READ_TIMEOUT)
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:
                # o servidor diz que não há mais nada a partir de `have`: confere o que temos
                if have != size:
                    os.remove(part)
                    self._progress(-have)
                    raise _Retryable('range not satisfiable')
                self._verify(b, h, part)
                return
            if e.code in (401, 403):
                raise DownloadFailed('gated', 'Hugging Face refused %s (HTTP %d): gated or private model — accept '
                                     'its license on huggingface.co and set an HF token (HF_TOKEN).'
                                     % (b['names'][0], e.code))
            if e.code == 404:
                raise DownloadFailed('not_found', 'File %s was not found on the hub (HTTP 404).' % b['names'][0])
            raise _Retryable('HTTP %d' % e.code)
        except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
            raise _Retryable(str(getattr(e, 'reason', None) or e))
        try:
            status = resp.status
            append = False
            if have and status == 206:
                m = re.match(r'bytes (\d+)-', resp.headers.get('Content-Range') or '')
                if not m or int(m.group(1)) != have:
                    raise _Retryable('unexpected Content-Range %r' % resp.headers.get('Content-Range'))
                append = True
            elif status != 200 and status != 206:
                raise _Retryable('HTTP %d' % status)
            elif have:
                # o servidor ignorou o Range: recomeça do zero
                self._progress(-have)
                have = 0
                h = self._new_hasher(b)
            flags = os.O_WRONLY | os.O_CREAT | O_NOFOLLOW | (os.O_APPEND if append else os.O_TRUNC)
            try:
                fd = os.open(part, flags, 0o644)
            except OSError as e:
                if e.errno == errno.ELOOP:
                    raise DownloadFailed('unsafe_path', 'Refusing to follow a symlink at %s.' % part)
                raise
            with os.fdopen(fd, 'wb') as out:
                while True:
                    if ctx.stopped():
                        raise _Canceled()
                    try:
                        chunk = resp.read(CHUNK)
                    except (OSError, http.client.HTTPException) as e:
                        raise _Retryable(str(e) or e.__class__.__name__)
                    if not chunk:
                        break
                    if have + len(chunk) > size:
                        out.close()
                        os.remove(part)
                        self._progress(-have)
                        raise DownloadFailed('size_mismatch', 'The hub sent more bytes than expected for %s.'
                                             % b['names'][0])
                    out.write(chunk)
                    h.update(chunk)
                    have += len(chunk)
                    self._progress(len(chunk))
        finally:
            try:
                resp.close()
            except Exception:
                pass
        if have != size:
            raise _Retryable('connection closed at %s of %s' % (human_bytes(have), human_bytes(size)))
        self._verify(b, h, part)

    # -------------------------------------------------------------------------------------------- delete

    def delete_model(self, repo_id):
        if not valid_repo_id(repo_id):
            raise ApiError(400, 'bad_id', 'Invalid Hugging Face model id (expected org/name).')
        with self._ctl:
            with self.lock:
                if self.download['state'] == 'running' and self.same_repo(self.download['id'], repo_id):
                    raise ApiError(409, 'busy', '%s is downloading; cancel it first.' % repo_id)
                if self.server['state'] in ('starting', 'ready') and self.same_repo(self.server['id'], repo_id):
                    raise ApiError(409, 'busy', '%s is being served; stop the server first.' % repo_id)
            path = self.repo_dir(repo_id)
            if not os.path.lexists(path):
                raise ApiError(404, 'not_found', '%s is not in the HF cache.' % repo_id)
            if os.path.islink(path) or not os.path.isdir(path):
                raise ApiError(409, 'not_a_directory', 'Refusing to delete %s: it is not a plain folder.' % path)
            cache_real = os.path.realpath(self.cache_dir)
            if os.path.dirname(os.path.realpath(path)) != cache_real:
                raise ApiError(409, 'outside_cache', 'Refusing to delete a folder outside the HF cache.')
            freed = 0
            for dirpath, _dirs, files in os.walk(path):
                for n in files:
                    try:
                        st = os.lstat(os.path.join(dirpath, n))
                        if not os.path.islink(os.path.join(dirpath, n)):
                            freed += st.st_size
                    except OSError:
                        pass
            shutil.rmtree(path)
        self._logt('download', 'deleted %s (%s)' % (repo_id, human_bytes(freed)))
        self.local_models(force=True)
        self._bump()
        return {'freed_bytes': freed, 'id': repo_id}

    # -------------------------------------------------------------------------------------------- mlx_lm.server

    def _server_snapshot(self):
        with self.lock:
            return dict(self.server)

    def gateway(self):
        """MlxGateway do modelo gerenciado enquanto ele está subindo/pronto (senão None)."""
        with self.lock:
            if self._gateway is not None and self.server['state'] in ('starting', 'ready'):
                return self._gateway
        return None

    def start_server(self, repo_id, port, max_tokens, temp):
        if not valid_repo_id(repo_id):
            raise ApiError(400, 'bad_id', 'Invalid Hugging Face model id (expected org/name).')
        self._require_platform()
        with self._ctl:
            with self.lock:
                self._check_open()
                if not self.mlx['installed'] or not self.mlx['server_bin']:
                    raise ApiError(409, 'mlx_missing', 'mlx-lm is not installed.')
                server_bin = self.mlx['server_bin']
                python = self.mlx.get('python')
            entry = self.local_entry(repo_id, force=True)
            if not entry or not entry['complete']:
                raise ApiError(409, 'not_downloaded', '%s is not fully downloaded yet.' % repo_id)
            with self.lock:
                alive = self._proc is not None and self._proc.poll() is None
                if (alive and self.server['id'] == repo_id and self.server['port'] == port
                        and self.server['state'] in ('starting', 'ready')):
                    return 200, self.status()
                leftover = self._gateway is not None
            if alive or leftover:
                self.stop_server()
            if port_listening(port):
                pr = self.probe(port)
                raise ApiError(409, 'port_busy', 'Port %d is already in use by another program.' % port,
                               port=port, openai=pr['openai'], models=pr['models'])
            # porta pública = proxy deste processo; o mlx_lm.server de verdade fica numa porta interna aleatória
            # (sorteada só depois do bind da pública, para nunca cair nela)
            gateway = MlxGateway(port, None, secrets.token_hex(24), repo_id, max_tokens, self.port)
            try:
                gateway.listen()
            except OSError as e:
                raise ApiError(409, 'port_busy', 'Port %d is already in use by another program (%s).'
                               % (port, e.strerror or e), port=port, openai=False, models=[])
            try:
                gateway.upstream_port = free_local_port()
            except OSError as e:
                gateway.close()
                raise ApiError(500, 'spawn_failed', 'Could not find a free local port: %s' % e)
            origins = ','.join(['http://localhost:%d' % self.port, 'http://127.0.0.1:%d' % self.port,
                                'http://[::1]:%d' % self.port])
            flags = ['--model', repo_id, '--host', '127.0.0.1', '--port', str(gateway.upstream_port),
                     '--max-tokens', str(max_tokens), '--temp', repr(float(temp)), '--allowed-origins', origins]
            # com o python do mlx-lm: bootstrap (segredo por pedido + watchdog); sem ele, o entry point direto
            cmd = ([python, '-c', _BOOTSTRAP, server_bin] if python else [server_bin]) + flags
            env = dict(self.env)
            env.update({'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1', 'PYTHONUNBUFFERED': '1',
                        'HF_HUB_CACHE': self.cache_dir})
            with self.lock:
                self.logs['server'].clear()
                self._fatal = None
            self._log('server', '$ ' + ' '.join([server_bin] + flags))
            self._log('server', '[atlas] port %d (proxy) → mlx_lm.server on 127.0.0.1:%d'
                      % (port, gateway.upstream_port))
            if not python:
                self._log('server', "[atlas] warning: mlx-lm's python was not found; the internal server runs "
                                    'without the per-run key and is not stopped if this helper crashes')
            spawn_t = _now()
            try:
                with self.lock:
                    self._check_open()
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE if python else subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                        cwd=self.cache_dir, start_new_session=True)
            except ApiError:
                gateway.close()
                raise
            except OSError as e:
                gateway.close()
                with self.lock:
                    self.server.update({'state': 'error', 'id': repo_id, 'port': port, 'pid': None,
                                        'url': 'http://127.0.0.1:%d/v1' % port, 'started': spawn_t,
                                        'error': 'Could not start mlx_lm.server: %s' % e,
                                        'error_code': 'spawn_failed'})
                    self.seq += 1
                raise ApiError(500, 'spawn_failed', 'Could not start mlx_lm.server: %s' % e)
            if python:
                try:
                    proc.stdin.write(gateway.key.encode('ascii') + b'\n')
                    proc.stdin.flush()
                except OSError:
                    pass  # morreu logo no início: o leitor do log reporta o erro
            with self.lock:
                self._gen += 1
                gen = self._gen
                self._proc = proc
                self._gateway = gateway
                self._stopping = False
                self.server = {'state': 'starting', 'id': repo_id, 'port': port, 'pid': proc.pid,
                               'url': 'http://127.0.0.1:%d/v1' % port, 'started': spawn_t, 'ready_at': 0,
                               'load_s': None, 'tok_s': None, 'error': None, 'error_code': None,
                               'max_tokens': max_tokens, 'temp': temp}
                self.seq += 1
            threading.Thread(target=self._read_server_output, args=(proc, gen), name='mlx-server-log',
                             daemon=True).start()
            threading.Thread(target=self._await_ready, args=(proc, gen, gateway, spawn_t),
                             name='mlx-server-ready', daemon=True).start()
        return 202, self.status()

    def _is_current(self, gen):
        with self.lock:
            return gen == self._gen

    @staticmethod
    def _close_stdin(proc):
        if proc is not None and proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass

    def _read_server_output(self, proc, gen):
        try:
            for raw in bounded_lines(proc.stdout):
                text = raw.decode('utf-8', 'replace').rstrip('\r\n')
                text = text.split('\r')[-1]
                if not text.strip() or _HEALTH_LOG_RE.search(text):
                    continue
                fatal = _FATAL_LOG_RE.search(text)
                with self.lock:
                    current = gen == self._gen
                    if fatal and current:
                        self._fatal = (gen, clip_line(fatal.group(1).strip() or text.strip(), 400))
                    bump = current and self.server['state'] in ('starting', 'error')
                self._log('server', text, bump=bump)
        except (OSError, ValueError):
            pass
        try:
            proc.stdout.close()
        except Exception:
            pass
        code = proc.wait()
        self._close_stdin(proc)
        with self.lock:
            if gen != self._gen:
                return
            self._proc = None
            gateway, self._gateway = self._gateway, None
            self.server['pid'] = None
            if self._stopping:
                self.server['state'] = 'stopped'
            elif self.server['state'] != 'error':
                hint = self._failure_hint(any_line=True)
                self.server['state'] = 'error'
                self.server['error_code'] = 'exited'
                self.server['error'] = 'mlx_lm.server exited (code %s)%s' % (code, (': ' + hint) if hint else '')
            self.seq += 1
        if gateway is not None:
            gateway.close()

    def _failure_hint(self, any_line=False):
        """Última linha do log do servidor com cara de erro (p.ex. `ValueError: …`, `- ERROR -`); com
        any_line=True cai para a última linha qualquer. Ignora as linhas do próprio helper."""
        with self.lock:
            lines = [ln.strip() for ln in self.logs['server']]
        lines = [ln for ln in lines if ln and not ln.startswith('[atlas]') and not ln.startswith('$ ')]
        for ln in reversed(lines):
            if _ERROR_LOG_RE.search(ln):
                return clip_line(ln, 300)
        return clip_line(lines[-1], 300) if (any_line and lines) else None

    def _fatal_reason(self, gen):
        with self.lock:
            if self._fatal is not None and self._fatal[0] == gen:
                return self._fatal[1]
        return None

    def _fail_start(self, gen, proc, code, message):
        with self.lock:
            if gen != self._gen:
                return
            self.server.update({'state': 'error', 'error': message, 'error_code': code})
            self.seq += 1
        self._log('server', '[atlas] ' + message, bump=True)
        self._kill(proc)

    def _load_failed(self, gen, proc):
        reason = (self._fatal_reason(gen) or self._failure_hint()
                  or 'its generation thread stopped (/health answers 503).')
        self._fail_start(gen, proc, 'load_failed', 'mlx_lm.server could not load the model: %s' % reason)

    @staticmethod
    def _http_error_text(e):
        try:
            return e.read(4000).decode('utf-8', 'replace')
        except Exception:
            return ''
        finally:
            try:
                e.close()
            except Exception:
                pass

    def _await_ready(self, proc, gen, gateway, spawn_t):
        base = 'http://127.0.0.1:%d' % gateway.upstream_port
        gate = gateway.upstream_headers()
        deadline = spawn_t + HEALTH_TIMEOUT
        unavailable = 0      # /health 503 seguidos: a thread de geração morreu (o modelo não carregou)
        last_status = None
        while True:
            if not self._is_current(gen) or proc.poll() is not None:
                return  # parado/substituído ou morreu (o leitor do log reporta o erro)
            if self._fatal_reason(gen):
                return self._load_failed(gen, proc)
            status = None
            try:
                status, _ = local_request(base + '/health', timeout=2.0, headers=gate)
            except urllib.error.HTTPError as e:
                status = e.code
                self._http_error_text(e)
            except Exception:
                status = None
            if status in (200, 404):
                break  # 404: mlx-lm antigo, sem /health — o aquecimento decide
            unavailable = unavailable + 1 if status == 503 else 0
            if unavailable >= HEALTH_503_LIMIT:
                return self._load_failed(gen, proc)
            last_status = status if status is not None else last_status
            if _now() > deadline:
                if last_status is None:
                    msg = 'mlx_lm.server did not answer /health within %d s.' % HEALTH_TIMEOUT
                else:
                    msg = 'mlx_lm.server answered /health with HTTP %d for %d s.' % (last_status, HEALTH_TIMEOUT)
                self._fail_start(gen, proc, 'health_timeout', msg)
                return
            time.sleep(0.5)
        # aquecimento: só uma geração bem-sucedida garante que os pesos carregaram
        body = {'messages': [{'role': 'user', 'content': 'Hi'}], 'max_tokens': 8,
                'chat_template_kwargs': {'enable_thinking': False}}
        use_kwargs = True
        err = None
        for attempt in (0, 1):
            try:
                local_request(base + '/v1/chat/completions', body, timeout=WARMUP_TIMEOUT, headers=gate)
                err = None
                break
            except urllib.error.HTTPError as e:
                detail = self._http_error_text(e).strip()
                if attempt == 0 and 400 <= e.code < 500 and 'chat_template_kwargs' in detail:
                    use_kwargs = False
                    body.pop('chat_template_kwargs', None)
                    continue
                err = 'HTTP %d%s' % (e.code, (': ' + clip_line(detail, 200)) if detail else '')
                break
            except Exception as e:
                err = str(getattr(e, 'reason', None) or e) or e.__class__.__name__
                break
        if err is not None:
            if not self._is_current(gen) or proc.poll() is not None:
                return
            for _ in range(10):  # o leitor do log pode estar um instante atrás
                if self._fatal_reason(gen):
                    return self._load_failed(gen, proc)
                time.sleep(0.1)
            self._fail_start(gen, proc, 'warmup_failed', 'Warmup request failed: %s' % err)
            return
        load_s = _now() - spawn_t
        if not self._is_current(gen):
            return
        # medição rápida de velocidade (best effort)
        tok_s = None
        try:
            mbody = {'messages': [{'role': 'user', 'content': 'Write two sentences about the ocean.'}],
                     'max_tokens': 48}
            if use_kwargs:
                mbody['chat_template_kwargs'] = {'enable_thinking': False}
            t0 = _now()
            _, data = local_request(base + '/v1/chat/completions', mbody, timeout=120, headers=gate)
            elapsed = _now() - t0
            tokens = ((data or {}).get('usage') or {}).get('completion_tokens')
            if isinstance(tokens, (int, float)) and tokens > 0 and elapsed > 0:
                tok_s = round(tokens / elapsed, 1)
        except Exception:
            tok_s = None
        with self.lock:
            if gen != self._gen or self._proc is not proc or proc.poll() is not None:
                return
            self.server.update({'state': 'ready', 'ready_at': _now(), 'load_s': round(load_s, 1), 'tok_s': tok_s})
            self.seq += 1
        self._log('server', '[atlas] ready · loaded in %.1f s%s' % (load_s, (' · ~%s tok/s' % tok_s) if tok_s else ''))

    @staticmethod
    def _kill(proc, grace=STOP_GRACE):
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.terminate()
            except OSError:
                pass
        try:
            proc.wait(grace)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(3)
        except subprocess.TimeoutExpired:
            pass

    def stop_server(self):
        with self._ctl:
            with self.lock:
                proc = self._proc
                gateway, self._gateway = self._gateway, None
                self._stopping = True
                self._gen += 1     # leitor/aquecimento antigos param de mexer no estado
            if gateway is not None:
                gateway.close()    # a porta pública fecha primeiro: nada novo chega ao servidor que está saindo
            self._kill(proc)
            self._close_stdin(proc)
            with self.lock:
                self._proc = None
                self._stopping = False
                self.server.update({'state': 'stopped', 'id': None, 'pid': None, 'started': 0, 'ready_at': 0,
                                    'load_s': None, 'tok_s': None, 'error': None, 'error_code': None})
                self.seq += 1
            if proc is not None:
                self._log('server', '[atlas] stopped')
        return self.status()

    def probe(self, port):
        listening = port_listening(port)
        with self.lock:
            gw = self._gateway
            managed = gw is not None and gw.port == port and self.server['state'] in ('starting', 'ready')
        openai, models = False, []
        if listening:
            try:
                status, data = local_request('http://127.0.0.1:%d/v1/models' % port, timeout=1.5)
                if status == 200 and isinstance(data, dict) and isinstance(data.get('data'), list):
                    openai = True
                    models = [m.get('id') for m in data['data'] if isinstance(m, dict) and
                              isinstance(m.get('id'), str)][:50]
            except Exception:
                pass
        return {'port': port, 'listening': listening, 'managed': managed, 'openai': openai, 'models': models}

    # -------------------------------------------------------------------------------------------- ciclo de vida

    def shutdown(self):
        with self.lock:
            if self._closed:
                return
            self._closed = True
            cancel = self._dl_cancel
            inst = self._install_proc
        if cancel is not None:
            cancel.set()
        if inst is not None:
            self._kill(inst, grace=2.0)
        try:
            self.stop_server()
        except Exception:
            pass


# ----------------------------------------------------------------------------------------------------------------
# parâmetros
# ----------------------------------------------------------------------------------------------------------------

def _num(value, name):
    if isinstance(value, bool) or value is None:
        raise ApiError(400, 'bad_param', 'Invalid %s.' % name, param=name)
    try:
        if isinstance(value, (int, float)):
            out = float(value)          # inteiros JSON enormes → OverflowError
        elif isinstance(value, str):
            out = float(value.strip())
        else:
            raise ValueError(name)
    except (ValueError, OverflowError):
        raise ApiError(400, 'bad_param', 'Invalid %s.' % name, param=name)
    if not math.isfinite(out):
        raise ApiError(400, 'bad_param', 'Invalid %s.' % name, param=name)
    return out


def int_param(value, name, lo, hi):
    out = _num(value, name)
    if not out.is_integer() or not (lo <= out <= hi):
        raise ApiError(400, 'bad_param', '%s must be an integer between %d and %d.' % (name, lo, hi), param=name)
    return int(out)


def float_param(value, name, lo, hi):
    out = _num(value, name)
    if not (lo <= out <= hi):
        raise ApiError(400, 'bad_param', '%s must be between %s and %s.' % (name, lo, hi), param=name)
    return out


def port_param(value, helper_port):
    port = int_param(value, 'port', 1024, 65535)
    if port == helper_port:
        raise ApiError(400, 'bad_param', 'port cannot be the platform port (%d).' % helper_port, param='port')
    return port


# ----------------------------------------------------------------------------------------------------------------
# servidor HTTP
# ----------------------------------------------------------------------------------------------------------------

class AtlasHandler(http.server.SimpleHTTPRequestHandler):
    server_version = 'LatentAtlas/' + VERSION
    sys_version = ''
    timeout = HTTP_TIMEOUT   # cliente ocioso ou que manda o corpo pela metade não segura uma thread para sempre

    # ---------------------------------------------------------------------------- logging silencioso

    def log_request(self, code='-', size='-'):
        try:
            status = int(code)
        except (TypeError, ValueError):
            status = 0
        if status >= 400 and not self.server.quiet:
            http.server.SimpleHTTPRequestHandler.log_request(self, code, size)

    def log_error(self, fmt, *args):
        pass  # o log_request acima já registra os >= 400

    def end_headers(self):
        # esta origem tem a API do helper: nenhuma página dela pode ser emoldurada por outro site (clickjacking)
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Content-Security-Policy', "frame-ancestors 'none'")
        if not getattr(self, '_is_api', False):
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
        http.server.SimpleHTTPRequestHandler.end_headers(self)

    # ---------------------------------------------------------------------------- estático

    def _static_host_ok(self):
        """Com a API ligada (loopback), o estático também exige Host local: DNS rebinding não lê o repo."""
        srv = self.server
        if not srv.api_enabled:
            return True  # modo LAN (--host 0.0.0.0, sem API): igual ao http.server
        host = (self.headers.get('Host') or '').strip().lower()
        return host in allowed_hosts(srv.server_address[1])

    def send_head(self):
        if not self._static_host_ok():
            self.send_error(403, 'Forbidden host')
            return None
        raw = urllib.parse.urlsplit(self.path).path
        if raw.startswith('//'):
            self.send_error(404, 'Not found')
            return None
        plain = urllib.parse.unquote(raw)
        segments = plain.split('/')
        if any(seg.startswith('.') for seg in segments if seg) or '\\' in plain or '\x00' in plain:
            self.send_error(404, 'Not found')  # dotfiles, qualquer `..` e NUL
            return None
        try:
            return http.server.SimpleHTTPRequestHandler.send_head(self)
        except ValueError:  # caminho que o SO não aceita
            self.send_error(404, 'Not found')
            return None

    def list_directory(self, path):
        self.send_error(404, 'Not found')
        return None

    # ---------------------------------------------------------------------------- métodos

    def _is_api_path(self):
        p = urllib.parse.urlsplit(self.path).path
        return p == '/api' or p.startswith('/api/')

    def do_GET(self):
        if self._is_api_path():
            return self._api('GET')
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def do_HEAD(self):
        if self._is_api_path():
            return self._api('HEAD')
        return http.server.SimpleHTTPRequestHandler.do_HEAD(self)

    def do_POST(self):
        if self._is_api_path():
            return self._api('POST')
        self.send_error(405, 'Method not allowed')

    def do_OPTIONS(self):
        if self._is_api_path():
            return self._api('OPTIONS')
        self.send_error(405, 'Method not allowed')

    def do_PUT(self):
        return self.do_OPTIONS()

    def do_DELETE(self):
        return self.do_OPTIONS()

    def do_PATCH(self):
        return self.do_OPTIONS()

    # ---------------------------------------------------------------------------- API

    def _json(self, status, payload):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if status == 405:
            self.send_header('Allow', 'GET, POST')
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def _read_body(self):
        return read_json_body(self, MAX_BODY)

    def _api(self, method):
        self._is_api = True
        srv = self.server
        try:
            if not srv.api_enabled:
                raise ApiError(404, 'api_disabled', 'The helper API only runs when bound to 127.0.0.1/localhost.')
            port = srv.server_address[1]
            host = (self.headers.get('Host') or '').strip().lower()
            if host not in allowed_hosts(port):
                raise ApiError(403, 'forbidden_host', 'Unexpected Host header.')
            if method not in ('GET', 'POST'):
                raise ApiError(405, 'method_not_allowed', 'Method not allowed.')
            url = urllib.parse.urlsplit(self.path)
            path = url.path
            query = dict((k, v[0]) for k, v in urllib.parse.parse_qs(url.query, keep_blank_values=True).items())
            if path == '/api/health':
                if method != 'GET':
                    raise ApiError(405, 'method_not_allowed', 'Method not allowed.')
                return self._json(200, {'ok': True, 'app': APP, 'helper': HELPER_API, 'version': VERSION,
                                        'port': port})
            if not path.startswith('/api/mlx/'):
                raise ApiError(404, 'not_found', 'Unknown API endpoint.')
            if self.headers.get('X-Atlas') != '1':
                raise ApiError(403, 'forbidden', 'Missing X-Atlas header.')
            origin = self.headers.get('Origin')
            if origin is not None and origin not in allowed_origins(port):
                raise ApiError(403, 'forbidden_origin', 'Cross-origin requests are not allowed.')
            if method == 'POST' and not json_content_type(self):
                raise ApiError(403, 'forbidden', 'POST requests must be application/json.')
            if path.startswith('/api/mlx/v1/'):
                return self._mlx_v1(method, path[len('/api/mlx'):])
            body = self._read_body() if method == 'POST' else {}
            route = ROUTES.get((method, path))
            if route is None:
                if any(p == path for (_m, p) in ROUTES):
                    raise ApiError(405, 'method_not_allowed', 'Method not allowed.')
                raise ApiError(404, 'not_found', 'Unknown API endpoint.')
            status, payload = route(srv.helper, query, body)
            return self._json(status, payload)
        except ApiError as e:
            return self._safe_json(e.status, e.payload())
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            self.close_connection = True
            return None
        except Exception as e:
            if not srv.quiet:
                traceback.print_exc()
            return self._safe_json(500, {'error': 'internal', 'message': 'Internal error: %s' % e})

    def _mlx_v1(self, method, sub):
        """/api/mlx/v1/* — o mesmo proxy da porta pública, na origem da plataforma (mesma origem, sem CORS)."""
        gw = self.server.helper.gateway()
        if gw is None:
            raise ApiError(409, 'not_running', 'No model is running.')
        if sub == '/v1/models':
            if method != 'GET':
                raise ApiError(405, 'method_not_allowed', 'Method not allowed.')
            return self._json(200, gw.models_payload())
        if sub in ('/v1/chat/completions', '/v1/completions'):
            if method != 'POST':
                raise ApiError(405, 'method_not_allowed', 'Method not allowed.')
            body = gw.sanitize(read_json_body(self, PROXY_MAX_BODY))
            return gw.relay(self, 'POST', sub, json.dumps(body).encode('utf-8'), None)
        raise ApiError(404, 'not_found', 'Unknown API endpoint.')

    def _safe_json(self, status, payload):
        try:
            self._json(status, payload)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            self.close_connection = True


# rotas da API: (método, caminho) → função(helper, query, body) → (status, payload)

def _r_status(h, q, b):
    return 200, h.status()


def _r_logs(h, q, b):
    kind = q.get('kind', 'server')
    if kind not in ('server', 'install', 'download'):
        raise ApiError(400, 'bad_param', 'kind must be server, install or download.', param='kind')
    n = int_param(q.get('n', '200'), 'n', 1, LOG_LINES)
    return 200, {'kind': kind, 'lines': h.logs_for(kind, n)}


def _r_info(h, q, b):
    return 200, h.model_card(q.get('id', ''))


def _r_probe(h, q, b):
    return 200, h.probe(port_param(q.get('port', str(h.server['port'])), h.port))


def _r_install(h, q, b):
    upgrade = b.get('upgrade', False)
    if not isinstance(upgrade, bool):
        raise ApiError(400, 'bad_param', 'upgrade must be a boolean.', param='upgrade')
    return 202, h.start_install(upgrade)


def _r_download(h, q, b):
    return h.start_download(b.get('id'))


def _r_cancel(h, q, b):
    return 200, h.cancel_download()


def _r_start(h, q, b):
    repo_id = b.get('id')
    if not valid_repo_id(repo_id):
        raise ApiError(400, 'bad_id', 'Invalid Hugging Face model id (expected org/name).')
    port = port_param(b.get('port', h.server['port']), h.port)
    max_tokens = int_param(b.get('max_tokens', 2048), 'max_tokens', 16, MAX_GEN_TOKENS)
    temp = float_param(b.get('temp', 0.4), 'temp', 0.0, 2.0)
    return h.start_server(repo_id, port, max_tokens, temp)


def _r_stop(h, q, b):
    return 200, h.stop_server()


def _r_delete(h, q, b):
    return 200, h.delete_model(b.get('id'))


ROUTES = {
    ('GET', '/api/mlx/status'): _r_status,
    ('GET', '/api/mlx/logs'): _r_logs,
    ('GET', '/api/mlx/info'): _r_info,
    ('GET', '/api/mlx/probe'): _r_probe,
    ('POST', '/api/mlx/install'): _r_install,
    ('POST', '/api/mlx/download'): _r_download,
    ('POST', '/api/mlx/download/cancel'): _r_cancel,
    ('POST', '/api/mlx/start'): _r_start,
    ('POST', '/api/mlx/stop'): _r_stop,
    ('POST', '/api/mlx/delete'): _r_delete,
}


class AtlasHTTPServer(BoundedThreadingServer):
    def __init__(self, address, handler, helper, api_enabled, quiet=False):
        self.helper = helper
        self.api_enabled = api_enabled
        self.quiet = quiet
        BoundedThreadingServer.__init__(self, address, handler)


class AtlasHTTPServer6(AtlasHTTPServer):
    address_family = socket.AF_INET6


def make_server(host='127.0.0.1', port=8765, root=ROOT, helper=None, quiet=False, tries=11, api=None):
    """Cria o servidor (tenta as próximas `tries - 1` portas se a pedida estiver ocupada)."""
    bind_host = '127.0.0.1' if host == 'localhost' else host
    cls = AtlasHTTPServer6 if ':' in bind_host else AtlasHTTPServer
    handler = functools.partial(AtlasHandler, directory=os.path.abspath(root))
    api_enabled = is_loopback_host(host) if api is None else bool(api)
    helper = helper or Helper()
    candidates = [0] if port == 0 else list(range(port, min(port + tries, 65536)))
    last_error = None
    for p in candidates:
        if p and port_listening(p, host=bind_host, timeout=0.3):
            last_error = OSError(errno.EADDRINUSE, 'port %d busy' % p)
            continue
        try:
            httpd = cls((bind_host, p), handler, helper=helper, api_enabled=api_enabled, quiet=quiet)
        except OSError as e:
            if e.errno in (errno.EADDRINUSE, errno.EACCES):
                last_error = e
                continue
            raise
        helper.port = httpd.server_address[1]
        return httpd
    raise last_error or OSError(errno.EADDRINUSE, 'no free port')


def _open_browser(url):
    try:
        if sys.platform == 'darwin' and shutil.which('open'):
            subprocess.Popen(['open', url], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        else:
            import webbrowser
            webbrowser.open(url)
    except Exception:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description='Latent Atlas local server + MLX copilot helper')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--open', action='store_true', help='open the browser after binding')
    ap.add_argument('--mlx-port', type=int, default=8080)
    ap.add_argument('--quiet', action='store_true', help='no request logging')
    args = ap.parse_args(argv)

    helper = Helper(mlx_port=args.mlx_port)
    try:
        httpd = make_server(args.host, args.port, args.root, helper, quiet=args.quiet)
    except OSError as e:
        print('✕ Could not bind %s:%d (+10 ports): %s' % (args.host, args.port, e), file=sys.stderr)
        return 1
    port = httpd.server_address[1]
    detector = helper.start_detection() if httpd.api_enabled else None
    threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': 0.5}, name='http', daemon=True).start()

    stop = threading.Event()

    def on_signal(signum, frame):
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, 'SIGHUP', None)):
        if sig is not None:
            try:
                signal.signal(sig, on_signal)
            except (ValueError, OSError):
                pass
    atexit.register(helper.shutdown)

    url = 'http://localhost:%d' % port
    print('Latent Atlas → %s' % url + ('' if port == args.port else '   (port %d was busy)' % args.port),
          flush=True)
    if args.open:
        _open_browser(url)
    if detector is not None:
        detector.join(8)
        with helper.lock:
            mlx = dict(helper.mlx)
        if mlx['installed']:
            print('MLX helper → mlx-lm %s found' % (mlx['version'] or '(version unknown)'), flush=True)
        elif mlx['detecting']:
            print('MLX helper → still looking for mlx-lm…', flush=True)
        else:
            print('MLX helper → mlx-lm not installed (install it from the MLX button)', flush=True)
    else:
        print('MLX helper → disabled (the API only runs on 127.0.0.1 / localhost)', flush=True)
    print('Ctrl+C stops everything (including the MLX server)', flush=True)
    try:
        while not stop.wait(0.5):
            pass
    finally:
        print('\nStopping…', flush=True)
        httpd.shutdown()      # 1º para de aceitar pedidos (um /start no meio do caminho cai no shutting_down)
        helper.shutdown()
        httpd.server_close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
