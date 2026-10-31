#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Testes do helper local (tools/atlas_server.py) — só stdlib, roda no python3 3.9 do macOS.

  python3 tools/test_atlas_server.py

Usa um hub falso (ThreadingHTTPServer local) no lugar do Hugging Face, um "CDN" falso em outra porta (para o
redirect dos arquivos LFS) e um mlx_lm.server falso — nada sai da máquina e nada é escrito fora de pastas
temporárias.
"""

import functools
import hashlib
import http.client
import http.server
import io
import json
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.parse

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlas_server as A  # noqa: E402

COMMIT = 'c0ffee1234567890abcdef1234567890abcdef12'
REPO = 'org/model'
MOE_REPO = 'org/moe'


def git_oid(data):
    return hashlib.sha1(b'blob %d\x00' % len(data) + data).hexdigest()


def free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_for(pred, timeout=20.0, step=0.05):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(step)
    return False


# ----------------------------------------------------------------------------------------------------------------
# hub falso
# ----------------------------------------------------------------------------------------------------------------

class FakeHub(object):
    """API /api/models/<id>/revision/main + /<id>/resolve/<commit>/<arquivo>.
    Arquivos normais: 302 relativo para /api/resolve-cache/… (mesmo host). LFS: 302 absoluto para o CDN (outra porta)."""

    def __init__(self):
        rnd = random.Random(7)
        weights = bytes(rnd.getrandbits(8) for _ in range(4 * 1024 * 1024 + 123))
        cfg = {'model_type': 'qwen3', 'hidden_size': 64, 'quantization': {'bits': 4, 'group_size': 64}}
        self.repos = {
            REPO: {
                'config.json': (json.dumps(cfg).encode(), False),
                'tokenizer.json': (b'{"version": "1.0", "model": {"type": "BPE"}}', False),
                'model.safetensors': (weights, True),
                'sub/x.json': (b'{"x": 1}', False),
                'README.md': (b'# not downloaded', False),
            },
            MOE_REPO: {
                'config.json': (json.dumps({'model_type': 'qwen3_moe', 'num_experts': 128,
                                            'quantization_config': {'bits': 8}}).encode(), False),
                'model.safetensors': (b'\x01' * 1000, True),
                'tokenizer.json': (b'{}', False),
            },
        }
        self.lock = threading.Lock()
        self.requests = []          # (quem, método, caminho, headers)
        self.throttle = 0.0         # pausa por pedaço de 64 KiB no CDN
        self.corrupt = False        # CDN troca um byte → sha256 não bate
        self.ignore_range = False   # CDN responde 200 a pedidos com Range
        self.drop_once = set()      # nomes LFS cujo 1º envio corta no meio
        self.hub = self._serve('hub')
        self.cdn = self._serve('cdn')
        self.hub_port = self.hub.server_address[1]
        self.cdn_port = self.cdn.server_address[1]
        self.endpoint = 'http://127.0.0.1:%d' % self.hub_port

    def close(self):
        for srv in (self.hub, self.cdn):
            srv.shutdown()
            srv.server_close()

    def log(self, who):
        with self.lock:
            return [r for r in self.requests if r[0] == who]

    def sha(self, repo, name):
        return hashlib.sha256(self.repos[repo][name][0]).hexdigest()

    def api(self, repo):
        sib = []
        for name, (data, lfs) in sorted(self.repos[repo].items()):
            entry = {'rfilename': name, 'size': len(data), 'blobId': git_oid(data if not lfs else b'ptr' + name.encode())}
            if lfs:
                entry['lfs'] = {'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data), 'pointerSize': 130}
                entry['size'] = len(data)
            sib.append(entry)
        return {'id': repo, 'sha': COMMIT, 'siblings': sib, 'config': {'model_type': 'qwen3'},
                'library_name': 'mlx', 'pipeline_tag': 'text-generation', 'tags': ['mlx', '4-bit'],
                'gated': False, 'downloads': 10, 'likes': 2, 'lastModified': '2026-01-01T00:00:00.000Z',
                'safetensors': {'total': 1234}}

    def _serve(self, who):
        hub = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *a):
                pass

            def _send(self, code, body=b'', headers=None):
                self.send_response(code)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                with hub.lock:
                    hub.requests.append((who, 'GET', self.path, dict(self.headers.items())))
                path = urllib.parse.urlsplit(self.path).path
                if who == 'cdn':
                    m = re.match(r'^/xet/([0-9a-f]{64})$', path)
                    for repo, files in hub.repos.items():
                        for name, (data, lfs) in files.items():
                            if lfs and m and hashlib.sha256(data).hexdigest() == m.group(1):
                                return self._bytes(data, name, cdn=True)
                    return self._send(404, b'nope')
                m = re.match(r'^/api/models/([^/]+/[^/]+)/revision/main$', path)
                if m:
                    repo = m.group(1)
                    if repo not in hub.repos:
                        return self._send(404, b'{"error":"Repository not found"}')
                    return self._send(200, json.dumps(hub.api(repo)).encode(), {'Content-Type': 'application/json'})
                m = re.match(r'^/([^/]+/[^/]+)/resolve/([0-9a-f]+)/(.+)$', path)
                if m:
                    repo, commit, name = m.group(1), m.group(2), urllib.parse.unquote(m.group(3))
                    if repo not in hub.repos or name not in hub.repos[repo] or commit != COMMIT:
                        return self._send(404, b'not found')
                    data, lfs = hub.repos[repo][name]
                    if lfs:
                        loc = 'http://127.0.0.1:%d/xet/%s' % (hub.cdn_port, hashlib.sha256(data).hexdigest())
                    else:
                        loc = '/api/resolve-cache/models/%s/%s/%s' % (repo, commit, urllib.parse.quote(name))
                    return self._send(302, b'', {'Location': loc})
                m = re.match(r'^/api/resolve-cache/models/([^/]+/[^/]+)/([0-9a-f]+)/(.+)$', path)
                if m:
                    repo, name = m.group(1), urllib.parse.unquote(m.group(3))
                    if repo in hub.repos and name in hub.repos[repo]:
                        return self._bytes(hub.repos[repo][name][0], name)
                return self._send(404, b'not found')

            def _bytes(self, data, name, cdn=False):
                start = 0
                rng = self.headers.get('Range')
                headers = {'Content-Type': 'application/octet-stream'}
                if rng and not (cdn and hub.ignore_range):
                    start = int(re.match(r'^bytes=(\d+)-$', rng).group(1))
                    if start >= len(data):
                        return self._send(416, b'', {'Content-Range': 'bytes */%d' % len(data)})
                    code = 206
                    headers['Content-Range'] = 'bytes %d-%d/%d' % (start, len(data) - 1, len(data))
                else:
                    code = 200
                body = data[start:]
                if cdn and hub.corrupt and body:
                    body = bytes([body[0] ^ 0xFF]) + body[1:]
                drop = False
                with hub.lock:
                    if cdn and name in hub.drop_once:
                        hub.drop_once.discard(name)
                        drop = True
                self.send_response(code)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                limit = len(body) // 2 if drop else len(body)
                pos = 0
                try:
                    while pos < limit:
                        piece = body[pos:min(limit, pos + 65536)]
                        self.wfile.write(piece)
                        pos += len(piece)
                        if cdn and hub.throttle:
                            time.sleep(hub.throttle)
                except (BrokenPipeError, ConnectionResetError):
                    return
                if drop:
                    self.wfile.flush()
                    self.close_connection = True
                    try:
                        self.connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass

        srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, kwargs={'poll_interval': 0.1}, daemon=True).start()
        return srv


# ----------------------------------------------------------------------------------------------------------------
# mlx_lm.server falso
# ----------------------------------------------------------------------------------------------------------------

FAKE_MLX = r'''#!%(python)s
import argparse, json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
ap = argparse.ArgumentParser()
for flag in ("--model", "--host", "--port", "--max-tokens", "--temp", "--allowed-origins"):
    ap.add_argument(flag)
args, rest = ap.parse_known_args()
RECORD = os.environ["FAKE_MLX_RECORD"]
with open(RECORD, "w") as fh:
    json.dump({"argv": sys.argv[1:], "cwd": os.getcwd(), "pid": os.getpid(),
               "env": {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "HF_HUB_CACHE", "HF_HUB_DISABLE_TELEMETRY",
                                                       "PYTHONUNBUFFERED")}}, fh)
mode = os.environ.get("FAKE_MLX_MODE", "ok")
print("fake mlx_lm.server starting", flush=True)
if mode == "crash":
    print("Traceback: boom, model failed to load", flush=True)
    sys.exit(3)
if mode == "spam":
    sys.stdout.write("x" * (3 * 1024 * 1024) + "\n")
    print("after the long line", flush=True)
if mode == "dead":
    print("2026-10-02 00:38:07,100 - ERROR - mlx_lm.server generation thread died: "
          "Model type weirdnet not supported.", flush=True)
origins = (args.allowed_origins or "").split(",")
class H(BaseHTTPRequestHandler):
    # log de acesso padrão (stderr), como o mlx_lm.server: '"GET /health HTTP/1.1" 200 -'
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if self.headers.get("Origin") in origins:
            self.send_header("Access-Control-Allow-Origin", self.headers.get("Origin"))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path == "/health":
            if mode in ("dead", "dead_quiet"):
                return self._send(503, {"status": "unavailable"})
            if mode == "nohealth":
                return self._send(404, {})
            return self._send(200, {"status": "ok"})
        if self.path.startswith("/v1/models"):
            return self._send(200, {"object": "list", "data": [{"id": "everything/in-the-cache", "object": "model"},
                                                                {"id": args.model, "object": "model"}]})
        self._send(404, {})
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        with open(RECORD + ".posts", "a") as fh:
            fh.write(json.dumps({"path": self.path, "body": body, "key": self.headers.get("X-Atlas-Key")}) + "\n")
        if mode in ("dead", "dead_quiet"):
            return self._send(500, {"error": "generation thread died"})
        if mode == "no_kwargs" and "chat_template_kwargs" in body:
            return self._send(400, {"error": "unexpected keyword chat_template_kwargs"})
        time.sleep(0.05)
        toks = int(body.get("max_tokens") or 8)
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for piece in ("Hel", "lo"):
                chunk = {"choices": [{"delta": {"content": piece}}]}
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.flush()
                time.sleep(0.05)
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self._send(200, {"choices": [{"message": {"role": "assistant", "content": "hi"}}],
                         "usage": {"prompt_tokens": 5, "completion_tokens": toks, "total_tokens": toks + 5}})
ThreadingHTTPServer((args.host, int(args.port)), H).serve_forever()
'''


def http_call(port, method, path, body=None, headers=None, host=True, timeout=15):
    """Pedido HTTP cru → (status, headers, corpo em bytes). host=True manda o Host local padrão."""
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    hdrs = {}
    if host is True:
        hdrs['Host'] = '127.0.0.1:%d' % port
    elif host:
        hdrs['Host'] = host
    hdrs.update(headers or {})
    if isinstance(body, (dict, list)):
        body = json.dumps(body).encode()
    try:
        conn.request(method, path, body=body, headers=hdrs)
        resp = conn.getresponse()
        return resp.status, resp.headers, resp.read()
    finally:
        conn.close()


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def make_local_model(cache, repo_id, complete=True, tokenizer=True, broken_link=False, incomplete_blob=False,
                     index_missing_shard=False, refs=True):
    """Monta à mão um modelo no layout do cache do HF."""
    repo_dir = os.path.join(cache, A.repo_folder_name(repo_id))
    blobs = os.path.join(repo_dir, 'blobs')
    snap = os.path.join(repo_dir, 'snapshots', COMMIT)
    os.makedirs(blobs, exist_ok=True)
    os.makedirs(snap, exist_ok=True)
    files = {'config.json': b'{"model_type": "qwen3"}', 'model.safetensors': b'w' * 2048}
    if tokenizer:
        files['tokenizer.json'] = b'{}'
    if index_missing_shard:
        files['model.safetensors.index.json'] = json.dumps(
            {'weight_map': {'a': 'model.safetensors', 'b': 'model-00002-of-00002.safetensors'}}).encode()
    for name, data in files.items():
        etag = git_oid(data)
        with open(os.path.join(blobs, etag), 'wb') as fh:
            fh.write(data)
        os.symlink(os.path.join('..', '..', 'blobs', etag), os.path.join(snap, name))
    if broken_link:
        os.symlink('../../blobs/' + 'f' * 40, os.path.join(snap, 'special_tokens_map.json'))
    if incomplete_blob:
        with open(os.path.join(blobs, 'e' * 64 + '.incomplete'), 'wb') as fh:
            fh.write(b'partial')
    if refs:
        os.makedirs(os.path.join(repo_dir, 'refs'), exist_ok=True)
        with open(os.path.join(repo_dir, 'refs', 'main'), 'w') as fh:
            fh.write(COMMIT)
    return repo_dir


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='atlas-test-')
        self.cache = os.path.join(self.tmp, 'hf-cache')
        os.makedirs(self.cache)
        self.env = dict(os.environ)
        self.env['LA_MLX_SERVER'] = os.path.join(self.tmp, 'missing', 'mlx_lm.server')
        for k in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HF_HUB_OFFLINE'):
            self.env.pop(k, None)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def helper(self, endpoint='http://127.0.0.1:9', token='', **kw):
        h = A.Helper(port=kw.pop('port', 0), cache_dir=self.cache, endpoint=endpoint, token=token, env=self.env, **kw)
        self.addCleanup(h.shutdown)
        return h


class HubBase(Base):
    def setUp(self):
        Base.setUp(self)
        self.hub = FakeHub()
        self.addCleanup(self.hub.close)

    def download(self, h, repo=REPO, timeout=30):
        code, _ = h.start_download(repo)
        self.assertIn(code, (200, 202))
        self.assertTrue(wait_for(lambda: h.download['state'] != 'running', timeout), 'download did not finish')
        return h.download


# ----------------------------------------------------------------------------------------------------------------
# validação
# ----------------------------------------------------------------------------------------------------------------

class ValidationTests(unittest.TestCase):
    def test_repo_ids(self):
        for ok in ('mlx-community/Qwen3-0.6B-4bit', 'a/b', 'Org_1/model.v2', 'x' * 96 + '/' + 'y' * 96):
            self.assertTrue(A.valid_repo_id(ok), ok)
        bad = ['', 'a', 'a/', '/b', 'a/b/c', '-a/b', 'a/-b', '.a/b', 'a/.b', 'a--b/c', 'a/b--c', 'a/b..c', '../x',
               'a/..', 'a b/c', 'a/b\n', 'a/b?x=1', 'á/b', 'x' * 97 + '/b', 'a/' + 'y' * 97, None, 42, 'a\\b/c']
        for b in bad:
            self.assertFalse(A.valid_repo_id(b), repr(b))

    def test_safe_paths(self):
        for ok in ('config.json', 'sub/x.json', 'a/b/c.txt'):
            self.assertTrue(A.safe_rel_path(ok), ok)
        for b in ('', '/etc/passwd', '../x.json', 'a/../b.json', 'a/./b', 'a//b', 'a\\b.json', 'a\x00.json', '..'):
            self.assertFalse(A.safe_rel_path(b), repr(b))

    def test_allow_patterns(self):
        for ok in ('config.json', 'sub/x.json', 'model.safetensors', 'model-00001-of-00002.safetensors',
                   'tokenizer.model', 'merges.txt', 'chat_template.jinja', 'o200k.tiktoken', 'modeling.py'):
            self.assertTrue(A.allowed_file(ok), ok)
        for b in ('README.md', '.gitattributes', 'consolidated.safetensors', 'pytorch_model.bin', 'x.gguf'):
            self.assertFalse(A.allowed_file(b), b)

    def test_params(self):
        self.assertEqual(A.port_param(8080, 8765), 8080)
        self.assertEqual(A.port_param('8081', 8765), 8081)
        for bad in (80, 1023, 65536, 8765, 'abc', None, True, 8080.5, float('nan')):
            with self.assertRaises(A.ApiError):
                A.port_param(bad, 8765)
        self.assertEqual(A.int_param(2048, 'max_tokens', 16, 32768), 2048)
        for bad in (15, 32769, 'x'):
            with self.assertRaises(A.ApiError):
                A.int_param(bad, 'max_tokens', 16, 32768)
        self.assertEqual(A.float_param(0.4, 'temp', 0, 2), 0.4)
        for bad in (-0.1, 2.1, float('inf'), '1e9'):
            with self.assertRaises(A.ApiError):
                A.float_param(bad, 'temp', 0, 2)

    def test_hosts_origins(self):
        self.assertEqual(A.allowed_hosts(8765), ('127.0.0.1:8765', 'localhost:8765', '[::1]:8765'))
        self.assertIn('http://[::1]:8765', A.allowed_origins(8765))
        self.assertTrue(A.is_loopback_host('127.0.0.1'))
        self.assertFalse(A.is_loopback_host('0.0.0.0'))

    def test_find_python_for_trampolines(self):
        tmp = tempfile.mkdtemp()
        try:
            py = sys.executable
            tramp = os.path.join(tmp, 'tramp')
            with open(tramp, 'w') as fh:
                fh.write("#!/bin/sh\n'''exec' '%s' \"$0\" \"$@\"\n' '''\nimport sys\n" % py)
            shebang = os.path.join(tmp, 'direct')
            with open(shebang, 'w') as fh:
                fh.write('#!%s\nprint(1)\n' % py)
            for p in (tramp, shebang):
                os.chmod(p, 0o755)
            link = os.path.join(tmp, 'link')
            os.symlink(tramp, link)
            self.assertEqual(A.find_python_for(link), py)
            self.assertEqual(A.find_python_for(shebang), py)
        finally:
            shutil.rmtree(tmp)

    def test_cache_dir_and_token_env(self):
        self.assertEqual(A.default_cache_dir({'HF_HUB_CACHE': '/x/y'}), '/x/y')
        self.assertEqual(A.default_cache_dir({'HF_HOME': '/h'}), '/h/hub')
        self.assertEqual(A.default_token({'HF_TOKEN': ' t1 '}), 't1')
        tmp = tempfile.mkdtemp()
        try:
            with open(os.path.join(tmp, 'token'), 'w') as fh:
                fh.write('t2\n')
            self.assertEqual(A.default_token({'HF_HOME': tmp}), 't2')
        finally:
            shutil.rmtree(tmp)


# ----------------------------------------------------------------------------------------------------------------
# download / cache
# ----------------------------------------------------------------------------------------------------------------

class DownloadTests(HubBase):
    def test_cache_layout(self):
        h = self.helper(self.hub.endpoint)
        st = self.download(h)
        self.assertEqual(st['state'], 'done', st)
        repo_dir = os.path.join(self.cache, 'models--org--model')
        files = self.hub.repos[REPO]
        # refs/main = commit, sem quebra de linha
        with open(os.path.join(repo_dir, 'refs', 'main'), 'rb') as fh:
            self.assertEqual(fh.read(), COMMIT.encode())
        snap = os.path.join(repo_dir, 'snapshots', COMMIT)
        # blob LFS nomeado pelo sha256; arquivos normais pelo oid git
        self.assertTrue(os.path.isfile(os.path.join(repo_dir, 'blobs', self.hub.sha(REPO, 'model.safetensors'))))
        self.assertTrue(os.path.isfile(os.path.join(repo_dir, 'blobs', git_oid(files['config.json'][0]))))
        for name in ('config.json', 'tokenizer.json', 'model.safetensors', 'sub/x.json'):
            p = os.path.join(snap, name)
            self.assertTrue(os.path.islink(p), name)
            target = os.readlink(p)
            self.assertFalse(os.path.isabs(target), target)
            depth = name.count('/') + 2
            self.assertTrue(target.startswith('../' * depth + 'blobs/'), target)
            with open(p, 'rb') as fh:
                self.assertEqual(fh.read(), files[name][0])
        self.assertFalse(os.path.lexists(os.path.join(snap, 'README.md')))
        self.assertEqual([n for n in os.listdir(os.path.join(repo_dir, 'blobs')) if n.endswith('.incomplete')], [])
        total = sum(len(d) for n, (d, _l) in files.items() if n != 'README.md')
        self.assertEqual(st['total_bytes'], total)
        self.assertEqual(st['done_bytes'], total)
        self.assertEqual(st['files_total'], 4)
        self.assertEqual(st['files_done'], 4)
        local = h.local_entry(REPO, force=True)
        self.assertTrue(local['complete'])
        self.assertEqual(local['bytes'], total)
        self.assertEqual(local['commit'], COMMIT)
        # um 2º download do mesmo modelo não baixa nada de novo
        before = len(self.hub.log('cdn'))
        self.assertEqual(self.download(h)['state'], 'done')
        self.assertEqual(len(self.hub.log('cdn')), before)

    def test_resume_from_partial(self):
        h = self.helper(self.hub.endpoint)
        data = self.hub.repos[REPO]['model.safetensors'][0]
        blobs = os.path.join(self.cache, 'models--org--model', 'blobs')
        os.makedirs(blobs)
        part = os.path.join(blobs, self.hub.sha(REPO, 'model.safetensors') + '.incomplete')
        with open(part, 'wb') as fh:
            fh.write(data[:1000000])
        st = self.download(h)
        self.assertEqual(st['state'], 'done', st)
        ranges = [r[3].get('Range') for r in self.hub.log('cdn')]
        self.assertEqual(ranges, ['bytes=1000000-'])
        self.assertTrue(h.local_entry(REPO, force=True)['complete'])
        with open(os.path.join(blobs, self.hub.sha(REPO, 'model.safetensors')), 'rb') as fh:
            self.assertEqual(fh.read(), data)

    def test_partial_already_complete(self):
        # .incomplete com o arquivo inteiro (p.ex. cancelado logo antes do rename): verifica e renomeia sem baixar
        h = self.helper(self.hub.endpoint)
        data = self.hub.repos[REPO]['model.safetensors'][0]
        blobs = os.path.join(self.cache, 'models--org--model', 'blobs')
        os.makedirs(blobs)
        with open(os.path.join(blobs, self.hub.sha(REPO, 'model.safetensors') + '.incomplete'), 'wb') as fh:
            fh.write(data)
        st = self.download(h)
        self.assertEqual(st['state'], 'done', st)
        self.assertEqual(self.hub.log('cdn'), [])
        self.assertTrue(h.local_entry(REPO, force=True)['complete'])

    def test_server_ignores_range_restarts(self):
        h = self.helper(self.hub.endpoint)
        data = self.hub.repos[REPO]['model.safetensors'][0]
        blobs = os.path.join(self.cache, 'models--org--model', 'blobs')
        os.makedirs(blobs)
        with open(os.path.join(blobs, self.hub.sha(REPO, 'model.safetensors') + '.incomplete'), 'wb') as fh:
            fh.write(data[:500000])
        self.hub.ignore_range = True
        st = self.download(h)
        self.assertEqual(st['state'], 'done', st)
        self.assertEqual(st['done_bytes'], st['total_bytes'])
        self.assertTrue(h.local_entry(REPO, force=True)['complete'])

    def test_retry_after_dropped_connection(self):
        h = self.helper(self.hub.endpoint)
        self.hub.drop_once.add('model.safetensors')
        st = self.download(h)
        self.assertEqual(st['state'], 'done', st)
        cdn = self.hub.log('cdn')
        self.assertEqual(len(cdn), 2)
        self.assertIsNone(cdn[0][3].get('Range'))
        self.assertRegex(cdn[1][3].get('Range') or '', r'^bytes=\d+-$')
        self.assertTrue(h.local_entry(REPO, force=True)['complete'])

    def test_cancel_keeps_partial_and_resumes(self):
        h = self.helper(self.hub.endpoint)
        self.hub.throttle = 0.03
        code, _ = h.start_download(REPO)
        self.assertEqual(code, 202)
        part = os.path.join(self.cache, 'models--org--model', 'blobs',
                            self.hub.sha(REPO, 'model.safetensors') + '.incomplete')
        self.assertTrue(wait_for(lambda: os.path.isfile(part) and os.path.getsize(part) >= 1 << 20, 20))
        st = h.cancel_download(wait=10)
        self.assertEqual(st['download']['state'], 'canceled', st['download'])
        self.assertTrue(os.path.isfile(part))
        kept = os.path.getsize(part)
        self.assertGreater(kept, 0)
        self.assertFalse(os.path.exists(os.path.join(self.cache, 'models--org--model', 'refs', 'main')))
        self.assertFalse(h.local_entry(REPO, force=True)['complete'])
        self.hub.throttle = 0
        st = self.download(h)
        self.assertEqual(st['state'], 'done', st)
        self.assertEqual(self.hub.log('cdn')[-1][3].get('Range'), 'bytes=%d-' % kept)
        self.assertTrue(h.local_entry(REPO, force=True)['complete'])

    def test_sha_mismatch(self):
        h = self.helper(self.hub.endpoint)
        self.hub.corrupt = True
        st = self.download(h)
        self.assertEqual(st['state'], 'error')
        self.assertEqual(st['error_code'], 'sha_mismatch')
        self.assertIn('model.safetensors', st['error'])
        blobs = os.path.join(self.cache, 'models--org--model', 'blobs')
        self.assertEqual([n for n in os.listdir(blobs) if n.startswith(self.hub.sha(REPO, 'model.safetensors'))], [])
        self.assertFalse(os.path.exists(os.path.join(self.cache, 'models--org--model', 'refs', 'main')))

    def test_busy_and_no_space(self):
        h = self.helper(self.hub.endpoint)
        self.hub.throttle = 0.03
        h.start_download(REPO)
        with self.assertRaises(A.ApiError) as cm:
            h.start_download(MOE_REPO)
        self.assertEqual((cm.exception.status, cm.exception.code), (409, 'busy'))
        self.assertEqual(h.start_download(REPO)[0], 200)  # mesmo id: no-op
        with self.assertRaises(A.ApiError) as cm:
            h.delete_model(REPO)
        self.assertEqual(cm.exception.code, 'busy')
        h.cancel_download(wait=10)
        orig = A.disk_free
        A.disk_free = lambda path: 1000
        try:
            with self.assertRaises(A.ApiError) as cm:
                h.start_download(MOE_REPO)
        finally:
            A.disk_free = orig
        e = cm.exception
        self.assertEqual((e.status, e.code), (409, 'no_space'))
        self.assertEqual(e.extra['free_bytes'], 1000)
        self.assertGreater(e.extra['need_bytes'], A.GIB)

    def test_unsafe_names_rejected(self):
        h = self.helper(self.hub.endpoint)
        self.hub.repos[REPO]['../evil.json'] = (b'{}', False)
        with self.assertRaises(A.ApiError) as cm:
            h.start_download(REPO)
        self.assertEqual(cm.exception.code, 'unsafe_path')
        self.assertFalse(os.path.exists(os.path.join(self.cache, 'evil.json')))
        card = h.model_card(REPO)  # info só ignora o nome perigoso
        self.assertNotIn('../evil.json', [f['name'] for f in card['files']])

    def test_authorization_not_forwarded_cross_host(self):
        h = self.helper(self.hub.endpoint, token='hf_secret_token')
        self.assertEqual(self.download(h)['state'], 'done')
        hub_reqs = self.hub.log('hub')
        self.assertTrue(hub_reqs)
        for r in hub_reqs:  # API, resolve e resolve-cache (mesmo host) levam o token
            self.assertEqual(r[3].get('Authorization'), 'Bearer hf_secret_token', r[2])
        self.assertTrue(any('/api/resolve-cache/' in r[2] for r in hub_reqs))
        cdn_reqs = self.hub.log('cdn')
        self.assertTrue(cdn_reqs)
        for r in cdn_reqs:  # outro host:porta → nunca
            self.assertNotIn('Authorization', r[3])
        for r in hub_reqs + cdn_reqs:
            self.assertEqual(r[3].get('User-Agent'), A.USER_AGENT)
        self.assertNotIn('hf_secret_token', json.dumps(h.status()))
        self.assertTrue(h.status()['hf']['token'])

    def test_model_card(self):
        h = self.helper(self.hub.endpoint)
        card = h.model_card(REPO)
        names = sorted(f['name'] for f in card['files'])
        self.assertEqual(names, ['config.json', 'model.safetensors', 'sub/x.json', 'tokenizer.json'])
        self.assertEqual(card['download_bytes'], sum(f['size'] for f in card['files']))
        self.assertEqual(card['sha'], COMMIT)
        self.assertEqual(card['model_type'], 'qwen3')
        self.assertEqual(card['quant_bits'], 4)        # do config.json de verdade
        self.assertFalse(card['moe'])
        self.assertIsNone(card['supported'])           # mlx-lm "não instalado" neste teste
        self.assertEqual(card['library'], 'mlx')
        self.assertEqual(card['params'], 1234)
        self.assertEqual(card['local'], {'bytes_present': 0, 'complete': False})
        moe = h.model_card(MOE_REPO)
        self.assertTrue(moe['moe'])
        self.assertEqual(moe['quant_bits'], 8)
        with h.lock:
            h.mlx.update({'installed': True, 'model_types': ['qwen3', 'llama'], 'remap': {'qwen3_moe': 'llama'}})
        self.assertTrue(h.model_card(REPO)['supported'])
        self.assertEqual(h.model_card(MOE_REPO)['mapped_type'], 'llama')
        with h.lock:
            h.mlx['model_types'] = ['llama']
        self.assertFalse(h.model_card(REPO)['supported'])
        with self.assertRaises(A.ApiError) as cm:
            h.model_card('org/missing')
        self.assertEqual((cm.exception.status, cm.exception.code), (404, 'not_found'))
        down = self.helper('http://127.0.0.1:%d' % free_port())
        with self.assertRaises(A.ApiError) as cm:
            down.model_card(REPO)
        self.assertEqual((cm.exception.status, cm.exception.code), (502, 'hf_unreachable'))


class LocalScanTests(Base):
    def test_complete_detection(self):
        make_local_model(self.cache, 'a/ok')
        make_local_model(self.cache, 'a/notok', tokenizer=False)
        make_local_model(self.cache, 'a/norefs', refs=False)
        make_local_model(self.cache, 'a/broken', broken_link=True)
        make_local_model(self.cache, 'a/partial', incomplete_blob=True)
        make_local_model(self.cache, 'a/shard', index_missing_shard=True)
        os.makedirs(os.path.join(self.cache, 'datasets--x--y', 'blobs'))
        os.makedirs(os.path.join(self.cache, '.locks'))
        h = self.helper()
        got = dict((m['id'], m['complete']) for m in h.local_models(force=True))
        self.assertEqual(got, {'a/ok': True, 'a/notok': False, 'a/norefs': False, 'a/broken': False,
                               'a/partial': False, 'a/shard': False})
        ok = h.local_entry('a/ok')
        self.assertEqual(ok['commit'], COMMIT)
        self.assertGreater(ok['bytes'], 2048)

    def test_delete_safety(self):
        h = self.helper()
        make_local_model(self.cache, 'a/ok')
        outside = os.path.join(self.tmp, 'outside')
        os.makedirs(outside)
        with open(os.path.join(outside, 'keep.txt'), 'w') as fh:
            fh.write('keep')
        os.symlink(outside, os.path.join(self.cache, 'models--evil--link'))
        with self.assertRaises(A.ApiError) as cm:
            h.delete_model('evil/link')
        self.assertEqual(cm.exception.code, 'not_a_directory')
        self.assertTrue(os.path.isfile(os.path.join(outside, 'keep.txt')))
        for bad in ('../x', 'a/../../etc', 'a--b/c', 'a/b/c'):
            with self.assertRaises(A.ApiError) as cm:
                h.delete_model(bad)
            self.assertEqual(cm.exception.code, 'bad_id')
        with self.assertRaises(A.ApiError) as cm:
            h.delete_model('a/none')
        self.assertEqual(cm.exception.code, 'not_found')
        with h.lock:
            h.server.update({'state': 'ready', 'id': 'a/ok'})
        with self.assertRaises(A.ApiError) as cm:
            h.delete_model('a/ok')
        self.assertEqual(cm.exception.code, 'busy')
        with h.lock:
            h.server.update({'state': 'stopped', 'id': None})
        out = h.delete_model('a/ok')
        self.assertGreater(out['freed_bytes'], 2048)
        self.assertFalse(os.path.exists(os.path.join(self.cache, 'models--a--ok')))
        self.assertTrue(os.path.islink(os.path.join(self.cache, 'models--evil--link')))
        self.assertEqual(h.local_entry('a/ok', force=True), None)


# ----------------------------------------------------------------------------------------------------------------
# API HTTP: Host / Origin / X-Atlas, estático
# ----------------------------------------------------------------------------------------------------------------

class HttpBase(Base):
    def setUp(self):
        Base.setUp(self)
        self.root = os.path.join(self.tmp, 'site')
        os.makedirs(os.path.join(self.root, 'assets'))
        with open(os.path.join(self.root, 'index.html'), 'w') as fh:
            fh.write('<!doctype html><title>atlas</title>')
        with open(os.path.join(self.root, '.hidden'), 'w') as fh:
            fh.write('hidden')
        with open(os.path.join(self.tmp, 'secret.txt'), 'w') as fh:
            fh.write('TOP-SECRET')
        self.h = self.helper()
        self.httpd = A.make_server('127.0.0.1', 0, self.root, self.h, quiet=True)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, kwargs={'poll_interval': 0.1}, daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def req(self, method, path, body=None, headers=None, host=True, json_ct=True):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        hdrs = {}
        if host is True:
            hdrs['Host'] = '127.0.0.1:%d' % self.port
        elif host:
            hdrs['Host'] = host
        if body is not None:
            body = body if isinstance(body, bytes) else json.dumps(body).encode()
            if json_ct:
                hdrs['Content-Type'] = 'application/json'
        hdrs.update(headers or {})
        if host is None:
            conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
            for k, v in hdrs.items():
                conn.putheader(k, v)
            conn.endheaders(body)
        else:
            conn.request(method, path, body=body, headers=hdrs)
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        for k in resp.headers.keys():
            self.assertFalse(k.lower().startswith('access-control-'), k)
        try:
            payload = json.loads(data.decode()) if data else None
        except ValueError:
            payload = data
        return resp.status, payload, resp.headers

    def mlx(self, method, path, body=None, **kw):
        headers = dict(kw.pop('headers', {}))
        headers.setdefault('X-Atlas', '1')
        return self.req(method, path, body=body, headers=headers, **kw)


class HttpTests(HttpBase):
    def test_health(self):
        st, data, _ = self.req('GET', '/api/health')
        self.assertEqual(st, 200)
        self.assertEqual(data, {'ok': True, 'app': 'latent-atlas', 'helper': 1, 'version': '1.0', 'port': self.port})
        for host in ('localhost:%d' % self.port, '[::1]:%d' % self.port):
            self.assertEqual(self.req('GET', '/api/health', host=host)[0], 200)

    def test_host_rejections(self):
        for host in ('evil.com', 'evil.com:%d' % self.port, '127.0.0.1:%d' % (self.port + 1), '127.0.0.1',
                     'localhost.:%d' % self.port, '192.168.0.10:%d' % self.port):
            st, data, _ = self.req('GET', '/api/health', host=host)
            self.assertEqual(st, 403, host)
            self.assertEqual(data['error'], 'forbidden_host')
            self.assertEqual(self.mlx('GET', '/api/mlx/status', host=host)[0], 403)
        self.assertEqual(self.req('GET', '/api/health', host=None)[0], 403)

    def test_x_atlas_and_origin(self):
        self.assertEqual(self.req('GET', '/api/mlx/status')[0], 403)
        self.assertEqual(self.req('GET', '/api/mlx/status', headers={'X-Atlas': '0'})[0], 403)
        st, data, hdrs = self.mlx('GET', '/api/mlx/status')
        self.assertEqual(st, 200)
        for key in ('seq', 'mac', 'mlx', 'hf', 'download', 'server', 'local_models'):
            self.assertIn(key, data)
        self.assertEqual(data['server']['state'], 'stopped')
        self.assertEqual(data['download']['state'], 'idle')
        self.assertFalse(data['mlx']['installed'])
        self.assertIn('install', data['mlx'])
        self.assertIn('installers', data['mlx'])
        self.assertEqual(hdrs.get('Content-Type'), 'application/json; charset=utf-8')
        # POST: X-Atlas + JSON + Origin local (ou ausente)
        self.assertEqual(self.req('POST', '/api/mlx/stop', body={})[0], 403)
        self.assertEqual(self.mlx('POST', '/api/mlx/stop', body={})[0], 200)
        for origin in ('http://localhost:%d' % self.port, 'http://127.0.0.1:%d' % self.port,
                       'http://[::1]:%d' % self.port):
            self.assertEqual(self.mlx('POST', '/api/mlx/stop', body={}, headers={'Origin': origin})[0], 200, origin)
        for origin in ('http://evil.com', 'null', 'http://localhost:%d' % (self.port + 1),
                       'https://localhost:%d' % self.port):
            st, data, _ = self.mlx('POST', '/api/mlx/stop', body={}, headers={'Origin': origin})
            self.assertEqual(st, 403, origin)
            self.assertEqual(data['error'], 'forbidden_origin')
        self.assertEqual(self.mlx('GET', '/api/mlx/status', headers={'Origin': 'http://evil.com'})[0], 403)
        st, _, _ = self.mlx('POST', '/api/mlx/stop', body=b'{}', json_ct=False,
                            headers={'Content-Type': 'text/plain'})
        self.assertEqual(st, 403)
        st, _, _ = self.mlx('POST', '/api/mlx/stop', body=b'{}', json_ct=False)
        self.assertEqual(st, 403)

    def test_methods_and_bodies(self):
        st, _, hdrs = self.mlx('OPTIONS', '/api/mlx/status', headers={'Origin': 'http://evil.com',
                                                                       'Access-Control-Request-Method': 'POST'})
        self.assertEqual(st, 405)
        self.assertEqual(self.req('OPTIONS', '/api/health')[0], 405)
        self.assertEqual(self.mlx('GET', '/api/mlx/stop')[0], 405)
        self.assertEqual(self.mlx('GET', '/api/mlx/nope')[0], 404)
        self.assertEqual(self.mlx('POST', '/api/mlx/stop', body=b'x' * (A.MAX_BODY + 1))[0], 413)
        st, data, _ = self.mlx('POST', '/api/mlx/stop', body=b'[1,2]')
        self.assertEqual((st, data['error']), (400, 'bad_json'))
        st, data, _ = self.mlx('POST', '/api/mlx/stop', body=b'{not json')
        self.assertEqual((st, data['error']), (400, 'bad_json'))

    def test_param_validation_over_http(self):
        st, data, _ = self.mlx('POST', '/api/mlx/download', body={'id': '../../etc'})
        self.assertEqual((st, data['error']), (400, 'bad_id'))
        st, data, _ = self.mlx('GET', '/api/mlx/info?id=a--b/c')
        self.assertEqual((st, data['error']), (400, 'bad_id'))
        for body, param in (({'id': 'a/b', 'port': 80}, 'port'), ({'id': 'a/b', 'port': self.port}, 'port'),
                            ({'id': 'a/b', 'port': 70000}, 'port'), ({'id': 'a/b', 'port': 'x'}, 'port'),
                            ({'id': 'a/b', 'max_tokens': 8}, 'max_tokens'),
                            ({'id': 'a/b', 'max_tokens': 40000}, 'max_tokens'),
                            ({'id': 'a/b', 'temp': 3}, 'temp'), ({'id': 'a/b', 'temp': -1}, 'temp')):
            st, data, _ = self.mlx('POST', '/api/mlx/start', body=body)
            self.assertEqual((st, data['error'], data.get('param')), (400, 'bad_param', param), body)
        st, data, _ = self.mlx('GET', '/api/mlx/logs?kind=passwd')
        self.assertEqual(st, 400)
        st, data, _ = self.mlx('GET', '/api/mlx/logs?kind=download&n=5')
        self.assertEqual((st, data), (200, {'kind': 'download', 'lines': []}))
        st, data, _ = self.mlx('GET', '/api/mlx/probe?port=%d' % self.port)
        self.assertEqual(st, 400)
        free = free_port()
        st, data, _ = self.mlx('GET', '/api/mlx/probe?port=%d' % free)
        self.assertEqual((st, data['listening'], data['openai'], data['managed']), (200, False, False, False))
        self.h.mac.update({'os': 'Darwin', 'arm64': True})
        st, data, _ = self.mlx('POST', '/api/mlx/start', body={'id': 'a/b', 'port': free})
        self.assertEqual((st, data['error']), (409, 'mlx_missing'))
        self.h.mac['arm64'] = False
        st, data, _ = self.mlx('POST', '/api/mlx/start', body={'id': 'a/b', 'port': free})
        self.assertEqual((st, data['error']), (409, 'unsupported_platform'))
        st, data, _ = self.mlx('POST', '/api/mlx/install', body={})
        self.assertEqual((st, data['error']), (409, 'unsupported_platform'))

    def test_install_without_installer(self):
        self.h.mac.update({'os': 'Darwin', 'arm64': True})
        orig = A.find_installers
        A.find_installers = lambda env: {'uv': None, 'brew': None}
        try:
            st, data, _ = self.mlx('POST', '/api/mlx/install', body={'upgrade': False})
        finally:
            A.find_installers = orig
        self.assertEqual((st, data['error']), (409, 'no_installer'))
        self.assertEqual(len(data['hint']), 2)

    def test_static_files_and_traversal(self):
        st, data, hdrs = self.req('GET', '/')
        self.assertEqual(st, 200)
        self.assertIn(b'atlas', data)
        self.assertEqual(hdrs.get('Cache-Control'), 'no-cache')
        self.assertEqual(self.req('GET', '/index.html?x=1')[0], 200)
        for path in ('/../secret.txt', '/%2e%2e/secret.txt', '/..%2fsecret.txt', '/assets/../../secret.txt',
                     '/%2e%2e%2f%2e%2e%2fsecret.txt', '/.hidden', '/assets/', '//evil.com/%2e%2e',
                     '/..\\secret.txt', '/nope.txt'):
            st, data, hdrs = self.req('GET', path)
            self.assertIn(st, (404,), path)
            self.assertNotIn(b'TOP-SECRET', data if isinstance(data, bytes) else json.dumps(data).encode())
            self.assertNotIn('Location', hdrs)
        self.assertEqual(self.req('POST', '/index.html', body=b'{}')[0], 405)

    def test_api_disabled_off_loopback(self):
        httpd = A.make_server('127.0.0.1', 0, self.root, self.helper(), quiet=True, api=False)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': 0.1}, daemon=True).start()
        try:
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
            conn.request('GET', '/api/health', headers={'Host': '127.0.0.1:%d' % port})
            resp = conn.getresponse()
            self.assertEqual(resp.status, 404)
            self.assertEqual(json.loads(resp.read().decode())['error'], 'api_disabled')
            conn.close()
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_port_fallback(self):
        httpd = A.make_server('127.0.0.1', self.port, self.root, self.helper(), quiet=True)
        try:
            self.assertNotEqual(httpd.server_address[1], self.port)
            self.assertLessEqual(httpd.server_address[1], self.port + 10)
        finally:
            httpd.server_close()

    def test_download_over_http(self):
        hub = FakeHub()
        self.addCleanup(hub.close)
        self.h.endpoint = hub.endpoint
        self.h.hub = A.HubClient(hub.endpoint, None)
        st, data, _ = self.mlx('GET', '/api/mlx/info?id=' + urllib.parse.quote(REPO, safe=''))
        self.assertEqual(st, 200)
        self.assertEqual(data['id'], REPO)
        st, data, _ = self.mlx('GET', '/api/mlx/info?id=org/missing')
        self.assertEqual((st, data['error']), (404, 'not_found'))
        st, data, _ = self.mlx('POST', '/api/mlx/download', body={'id': REPO})
        self.assertEqual(st, 202)
        self.assertEqual(data['download']['state'], 'running')
        self.assertTrue(wait_for(lambda: self.mlx('GET', '/api/mlx/status')[1]['download']['state'] == 'done', 20))
        st, data, _ = self.mlx('GET', '/api/mlx/status')
        self.assertEqual([(m['id'], m['complete']) for m in data['local_models']], [(REPO, True)])
        st, data, _ = self.mlx('GET', '/api/mlx/logs?kind=download')
        self.assertTrue(any('complete' in ln for ln in data['lines']))
        st, data, _ = self.mlx('POST', '/api/mlx/delete', body={'id': REPO})
        self.assertEqual(st, 200)
        self.assertGreater(data['freed_bytes'], 4 << 20)
        self.assertEqual(self.mlx('GET', '/api/mlx/status')[1]['local_models'], [])


# ----------------------------------------------------------------------------------------------------------------
# ciclo de vida do mlx_lm.server (com um servidor falso)
# ----------------------------------------------------------------------------------------------------------------

class MlxBase(Base):
    """Helper com um mlx_lm.server falso (rodando pelo bootstrap de verdade) e um modelo local completo."""

    def setUp(self):
        Base.setUp(self)
        self.fake = os.path.join(self.tmp, 'bin', 'mlx_lm.server')
        os.makedirs(os.path.dirname(self.fake))
        with open(self.fake, 'w') as fh:
            fh.write(FAKE_MLX % {'python': sys.executable})
        os.chmod(self.fake, 0o755)
        self.record_path = os.path.join(self.tmp, 'record.json')
        self.env['LA_MLX_SERVER'] = self.fake
        self.env['FAKE_MLX_RECORD'] = self.record_path
        make_local_model(self.cache, 'org/local')

    def started_helper(self, mode='ok'):
        self.env['FAKE_MLX_MODE'] = mode
        h = self.helper(port=8765)
        h.mac.update({'os': 'Darwin', 'arm64': True})
        info = h.detect_mlx()
        self.assertTrue(info['installed'])
        self.assertEqual(info['python'], sys.executable)
        return h

    def record(self):
        with open(self.record_path) as fh:
            return json.load(fh)

    def posts(self):
        try:
            with open(self.record_path + '.posts') as fh:
                return [json.loads(ln) for ln in fh if ln.strip()]
        except OSError:
            return []

    def ready(self, mode='ok', max_tokens=2048):
        """Sobe o modelo e espera ficar pronto → (helper, porta pública, porta interna)."""
        h = self.started_helper(mode)
        port = free_port()
        code, _ = h.start_server('org/local', port, max_tokens, 0.4)
        self.assertEqual(code, 202)
        self.assertTrue(wait_for(lambda: h.server['state'] in ('ready', 'error'), 20))
        self.assertEqual(h.server['state'], 'ready', h.server)
        argv = self.record()['argv']
        return h, port, int(argv[argv.index('--port') + 1])


class ServerLifecycleTests(MlxBase):

    def test_start_ready_probe_stop(self):
        h = self.started_helper()
        port = free_port()
        code, st = h.start_server('org/local', port, 2048, 0.4)
        self.assertEqual(code, 202)
        self.assertEqual(st['server']['state'], 'starting')
        self.assertTrue(wait_for(lambda: h.server['state'] in ('ready', 'error'), 20))
        srv = h.status()['server']
        self.assertEqual(srv['state'], 'ready', srv)
        self.assertEqual(srv['url'], 'http://127.0.0.1:%d/v1' % port)
        self.assertIsNotNone(srv['load_s'])
        self.assertGreater(srv['tok_s'], 0)
        rec = self.record()
        argv = rec['argv']
        self.assertNotIn('--trust-remote-code', argv)
        self.assertEqual(argv[argv.index('--model') + 1], 'org/local')
        self.assertEqual(argv[argv.index('--host') + 1], '127.0.0.1')
        # o mlx_lm.server de verdade fica numa porta interna; a pública é o proxy do helper
        internal = int(argv[argv.index('--port') + 1])
        self.assertNotEqual(internal, port)
        self.assertEqual(argv[argv.index('--max-tokens') + 1], '2048')
        self.assertEqual(argv[argv.index('--temp') + 1], '0.4')
        self.assertEqual(argv[argv.index('--allowed-origins') + 1],
                         'http://localhost:8765,http://127.0.0.1:8765,http://[::1]:8765')
        self.assertEqual(rec['env'], {'HF_HUB_OFFLINE': '1', 'HF_HUB_CACHE': self.cache,
                                      'HF_HUB_DISABLE_TELEMETRY': '1', 'PYTHONUNBUFFERED': '1'})
        self.assertEqual(rec['pid'], srv['pid'])
        # aquecimento e medição passaram com o segredo
        self.assertTrue(self.posts())
        self.assertTrue(all(p['key'] for p in self.posts()))
        # o log não guarda o /health que o helper consulta, mas guarda o resto
        lines = h.logs_for('server', 600)
        self.assertFalse([ln for ln in lines if '/health' in ln])
        self.assertTrue(any('POST /v1/chat/completions' in ln for ln in lines))
        pr = h.probe(port)
        self.assertEqual(pr, {'port': port, 'listening': True, 'managed': True, 'openai': True,
                              'models': ['org/local']})
        self.assertEqual(h.start_server('org/local', port, 2048, 0.4)[0], 200)  # no-op
        with self.assertRaises(A.ApiError) as cm:
            h.delete_model('org/local')
        self.assertEqual(cm.exception.code, 'busy')
        pid = srv['pid']
        st = h.stop_server()
        self.assertEqual(st['server']['state'], 'stopped')
        self.assertIsNone(st['server']['pid'])
        with self.assertRaises(OSError):
            os.kill(pid, 0)
        self.assertFalse(A.port_listening(port))
        self.assertFalse(A.port_listening(internal))
        self.assertIsNone(h.gateway())

    def test_switch_model_stops_previous(self):
        make_local_model(self.cache, 'org/other')
        h = self.started_helper()
        port = free_port()
        h.start_server('org/local', port, 2048, 0.4)
        self.assertTrue(wait_for(lambda: h.server['state'] == 'ready', 20))
        first = h.server['pid']
        code, _ = h.start_server('org/other', port, 1024, 0.2)
        self.assertEqual(code, 202)
        with self.assertRaises(OSError):
            os.kill(first, 0)
        self.assertTrue(wait_for(lambda: h.server['state'] == 'ready', 20))
        self.assertEqual(h.server['id'], 'org/other')
        h.stop_server()

    def test_retry_without_chat_template_kwargs(self):
        h = self.started_helper('no_kwargs')
        port = free_port()
        h.start_server('org/local', port, 2048, 0.4)
        self.assertTrue(wait_for(lambda: h.server['state'] in ('ready', 'error'), 20))
        self.assertEqual(h.server['state'], 'ready', h.server)
        h.stop_server()

    def test_crash_reports_error(self):
        h = self.started_helper('crash')
        port = free_port()
        h.start_server('org/local', port, 2048, 0.4)
        self.assertTrue(wait_for(lambda: h.server['state'] in ('ready', 'error'), 20))
        srv = h.status()['server']
        self.assertEqual(srv['state'], 'error')
        self.assertEqual(srv['error_code'], 'exited')
        self.assertIn('code 3', srv['error'])
        self.assertTrue(any('boom' in ln for ln in srv['log_tail']))
        self.assertIsNone(srv['pid'])

    def test_not_downloaded_and_port_busy(self):
        h = self.started_helper()
        with self.assertRaises(A.ApiError) as cm:
            h.start_server('org/absent', free_port(), 2048, 0.4)
        self.assertEqual(cm.exception.code, 'not_downloaded')
        make_local_model(self.cache, 'org/notok', tokenizer=False)
        with self.assertRaises(A.ApiError) as cm:
            h.start_server('org/notok', free_port(), 2048, 0.4)
        self.assertEqual(cm.exception.code, 'not_downloaded')

        class OpenAI(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = json.dumps({'data': [{'id': 'someone-else'}]}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        other = http.server.ThreadingHTTPServer(('127.0.0.1', 0), OpenAI)
        threading.Thread(target=other.serve_forever, kwargs={'poll_interval': 0.1}, daemon=True).start()
        try:
            busy = other.server_address[1]
            with self.assertRaises(A.ApiError) as cm:
                h.start_server('org/local', busy, 2048, 0.4)
            e = cm.exception
            self.assertEqual((e.status, e.code), (409, 'port_busy'))
            self.assertTrue(e.extra['openai'])
            self.assertEqual(e.extra['models'], ['someone-else'])
            pr = h.probe(busy)
            self.assertEqual((pr['listening'], pr['managed'], pr['openai']), (True, False, True))
        finally:
            other.shutdown()
            other.server_close()

    def test_shutdown_stops_server(self):
        h = self.started_helper()
        port = free_port()
        h.start_server('org/local', port, 2048, 0.4)
        self.assertTrue(wait_for(lambda: h.server['state'] == 'ready', 20))
        pid = h.server['pid']
        h.shutdown()
        with self.assertRaises(OSError):
            os.kill(pid, 0)


# ----------------------------------------------------------------------------------------------------------------
# regressões de segurança: proxy da porta pública + segredo do servidor interno (SEC-1)
# ----------------------------------------------------------------------------------------------------------------

LOCAL_ORIGIN = 'http://localhost:8765'
CHAT = {'messages': [{'role': 'user', 'content': 'Count from 1 to 200 in words.'}], 'max_tokens': 300}


class ProxyTests(MlxBase):
    def test_cross_site_and_rebinding_never_reach_mlx(self):
        h, port, internal = self.ready()
        before = len(self.posts())
        raw = json.dumps(CHAT).encode()
        # POST "simples" de outro site (text/plain, sem preflight): recusado antes de chegar ao mlx_lm.server
        st, hd, _ = http_call(port, 'POST', '/v1/chat/completions', raw,
                              {'Origin': 'https://evil.example', 'Content-Type': 'text/plain;charset=UTF-8'})
        self.assertEqual(st, 403)
        self.assertIsNone(hd.get('Access-Control-Allow-Origin'))
        st, _, _ = http_call(port, 'POST', '/v1/chat/completions', raw,
                             {'Origin': 'https://evil.example', 'Content-Type': 'application/json'})
        self.assertEqual(st, 403)
        st, _, _ = http_call(port, 'POST', '/v1/chat/completions', raw, {'Content-Type': 'text/plain'})
        self.assertEqual(st, 415)
        st, _, _ = http_call(port, 'POST', '/v1/chat/completions', raw)  # sem Content-Type
        self.assertEqual(st, 415)
        # DNS rebinding: Host de outro nome
        for method, path, body in (('GET', '/v1/models', None), ('GET', '/health', None),
                                   ('POST', '/v1/chat/completions', raw)):
            st, _, data = http_call(port, method, path, body, {'Content-Type': 'application/json'},
                                    host='rebind.evil.example:%d' % port)
            self.assertEqual(st, 403, (method, path))
            self.assertNotIn(b'org/local', data)
        # preflight: só a origem da plataforma ganha CORS
        st, hd, _ = http_call(port, 'OPTIONS', '/v1/chat/completions', None,
                              {'Origin': 'https://evil.example', 'Access-Control-Request-Method': 'POST'})
        self.assertEqual(st, 403)
        self.assertIsNone(hd.get('Access-Control-Allow-Origin'))
        st, hd, _ = http_call(port, 'OPTIONS', '/v1/chat/completions', None,
                              {'Origin': LOCAL_ORIGIN, 'Access-Control-Request-Method': 'POST',
                               'Access-Control-Request-Headers': 'content-type'})
        self.assertEqual(st, 204)
        self.assertEqual(hd.get('Access-Control-Allow-Origin'), LOCAL_ORIGIN)
        self.assertIn('POST', hd.get('Access-Control-Allow-Methods'))
        self.assertEqual(hd.get('Access-Control-Allow-Headers'), 'content-type')
        # o servidor interno não atende ninguém sem o segredo (nem /health, nem preflight)
        st, _, _ = http_call(internal, 'POST', '/v1/chat/completions', raw, {'Content-Type': 'text/plain'})
        self.assertEqual(st, 403)
        for method, path in (('GET', '/v1/models'), ('GET', '/health')):
            self.assertEqual(http_call(internal, method, path)[0], 403, path)
        st, hd, _ = http_call(internal, 'OPTIONS', '/v1/chat/completions', None,
                              {'Origin': 'https://evil.example', 'Access-Control-Request-Method': 'POST'})
        self.assertEqual(st, 403)
        self.assertIsNone(hd.get('Access-Control-Allow-Origin'))
        st, _, _ = http_call(internal, 'GET', '/health', headers={'X-Atlas-Key': 'guess'})
        self.assertEqual(st, 403)
        self.assertEqual(len(self.posts()), before)  # nada chegou ao mlx_lm.server
        self.assertEqual(h.server['state'], 'ready')

    def test_model_pinned_and_tokens_capped(self):
        h, port, internal = self.ready(max_tokens=2048)
        body = dict(CHAT, model='/etc', adapters='/tmp/evil', draft_model='evil/redirect-cross', max_tokens=99999)
        st, hd, data = http_call(port, 'POST', '/v1/chat/completions', body,
                                 {'Origin': LOCAL_ORIGIN, 'Content-Type': 'application/json'})
        self.assertEqual(st, 200, data)
        self.assertEqual(hd.get('Access-Control-Allow-Origin'), LOCAL_ORIGIN)
        sent = self.posts()[-1]
        self.assertEqual(sent['path'], '/v1/chat/completions')
        self.assertEqual(sent['body']['model'], 'org/local')
        self.assertNotIn('adapters', sent['body'])
        self.assertNotIn('draft_model', sent['body'])
        self.assertEqual(sent['body']['max_tokens'], A.MAX_GEN_TOKENS)
        self.assertTrue(sent['key'])
        # max_completion_tokens também; valores menores passam intactos
        http_call(port, 'POST', '/chat/completions', dict(CHAT, max_completion_tokens=10 ** 6, max_tokens=None),
                  {'Content-Type': 'application/json'})
        self.assertEqual(self.posts()[-1]['body']['max_completion_tokens'], A.MAX_GEN_TOKENS)
        # acima do --max-tokens do start, mas dentro do teto: passa (o copilot pode pedir mais)
        http_call(port, 'POST', '/v1/chat/completions', dict(CHAT, max_tokens=4096), {'Content-Type': 'application/json'})
        self.assertEqual(self.posts()[-1]['body']['max_tokens'], 4096)
        st, hd, _ = http_call(port, 'POST', '/v1/completions', {'prompt': 'hi', 'max_tokens': 64},
                              {'Content-Type': 'application/json'})
        self.assertEqual(st, 200)
        self.assertIsNone(hd.get('Access-Control-Allow-Origin'))  # sem Origin (curl/SDK): sem CORS
        self.assertEqual(self.posts()[-1]['body']['max_tokens'], 64)
        # /v1/models mostra só o modelo gerenciado (o do mlx lista o cache inteiro)
        st, hd, data = http_call(port, 'GET', '/v1/models', headers={'Origin': LOCAL_ORIGIN})
        self.assertEqual(st, 200)
        self.assertEqual([m['id'] for m in json.loads(data)['data']], ['org/local'])
        self.assertEqual(hd.get('Access-Control-Allow-Origin'), LOCAL_ORIGIN)
        self.assertEqual(http_call(port, 'GET', '/health')[0], 200)
        # streaming (SSE) passa pedaço a pedaço
        st, hd, data = http_call(port, 'POST', '/v1/chat/completions', dict(CHAT, stream=True),
                                 {'Origin': LOCAL_ORIGIN, 'Content-Type': 'application/json'})
        self.assertEqual(st, 200)
        self.assertEqual(hd.get('Content-Type'), 'text/event-stream')
        self.assertIn(b'"Hel"', data)
        self.assertTrue(data.rstrip().endswith(b'data: [DONE]'))
        # corpos inválidos não chegam ao mlx (que registraria o corpo cru no log)
        n = len(self.posts())
        for raw in (b'{nope', b'[1, 2]', b'[' * 30000 + b']' * 30000, b'\xff\xfe'):
            st, _, data = http_call(port, 'POST', '/v1/chat/completions', raw, {'Content-Type': 'application/json'})
            self.assertEqual((st, json.loads(data)['error']), (400, 'bad_json'))
        st, _, _ = http_call(port, 'POST', '/v1/chat/completions', b'{}',
                             {'Content-Type': 'application/json', 'Content-Length': str(A.PROXY_MAX_BODY + 1)})
        self.assertEqual(st, 413)
        self.assertEqual(len(self.posts()), n)
        self.assertEqual(http_call(port, 'GET', '/v1/chat/completions')[0], 405)
        self.assertEqual(http_call(port, 'PUT', '/v1/models', b'{}')[0], 405)
        self.assertEqual(http_call(port, 'GET', '/v1/../health')[0], 404)
        self.assertEqual(http_call(port, 'GET', '/nope')[0], 404)
        self.assertEqual(h.server['state'], 'ready')

    def test_same_origin_route_on_platform_port(self):
        h = self.started_helper()
        root = os.path.join(self.tmp, 'site')
        os.makedirs(root)
        httpd = A.make_server('127.0.0.1', 0, root, h, quiet=True)
        hp = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': 0.1}, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        api = {'X-Atlas': '1', 'Content-Type': 'application/json'}
        st, _, data = http_call(hp, 'GET', '/api/mlx/v1/models', headers=api)
        self.assertEqual((st, json.loads(data)['error']), (409, 'not_running'))
        port = free_port()
        h.start_server('org/local', port, 512, 0.4)
        self.assertTrue(wait_for(lambda: h.server['state'] == 'ready', 20))
        st, _, data = http_call(hp, 'GET', '/api/mlx/v1/models', headers=api)
        self.assertEqual([m['id'] for m in json.loads(data)['data']], ['org/local'])
        st, hd, data = http_call(hp, 'POST', '/api/mlx/v1/chat/completions', dict(CHAT, model='x/y', max_tokens=9999),
                                 dict(api, Origin='http://localhost:%d' % hp))
        self.assertEqual(st, 200, data)
        self.assertIsNone(hd.get('Access-Control-Allow-Origin'))
        self.assertEqual(self.posts()[-1]['body']['model'], 'org/local')
        self.assertEqual(self.posts()[-1]['body']['max_tokens'], 9999)
        n = len(self.posts())
        self.assertEqual(http_call(hp, 'POST', '/api/mlx/v1/chat/completions', CHAT,
                                   {'Content-Type': 'application/json'})[0], 403)          # sem X-Atlas
        self.assertEqual(http_call(hp, 'POST', '/api/mlx/v1/chat/completions', CHAT,
                                   dict(api, Origin='https://evil.example'))[0], 403)
        self.assertEqual(http_call(hp, 'POST', '/api/mlx/v1/chat/completions', CHAT,
                                   {'X-Atlas': '1', 'Content-Type': 'text/plain'})[0], 403)
        self.assertEqual(http_call(hp, 'POST', '/api/mlx/v1/chat/completions', CHAT, api,
                                   host='rebind.evil:%d' % hp)[0], 403)
        self.assertEqual(len(self.posts()), n)
        h.stop_server()


# ----------------------------------------------------------------------------------------------------------------
# mlx_lm.server morre junto com o helper, até com SIGKILL (SEC-5)
# ----------------------------------------------------------------------------------------------------------------

DRIVER = r'''
import json, os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, %(tools)r)
import atlas_server as A
h = A.Helper(port=8765, cache_dir=%(cache)r, endpoint='http://127.0.0.1:9', token='', env=dict(os.environ))
h.mac.update({'os': 'Darwin', 'arm64': True})
h.detect_mlx()
h.start_server('org/local', %(port)d, 2048, 0.4)
end = time.time() + 20
while h.server['state'] == 'starting' and time.time() < end:
    time.sleep(0.05)
print(json.dumps({'state': h.server['state'], 'pid': h.server['pid']}), flush=True)
time.sleep(120)
'''


class WatchdogTests(MlxBase):
    def test_server_dies_with_sigkilled_helper(self):
        self.env['FAKE_MLX_MODE'] = 'ok'
        port = free_port()
        code = DRIVER % {'tools': os.path.dirname(os.path.abspath(__file__)), 'cache': self.cache, 'port': port}
        drv = subprocess.Popen([sys.executable, '-c', code], env=self.env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
        pid = None
        try:
            info = json.loads(drv.stdout.readline().decode() or '{}')
            self.assertEqual(info.get('state'), 'ready', info)
            pid = info['pid']
            self.assertTrue(pid_alive(pid))
            self.assertTrue(A.port_listening(port))
            os.kill(drv.pid, signal.SIGKILL)
            drv.wait(10)
            self.assertTrue(wait_for(lambda: not pid_alive(pid), 15), 'mlx_lm.server outlived its helper')
            self.assertFalse(A.port_listening(port))
        finally:
            if drv.poll() is None:
                drv.kill()
                drv.wait(5)
            drv.stdout.close()
            if pid and pid_alive(pid):
                try:
                    os.killpg(pid, signal.SIGKILL)
                except OSError:
                    pass


# ----------------------------------------------------------------------------------------------------------------
# falha ao carregar o modelo (E2E-2) e logs com tamanho limitado (SEC-2)
# ----------------------------------------------------------------------------------------------------------------

class LoadFailureTests(MlxBase):
    def start_and_wait(self, mode):
        h = self.started_helper(mode)
        t0 = time.time()
        h.start_server('org/local', free_port(), 2048, 0.4)
        self.assertTrue(wait_for(lambda: h.server['state'] in ('ready', 'error'), 20))
        return h, time.time() - t0

    def test_generation_thread_died_fails_fast(self):
        h, took = self.start_and_wait('dead')
        self.assertLess(took, 10)   # antes: 90 s de "Loading" e um health_timeout enganoso
        self.assertTrue(wait_for(lambda: h.server['pid'] is None, 15))
        srv = h.status()['server']
        self.assertEqual((srv['state'], srv['error_code']), ('error', 'load_failed'))
        self.assertIn('Model type weirdnet not supported', srv['error'])
        self.assertTrue(srv['log_tail'])
        self.assertFalse([ln for ln in srv['log_tail'] if '/health' in ln])
        self.assertIsNone(h.gateway())

    def test_health_503_streak_fails_fast(self):
        h, took = self.start_and_wait('dead_quiet')
        self.assertLess(took, 10)
        srv = h.status()['server']
        self.assertEqual((srv['state'], srv['error_code']), ('error', 'load_failed'))
        self.assertIn('503', srv['error'])

    def test_server_without_health_endpoint(self):
        h, _ = self.start_and_wait('nohealth')   # mlx-lm antigo: o aquecimento decide
        self.assertEqual(h.server['state'], 'ready', h.server)
        h.stop_server()

    def test_long_output_lines_are_bounded(self):
        h, _ = self.start_and_wait('spam')       # o servidor escreve uma linha de 3 MB
        self.assertEqual(h.server['state'], 'ready', h.server)
        lines = h.logs_for('server', A.LOG_LINES)
        self.assertTrue(any('after the long line' in ln for ln in lines))
        self.assertLessEqual(max(len(ln) for ln in lines), A.LOG_LINE_MAX + 1)
        self.assertLess(sum(len(ln) for ln in lines), 64 * 1024)
        h.stop_server()


class LogBoundTests(unittest.TestCase):
    def test_ring_caps_lines_and_chars(self):
        ring = A.LogRing(max_lines=5, max_chars=12000)
        ring.append('x' * 100000)
        self.assertEqual(len(list(ring)[0]), A.LOG_LINE_MAX + 1)
        self.assertTrue(list(ring)[0].endswith('…'))
        for i in range(10):
            ring.append('line %d' % i)
        self.assertEqual(list(ring), ['line %d' % i for i in range(5, 10)])
        for _ in range(10):
            ring.append('y' * 5000)
        self.assertLessEqual(sum(len(ln) for ln in ring), 12000)
        ring.clear()
        self.assertEqual(len(ring), 0)

    def test_bounded_lines_drop_the_rest(self):
        data = b'a' * 10000 + b'\nshort\n' + b'b' * 5000
        out = list(A.bounded_lines(io.BytesIO(data), limit=4096))
        self.assertEqual(len(out), 3)
        self.assertTrue(out[0].startswith(b'a' * 4096) and out[0].endswith('…'.encode()))
        self.assertLessEqual(len(out[0]), 4096 + 3)
        self.assertEqual(out[1], b'short\n')
        self.assertEqual(out[2], b'b' * 4096 + '…'.encode())

    def test_helper_log_lines_are_clipped(self):
        h = A.Helper(port=0, cache_dir=tempfile.gettempdir(), endpoint='http://127.0.0.1:9', token='',
                     env={'LA_MLX_SERVER': '/nonexistent'})
        for _ in range(700):
            h._log('server', 'z' * 5000)
        self.assertLessEqual(sum(len(ln) for ln in h.logs_for('server', A.LOG_LINES)), A.LOG_BYTES)
        self.assertTrue(all(len(ln) <= A.LOG_LINE_MAX + 1 for ln in h.logs_for('server', A.LOG_LINES)))


# ----------------------------------------------------------------------------------------------------------------
# nada novo começa durante o shutdown (SEC-4)
# ----------------------------------------------------------------------------------------------------------------

class ShutdownGuardTests(MlxBase):
    def test_nothing_starts_after_shutdown(self):
        h = self.started_helper()
        h.shutdown()
        calls = (lambda: h.start_server('org/local', free_port(), 2048, 0.4), lambda: h.start_install(False),
                 lambda: h.start_download('org/local'))
        for fn in calls:
            with self.assertRaises(A.ApiError) as cm:
                fn()
            self.assertEqual((cm.exception.status, cm.exception.code), (503, 'shutting_down'))
        self.assertFalse(os.path.exists(self.record_path))   # nenhum mlx_lm.server foi lançado

    def test_start_waiting_on_lock_during_shutdown(self):
        h = self.started_helper()
        out = {}

        def start():
            try:
                out['res'] = h.start_server('org/local', free_port(), 2048, 0.4)
            except A.ApiError as e:
                out['err'] = e

        h._ctl.acquire()       # simula um stop lento em andamento
        t1 = threading.Thread(target=start)
        t1.start()
        time.sleep(0.2)
        t2 = threading.Thread(target=h.shutdown)
        t2.start()
        self.assertTrue(wait_for(lambda: h._closed, 5))
        h._ctl.release()
        t1.join(20)
        t2.join(20)
        self.assertEqual(out.get('err') and out['err'].code, 'shutting_down', out)
        self.assertFalse(os.path.exists(self.record_path))
        self.assertIsNone(h.gateway())

    def test_install_does_not_spawn_after_shutdown(self):
        h = self.started_helper()
        h.shutdown()
        h._run_install([sys.executable, '-c', 'import time; time.sleep(30)'])
        self.assertEqual(h.install['state'], 'error')
        self.assertIsNone(h._install_proc)


# ----------------------------------------------------------------------------------------------------------------
# downloader não segue symlinks plantados no cache (SEC-8)
# ----------------------------------------------------------------------------------------------------------------

class DownloadSymlinkTests(HubBase):
    def victim(self, data=b'victim data, 21 bytes'):
        path = os.path.join(self.tmp, 'victim.txt')
        with open(path, 'wb') as fh:
            fh.write(data)
        return path, data

    def test_symlinked_partial_is_not_followed(self):
        victim, data = self.victim()
        blobs = os.path.join(self.cache, 'models--org--model', 'blobs')
        os.makedirs(blobs)
        os.symlink(victim, os.path.join(blobs, self.hub.sha(REPO, 'model.safetensors') + '.incomplete'))
        h = self.helper(self.hub.endpoint)
        st = self.download(h)
        self.assertEqual(st['state'], 'done', st)
        with open(victim, 'rb') as fh:
            self.assertEqual(fh.read(), data)
        self.assertEqual([r[3].get('Range') for r in self.hub.log('cdn')], [None])  # recomeçou do zero
        self.assertTrue(h.local_entry(REPO, force=True)['complete'])

    def test_symlinked_repo_folder_creates_nothing_outside(self):
        outside = os.path.join(self.tmp, 'outside')
        os.makedirs(outside)
        os.symlink(outside, os.path.join(self.cache, 'models--org--model'))
        h = self.helper(self.hub.endpoint)
        st = self.download(h)
        self.assertEqual((st['state'], st['error_code']), ('error', 'unsafe_path'))
        self.assertEqual(os.listdir(outside), [])
        self.assertEqual(self.hub.log('cdn'), [])

    def test_symlinked_refs_tmp_is_not_followed(self):
        victim, data = self.victim()
        refs = os.path.join(self.cache, 'models--org--model', 'refs')
        os.makedirs(refs)
        os.symlink(victim, os.path.join(refs, '.main.tmp'))
        h = self.helper(self.hub.endpoint)
        self.assertEqual(self.download(h)['state'], 'done')
        with open(victim, 'rb') as fh:
            self.assertEqual(fh.read(), data)
        main = os.path.join(refs, 'main')
        self.assertFalse(os.path.islink(main))
        with open(main) as fh:
            self.assertEqual(fh.read(), COMMIT)

    def test_symlinked_snapshot_subfolder(self):
        outside = os.path.join(self.tmp, 'outside')
        os.makedirs(outside)
        snap = os.path.join(self.cache, 'models--org--model', 'snapshots', COMMIT)
        os.makedirs(snap)
        os.symlink(outside, os.path.join(snap, 'sub'))
        h = self.helper(self.hub.endpoint)
        st = self.download(h)
        self.assertEqual((st['state'], st['error_code']), ('error', 'unsafe_path'))
        self.assertEqual(os.listdir(outside), [])


# ----------------------------------------------------------------------------------------------------------------
# servidor HTTP da plataforma: timeouts, teto de conexões, entradas malformadas, headers, Host no estático
# ----------------------------------------------------------------------------------------------------------------

class HttpHardeningTests(HttpBase):
    def test_idle_and_half_sent_connections_time_out(self):     # SEC-6
        orig = A.AtlasHandler.timeout
        A.AtlasHandler.timeout = 1
        self.addCleanup(setattr, A.AtlasHandler, 'timeout', orig)
        idle = socket.create_connection(('127.0.0.1', self.port))
        half = socket.create_connection(('127.0.0.1', self.port))
        half.sendall(('POST /api/mlx/stop HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nX-Atlas: 1\r\n'
                      'Content-Type: application/json\r\nContent-Length: 65536\r\n\r\n{' % self.port).encode())
        t0 = time.time()
        for s in (idle, half):
            s.settimeout(10)
            try:
                self.assertEqual(s.recv(4096), b'')
            except ConnectionResetError:
                pass
            s.close()
        self.assertLess(time.time() - t0, 8)
        self.assertEqual(self.req('GET', '/api/health')[0], 200)

    def test_connection_cap(self):                                # SEC-6
        class Small(A.AtlasHTTPServer):
            max_connections = 2
        orig = A.AtlasHandler.timeout
        A.AtlasHandler.timeout = 5
        self.addCleanup(setattr, A.AtlasHandler, 'timeout', orig)
        handler = functools.partial(A.AtlasHandler, directory=self.root)
        httpd = Small(('127.0.0.1', 0), handler, helper=self.helper(), api_enabled=True, quiet=True)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': 0.1}, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        held = [socket.create_connection(('127.0.0.1', port)) for _ in range(2)]
        time.sleep(0.3)
        extra = socket.create_connection(('127.0.0.1', port))
        extra.settimeout(3)
        try:
            self.assertEqual(extra.recv(10), b'')   # fechada na hora, sem thread nova
        except ConnectionResetError:
            pass
        extra.close()
        for s in held:
            s.close()
        self.assertTrue(wait_for(lambda: http_call(port, 'GET', '/api/health', timeout=3)[0] == 200, 10))

    def test_malformed_inputs(self):                              # SEC-7
        for path in ('/index.html%00.js', '/%00', '/assets/%00/x'):
            st, _, _ = self.req('GET', path)
            self.assertEqual(st, 404, path)
        huge = b'1' + b'0' * 400
        for field in (b'port', b'max_tokens', b'temp'):
            body = b'{"id": "a/b", "' + field + b'": ' + huge + b'}'
            st, data, _ = self.mlx('POST', '/api/mlx/start', body=body)
            self.assertEqual((st, data['error'], data['param']), (400, 'bad_param', field.decode()))
        st, data, _ = self.mlx('GET', '/api/mlx/probe?port=' + '9' * 400)
        self.assertEqual((st, data['error']), (400, 'bad_param'))
        for deep in (b'[' * 30000 + b']' * 30000, b'{"a":' * 9000 + b'1' + b'}' * 9000):
            st, data, _ = self.mlx('POST', '/api/mlx/stop', body=deep)
            self.assertEqual((st, data['error']), (400, 'bad_json'))
        with self.assertRaises(A.ApiError):
            A.int_param(10 ** 400, 'port', 1024, 65535)
        self.assertEqual(self.req('GET', '/api/health')[0], 200)

    def test_frame_and_sniff_headers(self):                       # SEC-9
        for path in ('/', '/index.html', '/nope.txt'):
            _, _, hdrs = self.req('GET', path)
            self.assertEqual(hdrs.get('X-Frame-Options'), 'DENY', path)
            self.assertEqual(hdrs.get('Content-Security-Policy'), "frame-ancestors 'none'", path)
            self.assertEqual(hdrs.get('X-Content-Type-Options'), 'nosniff', path)
        _, _, hdrs = self.mlx('GET', '/api/mlx/status')
        self.assertEqual(hdrs.get('X-Frame-Options'), 'DENY')

    def test_static_requires_local_host(self):                    # SEC-11
        for host in ('rebind.evil:%d' % self.port, 'evil.com', '127.0.0.1:%d' % (self.port + 1),
                     'localhost.:%d' % self.port):
            st, data, _ = self.req('GET', '/index.html', host=host)
            self.assertEqual(st, 403, host)
            self.assertNotIn(b'atlas', data if isinstance(data, bytes) else b'')
        self.assertEqual(self.req('GET', '/index.html', host=None)[0], 403)
        self.assertEqual(self.req('HEAD', '/index.html', host='rebind.evil:%d' % self.port)[0], 403)
        for host in ('127.0.0.1:%d' % self.port, 'localhost:%d' % self.port, 'LOCALHOST:%d' % self.port):
            self.assertEqual(self.req('GET', '/index.html', host=host)[0], 200, host)
        # modo LAN (API desligada): o estático continua como o http.server
        httpd = A.make_server('127.0.0.1', 0, self.root, self.helper(), quiet=True, api=False)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': 0.1}, daemon=True).start()
        try:
            self.assertEqual(http_call(port, 'GET', '/index.html', host='mymac.local:%d' % port)[0], 200)
        finally:
            httpd.shutdown()
            httpd.server_close()


# ----------------------------------------------------------------------------------------------------------------
# "busy" do delete/download não é driblado com outra caixa no id (SEC-3)
# ----------------------------------------------------------------------------------------------------------------

class CaseInsensitiveIdTests(HubBase):
    def test_delete_busy_with_other_case_while_served(self):
        make_local_model(self.cache, 'a/ok')
        h = self.helper()
        with h.lock:
            h.server.update({'state': 'ready', 'id': 'a/ok'})
        for variant in ('A/OK', 'a/OK', 'A/ok'):
            with self.assertRaises(A.ApiError) as cm:
                h.delete_model(variant)
            self.assertEqual(cm.exception.code, 'busy', variant)
        self.assertTrue(os.path.isdir(os.path.join(self.cache, 'models--a--ok')))

    def test_delete_busy_with_other_case_while_downloading(self):
        h = self.helper(self.hub.endpoint)
        self.hub.throttle = 0.03
        self.assertEqual(h.start_download(REPO)[0], 202)
        self.assertEqual(h.start_download(REPO.upper())[0], 200)   # o mesmo repo: no-op
        with self.assertRaises(A.ApiError) as cm:
            h.delete_model(REPO.upper())
        self.assertEqual(cm.exception.code, 'busy')
        h.cancel_download(wait=10)
        self.assertTrue(os.path.isdir(os.path.join(self.cache, 'models--org--model')))

    def test_same_repo(self):
        h = self.helper()
        self.assertTrue(h.same_repo('Org/Model', 'org/model'))
        self.assertFalse(h.same_repo('org/model', 'org/other'))
        self.assertFalse(h.same_repo(None, 'org/model'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
