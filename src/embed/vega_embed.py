#!/usr/bin/env python3
r"""
vega_embed.py - embed the scraped Aesthetics Wiki images with Qwen3-VL-Embedding
on the 4080, using the same llama-server setup as qwen-native-benchmark.py.

The images can stay on the server. Point --wiki at the remote folder and they
are streamed over one ssh connection, embedded, and never written to this PC:

    python vega_embed.py 2b --wiki erika@ml-db:~/shared/vega/wiki --out E:\vega\emb
    python vega_embed.py 8b --wiki erika@ml-db:~/shared/vega/wiki --out E:\vega\emb

A local folder still works:   --wiki E:\vega\wiki

Reads   <wiki>/manifest.csv and <wiki>/images/...   (output of vega_scrape.py)
Writes  <out>\qwen<model>-<edge>\embeddings.npy     float32, one L2-normalised row per image
        <out>\qwen<model>-<edge>\ids.txt            image sha1 for each row (joins to manifest.csv)
        <out>\qwen<model>-<edge>\meta.json          model, settings, counts
        <out>\qwen<model>-<edge>\failed.txt         images that could not be embedded
With a remote --wiki, those four files are also copied back to the server,
next to the wiki folder:  .../vega/emb/qwen<model>-<edge>/   (--no-upload skips this)

Safe to stop and re-run: finished images are kept in shards\ and skipped.
Needs Pillow and numpy (the same Python the benchmark used) and, for a remote
wiki, the ssh/scp that ship with Windows, with key login already working.
"""
import argparse, base64, csv, io, json, os, posixpath, re, shlex, subprocess, sys, tarfile, threading, time
import urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# ---- paths from the benchmark run; edit here if anything moves ------------
BENCH = Path(os.environ.get("VEGA_LLAMA_DIR", "llama-runtime"))             # the folder that holds runtime/llama-server.exe
LMS = Path(os.environ.get("VEGA_LMSTUDIO_DIR", str(Path.home() / ".lmstudio")))   # the LM Studio folder: models and CUDA libraries
SERVER = BENCH / "runtime" / "llama-server.exe"
CUDA_DLLS = LMS / "extensions" / "backends" / "vendor" / "win-llama-cuda12-vendor-v2"
MODELS = {  # name: (model gguf, vision projector gguf, embedding dimensions)
    "2b": (LMS / "models" / "DevQuasar" / "Qwen.Qwen3-VL-Embedding-2B-GGUF" / "Qwen.Qwen3-VL-Embedding-2B.f16.gguf",
           LMS / "models" / "DevQuasar" / "Qwen.Qwen3-VL-Embedding-2B-GGUF" / "mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf",
           2048),
    "8b": (LMS / "models" / "dam2452" / "Qwen3-VL-Embedding-8B-GGUF" / "Qwen3-VL-Embedding-8B-Q8_0.gguf",
           BENCH / "mmproj-8b-f16.gguf",
           4096),
}
PROMPT = ("<|im_start|>system\nRepresent the user's input.<|im_end|>\n"
          "<|im_start|>user\n<__media__><|im_end|>\n<|im_start|>assistant\n")
PORT = 18081
SHARD = 256  # images per shard file
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30"]
SCP = ["scp", "-q", "-o", "BatchMode=yes"]
NOWINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# ---------- where the wiki lives -------------------------------------------

class Remote:
    """A wiki folder on another machine, e.g. erika@ml-db:~/shared/vega/wiki"""
    # host must be 2+ characters so a Windows drive like E:\vega is not mistaken for one
    PATTERN = re.compile(r"^((?:[^@/\\:\s]+@)?[^@/\\:\s]{2,}):(.+)$")

    def __init__(self, host, path):
        self.host, self.path = host, path.rstrip("/") or "/"

    @staticmethod
    def shell(path):
        """Quote a remote path for the remote shell, keeping a leading ~ working."""
        if path == "~":
            return '"$HOME"'
        if path.startswith("~/"):
            return '"$HOME"/' + shlex.quote(path[2:])
        return shlex.quote(path)

    def run(self, command, **kw):
        return subprocess.run(SSH + [self.host, command], creationflags=NOWINDOW, **kw)

    def read(self, name):
        r = self.run("cat " + self.shell(posixpath.join(self.path, name)), capture_output=True)
        if r.returncode:
            sys.exit(f"could not read {name} from {self.host}:{self.path}\n{r.stderr.decode(errors='replace').strip()}")
        return r.stdout

    def stream(self, items):
        """Yield (sha1, image bytes) for every item, over a single ssh connection."""
        by_name = {rel: sha for sha, rel in items}
        proc = subprocess.Popen(SSH + [self.host, f"cd {self.shell(self.path)} && tar -cf - -T -"],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                creationflags=NOWINDOW)

        def feed():  # send the file list on a thread so a full pipe can never deadlock us
            try:
                proc.stdin.write(("\n".join(rel for _, rel in items) + "\n").encode())
                proc.stdin.close()
            except OSError:
                pass
        errors = []
        threading.Thread(target=feed, daemon=True).start()
        threading.Thread(target=lambda: errors.append(proc.stderr.read()), daemon=True).start()
        seen = set()
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
                for member in tar:
                    name = member.name[2:] if member.name.startswith("./") else member.name
                    sha = by_name.get(name)
                    if sha is None or not member.isfile():
                        continue
                    seen.add(sha)
                    yield sha, tar.extractfile(member).read()
        except tarfile.TarError as e:
            if not seen:
                proc.kill()
                time.sleep(0.2)
                msg = b"".join(errors).decode(errors="replace").strip()
                sys.exit(f"could not stream images from {self.host}: {msg or e}")
        finally:
            if proc.poll() is None:
                proc.kill()
        for sha, rel in items:  # anything the server did not send
            if sha not in seen:
                yield sha, FileNotFoundError(f"not received from {self.host}: {rel}")

    def upload(self, files, run_name):
        dest = posixpath.join(posixpath.dirname(self.path), "emb", run_name)
        if self.run("mkdir -p " + self.shell(dest), capture_output=True).returncode:
            return None
        scp_dest = dest[2:] if dest.startswith("~/") else dest  # relative paths start at the home folder
        r = subprocess.run(SCP + [str(f) for f in files] + [f"{self.host}:{scp_dest}/"],
                           capture_output=True, creationflags=NOWINDOW)
        return dest if r.returncode == 0 else None


class Local:
    def __init__(self, path):
        self.path = Path(path)

    def read(self, name):
        return (self.path / name).read_bytes()

    def stream(self, items):
        for sha, rel in items:
            try:
                yield sha, (self.path / rel).read_bytes()
            except OSError as e:
                yield sha, e


def open_wiki(spec):
    m = Remote.PATTERN.match(spec)
    return Remote(m.group(1), m.group(2)) if m else Local(spec)


# ---------- model server ----------------------------------------------------

def api(url, route, data=None, timeout=300):
    req = urllib.request.Request(url + route,
                                 data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return json.load(res)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from e


def prepare(data, edge):
    """Same preprocessing as the benchmark: EXIF rotate, RGB, cap longest edge, JPEG q90."""
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))  # transparent areas become white
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        if edge:
            im.thumbnail((edge, edge), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    return {"prompt_string": PROMPT, "multimodal_data": [base64.b64encode(buf.getvalue()).decode()]}


def embed(url, contents, dim):
    ans = api(url, "/embedding", {"content": contents[0] if len(contents) == 1 else contents,
                                  "embd_normalize": 2, "cache_prompt": False})
    if isinstance(ans, dict):
        ans = [ans]
    ans = sorted(ans, key=lambda a: a.get("index", 0))
    vecs = []
    for item in ans:
        e = item["embedding"]
        v = np.asarray(e[0] if isinstance(e[0], list) else e, dtype=np.float32)
        if v.shape != (dim,) or not np.isfinite(v).all():
            raise RuntimeError(f"bad vector: shape {v.shape}")
        vecs.append(v)
    if len(vecs) != len(contents):
        raise RuntimeError(f"asked for {len(contents)} vectors, got {len(vecs)}")
    return vecs


def start_server(model, parallel, image_cap, log_path):
    path, projector, _ = MODELS[model]
    for p in (SERVER, path, projector):
        if not Path(p).exists():
            sys.exit(f"missing file: {p}\n(edit the paths at the top of this script)")
    env = os.environ.copy()
    env["LLAMA_MEDIA_MARKER"] = "<__media__>"
    env["PATH"] = str(CUDA_DLLS) + os.pathsep + env["PATH"]
    args = [str(SERVER), "-m", str(path), "--mmproj", str(projector), "-ngl", "all", "-c", "8192",
            "-b", "2048", "-ub", "2048", "-np", str(parallel), "--embedding", "--pooling", "last",
            "-fa", "on", "--host", "127.0.0.1", "--port", str(PORT), "--no-webui", "--no-warmup",
            "--image-min-tokens", "4", "--image-max-tokens", str(image_cap),
            "--override-kv", "tokenizer.ggml.add_eos_token=bool:false"]
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(args, stdout=log, stderr=log, env=env, creationflags=NOWINDOW)
    url, begin = f"http://127.0.0.1:{PORT}", time.perf_counter()
    while True:
        if proc.poll() is not None:
            sys.exit(f"llama-server exited with code {proc.returncode}; see {log_path}")
        try:
            api(url, "/health", timeout=1)
            break
        except Exception:
            if time.perf_counter() - begin > 180:
                proc.terminate()
                sys.exit("llama-server did not become ready in 180s")
            time.sleep(0.4)
    print(f"model loaded in {time.perf_counter() - begin:.1f}s")
    return proc, log, url


def batches(stream, size):
    group = []
    for item in stream:
        group.append(item)
        if len(group) == size:
            yield group
            group = []
    if group:
        yield group


# ---------- main ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("--wiki", required=True, help="user@host:path on the server, or a local folder")
    ap.add_argument("--out", required=True, help="local folder where embeddings are written")
    ap.add_argument("--edge", type=int, default=448, help="cap on the longest image edge; 0 = originals")
    ap.add_argument("--batch", type=int, default=4, help="images per request (= server slots)")
    ap.add_argument("--limit", type=int, help="only the first N images (for a test run)")
    ap.add_argument("--url", help="use an already-running llama-server instead of starting one")
    ap.add_argument("--no-upload", action="store_true", help="do not copy results back to a remote wiki")
    a = ap.parse_args()

    wiki, dim = open_wiki(a.wiki), MODELS[a.model][2]
    run_name = f"qwen{a.model}-{a.edge or 'orig'}"
    out = Path(a.out) / run_name
    shards = out / "shards"
    shards.mkdir(parents=True, exist_ok=True)

    # unique images, in manifest order: [(sha1, path relative to the wiki folder)]
    seen, images = set(), []
    for row in csv.DictReader(io.StringIO(wiki.read("manifest.csv").decode("utf-8"), newline="")):
        if row["sha1"] not in seen:
            seen.add(row["sha1"])
            images.append((row["sha1"], row["path"].replace("\\", "/")))
    if a.limit:
        images = images[:a.limit]
    done = set()
    for f in sorted(shards.glob("part-*.npz")):
        done.update(np.load(f)["ids"].tolist())
    todo = [(s, p) for s, p in images if s not in done]
    where = f"{wiki.host}:{wiki.path}" if isinstance(wiki, Remote) else str(wiki.path)
    print(f"{len(images)} images in {where}: {len(images) - len(todo)} already embedded, {len(todo)} to do")

    proc = log = None
    failed, rel_of = [], dict(images)
    elapsed = load_time = 0.0
    embedded_now = 0
    if todo:
        if a.url:
            url = a.url.rstrip("/")
        else:
            t0 = time.perf_counter()
            proc, log, url = start_server(a.model, a.batch, 1800 if not a.edge else 1024, out / "llama-server.log")
            load_time = time.perf_counter() - t0
        next_part = 1 + max([int(f.stem.split("-")[1]) for f in shards.glob("part-*.npz")], default=-1)
        buf_ids, buf_vecs = [], []

        def flush():
            nonlocal next_part
            if buf_ids:
                tmp = shards / f"part-{next_part:06d}.tmp.npz"
                np.savez(tmp, ids=np.array(buf_ids), vecs=np.stack(buf_vecs))
                os.replace(tmp, shards / f"part-{next_part:06d}.npz")
                next_part += 1
                buf_ids.clear(); buf_vecs.clear()

        def safe_prepare(item):
            sha, data = item
            if isinstance(data, Exception):
                return data
            try:
                return prepare(data, a.edge)
            except Exception as e:
                return RuntimeError(f"unreadable: {e}")

        start, n = time.perf_counter(), 0
        try:
            with ThreadPoolExecutor(a.batch) as pool:
                for k, group in enumerate(batches(wiki.stream(todo), a.batch)):
                    ok = []
                    for (sha, _), c in zip(group, pool.map(safe_prepare, group)):
                        if isinstance(c, Exception):
                            failed.append((sha, rel_of[sha], str(c)[:200]))
                        else:
                            ok.append((sha, c))
                    if ok:
                        try:
                            results = list(zip([s for s, _ in ok], embed(url, [c for _, c in ok], dim)))
                        except Exception:
                            results = []  # retry one at a time to isolate the bad image
                            for sha, c in ok:
                                try:
                                    results.append((sha, embed(url, [c], dim)[0]))
                                except Exception as e:
                                    failed.append((sha, rel_of[sha], str(e)[:200]))
                        for sha, v in results:
                            buf_ids.append(sha); buf_vecs.append(v)
                        embedded_now += len(results)
                    n += len(group)
                    if len(buf_ids) >= SHARD:
                        flush()
                    if k % 25 == 0 or n == len(todo):
                        rate = n / (time.perf_counter() - start)
                        eta = (len(todo) - n) / rate / 60 if rate else 0
                        print(f"\r  {n}/{len(todo)}  {rate:.1f} img/s  about {eta:.0f} min left   ", end="", flush=True)
                    if proc and proc.poll() is not None:
                        raise RuntimeError(f"llama-server died; see {out / 'llama-server.log'}")
        except KeyboardInterrupt:
            print("\nstopped; progress is saved, re-run the same command to continue")
        finally:
            elapsed = time.perf_counter() - start
            flush()
            if proc:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                log.close()
        print()

    # merge shards into one matrix, in manifest order
    table = {}
    for f in sorted(shards.glob("part-*.npz")):
        z = np.load(f)
        for sha, v in zip(z["ids"].tolist(), z["vecs"]):
            table[sha] = v
    ids = [s for s, _ in images if s in table]
    if not ids:
        sys.exit("nothing embedded")
    mat = np.stack([table[s] for s in ids]).astype(np.float32)
    np.save(out / "embeddings.npy", mat)
    (out / "ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    (out / "failed.txt").write_text("".join(f"{s}\t{p}\t{e}\n" for s, p, e in failed), encoding="utf-8")
    norms = np.linalg.norm(mat, axis=1)
    (out / "meta.json").write_text(json.dumps({
        "model": a.model, "model_file": str(MODELS[a.model][0]), "edge": a.edge, "dimensions": int(mat.shape[1]),
        "rows": int(mat.shape[0]), "images_in_manifest": len(images), "failed": len(failed), "wiki": where,
        "norm_min": float(norms.min()), "norm_max": float(norms.max()),
        "prompt": PROMPT, "pooling": "last", "normalized": True,
        "this_run": {"embedded": embedded_now, "seconds": round(elapsed, 1), "model_load_seconds": round(load_time, 1),
                     "images_per_second": round(embedded_now / elapsed, 2) if elapsed else None},
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2), encoding="utf-8")
    if embedded_now:
        m, sec = divmod(int(round(elapsed)), 60)
        took = f"{m}m {sec:02d}s" if m else f"{elapsed:.1f}s"
        print(f"time elapsed: {took} for {embedded_now} images ({embedded_now / elapsed:.1f} img/s)"
              + (f", plus {load_time:.1f}s to load the model" if load_time else ""))
    print(f"embeddings.npy: {mat.shape[0]} x {mat.shape[1]}  (norms {norms.min():.4f} to {norms.max():.4f})")
    print(f"missing: {len(images) - len(ids)}" + (f", of which {len(failed)} failed this run (failed.txt)" if failed else ""))
    print("saved in", out)

    complete = len(ids) + len(failed) >= len(images)
    if isinstance(wiki, Remote) and not a.no_upload and not a.limit and complete:
        dest = wiki.upload([out / n for n in ("embeddings.npy", "ids.txt", "meta.json", "failed.txt")], run_name)
        print(f"copied to {wiki.host}:{dest}" if dest else
              "could not copy the results to the server; they are still saved locally")


if __name__ == "__main__":
    main()
