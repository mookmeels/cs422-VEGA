"""VEGA, the app: give it a few photos (3 to 5 is the idea, up to 50 are taken), get an aesthetic profile, nearby aesthetics, surprising ones, and the photo behind each.

    E:\\vega\\.venv\\Scripts\\python.exe -B vega_app.py [--root E:/vega] [--port 8770]       then open http://127.0.0.1:8770

A demonstration on this PC only (it listens on 127.0.0.1). Nothing is installed: Python's own web server, the team's
embedder (llama.cpp with Qwen3-VL-Embedding-2B on the GPU, through code/experiments/embed_eos_check.py), the NN from
code/nn/train_nn.py, and the engine in vega_profile.py.

At start it trains the NN once on the GPU (frozen settings, street photos, seed 422) on the wiki training AND validation
images, because this network is for showing, not for scoring: it is not a row of the team's table. The test split is not read.
Uploaded photos are embedded, scored and deleted; nothing about them is kept.
"""
import base64
import io
import json
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
ROOT, PORT = Path(arg("--root", "E:/vega")), int(arg("--port", 8770))
IMAGES = Path(arg("--images", "E:/pack-test/wiki"))
EMBED_PY = arg("--embed-python", sys.executable)
SURPRISE_MIN = 0.41        # below this, "nothing surprising". Set on expert-labelled outfit photos so that 1 set in 10 with no off-style
#                            photo is shown a surprise; 29% of sets that do hold one clear it (runs/personas-v1/weight-variants.md)
POOL = "agree"             # how a person's photos are combined: each counts by how much it agrees with the others (vega_profile.pooled)
MAX_PHOTOS = 50
AGREE_UP_TO = 16           # beyond this many photos the same weights are used with an ordinary average ("agree_mean"): the geometric mean
#                            starts to lose to the plain average at about 20 photos (runs/personas-v1/many-photos.md)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code" / "shared"))
sys.path.insert(0, str(ROOT / "code" / "nn"))
sys.path.insert(0, str(HERE))
import train_nn  # noqa: E402
import vega_data  # noqa: E402
import vega_profile as vp  # noqa: E402

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
log("loading the wiki images' vectors and training the network once ...")
d = vega_data.load(root=ROOT)                                          # training and validation; the test split is not read
classes = [str(c) for c in d["classes"]]
X = np.concatenate([d["fit"]["x"], d["eval"]["x"]]).astype(np.float32)
Y = np.concatenate([d["fit"]["y"], d["eval"]["y"]]).astype(np.float32)
PATHS = [IMAGES / Path(str(p)).relative_to(ROOT / "wiki") for p in list(d["fit"]["paths"]) + list(d["eval"]["paths"])]
import csv  # noqa: E402
_cuts = json.loads((HERE / "surprise_cuts.json").read_text(encoding="utf-8"))      # made by many_photos.py: the cut per number of photos
assert _cuts["agree_up_to"] == AGREE_UP_TO and _cuts["small_cut"] == SURPRISE_MIN, "surprise_cuts.json was made for other settings: rerun many_photos.py"


def surprise_cut(n):
    """From which score a surprise is shown for n photos: the validated cut up to 5 photos, then the table (1 one-style set in 10)."""
    return SURPRISE_MIN if n <= 5 else float(_cuts["cut"][str(min(n, MAX_PHOTOS))])


fam_rows = list(csv.DictReader(open(ROOT / "runs" / "family-v1" / "label_family.csv", encoding="utf-8", newline="")))
FAMILY = {r["wiki_label"]: r["family_name"] for r in fam_rows}
_keys = sorted({r["family"] for r in fam_rows})
_key_of = {r["wiki_label"]: r["family"] for r in fam_rows}
LAB_FAM = np.asarray([_keys.index(_key_of[c]) for c in classes])       # each aesthetic's family, as a number
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
SIM = vp.aesthetic_map(Xn.astype(np.float64), Y > 0)
SIMF = vp.family_map(SIM, LAB_FAM, len(_keys))
CEN = ((Y > 0).T.astype(np.float64) @ Xn) / np.maximum((Y > 0).sum(0), 1)[:, None]
CEN /= np.linalg.norm(CEN, axis=1, keepdims=True)
MEAN = X.mean(0)                                                       # for "closeness": how near a photo is to its nearest aesthetic once the
CENC = ((Y > 0).T.astype(np.float64) @ (X - MEAN)) / np.maximum((Y > 0).sum(0), 1)[:, None]      # average image is taken away (as in one_photo.py)
CENC /= np.linalg.norm(CENC, axis=1, keepdims=True) + 1e-9
one =(Y > 0).sum(1) == 1
EXAMPLES = {}                                                          # a few example images of each aesthetic: those closest to its average
for j in range(len(classes)):
    rows = np.flatnonzero((Y[:, j] > 0) & one)
    rows = rows if len(rows) else np.flatnonzero(Y[:, j] > 0)
    EXAMPLES[j] = [int(r) for r in rows[np.argsort(-(Xn[rows] @ CEN[j]))][:3]]

_net = {}
_predict = train_nn.predict


def _spy(net, x):                                                      # keep the trained network; train() itself is left untouched
    _net["net"] = net
    return _predict(net, x)


train_nn.predict = _spy
t0 = time.time()
train_nn.train(X, Y, X[:4], train_nn.SETTINGS, 422, "cuda", real=train_nn.real_photos(ROOT, classes))
train_nn.predict = _predict
SCALE = train_nn.input_scaling(X, train_nn.SETTINGS["prep"])
log(f"network ready in {time.time() - t0:.0f}s: {len(classes)} aesthetics, trained on {len(X):,} wiki images and street photos")
LOCK = threading.Lock()


def embed(folder, names):
    """The team's embedder on these files; returns one vector per file, in order."""
    (folder / "list.tsv").write_text("".join(f"p{i}\t{n}\n" for i, n in enumerate(names)), encoding="utf-8")
    cmd = [EMBED_PY, "-u", str(ROOT / "code" / "experiments" / "embed_eos_check.py"), "2b", "--list", str(folder / "list.tsv"), "--out", str(folder / "out"),
           "--wiki", str(folder / "photo"), "--no-text", "--name", "one", "--port", "18083", "--instruction", "Represent the fashion aesthetic of this image."]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT / "code" / "experiments"))
    out = folder / "out" / "one-448"
    if r.returncode != 0 or not (out / "embeddings.npy").exists():
        raise RuntimeError("the embedder failed: " + (r.stderr or r.stdout)[-400:])
    ids = (out / "ids.txt").read_text(encoding="utf-8").split()
    vec = np.load(out / "embeddings.npy")
    if len(ids) != len(names):
        raise RuntimeError(f"{len(names) - len(ids)} photo(s) could not be read")
    return vec[[ids.index(f"p{i}") for i in range(len(names))]]


def analyse(data_urls):
    from PIL import Image
    folder = ROOT / "runs" / "app-uploads" / time.strftime("%Y%m%d-%H%M%S")
    (folder / "photo").mkdir(parents=True)
    try:
        names = []
        for i, u in enumerate(data_urls):
            im = Image.open(io.BytesIO(base64.b64decode(u.split(",", 1)[1])))
            im.seek(0)
            im.convert("RGB").save(folder / "photo" / f"p{i}.png")
            names.append(f"p{i}.png")
        t1 = time.time()
        v = embed(folder, names).astype(np.float32)
        t2 = time.time()
        with torch.no_grad():
            z = _predict(_net["net"], torch.from_numpy(SCALE(v)).to("cuda")).astype(np.float64)
        z -= z.max(1, keepdims=True)
        P = np.exp(z)
        P /= P.sum(1, keepdims=True)
        cut = surprise_cut(len(P))
        out = vp.suggest(P, np.ones(len(P), bool), SIM, mode=POOL if len(P) <= AGREE_UP_TO else "agree_mean", lab_fam=LAB_FAM, simf=SIMF,
                         cut=cut, rank=vp.support_rank(len(P)))      # surprise per family; a family
        #                                                                 shown as surprising is not also listed under "your aesthetics"
        q = v - MEAN
        close = ((q / np.linalg.norm(q, axis=1, keepdims=True)) @ CENC.T).max(1)      # about 0.3 for a logo or a screenshot, 0.6 and up for a clear aesthetic photo
        card = lambda a: {"id": a, "name": classes[a], "family": FAMILY[classes[a]], "examples": [f"/ex/{a}/{k}" for k in range(len(EXAMPLES[a]))]}
        return {"profile": [{**card(a), "share": p, "photo": ph} for a, p, ph in out["profile"]],
                "nearest": [{**card(a), "similarity": s} for a, s in out["nearest"]],
                "surprising": [{**card(a), "score": s, "relevance": rel, "unexpected": un, "photo": ph}
                               for a, s, rel, un, ph in out["surprising"] if s >= cut],
                "best_surprise": out["surprising"][0][1], "surprise_min": cut,
                "photos": [{"first": classes[int(P[i].argmax())], "share": float(P[i].max()), "closeness": float(close[i])} for i in range(len(P))],
                "seconds": {"embedding": round(t2 - t1, 1), "everything": round(time.time() - t1, 1)}}
    finally:
        shutil.rmtree(folder, ignore_errors=True)                      # the photos are not kept


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, kind):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if kind.startswith("image/") else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self.send(200, (HERE / "vega_app.html").read_bytes(), "text/html; charset=utf-8")
        parts = self.path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "ex" and parts[1].isdigit() and parts[2].isdigit():
            a, k = int(parts[1]), int(parts[2])
            if a in EXAMPLES and k < len(EXAMPLES[a]) and PATHS[EXAMPLES[a][k]].exists():
                from PIL import Image
                im = Image.open(PATHS[EXAMPLES[a][k]])
                im.seek(0)
                im = im.convert("RGB")
                im.thumbnail((360, 360))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=82)
                return self.send(200, buf.getvalue(), "image/jpeg")
        self.send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/analyse":
            return self.send(404, b"not found", "text/plain")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            photos = body.get("photos", [])
            if not 1 <= len(photos) <= MAX_PHOTOS:
                raise ValueError(f"send 1 to {MAX_PHOTOS} photos")
            with LOCK:
                result = analyse(photos)
            log(f"read {len(photos)} photo(s) in {result['seconds']['everything']}s; first: {result['profile'][0]['name']}")
            self.send(200, json.dumps(result).encode("utf-8"), "application/json")
        except Exception as e:                                         # tell the page what went wrong
            log("failed:", repr(e)[:300])
            self.send(500, json.dumps({"error": str(e)[:400]}).encode("utf-8"), "application/json")


log(f"VEGA is at http://127.0.0.1:{PORT}")
ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
