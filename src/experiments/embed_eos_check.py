r"""Embedding diagnostic: does the end-of-text token matter?   (Runs on the PC, next to E:\vega\scripts.)

vega_embed.py starts the model server with `--override-kv tokenizer.ggml.add_eos_token=bool:false`, so the
vector is read at the last token of the prompt ("assistant\n"). Both model files say add_eos_token = True:
left alone, the server appends <|endoftext|> and the vector is read there instead. Qwen's description of the
model says the vector is the hidden state of the end-of-sequence token. This script embeds the same images and
the same "intro" texts with that token left ON, so the two settings can be compared on validation images.

    python embed_eos_check.py 2b --list images.tsv --out E:\vega\emb
    python embed_eos_check.py 8b --list images.tsv --out E:\vega\emb --limit 300

images.tsv: one "sha1<TAB>path relative to the wiki folder" per line.
Writes  <out>\qwen<model>eos-448\embeddings.npy, ids.txt, meta.json
        <out>\qwen<model>eos-text\text-intro.npy, titles.txt      (texts taken from <out>\qwen2b-text\texts.jsonl)
Safe to stop and re-run: finished images are kept in <out>\qwen<model>eos-448\parts and skipped.
Nothing here touches qwen2b-448 or any other existing folder.
"""
import argparse, datetime, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import vega_embed as ve                                         # noqa: E402
import vega_text as vt                                          # noqa: E402

EOS_ID = 151643                                                 # <|endoftext|>


def prepare_cropped(data, edge, box):
    """vega_embed.prepare with one extra step: crop to box (x0,y0,x1,y1 in the EXIF-rotated image) before shrinking."""
    import base64, io
    from PIL import Image, ImageOps
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        im = im.crop(box)
        im.thumbnail((edge, edge), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    return {"prompt_string": ve.PROMPT, "multimodal_data": [base64.b64encode(buf.getvalue()).decode()]}


def start_server(model, parallel, log_path):
    """vega_embed.start_server without the add_eos_token override."""
    path, projector, _ = ve.MODELS[model]
    env = os.environ.copy()
    env["LLAMA_MEDIA_MARKER"] = "<__media__>"
    env["PATH"] = str(ve.CUDA_DLLS) + os.pathsep + env["PATH"]
    args = [str(ve.SERVER), "-m", str(path), "--mmproj", str(projector), "-ngl", "all", "-c", "8192",
            "-b", "2048", "-ub", "2048", "-np", str(parallel), "--embedding", "--pooling", "last",
            "-fa", "on", "--host", "127.0.0.1", "--port", str(ve.PORT), "--no-webui", "--no-warmup",
            "--image-min-tokens", "4", "--image-max-tokens", "1024"]
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(args, stdout=log, stderr=log, env=env, creationflags=ve.NOWINDOW)
    url, begin = f"http://127.0.0.1:{ve.PORT}", time.perf_counter()
    while True:
        if proc.poll() is not None:
            sys.exit(f"llama-server exited with code {proc.returncode}; see {log_path}")
        try:
            ve.api(url, "/health", timeout=1)
            return proc, log, url
        except Exception:
            if time.perf_counter() - begin > 180:
                proc.terminate()
                sys.exit("llama-server did not become ready in 180s")
            time.sleep(0.4)


def embed_retry(url, contents, dim, tries=4):
    """The local server occasionally fails to accept a connection; wait and ask again."""
    for k in range(tries):
        try:
            return ve.embed(url, contents, dim)
        except Exception as e:
            if k == tries - 1:
                raise
            print(f"  request failed ({str(e)[:90]}); trying again", flush=True)
            time.sleep(2 + 4 * k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(ve.MODELS))
    ap.add_argument("--list", required=True, help="tsv of sha1 and wiki-relative path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wiki", default="ml-db-lan:/srv/ml/vega/wiki")
    ap.add_argument("--limit", type=int, help="only the first N images of the list")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--all-texts", action="store_true", help="embed all four text variants, not only 'intro'")
    ap.add_argument("--no-text", action="store_true", help="images only (for image sets other than the wiki)")
    ap.add_argument("--stop-at", help="local clock time HH:MM: once it passes, save what is done and stop (rerun the same "
                                      "command later to continue). Refuses to start if that time has already passed today.")
    ap.add_argument("--instruction", help="system instruction for IMAGES instead of Qwen's default \"Represent the user's input.\"")
    ap.add_argument("--fast-cores", help="Windows: processor mask for the model server, e.g. 0xFFF = the six fast cores of an i5-13600K. "
                                         "Without it Windows sometimes runs the server on the small cores, at about half the speed.")
    ap.add_argument("--cache-prompt", action="store_true", help="ask the server to reuse the part of the prompt before the image (the instruction) between images")
    ap.add_argument("--instruction-file", help="read the IMAGE instruction from this UTF-8 text file (long, multi-line instructions)")
    ap.add_argument("--user-text", help="experimental: text placed in the user turn right after each IMAGE (a question, for example)")
    ap.add_argument("--name", help="output folder base instead of qwen<model>eos (writes <name>-448 and <name>-text)")
    ap.add_argument("--port", type=int, default=ve.PORT, help=f"llama-server port (default {ve.PORT})")
    ap.add_argument("--edge", type=int, default=448, help="longest edge in pixels after shrinking (default 448; smaller images are not enlarged)")
    ap.add_argument("--crops", help="csv from detect_people.py: images with a crop_box are cropped to it before shrinking")
    a = ap.parse_args()
    crops = {}
    if a.crops:
        import csv as _csv
        for r in _csv.DictReader(open(a.crops, encoding="utf-8", newline="")):
            if r["crop_box"]:
                crops[r["id"]] = tuple(int(v) for v in r["crop_box"].split(","))
        print(f"crops: {len(crops)} images will be cropped to their people", flush=True)
    ve.PORT = a.port
    if a.cache_prompt:                                          # let the server keep the instruction between images
        _api = ve.api

        def api(url, route, data=None, timeout=300):
            if route == "/embedding" and isinstance(data, dict):
                data = dict(data, cache_prompt=True)
            return _api(url, route, data, timeout)
        ve.api = api
        print("prompt caching: on", flush=True)
    if a.instruction_file:                                      # long or multi-line instructions come from a file
        a.instruction = Path(a.instruction_file).read_text(encoding="utf-8")
    if a.instruction:
        instr = a.instruction.strip()
        instr = instr if instr[-1] in ".!?" else instr + "."           # Qwen's reference code adds the period too
        ve.PROMPT = ve.PROMPT.replace("Represent the user's input.", instr)
        assert instr in ve.PROMPT, "could not set the instruction"
        print(f"image instruction: {instr!r}", flush=True)
    if a.user_text:                                             # experimental: text in the user turn, right after the image
        assert "<__media__><|im_end|>" in ve.PROMPT, "the prompt has no image slot to put text after"
        ve.PROMPT = ve.PROMPT.replace("<__media__><|im_end|>", "<__media__>" + a.user_text.strip() + "<|im_end|>")
        print(f"text after the image: {a.user_text.strip()!r}", flush=True)
    t0 = time.perf_counter()
    stop_at = None
    if a.stop_at:
        hh, mm = (int(x) for x in a.stop_at.split(":"))
        stop_at = datetime.datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
        if datetime.datetime.now() >= stop_at:
            sys.exit(f"it is already past {a.stop_at}; not starting")
    stopped = False
    dim = ve.MODELS[a.model][2]
    base = a.name or f"qwen{a.model}eos"
    out, tout = Path(a.out) / f"{base}-448", Path(a.out) / f"{base}-text"
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    tout.mkdir(parents=True, exist_ok=True)
    images = [tuple(l.split("\t")) for l in Path(a.list).read_text(encoding="utf-8").split("\n") if l.strip()]
    if a.limit:
        images = images[:a.limit]
    table = {}
    for f in parts.glob("part-*.tmp.npz"):                      # a part left half-written by a run that was killed
        f.unlink()
    for f in sorted(parts.glob("part-*.npz")):
        z = np.load(f)
        table.update(zip(z["ids"].tolist(), z["vecs"]))
    todo = [(s, p) for s, p in images if s not in table]
    print(f"{len(images)} images listed, {len(images) - len(todo)} already embedded, {len(todo)} to do", flush=True)

    texts = [] if a.no_text else [json.loads(l) for l in open(Path(a.out) / "qwen2b-text" / "texts.jsonl", encoding="utf-8") if l.strip()]
    proc, log, url = start_server(a.model, a.batch, out / "llama-server.log")
    if a.fast_cores:                                            # keep the model server off the processor's small cores
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            f"$p = Get-Process -Id {proc.pid}; $p.ProcessorAffinity = {int(a.fast_cores, 0)}; $p.PriorityClass = 'High'"],
                           capture_output=True, text=True)
        print(f"model server kept on processors {a.fast_cores} at high priority" if r.returncode == 0
              else f"could not set the model server's processors: {r.stderr.strip()[:200]}", flush=True)
    failed, last_tokens = [], None
    try:
        probe = "<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        last_tokens = ve.api(url, "/tokenize", {"content": probe, "add_special": True})["tokens"][-3:]
        print(f"last tokens of a prompt: {last_tokens}  (end-of-text {EOS_ID} appended: {last_tokens[-1] == EOS_ID})", flush=True)
        if last_tokens[-1] != EOS_ID:
            sys.exit("the server is not appending the end-of-text token, so this run would repeat the existing setting")

        for v in ([] if a.no_text else vt.VARIANTS if a.all_texts else ["intro"]):
            f = tout / f"text-{v}.npy"
            if f.exists() and np.load(f, mmap_mode="r").shape == (len(texts), dim):
                print(f"  '{v}' texts already embedded")
                continue
            print(f"  embedding '{v}' texts")
            np.save(f, vt.embed_texts(url, [t[v] for t in texts], dim, a.batch))
        if not a.no_text:
            (tout / "titles.txt").write_text("\n".join(t["aesthetic"] for t in texts) + "\n", encoding="utf-8")
            with open(tout / "texts.jsonl", "w", encoding="utf-8") as fh:      # the exact text that was embedded
                fh.writelines(json.dumps(t, ensure_ascii=False) + "\n" for t in texts)

        wiki = ve.open_wiki(a.wiki)
        nxt = 1 + max([int(f.stem.split("-")[1]) for f in parts.glob("part-*.npz")], default=-1)
        buf, start, n = [], time.perf_counter(), 0

        def flush():
            nonlocal nxt
            if buf:
                tmp = parts / f"part-{nxt:05d}.tmp.npz"
                np.savez(tmp, ids=np.array([s for s, _ in buf]), vecs=np.stack([v for _, v in buf]))
                os.replace(tmp, parts / f"part-{nxt:05d}.npz")
                table.update(buf)
                nxt += 1
                buf.clear()

        def safe_prepare(item):
            try:
                if isinstance(item[1], Exception):
                    return item[1]
                box = crops.get(item[0])
                return prepare_cropped(item[1], a.edge, box) if box else ve.prepare(item[1], a.edge)
            except Exception as e:
                return RuntimeError(f"unreadable: {e}")

        src = wiki.stream(todo)
        with ThreadPoolExecutor(a.batch) as pool:
            for k, group in enumerate(ve.batches(src, a.batch)):
                ok = []
                for (sha, _), c in zip(group, pool.map(safe_prepare, group)):
                    (failed if isinstance(c, Exception) else ok).append((sha, c))
                if ok:
                    buf.extend(zip([s for s, _ in ok], embed_retry(url, [c for _, c in ok], dim)))
                n += len(group)
                if len(buf) >= 512:
                    flush()
                if k % 50 == 0 or n == len(todo):
                    rate = n / (time.perf_counter() - start)
                    print(f"\r  {n}/{len(todo)}  {rate:.1f} img/s  about {(len(todo) - n) / rate / 60:.0f} min left   ", end="", flush=True)
                if stop_at and datetime.datetime.now() >= stop_at:
                    stopped = True
                    print(f"\n  {a.stop_at} reached: stopping with {n} of {len(todo)} done this run (rerun to continue)", flush=True)
                    break
        src.close()                                             # ends the ssh/tar stream if we stopped early
        flush()
        print()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    ids = [s for s, _ in images if s in table]
    mat = np.stack([table[s] for s in ids]).astype(np.float32)
    np.save(out / "embeddings.npy", mat)
    (out / "ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    took = time.perf_counter() - t0
    (out / "meta.json").write_text(json.dumps({
        "model": a.model, "model_file": str(ve.MODELS[a.model][0]), "edge": a.edge, "dimensions": dim, "rows": len(ids),
        "images_listed": len(images), "complete": len(ids) + len(failed) >= len(images), "stopped_at_clock_time": a.stop_at if stopped else None,
        "failed": [str(s) for s, _ in failed], "add_eos_token": True, "last_prompt_tokens": last_tokens,
        "pooling": "last", "normalized": True, "image_prompt": ve.PROMPT, "crops_file": a.crops, "images_cropped": sum(s in crops for s in ids), "seconds": round(took, 1), "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2), encoding="utf-8")
    (out / "failed.txt").write_text("".join(f"{s}\t{str(e)[:200]}\n" for s, e in failed), encoding="utf-8")
    norms = np.linalg.norm(mat, axis=1)
    print(f"embeddings.npy: {mat.shape[0]} x {mat.shape[1]} (norms {norms.min():.4f} to {norms.max():.4f}); "
          f"texts: {len(texts)}; failed: {len(failed)} (failed.txt)")
    print(f"time elapsed: {took:.0f}s. saved in {out} and {tout}")


if __name__ == "__main__":
    main()
