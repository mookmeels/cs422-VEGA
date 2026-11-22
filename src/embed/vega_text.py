#!/usr/bin/env python3
r"""
vega_text.py - embed each aesthetic's description as text, several ways, and
measure which way labels images best.

Keep this file next to vega_embed.py (it reuses its server and ssh code).

    python vega_text.py 2b --wiki ml-db-lan:/srv/ml/vega/wiki --out E:\vega\emb

For every aesthetic with a gallery it builds four descriptions from the wiki page:
    name      just the name
    infobox   name + key motifs, colours and values
    fashion   infobox + the page's Fashion section
    intro     name + the page's opening paragraph
embeds each with the same Qwen model as the images, then scores every variant
against the wiki gallery images you already embedded: an image counts as a hit
when the aesthetic whose gallery it came from is among the top-k text matches.
The variant with the best score is the one to label the fashion photos with.

Writes  <out>\qwen<model>-text\titles.txt          aesthetic for each row
        <out>\qwen<model>-text\text-<variant>.npy  float32, rows x dimensions, L2-normalised
        <out>\qwen<model>-text\texts.jsonl         the exact text that was embedded
        <out>\qwen<model>-text\report.json         scores per variant
and copies them next to the wiki folder on the server when --wiki is remote.
"""
import argparse, csv, io, json, re, sys, time
from pathlib import Path

import numpy as np

import vega_embed as ve

SYSTEM = "<|im_start|>system\nRepresent the user's input.<|im_end|>\n"
VARIANTS = ["name", "infobox", "fashion", "intro"]
MAX_WORDS = 160


# ---------- turning wikitext into plain sentences ---------------------------

def strip_templates(s):
    out, depth, i = [], 0, 0
    while i < len(s):
        two = s[i:i + 2]
        if two == "{{":
            depth += 1; i += 2
        elif two == "}}" and depth:
            depth -= 1; i += 2
        else:
            if not depth:
                out.append(s[i])
            i += 1
    return "".join(out)


def clean(s):
    s = re.sub(r"<!--.*?-->", " ", s, flags=re.S)
    s = re.sub(r"<ref[^>]*/>", " ", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<gallery[^>]*>.*?</gallery>", " ", s, flags=re.S | re.I)
    s = strip_templates(s)
    s = re.sub(r"\[\[(?:File|Image|Category):[^\]]*\]\]", " ", s, flags=re.I)
    s = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", s)      # [[link|text]] -> text
    s = re.sub(r"\[\[([^\]]*)\]\]", r"\1", s)               # [[link]] -> link
    s = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", s)    # [url text] -> text
    s = re.sub(r"\[?https?://\S+\]?", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"'{2,}", "", s)
    s = re.sub(r"^[*#:;]+\s*", "", s, flags=re.M)           # list markers
    s = re.sub(r"\s*\n\s*", ", ", s.strip())
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"(,\s*)+", ", ", s).strip(" ,")
    return s


def truncate(s, words=MAX_WORDS):
    w = s.split()
    return " ".join(w[:words])


def section(wikitext, name):
    """Text of a ==Name== section (any heading level), up to the next heading of the same or higher level."""
    m = re.search(r"^(={2,})\s*" + re.escape(name) + r"\s*\1\s*$", wikitext, flags=re.M | re.I)
    if not m:
        return ""
    level = len(m.group(1))
    rest = wikitext[m.end():]
    nxt = re.search(r"^={2," + str(level) + r"}[^=].*$", rest, flags=re.M)
    body = rest[:nxt.start()] if nxt else rest
    return clean(re.sub(r"^=+.*?=+\s*$", "", body, flags=re.M))


def intro(wikitext, name=""):
    wikitext = re.sub(r"\{\{\s*PAGENAME\s*\}\}", name, wikitext)
    head = re.split(r"^==", wikitext, maxsplit=1, flags=re.M)[0]
    for para in re.split(r"\n\s*\n", strip_templates(head)):
        text = clean(para)
        if len(text.split()) >= 8:
            return text
    return clean(head)


def describe(page):
    name, box, wt = page["title"], page.get("infobox") or {}, page.get("wikitext") or ""
    def field(*keys):
        for k in keys:
            if box.get(k):
                return clean(box[k]).rstrip(" .")
        return ""
    bits = []
    for label, keys in (("Key motifs", ("key_motifs",)),
                        ("Key colours", ("key_colours", "key_colors")),
                        ("Key values", ("key_values",))):
        v = field(*keys)
        if v:
            bits.append(f"{label}: {v}.")
    info = " ".join(bits)
    fashion = section(wt, "Fashion")
    first = intro(wt, name)
    base = f"{name} aesthetic."
    return {
        "name": base,
        "infobox": truncate(f"{base} {info}" if info else f"{base} {first}"),
        "fashion": truncate(f"{base} {info} Fashion: {fashion}" if fashion else (f"{base} {info}" if info else f"{base} {first}")),
        "intro": truncate(f"{base} {first}"),
    }, bool(fashion), bool(info)


# ---------- embedding and scoring -------------------------------------------

def embed_texts(url, texts, dim, batch):
    out = []
    for i in range(0, len(texts), batch):
        group = [f"{SYSTEM}<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n" for t in texts[i:i + batch]]
        out.extend(ve.embed(url, group, dim))
        print(f"\r    {min(i + batch, len(texts))}/{len(texts)}", end="", flush=True)
    print()
    return np.stack(out).astype(np.float32)


def score(img, truth, txt, chunk=2048):
    """Rank of each image's best true aesthetic among all text vectors."""
    ranks = []
    for i in range(0, len(img), chunk):
        s = img[i:i + chunk] @ txt.T
        for row, labels in zip(s, truth[i:i + chunk]):
            ranks.append(1 + int((row > row[labels].max()).sum()))
    r = np.asarray(ranks)
    return {"recall@1": float((r <= 1).mean()), "recall@5": float((r <= 5).mean()),
            "recall@10": float((r <= 10).mean()), "mrr": float((1 / r).mean()),
            "median_rank": float(np.median(r)), "images": int(len(r))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(ve.MODELS))
    ap.add_argument("--wiki", required=True, help="user@host:path on the server, or a local folder")
    ap.add_argument("--out", required=True, help="local folder holding the embeddings (same --out as vega_embed.py)")
    ap.add_argument("--images", help="folder with the image embeddings to score against (default <out>\\qwen<model>-448)")
    ap.add_argument("--min-gallery", type=int, default=5, help="leave out aesthetics with fewer gallery images")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--url", help="use an already-running llama-server")
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()

    wiki, dim = ve.open_wiki(a.wiki), ve.MODELS[a.model][2]
    run_name = f"qwen{a.model}-text"
    out = Path(a.out) / run_name
    out.mkdir(parents=True, exist_ok=True)
    img_dir = Path(a.images) if a.images else Path(a.out) / f"qwen{a.model}-448"

    # which aesthetic each image belongs to
    labels_of, count = {}, {}
    for row in csv.DictReader(io.StringIO(wiki.read("manifest.csv").decode("utf-8"), newline="")):
        labels_of.setdefault(row["sha1"], set()).add(row["aesthetic"])
    for names in labels_of.values():
        for n in names:
            count[n] = count.get(n, 0) + 1

    pages = [json.loads(l) for l in wiki.read("pages.jsonl").decode("utf-8").split("\n") if l.strip()]
    pages = [p for p in pages if count.get(p["title"], 0) >= a.min_gallery]
    pages.sort(key=lambda p: p["title"])
    titles = [p["title"] for p in pages]
    index = {t: i for i, t in enumerate(titles)}
    texts, has_fashion, has_info = [], 0, 0
    for p in pages:
        d, f, i = describe(p)
        texts.append(d); has_fashion += f; has_info += i
    print(f"{len(titles)} aesthetics with {a.min_gallery}+ gallery images "
          f"({has_info} have infobox fields, {has_fashion} have a Fashion section)")

    with open(out / "texts.jsonl", "w", encoding="utf-8") as fh:
        for t, d in zip(titles, texts):
            fh.write(json.dumps({"aesthetic": t, **d}, ensure_ascii=False) + "\n")
    (out / "titles.txt").write_text("\n".join(titles) + "\n", encoding="utf-8")

    proc = log = None
    start = time.perf_counter()
    try:
        if a.url:
            url = a.url.rstrip("/")
        else:
            proc, log, url = ve.start_server(a.model, a.batch, 1024, out / "llama-server.log")
        vecs = {}
        for v in VARIANTS:
            print(f"  embedding '{v}' descriptions")
            vecs[v] = embed_texts(url, [d[v] for d in texts], dim, a.batch)
            np.save(out / f"text-{v}.npy", vecs[v])
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except Exception:
                proc.kill()
            log.close()
    print(f"time elapsed: {time.perf_counter() - start:.1f}s")

    report = {"model": a.model, "aesthetics": len(titles), "min_gallery": a.min_gallery, "variants": {}}
    if (img_dir / "embeddings.npy").exists():
        mat = np.load(img_dir / "embeddings.npy")
        ids = (img_dir / "ids.txt").read_text(encoding="utf-8").split()
        fashion_pages = {p["title"] for p in pages if "Fashion" in p.get("categories", [])}
        keep, truth, is_fashion = [], [], []
        for row, sha in enumerate(ids):
            lab = [index[n] for n in labels_of.get(sha, ()) if n in index]
            if lab:
                keep.append(row); truth.append(np.asarray(lab))
                is_fashion.append(any(titles[j] in fashion_pages for j in lab))
        img, is_fashion = mat[keep], np.asarray(is_fashion)
        chance5 = 5 / len(titles)
        print(f"\nscored on {len(img)} wiki gallery images; picking 5 of {len(titles)} at random would hit {chance5:.1%}")
        print(f"  {'variant':<9}{'recall@1':>10}{'recall@5':>10}{'recall@10':>11}{'MRR':>8}{'median rank':>13}   recall@5, fashion-tagged only")
        for v in VARIANTS:
            s = score(img, truth, vecs[v])
            f = score(img[is_fashion], [t for t, k in zip(truth, is_fashion) if k], vecs[v]) if is_fashion.any() else None
            report["variants"][v] = {"all": s, "fashion_tagged": f}
            print(f"  {v:<9}{s['recall@1']:>10.1%}{s['recall@5']:>10.1%}{s['recall@10']:>11.1%}{s['mrr']:>8.3f}{s['median_rank']:>13.0f}"
                  + (f"   {f['recall@5']:.1%} of {f['images']}" if f else ""))
        best = max(VARIANTS, key=lambda v: report["variants"][v]["all"]["recall@5"])
        report["best_by_recall@5"] = best
        print(f"\nbest by recall@5: {best}")
    else:
        print(f"\nno image embeddings found in {img_dir}, so the variants were saved but not scored")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved in", out)

    if isinstance(wiki, ve.Remote) and not a.no_upload:
        files = [out / n for n in ["titles.txt", "texts.jsonl", "report.json"] + [f"text-{v}.npy" for v in VARIANTS]]
        dest = wiki.upload(files, run_name)
        print(f"copied to {wiki.host}:{dest}" if dest else "could not copy to the server; results are still saved locally")


if __name__ == "__main__":
    main()
