"""Family table v1: a PROPOSAL for Andrew to accept or correct. Run on ml-db.

Decision it follows (Andrew, 21 Nov): ONE family level for the overall aesthetic. The draft's set-aside family
("mood, season, setting, art and design", 91 labels) is dissolved: each of its labels joins the family whose overall
aesthetic it shares, and the few with no such family form two new ones (seasons and holidays; design, graphics and
screens). Outfit against vibe is from now on a property of the photo, not of the label.

How the placements were made
  - Every placement is by MEANING (Claude Code's reading of what the aesthetic is), to be approved by Andrew.
  - The image vectors were used to decide WHICH labels to look at again: for the 91, the clothing family their
    training images sit nearest; for labels the draft placed by guess, those whose images sit clearly nearer another
    family (cosine gap of 0.20 or more). Where I do not know an aesthetic, the placement follows its images and is
    marked for Andrew.
  - Caveat to report: families shaped with help from the Qwen vectors may favour models that use those vectors.

    python family_v1.py        -> runs/family-v1/{proposal.csv, families.csv, summary.md, review.html}

Training and validation images only. No training, no GPU, no test image. Refuses to run unless all 503 labels are placed.
"""
import base64
import html
import io
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

t0 = time.perf_counter()
ROOT = Path(sys.argv[sys.argv.index("--root") + 1]) if "--root" in sys.argv else Path("/srv/ml/vega")
sys.path.insert(0, str(ROOT / "code" / "shared"))
import vega_data  # noqa: E402

DRAFT = ROOT / "runs" / "family-draft"
OUT = ROOT / "runs" / "family-v1"
OUT.mkdir(parents=True, exist_ok=True)

NEW = {"seasonal": ("Seasons and holidays", "pages about a season or a holiday: scenery, decorations, festive dress"),
       "design": ("Design, graphics and screens", "graphic and interior design looks, art movements, games and screen culture")}

Q = "?"          # marks a placement that is Andrew's call
# The 91 labels the draft set aside: label -> (family, "?" if it is Andrew's call, note)
SET_ASIDE = {
    "Japanese Horror Game Protagonist": ("goth", Q, "plain dresses in a horror setting"),
    "LOLcore": ("glam", Q, "I do not know this one well; placed by its images"),
    "2000s Tuscan Mom": ("casual", Q, "could also be bohemian"),
    "90s Butter Mom": ("casual", "", ""),
    "Too Cool": ("street", Q, "I do not know this one well"),
    "Mall Ninja": ("street", Q, "tactical gear and blades; no clothing family is close"),
    "Sots Art": ("design", "", ""),
    "Slimepunk": ("club", "", ""),
    "Mid-Century Medieval": ("vintage", "", ""),
    "Tropicabana": ("boho", "", ""),
    "Graffiti Pop": ("street", "", ""),
    "Pen & Pixel": ("street", "", ""),
    "Grandmacore": ("outdoors", "", ""),
    "Nerdcore": ("casual", Q, "could also be club and rave"),
    "Yugo-Nostalgia": ("vintage", "", ""),
    "Horror": ("goth", Q, "could also be fantasy and costume"),
    "Corporate Grunge": ("street", Q, "a 1990s advertising look, 93 images; could also be rock and grunge, or design"),
    "Holosexual": ("glam", "", ""),
    "New Partisans": ("club", Q, "I do not know this one well; placed by its images"),
    "Shibuya-Kei": ("vintage", Q, "1990s Tokyo pop with a 1960s look; could also be club and rave"),
    "Incelcore": ("street", Q, "placed by its images"),
    "Tropical": ("boho", "", ""),
    "Pacific Punk Wave": ("street", Q, "1990s surf and skate graphics; could also be club and rave, or design"),
    "Pianocore": ("elegant", Q, "no clothing family is close"),
    "Twilightcore": ("fantasy", "", ""),
    "Gen X Soft Club": ("design", Q, "a late-1990s design look; 4 in 10 images show outfits"),
    "Bastardcore": ("street", "", ""),
    "Winter Aesthetic": ("seasonal", "", ""),
    "Plaguecore": ("fantasy", "", ""),
    "Shibuya Punk": ("design", "", ""),
    "Galaxy Print": ("casual", Q, "a 2010s Tumblr print; could also be cute and kawaii"),
    "Craftcore": ("casual", "", ""),
    "Halloween": ("seasonal", "", ""),
    "Post-Grunge Maximalism": ("street", "", ""),
    "Vaporwave": ("design", "", ""),
    "B-Movie Horror": ("fantasy", "", ""),
    "Ostalgie": ("vintage", Q, "nostalgia for East Germany; no clothing family is close"),
    "Soviet Nostalgia": ("vintage", Q, "its images sit nearest natural and outdoors"),
    "Zen-X": ("design", Q, "a late-1990s design look; could also be bohemian"),
    "Soviet Chic": ("vintage", Q, "could also be club and rave"),
    "Memphis Lite": ("design", "", ""),
    "Pop Art": ("design", "", ""),
    "Beach Day": ("seasonal", Q, "a summer setting; could also be casual"),
    "Diner": ("vintage", "", ""),
    "Ghostcore": ("goth", Q, "could also be fantasy and costume"),
    "Northerness": ("outdoors", "", ""),
    "Psychedelia": ("boho", Q, "1960s hippie art; could also be club and rave, or design"),
    "Rainforest Chic": ("outdoors", Q, "a 1990s jungle-themed design look; could also be design"),
    "Wishcore": ("kawaii", "", ""),
    "Avantropop": ("club", "", ""),
    "Grandmillennial": ("classic", "", ""),
    "Spring Aesthetic": ("seasonal", "", ""),
    "Picniccore": ("romantic", "", ""),
    "2010s Summer Nostalgia": ("casual", Q, "could also be seasons and holidays"),
    "Danish Pastel": ("romantic", Q, "pastel rooms and decor; could also be casual, or design"),
    "Arcadecore": ("design", "", ""),
    "Cybersigilism": ("goth", "", ""),
    "Retrofuturism": ("fantasy", "", ""),
    "Cozycore": ("casual", "", ""),
    "Animecore": ("kawaii", "", ""),
    "Nordic Twee": ("romantic", Q, "no clothing family is close; could also be cute and kawaii"),
    "Spacecore": ("fantasy", "", ""),
    "UrBling": ("street", "", ""),
    "Christmas": ("seasonal", "", ""),
    "Chinese New Year": ("seasonal", Q, "a holiday; its images sit close to regional and traditional dress"),
    "Dark Nautical": ("outdoors", Q, "no clothing family is close; could also be goth and dark"),
    "Starrflesh": ("fantasy", Q, "I do not know this one; placed by its images"),
    "Autumn Aesthetic": ("seasonal", "", ""),
    "Bubble Bujo": ("kawaii", "", ""),
    "Unicorn Trend": ("kawaii", "", ""),
    "Casino": ("glam", Q, "no clothing family is close"),
    "Cleancore": ("design", Q, "soap, tiles and clean rooms, 88 images; no clothing family is close"),
    "Cryptidcore": ("outdoors", "", ""),
    "Raygun Gothic": ("fantasy", "", ""),
    "Whimsicraft": ("design", "", ""),
    "Art Deco": ("vintage", Q, "1920s and 1930s; could also be design, or historical dress"),
    "Feralcore": ("outdoors", Q, "no clothing family is close"),
    "Medicalcore": ("kawaii", Q, "no clothing family is close; could also be goth and dark"),
    "Cozy Gamer": ("casual", "", ""),
    "Café Aesthetic": ("classic", "", ""),
    "Honeycore": ("outdoors", "", ""),
    "Retro Gamer": ("design", "", ""),
    "Tropicalcore": ("boho", Q, "placed with Tropical and Tropicabana; no clothing family is close"),
    "Superflat Pop": ("kawaii", Q, "Murakami-style pop art, 104 images; could also be design"),
    "Bright Tertiaries": ("design", Q, "a colour-palette design look; could also be casual"),
    "Chinoiserie": ("historical", "", ""),
    "Flat Design": ("design", "", ""),
    "Futurism": ("design", "", ""),
    "Glitchcore": ("kawaii", Q, "glitchy rainbow edits; could also be club and rave, or design"),
    "Gloomcore": ("goth", Q, "gloomy scenery, no outfits; its images sit nearest natural and outdoors"),
    "Saint Patrick's Day": ("seasonal", "", ""),
}
# Labels the draft placed in a clothing family by guess, whose images sit clearly nearer another family: MOVED
MOVED = {
    "Cumbiero": ("street", "", "Argentine cumbia street style"),
    "Milipili": ("glam", Q, "I do not know this one well; its images sit far from classic and preppy"),
    "Herki": ("goth", Q, "I do not know this one; placed by its images"),
    "Qingleng Feng": ("regional", "", "a Chinese style; sits with the other Chinese styles"),
    "Peach Girl": ("romantic", "", ""),
    "Hime Gyaru": ("lolita", Q, "a gal style by name, doll-like in its images"),
    "Genderless Kei": ("kawaii", "", ""),
    "Haunted Mound": ("goth", "", ""),
    "Trailer Park Princess": ("romantic", Q, "52 images"),
    "Magical Girl": ("kawaii", "", ""),
    "Playground Prep": ("kawaii", Q, "I do not know this one; placed by its images"),
    "Lulicore": ("classic", Q, "I do not know this one; placed by its images"),
    "Jirai Kei": ("kawaii", "", ""),
    "Hedi Boy": ("mode", "", ""),
    "Tango": ("elegant", Q, "dance dress; was regional and traditional dress"),
    "Boujee": ("elegant", "", ""),
    "Tumbette": ("street", Q, "I do not know this one well; placed by its images"),
    "Cyber Ghetto": ("glam", Q, ""),
    "Heroin Chic": ("mode", Q, "a 1990s fashion-model look; its images sit nearest elegant and dressy"),
    "Shoujo Girl": ("romantic", Q, ""),
    "Keller Synth": ("goth", "", ""),
    "Indie": ("rock", "", ""),
    "Tweencore": ("kawaii", Q, ""),
}
# Kept where the draft put them, but worth Andrew's eye: label -> note
KEPT_ASK = {
    "Clowncore": "the largest label, 141 images; they sit nearest cute and kawaii",
    "Y2K": "102 images, mostly turn-of-the-millennium graphics; could move to design, graphics and screens",
    "Soft Grunge": "84 images; they sit nearest casual and everyday",
    "Industrial": "43 images; they sit nearest rock and punk",
    "Seapunk": "its images sit nearest fantasy and costume",
    "Draincore": "its images sit nearest fantasy and costume",
    "Dark Naturalism": "no outfits in its images, which sit nearest natural and outdoors",
    "Qianxi Feng": "could also sit with the Chinese styles in regional dress",
}

# ---------------------------------------------------------------- load
d = vega_data.load(root=ROOT)
classes = list(d["classes"])
draft = pl.read_csv(DRAFT / "label_family.csv")
dfam = dict(zip(draft["wiki_label"], draft["family"]))
dconf = dict(zip(draft["wiki_label"], draft["first_pass_confidence"]))
name = {f: n for f, n in zip(draft["family"], draft["family_name"])}
meaning = dict(zip(pl.read_csv(DRAFT / "families.csv")["family"], pl.read_csv(DRAFT / "families.csv")["meaning"]))
for k, (n, m) in NEW.items():
    name[k], meaning[k] = n, m
people = pl.read_csv(DRAFT / "label_people.csv")
outfit_share = dict(zip(people["wiki_label"], people["share_with_outfit"]))

aside = {c for c in classes if dfam[c] == "mood"}
assert set(SET_ASIDE) == aside, (sorted(set(SET_ASIDE) - aside), sorted(aside - set(SET_ASIDE)))
for table in (MOVED, KEPT_ASK):
    assert set(table) <= set(classes) - aside, sorted(set(table) - (set(classes) - aside))
assert not set(MOVED) & set(KEPT_ASK)
families = [f for f in name if f != "mood"]
for lab, (f, _, _) in {**SET_ASIDE, **MOVED}.items():
    assert f in families, (lab, f)

prop, call, note = {}, {}, {}
for c in classes:
    if c in SET_ASIDE:
        prop[c], call[c], note[c] = SET_ASIDE[c]
    elif c in MOVED:
        prop[c], call[c], note[c] = MOVED[c]
    else:
        prop[c], call[c], note[c] = dfam[c], (Q if c in KEPT_ASK else ""), KEPT_ASK.get(c, "")
assert len(prop) == 503 and "mood" not in set(prop.values())

# ---------------------------------------------------------------- what the images say (training vectors)
x, y = d["fit"]["x"].astype(np.float32), d["fit"]["y"]
x = x - x.mean(0)
has = y > 0
cent = np.stack([x[has[:, j]].mean(0) for j in range(len(classes))])
cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9

def nearest(assign, fams):
    """For every label: cosine to each family's centre (mean of its labels' centres, the label itself left out)."""
    lab = np.asarray([assign[c] for c in classes])
    mem = {f: np.where(lab == f)[0] for f in fams}
    sums = {f: cent[mem[f]].sum(0) for f in fams}
    res = []
    for j in range(len(classes)):
        cs = []
        for f in fams:
            s, n = sums[f], len(mem[f])
            if lab[j] == f:
                s, n = s - cent[j], n - 1
            v = s / max(n, 1)
            cs.append(float(cent[j] @ v / (np.linalg.norm(v) + 1e-9)) if n > 0 else -1.0)
        res.append(np.asarray(cs))
    return res

cloth = [f for f in families if f not in NEW]
by_draft = nearest(dfam, cloth)                       # against the draft's 18 clothing families
by_v1 = nearest(prop, families)                       # against the proposal's 20 families
near_draft = {c: (cloth[int(np.argmax(v))], float(v.max())) for c, v in zip(classes, by_draft)}
agree_draft = sum(near_draft[c][0] == dfam[c] for c in classes if dfam[c] != "mood")
agree_v1 = sum(families[int(np.argmax(v))] == prop[c] for c, v in zip(classes, by_v1))

# ---------------------------------------------------------------- tables
n_fit = {c: int(has[:, j].sum()) for j, c in enumerate(classes)}
ev_truth = d["eval"]["truth"]
n_val = defaultdict(int)
fam_val = defaultdict(int)
for t in ev_truth:
    for j in t:
        n_val[classes[j]] += 1
    for f in {prop[classes[j]] for j in t}:
        fam_val[f] += 1

review = [c for c in classes if c in SET_ASIDE or c in MOVED or c in KEPT_ASK]
review.sort(key=lambda c: (call[c] != Q, families.index(prop[c]), c.lower()))
number = {c: i + 1 for i, c in enumerate(review)}

pl.DataFrame({
    "wiki_label": classes,
    "family": [prop[c] for c in classes],
    "family_name": [name[prop[c]] for c in classes],
    "draft_family": [dfam[c] for c in classes],
    "changed": ["yes" if prop[c] != dfam[c] else "" for c in classes],
    "status": ["your call" if call[c] == Q else ("proposed" if c in number else "draft kept") for c in classes],
    "note": [note[c] for c in classes],
    "draft_confidence": [dconf[c] for c in classes],
    "images_sit_nearest": [near_draft[c][0] for c in classes],
    "cosine_to_nearest": [round(near_draft[c][1], 3) for c in classes],
    "training_images": [n_fit[c] for c in classes],
    "validation_images": [n_val[c] for c in classes],
    "share_with_outfit": [outfit_share.get(c) for c in classes],
    "review_number": [number.get(c) for c in classes],
}).write_csv(OUT / "proposal.csv")

rows = []
for f in families:
    labs = [c for c in classes if prop[c] == f]
    rows.append((f, name[f], meaning[f], len(labs), sum(c in aside for c in labs), sum(n_fit[c] for c in labs), fam_val[f],
                 ", ".join(f"{c} ({n_fit[c]})" for c in sorted(labs, key=lambda c: -n_fit[c])[:6])))
pl.DataFrame(rows, schema=["family", "name", "meaning", "wiki_labels", "of_them_from_the_set_aside_91", "wiki_training_images",
                           "wiki_validation_images", "largest_wiki_labels"], orient="row").write_csv(OUT / "families.csv")

n_ev = len(ev_truth)
big = max(families, key=lambda f: fam_val[f])
n_call = sum(call[c] == Q for c in review)
summary = ["# Family table v1: PROPOSAL (not accepted; Andrew's review pending)", "",
           "One family level for the overall aesthetic (Andrew, 21 Nov). The draft's set-aside family is dissolved: its 91 labels join "
           f"the family whose overall aesthetic they share, or one of two new families. {len(families)} families over 503 labels.", "",
           f"- Set-aside labels placed: 91 ({sum(prop[c] in NEW for c in aside)} into the two new families, {sum(prop[c] not in NEW for c in aside)} into clothing families).",
           f"- Labels moved between clothing families: {len(MOVED)} (placed by guess in the draft; their images sit at least 0.20 cosine nearer another family).",
           f"- Shown to Andrew: {len(review)} labels (review.html), {n_call} marked as his call. The other {503 - len(review)} keep the draft's family.",
           f"- Largest family on validation: {name[big]}, {fam_val[big]:,} of {n_ev:,} images ({100 * fam_val[big] / n_ev:.1f}%). In the draft it was the set-aside family with 20.9%.",
           f"- Labels whose images sit nearest their own family (the label left out of its family's centre): {agree_v1} of 503 "
           f"({100 * agree_v1 / 503:.0f}%) under this proposal; {agree_draft} of 412 ({100 * agree_draft / 412:.0f}%) for the draft's clothing families.",
           "- Caveat: the image vectors chose which labels were looked at again and decided the placement of the labels I do not know, "
           "so the table may favour models that use those vectors. To be stated in the report.", "",
           "| family | labels | from the set-aside 91 | training images | validation images | largest labels |", "|---|---|---|---|---|---|"]
for r in sorted(rows, key=lambda r: -r[5]):
    summary.append(f"| {r[1]} | {r[3]} | {r[4]} | {r[5]:,} | {r[6]:,} | {r[7]} |")
summary += ["", f"Time elapsed: {time.perf_counter() - t0:.0f} s (before the review page)."]
(OUT / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- review page (photos from training and validation only)
paths = defaultdict(list)
for part in ("fit", "eval"):
    for p, t in zip(d[part]["paths"], d[part]["truth"]):
        for j in t:
            paths[classes[j]].append(p)
rng = np.random.default_rng(422)

def strip(files, h=116, k=8):
    files = sorted(files)
    pick = [files[i] for i in sorted(rng.choice(len(files), min(k, len(files)), replace=False))]
    ims = []
    for p in pick:
        try:
            with Image.open(p) as src:
                im = src.convert("RGB")
            w = max(1, round(im.width * h / im.height))
            im = im.resize((w, h), Image.LANCZOS)
            if w > 2 * h:
                im = im.crop(((w - 2 * h) // 2, 0, (w - 2 * h) // 2 + 2 * h, h))
            ims.append(im)
        except Exception:
            continue
    if not ims:
        return "", 0
    canvas = Image.new("RGB", (sum(i.width for i in ims) + 3 * (len(ims) - 1), h), "white")
    at = 0
    for im in ims:
        canvas.paste(im, (at, 0)); at += im.width + 3
    buf = io.BytesIO()
    canvas.save(buf, "WEBP", quality=55)
    return base64.b64encode(buf.getvalue()).decode(), canvas.width

E = html.escape
cards = {True: [], False: []}
for c in review:
    b64, width = strip(paths[c])
    was = "set aside in the draft" if c in aside else f"draft: {name[dfam[c]]}"
    verb = "stays in" if prop[c] == dfam[c] else "&rarr;"
    o = outfit_share.get(c)
    facts = [was, f"{n_fit[c] + n_val[c]} images", f"outfit in {100 * o:.0f}%" if o is not None else "",
             f"images sit nearest {name[near_draft[c][0]]} ({near_draft[c][1]:.2f})"]
    cards[call[c] == Q].append(
        f"<div class='card'><div class='head'><span class='num'>{number[c]}</span> <b>{E(c)}</b> {verb} <span class='fam'>{E(name[prop[c]])}</span></div>"
        + (f"<div class='note'>{E(note[c])}</div>" if note[c] else "")
        + f"<div class='facts'>{E(' · '.join(x for x in facts if x))}</div>"
        + (f"<div class='scroll'><img src='data:image/webp;base64,{b64}' width='{width}' height='116' alt=''></div>" if b64 else "")
        + "</div>")
legend = "".join(f"<li><b>{E(name[f])}</b>: {E(meaning[f])}</li>" for f in families)
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Family table v1: review</title><style>
:root{{--bg:#fafaf7;--fg:#1c1c1a;--mut:#666;--card:#fff;--line:#ddd;--acc:#8a3b12}}
@media (prefers-color-scheme: dark){{:root{{--bg:#161614;--fg:#ecece6;--mut:#a3a39b;--card:#20201d;--line:#3a3a35;--acc:#f0a070}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}}
main{{max-width:980px;margin:0 auto}} h1{{font-size:22px;margin:0 0 6px}} h2{{font-size:18px;margin:28px 0 8px}}
p,li{{max-width:70ch}} .mut{{color:var(--mut)}} ul{{padding-left:20px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:10px 0}}
.num{{display:inline-block;min-width:28px;text-align:center;border:1px solid var(--line);border-radius:6px;padding:0 5px;margin-right:4px;font-variant-numeric:tabular-nums}}
.fam{{color:var(--acc);font-weight:600}} .note{{margin-top:3px}} .facts{{color:var(--mut);font-size:13px;margin-top:2px}}
.scroll{{overflow-x:auto;margin-top:8px}} .scroll img{{display:block;border-radius:4px}}
details{{margin:8px 0}}
</style></head><body><main>
<h1>Family table v1: your yes or no</h1>
<p>One family level for the overall aesthetic. The 91 labels the draft had set aside are placed here, and {len(MOVED)} labels move
between clothing families because their images disagreed with my first guess. Photos are from the training and validation images only.</p>
<p><b>How to answer:</b> reply with the numbers you disagree with and the family you want, for example
<i>&ldquo;12 goth, 40 design&rdquo;</i>. Anything you do not mention stays as proposed.</p>
<details><summary>The {len(families)} families</summary><ul>{legend}</ul></details>
<h2>Your call ({len(cards[True])})</h2>
<p class="mut">Placements I am unsure of, or large labels where the choice moves many images.</p>
{''.join(cards[True])}
<h2>Probably fine ({len(cards[False])})</h2>
<p class="mut">Skim these; speak up only if one looks wrong.</p>
{''.join(cards[False])}
</main></body></html>"""
(OUT / "review.html").write_text(page, encoding="utf-8")
print("\n".join(summary))
print(f"review.html: {len(review)} labels, {len(page) / 1e6:.1f} MB; total time elapsed {time.perf_counter() - t0:.0f} s")
