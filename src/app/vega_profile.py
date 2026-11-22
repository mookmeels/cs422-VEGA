"""The application layer: from one person's photos to a profile, nearby aesthetics, surprising aesthetics and an explanation.

Everything here works on what a model already gives for each photo: one probability per aesthetic (a row of P). It follows
the proposal's definitions:

  profile        one ranking for the person, pooled over their photos (three ways, see pooled)
  nearest        the aesthetics closest to the person's top aesthetic on the map of aesthetics
  surprising     S(a) = relevance(a) x unexpectedness(a)
                   relevance(a)      = the strongest match between a and any ONE of the person's photos
                   unexpectedness(a) = 1 - similarity between a and the person's top aesthetic
  explanation    for a suggested aesthetic, the photo that matches it most

The map: SIM[a, b] in 0..1, how alike two aesthetics look (see aesthetic_map).
"""
import numpy as np


def aesthetic_map(x, y):
    """How alike every two aesthetics look, 0..1, from training images only.

    x: image vectors (rows of length 1); y: images x labels, > 0 where the image has the label. Each aesthetic is the average
    of its images' vectors. The average aesthetic is subtracted first (all fashion images share a lot), cosines are taken, and
    the result is stretched so the least alike pair is 0 and the most alike pair is 1."""
    cen = (y.T.astype(np.float64) @ x) / np.maximum(y.sum(0), 1e-12)[:, None]
    cen -= cen.mean(0, keepdims=True)
    cen /= np.linalg.norm(cen, axis=1, keepdims=True) + 1e-12
    cos = cen @ cen.T
    off = cos[~np.eye(len(cos), dtype=bool)]
    sim = np.clip((cos - off.min()) / (off.max() - off.min()), 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    return sim


def pooled(P, outfit, mode="weighted", w_outfit=2.0, power=1.0, lab_fam=None):
    """One ranking for the person. P: photos x aesthetics (probabilities); outfit: True where the photo shows an outfit.

    "outfit"    only the outfit photos (all photos if there is none)
    "equal"     every photo counts the same
    "weighted"  a photo counts w_outfit times if it shows an outfit, times (its strongest match) ** power, so a photo that
                matches nothing in particular says less
    "agree"     (needs lab_fam) a photo counts by how much its family shares overlap with the other photos', and the photos
                are combined by a weighted GEOMETRIC mean: an aesthetic ranks high only if the photos that agree with each
                other all support it. Names the home aesthetic or family first more often than "equal" on every kind of set
                tried (runs/personas-v1/weight-variants.md).
    "agree_mean" the same weights with an ordinary average. For more than about 16 photos: there the geometric mean starts
                to lose to the plain average and this does not (runs/personas-v1/many-photos.md)."""
    P, outfit = np.asarray(P, np.float64), np.asarray(outfit, bool)
    if mode in ("agree", "agree_mean"):
        w = np.ones(len(P))
        if len(P) > 1:
            F = np.zeros((len(P), int(np.max(lab_fam)) + 1))
            np.add.at(F.T, lab_fam, P.T)
            A = np.minimum(F[:, None, :], F[None, :, :]).sum(2)
            w = (A.sum(1) - np.diag(A)) / (len(P) - 1)
            w = w / w.max()
        if mode == "agree_mean":
            return (w[:, None] * P).sum(0) / w.sum()
        g = np.exp((w[:, None] * np.log(P + 1e-9)).sum(0) / w.sum())
        return g / g.sum()
    if mode == "outfit":
        return P[outfit].mean(0) if outfit.any() else P.mean(0)
    if mode == "equal":
        return P.mean(0)
    w = np.where(outfit, w_outfit, 1.0) * P.max(1) ** power
    return (w[:, None] * P).sum(0) / w.sum()


def surprise(P, profile, sim):
    """S for every aesthetic, with its two parts. The top aesthetic itself gets 0 (it is not unexpected)."""
    rel = np.asarray(P, np.float64).max(0)
    unexp = 1.0 - sim[int(np.argmax(profile))]
    return rel * unexp, rel, unexp


def family_map(sim, lab_fam, n_fam):
    """How alike every two FAMILIES look, 0..1: the average of sim over their pairs of aesthetics, stretched like sim."""
    m = np.zeros((len(lab_fam), n_fam))
    m[np.arange(len(lab_fam)), lab_fam] = 1
    count = np.maximum(m.sum(0), 1)
    f = (m.T @ sim @ m) / np.outer(count, count)
    off = f[~np.eye(n_fam, dtype=bool)]
    f = np.clip((f - off.min()) / (off.max() - off.min()), 0.0, 1.0)
    np.fill_diagonal(f, 1.0)
    return f


def support_rank(n):
    """Which photo a family's relevance is read from when a person gives n photos: the strongest up to 10 photos, then the one
    a tenth of the way down (the 2nd strongest of 20, the 5th of 50; in between, a blend of the two neighbours). With many
    photos some stray photo always matches a far family, so one photo is no longer enough (runs/personas-v1/many-photos.md)."""
    return max(1.0, n / 10.0)


def surprise_family(P, profile, simf, lab_fam, rank=1.0):
    """The surprise score worked out one level up, for photos whose probability is spread over a family's aesthetics.

    relevance(F)      = the share of family F in the photo at `rank` from the top for F: rank 1 (default) is the largest
                        share any ONE photo gives F in total; see support_rank for sets of many photos
    unexpectedness(F) = 1 - similarity between F and the person's top family
    Returns S per family, its two parts, the person's top family, and each photo's share per family."""
    P = np.asarray(P, np.float64)
    n_fam = simf.shape[0]
    M = np.zeros((P.shape[1], n_fam))
    M[np.arange(P.shape[1]), lab_fam] = 1
    PF = P @ M
    top = int(np.argmax(np.asarray(profile, np.float64) @ M))
    down = -np.sort(-PF, axis=0)                                      # each family's shares, strongest photo first
    lo = min(int(rank) - 1, len(P) - 1)
    hi = min(lo + 1, len(P) - 1)
    rel = down[lo] + (rank - int(rank)) * (down[hi] - down[lo])
    unexp = 1.0 - simf[top]
    return rel * unexp, rel, unexp, top, PF


def suggest(P, outfit, sim, mode="weighted", w_outfit=2.0, power=1.0, top=5, n=3, lab_fam=None, simf=None, cut=None, rank=1.0):
    """Everything the app shows for one person. Aesthetics are returned as column numbers, best first.

    Surprising suggestions are (aesthetic, S, relevance, unexpectedness, photo). With lab_fam and simf they are worked out per
    family (surprise_family) and each family is shown by the aesthetic in it that one photo matches most; that version names
    the off-style family more often on real outfit photos (runs/personas-v1/surprise-variants.md). Without them, the
    proposal's per-aesthetic score is used.

    cut (with lab_fam): the score from which a surprise is shown. A family that reaches it is shown as surprising, so its
    aesthetics are left out of the profile and of the nearby list. Without this, one off-style photo can lead the profile
    (the other photos spread over several aesthetics of their family, the odd one piles onto one aesthetic) and the same
    aesthetic is then shown as "yours" and as "surprising"."""
    P = np.asarray(P, np.float64)
    profile = pooled(P, outfit, mode, w_outfit, power, lab_fam)
    because = lambda a: int(np.argmax(P[:, a]))                       # the photo that matches this aesthetic most
    hide = np.zeros(P.shape[1], bool)
    if lab_fam is None:
        s, rel, unexp = surprise(P, profile, sim)
        surprising = [(int(a), float(s[a]), float(rel[a]), float(unexp[a]), because(a)) for a in np.argsort(-s)[:n]]
    else:
        s, rel, unexp, _, PF = surprise_family(P, profile, simf, lab_fam, rank)
        surprising = []
        for f in np.argsort(-s)[:n]:
            inside = np.flatnonzero(lab_fam == f)
            a = int(inside[P[:, inside].max(0).argmax()])
            surprising.append((a, float(s[f]), float(rel[f]), float(unexp[f]), int(PF[:, f].argmax())))
            if cut is not None and s[f] >= cut:
                hide[inside] = True
    order = np.asarray([a for a in np.argsort(-profile) if not hide[a]])
    first = int(order[0])
    shown = set(order[:top].tolist())
    near = [int(a) for a in np.argsort(-sim[first]) if a not in shown and not hide[a]][:n]
    return {"profile": [(int(a), float(profile[a]), because(a)) for a in order[:top]],
            "nearest": [(a, float(sim[first, a])) for a in near],
            "surprising": surprising}
