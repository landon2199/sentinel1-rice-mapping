"""Labels, spatial blocks, sampling, models and cross-validation."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, KFold

# Map classes. CDL codes: 3 = Rice, 1 = Corn (USDA NASS CDL legend).
RICE, CORN, OTHER = 1, 2, 3
CLASS_NAMES = {RICE: "Rice", CORN: "Corn", OTHER: "Other"}
RICE_CODES = (3,)
CORN_CODES = (1,)
# Excluded, because their season does not match field corn or rice: sweet corn (12), pop or
# ornamental corn (13) and double crops that include corn (225, 226, 237, 241). 0 is no data and
# 81 is clouds/no data in the CDL legend.
EXCLUDED_CODES = (0, 81, 12, 13, 225, 226, 237, 241)


def cdl_to_classes(cdl: np.ndarray, excluded=EXCLUDED_CODES):
    """CDL codes -> 1 rice, 2 corn, 3 other, 0 excluded. Returns (labels, report)."""
    cdl = np.asarray(cdl)
    labels = np.full(cdl.shape, OTHER, dtype="uint8")
    labels[np.isin(cdl, RICE_CODES)] = RICE
    labels[np.isin(cdl, CORN_CODES)] = CORN
    labels[np.isin(cdl, excluded)] = 0
    codes, counts = np.unique(cdl[cdl > 0], return_counts=True)
    total = counts.sum() if counts.size else 1
    report = {
        "cdl_fractions": {int(c): round(n / total, 5) for c, n in sorted(zip(codes, counts), key=lambda t: -t[1])},
        "excluded_present": {int(c): round(n / total, 5) for c, n in zip(codes, counts) if c in excluded},
        "class_fractions": {CLASS_NAMES[k]: round(float((labels == k).sum() / max((labels > 0).sum(), 1)), 4)
                            for k in (RICE, CORN, OTHER)},
    }
    return labels, report


def interior_mask(labels: np.ndarray, radius: int = 1) -> np.ndarray:
    """True where every pixel in the (2r+1) x (2r+1) window has the same, non-zero label.

    Used to keep field-edge pixels (mixed rice/road/ditch/neighbour field at 20-30 m) out of
    the training sample.
    """
    size = 2 * radius + 1
    lo = ndimage.minimum_filter(labels, size=size, mode="nearest")
    hi = ndimage.maximum_filter(labels, size=size, mode="nearest")
    return (lo == hi) & (labels > 0)


def block_ids(shape, transform, block_size_m: float) -> np.ndarray:
    """Integer id of the square ``block_size_m`` block containing each pixel centre.

    Blocks are aligned to the grid origin, so ids are reproducible for a given grid.
    """
    a, _, c, _, e, f = transform[:6]
    rows, cols = shape
    x = c + (np.arange(cols) + 0.5) * a
    y = f + (np.arange(rows) + 0.5) * e
    bx = np.floor((x - c) / block_size_m).astype(int)
    by = np.floor((f - y) / block_size_m).astype(int)
    nbx = bx.max() + 1
    return by[:, None] * nbx + bx[None, :]


def split_test_blocks(blocks: np.ndarray, usable: np.ndarray, test_fraction: float, seed: int) -> np.ndarray:
    """Randomly hold out ``test_fraction`` of the blocks that contain usable pixels."""
    rng = np.random.default_rng(seed)
    ids = np.unique(blocks[usable])
    n_test = max(1, int(round(test_fraction * ids.size)))
    test_ids = rng.choice(ids, size=n_test, replace=False)
    return np.isin(blocks, test_ids)


def buffer_zone(mask: np.ndarray, distance_px: int) -> np.ndarray:
    """Pixels outside ``mask`` but within ``distance_px`` of it."""
    if distance_px <= 0:
        return np.zeros_like(mask, dtype=bool)
    dist = ndimage.distance_transform_edt(~mask)
    return (dist > 0) & (dist <= distance_px)


def stratified_sample(strata: np.ndarray, eligible: np.ndarray, n_per_class: int, rng) -> np.ndarray:
    """Flat pixel indices: up to ``n_per_class`` random pixels from each stratum (> 0)."""
    flat_strata = strata.ravel()
    flat_ok = eligible.ravel()
    picks = []
    for code in np.unique(flat_strata[flat_ok]):
        if code == 0:
            continue
        idx = np.flatnonzero(flat_ok & (flat_strata == code))
        take = min(n_per_class, idx.size)
        picks.append(rng.choice(idx, size=take, replace=False))
    return np.sort(np.concatenate(picks))


def clustered_sample(labels: np.ndarray, eligible: np.ndarray, n_per_class: int, cluster_px: int, rng):
    """Polygon-like training data: whole ``cluster_px`` x ``cluster_px`` patches of one class.

    Mimics hand-digitized training polygons: random seed pixels are expanded to the
    surrounding window, keeping eligible pixels of the seed's class, until ``n_per_class``
    pixels are collected. Neighbouring pixels are near-duplicates, which is exactly what
    makes random cross-validation optimistic.

    Returns (flat pixel indices, flat index of each pixel's patch seed). Grouping CV folds by
    the seed's block keeps a whole patch in one fold even when it straddles a block edge.
    """
    rows, cols = labels.shape
    h = cluster_px // 2
    picks, seeds_all = [], []
    for code in np.unique(labels[eligible]):
        if code == 0:
            continue
        ok = eligible & (labels == code)
        seeds = rng.permutation(np.flatnonzero(ok.ravel()))
        taken = np.zeros(ok.shape, dtype=bool)
        seed_of = np.full(ok.shape, -1, dtype=np.int64)
        count = 0
        for s in seeds:
            if count >= n_per_class:
                break
            r, c = divmod(int(s), cols)
            if taken[r, c]:
                continue
            r0, r1, c0, c1 = max(r - h, 0), min(r + h + 1, rows), max(c - h, 0), min(c + h + 1, cols)
            win = ok[r0:r1, c0:c1] & ~taken[r0:r1, c0:c1]
            taken[r0:r1, c0:c1] |= win
            seed_of[r0:r1, c0:c1][win] = s
            count += int(win.sum())
        picks.append(np.flatnonzero(taken.ravel()))
        seeds_all.append(seed_of.ravel()[picks[-1]])
    idx = np.concatenate(picks)
    seeds = np.concatenate(seeds_all)
    order = np.argsort(idx)
    return idx[order], seeds[order]


def make_models(seed: int) -> dict:
    """Random Forest (main model) and a histogram gradient-boosting baseline."""
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_features="sqrt", min_samples_leaf=1, n_jobs=-1, random_state=seed
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, early_stopping=False, random_state=seed
        ),
    }


def cross_validate(model, X, y, groups=None, n_splits: int = 5, seed: int = 0, coords=None,
                   buffer_m: float = 0.0) -> dict:
    """K-fold CV; random folds when ``groups`` is None, otherwise grouped (spatial) folds.

    With ``coords`` (n x 2, metres) and ``buffer_m`` > 0, training samples closer than
    ``buffer_m`` to any sample of the held-out fold are dropped from that fold's training set.
    """
    from sklearn.base import clone

    if groups is None:
        splits = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X)
    else:
        splits = GroupKFold(n_splits=n_splits).split(X, y, groups)
    oa, ba, f1 = [], [], []
    for tr, te in splits:
        if buffer_m > 0 and coords is not None:
            from scipy.spatial import cKDTree

            dist, _ = cKDTree(coords[te]).query(coords[tr], k=1, distance_upper_bound=buffer_m)
            tr = tr[~np.isfinite(dist)]
        m = clone(model).fit(X[tr], y[tr])
        p = m.predict(X[te])
        oa.append(accuracy_score(y[te], p))
        ba.append(balanced_accuracy_score(y[te], p))
        f1.append(f1_score(y[te], p, average="macro"))
    return {
        "overall_accuracy": float(np.mean(oa)),
        "overall_accuracy_sd": float(np.std(oa)),
        "balanced_accuracy": float(np.mean(ba)),
        "balanced_accuracy_sd": float(np.std(ba)),
        "macro_f1": float(np.mean(f1)),
    }


def pixel_coords(flat_idx: np.ndarray, shape, transform) -> np.ndarray:
    """Map coordinates (x, y in CRS units) of pixel centres given flat indices."""
    a, _, c, _, e, f = transform[:6]
    r, col = np.divmod(np.asarray(flat_idx), shape[1])
    return np.column_stack([c + (col + 0.5) * a, f + (r + 0.5) * e])


def predict_raster(model, features: np.ndarray, valid: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """Classify every valid pixel of a [n_features, rows, cols] stack (0 = no prediction)."""
    nf, rows, cols = features.shape
    X = features.reshape(nf, -1).T
    idx = np.flatnonzero(valid.ravel())
    out = np.zeros(rows * cols, dtype="uint8")
    for s in range(0, idx.size, chunk):
        sl = idx[s : s + chunk]
        out[sl] = model.predict(X[sl])
    return out.reshape(rows, cols)
