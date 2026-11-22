"""End-to-end run: Sentinel-1 + Sentinel-2 + CDL -> features -> S1 vs S2 vs S1+S2 models ->
random vs spatial CV -> held-out assessment (Olofsson et al. 2014) -> figures and metrics.

Example:
    python run.py                                   # Wharton County, Texas, 2021, 20 m
    python run.py --bbox -96.49 29.26 -96.23 29.48 --year 2021 --resolution 20 --seed 42
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable even when Python runs in isolated mode (python -I run.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ricemap  # noqa: E402,F401  (sets MKL/DLL environment before numpy loads)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from ricemap import assess, data, features, plots, train  # noqa: E402

ROOT = Path(__file__).resolve().parent
SETS = {"S1": ("S1",), "S2": ("S2",), "S1+S2": ("S1", "S2")}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bbox", nargs=4, type=float, default=[-96.49, 29.26, -96.23, 29.48],
                   metavar=("WEST", "SOUTH", "EAST", "NORTH"), help="WGS 84 bounding box")
    p.add_argument("--place", default="Wharton County, Texas", help="name used in figure titles")
    p.add_argument("--year", type=int, default=2021, help="season and CDL label year (CDL on Planetary Computer: 2008-2021)")
    p.add_argument("--start", default="02-01", help="season start (MM-DD)")
    p.add_argument("--end", default="11-30", help="season end (MM-DD)")
    p.add_argument("--orbit", default="descending", choices=["ascending", "descending"], help="Sentinel-1 orbit direction")
    p.add_argument("--relative-orbit", type=int, default=None, help="Sentinel-1 relative orbit (default: most frequent)")
    p.add_argument("--resolution", type=float, default=20.0, help="pixel size in metres")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--block-km", type=float, default=2.0, help="spatial block size for CV and the test split")
    p.add_argument("--test-fraction", type=float, default=0.2, help="share of blocks held out for testing")
    p.add_argument("--buffer-m", type=float, default=200.0, help="training pixels this close to test blocks are dropped")
    p.add_argument("--n-train", type=int, default=3000, help="training pixels per class")
    p.add_argument("--n-test", type=int, default=500, help="test pixels per mapped class (Olofsson sample)")
    p.add_argument("--n-eval", type=int, default=1000, help="test pixels per reference class (in-season and importance)")
    p.add_argument("--cluster-px", type=int, default=5, help="patch width (pixels) for the clustered training design")
    p.add_argument("--min-clear-fraction", type=float, default=0.05,
                   help="Sentinel-2 dates with less clear sky over the box are counted but not loaded")
    p.add_argument("--cache", type=Path, default=ROOT / "data", help="folder for cached rasters (not in git)")
    return p.parse_args()


def cache_paths(args):
    tag = (f"{'_'.join(f'{v:.3f}' for v in args.bbox)}_{args.year}_{args.start}_{args.end}_{args.orbit}"
           f"{args.relative_orbit or ''}_{args.resolution:g}m")
    c = args.cache
    return {"features": c / f"features_{tag}.tif", "cdl": c / f"cdl_{tag}.tif", "clear": c / f"s2clear_{tag}.tif",
            "meta": c / f"meta_{tag}.json"}


def _prepare_inputs(args, status_file):
    """Child-process entry: build the cache, report back, then exit hard.

    On some Windows setups a process that has made GDAL network reads never finishes exiting,
    and threaded reads were seen to stall scikit-learn later in the same process. Doing the
    download here keeps the analysis process clean. The status goes to a file and the parent
    terminates this process instead of waiting for it.
    """
    import os
    import sys
    import traceback

    code = 0
    try:
        get_inputs(args)
    except Exception:  # noqa: BLE001 - report and signal failure to the parent
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    Path(status_file).write_text(str(code))
    os._exit(code)


def _class_stats_db(vals_lin_by_class):
    """Class mean (linear power -> dB) and 25th/75th percentiles in dB."""
    out = {}
    for k, v in vals_lin_by_class.items():
        v = v[np.isfinite(v) & (v > 0)]
        if v.size == 0:
            out[k] = [np.nan, np.nan, np.nan]
            continue
        q25, q75 = np.percentile(v, [25, 75])
        out[k] = [float(features.to_db(v.mean())), float(features.to_db(q25)), float(features.to_db(q75))]
    return out


def get_inputs(args):
    """Feature stack, CDL labels and Sentinel-2 clear-observation counts on one grid (cached)."""
    import psutil

    data.configure_gdal()
    paths = cache_paths(args)
    if all(p.exists() for p in paths.values()):
        print(f"Using cached inputs in {args.cache}")
        stack, names, grid = data.read_stack(paths["features"])
        cdl = data.read_stack(paths["cdl"])[0][0]
        clear = data.read_stack(paths["clear"])
        return stack, names, cdl, clear, grid, json.loads(paths["meta"].read_text())

    import faulthandler
    import sys

    # If a network read stalls, print where every thread is waiting (every 15 minutes).
    faulthandler.dump_traceback_later(900, repeat=True, file=sys.stderr)
    proc = psutil.Process()
    peak = [0.0]

    def mem(tag):
        rss = proc.memory_info().rss / 1e9
        peak[0] = max(peak[0], rss)
        free = psutil.virtual_memory().available / 1e9
        print(f"  [{tag}] process {rss:.1f} GB, system free {free:.1f} GB", flush=True)
        if free < 8:
            raise MemoryError(f"System free memory fell to {free:.1f} GB; stopping to protect the PC")

    y = args.year
    start, end = f"{y}-{args.start}", f"{y}-{args.end}"
    lon, lat = (args.bbox[0] + args.bbox[2]) / 2, (args.bbox[1] + args.bbox[3]) / 2
    epsg = data.utm_epsg(lon, lat)
    meta = {"crs": f"EPSG:{epsg}", "season": [start, end]}

    # ---- Sentinel-1 (one relative orbit), averaged in linear power
    t0 = time.time()
    s1_items, rel_orbit = data.search_sentinel1(args.bbox, f"{start}/{end}", args.orbit, args.relative_orbit)
    print(f"Sentinel-1: {len(s1_items)} RTC items, {args.orbit} relative orbit {rel_orbit}", flush=True)
    vv, vh, s1_dates, s1_platforms, grid, s1_valid, s1_reloaded = data.load_sentinel1(
        s1_items, args.bbox, epsg, args.resolution)
    print(f"  valid-pixel share per date: min {min(s1_valid):.3f}; dates re-read: {s1_reloaded or 'none'}", flush=True)
    if min(s1_valid) < 0.98:
        raise RuntimeError(f"Sentinel-1 dates with missing data after a re-read: {s1_valid}")
    mem("S1 loaded")
    med_vv = features.check_linear_power(vv[0], "VV")
    med_vh = features.check_linear_power(vh[0], "VH")
    print(f"  first date median VV {med_vv:.3f} ({features.to_db(med_vv):.1f} dB), "
          f"VH {med_vh:.3f} ({features.to_db(med_vh):.1f} dB) -> linear power confirmed", flush=True)
    windows_s1 = features.periods(start, end, "semimonthly")
    s1_layers, s1_counts = features.s1_composites(vv, vh, s1_dates, windows_s1)
    meta["s1"] = {
        "orbit_state": args.orbit, "relative_orbit": int(rel_orbit), "n_acquisitions": len(s1_dates),
        "dates": [d.isoformat() for d in s1_dates], "platforms": s1_platforms,
        "items": [it.id for it in s1_items], "windows": [w[0] for w in windows_s1],
        "acquisitions_per_window": s1_counts, "first_date_median_linear": {"VV": med_vv, "VH": med_vh},
        "valid_share_per_date": s1_valid, "dates_reread": s1_reloaded,
        "load_seconds": round(time.time() - t0, 1),
    }

    # ---- CDL labels on the same grid
    cdl_items = data.search_cdl(args.bbox, y)
    cdl = data.load_cdl(cdl_items, grid)
    mem(f"CDL loaded ({len(cdl_items)} items)")
    meta["cdl"] = {"items": [it.id for it in cdl_items],
                   "colors": data.cdl_colors(cdl_items[0], list(train.RICE_CODES) + list(train.CORN_CODES))}
    labels, _ = train.cdl_to_classes(cdl)
    interior = train.interior_mask(labels)

    # ---- class curves per acquisition date (interior pixels only)
    curves = {"dates": meta["s1"]["dates"], "VV": {}, "VH": {}, "VHVV": {}}
    by_class = {c: interior & (labels == c) for c in (train.RICE, train.CORN, train.OTHER)}
    for t in range(len(s1_dates)):
        for pol, arr in (("VV", vv), ("VH", vh)):
            st = _class_stats_db({c: arr[t][m] for c, m in by_class.items()})
            for c, v in st.items():
                curves[pol].setdefault(train.CLASS_NAMES[c], []).append(v)
        for c, m in by_class.items():
            a, b = vh[t][m], vv[t][m]
            ok = np.isfinite(a) & np.isfinite(b)
            r = features.to_db(a[ok]) - features.to_db(b[ok])
            mean_ratio = float(features.to_db(a[ok].mean()) - features.to_db(b[ok].mean())) if ok.any() else np.nan
            q25, q75 = (np.percentile(r, [25, 75]) if ok.any() else (np.nan, np.nan))
            curves["VHVV"].setdefault(train.CLASS_NAMES[c], []).append([mean_ratio, float(q25), float(q75)])
    meta["curves"] = curves
    del vv, vh
    mem("S1 class curves done")

    # ---- Sentinel-2: clear-sky mask for every date, bands only for dates with some clear sky
    t0 = time.time()
    s2_items = data.search_sentinel2(args.bbox, f"{start}/{end}")
    clear, s2_dates, scl_reread = data.load_scl(s2_items, grid)
    s2_reread = list(scl_reread)
    mem("S2 SCL loaded")
    in_box = np.ones(grid.shape, dtype=bool)
    clear_frac = clear.reshape(len(s2_dates), -1).mean(axis=1)
    months = features.periods(start, end, "monthly")
    m_idx = features.assign_periods(s2_dates, months)
    clear_by_month = np.stack([clear[m_idx == j].sum(axis=0) for j in range(len(months))]).astype("int16")
    s1_by_month = [int(np.sum(features.assign_periods(s1_dates, [w]) == 0)) for w in months]
    keep_dates = {d for d, f in zip(s2_dates, clear_frac) if f >= args.min_clear_fraction}
    monthly = {}
    for j, (code, a, b) in enumerate(months):
        its = [it for it in s2_items if a <= it.datetime.date() <= b and it.datetime.date() in keep_dates]
        if its:
            bands, _, rr = data.load_s2_median(its, grid)
            s2_reread += rr
        else:
            bands = {bnd: np.full(grid.shape, np.nan, dtype="float32") for bnd in data.S2_BANDS}
        monthly[code] = bands
        mem(f"S2 {code}: {len(its)} items")
    s2_layers = features.s2_layers(monthly)
    meta["s2"] = {
        "n_items": len(s2_items), "n_dates": len(s2_dates), "tiles": sorted({it.properties["s2:mgrs_tile"] for it in s2_items}),
        "processing_baselines": sorted({it.properties.get("s2:processing_baseline", "") for it in s2_items}),
        "dates": [d.isoformat() for d in s2_dates], "clear_fraction_per_date": [round(float(f), 4) for f in clear_frac],
        "dates_loaded": len(keep_dates), "months": [m[0] for m in months],
        "s1_acquisitions_per_month": s1_by_month, "load_seconds": round(time.time() - t0, 1),
        "dates_reread_after_incomplete_threaded_read": s2_reread,
    }
    del clear

    # ---- class NDVI per month (interior pixels), for the curve figure
    nd = {}
    for code in monthly:
        arr = s2_layers[f"S2_{code}_NDVI"]
        for c, m in by_class.items():
            v = arr[m]
            v = v[np.isfinite(v)]
            nd.setdefault(train.CLASS_NAMES[c], []).append(
                [float(np.median(v)), *map(float, np.percentile(v, [25, 75]))] if v.size else [np.nan] * 3)
    meta["curves"]["NDVI_monthly"] = nd

    names = list(s1_layers) + list(s2_layers)
    stack = np.stack([s1_layers[n] for n in s1_layers] + [s2_layers[n] for n in s2_layers]).astype("float32")
    del s1_layers, s2_layers, monthly
    mem("feature stack built")
    meta["peak_rss_gb_data_process"] = round(peak[0], 2)
    data.save_stack(paths["features"], stack, names, grid)
    data.save_stack(paths["cdl"], cdl, ["cdl"], grid, nodata=0)
    data.save_stack(paths["clear"], clear_by_month, [f"clear_{m[0]}" for m in months], grid, nodata=-1)
    paths["meta"].write_text(json.dumps(meta, indent=1, default=float))
    del in_box
    return stack, names, cdl, (clear_by_month, [f"clear_{m[0]}" for m in months], grid), grid, meta


def main():
    import psutil

    args = parse_args()
    data.configure_gdal()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    proc = psutil.Process()
    peak = [0.0]

    def mem(tag):
        rss = proc.memory_info().rss / 1e9
        peak[0] = max(peak[0], rss)
        free = psutil.virtual_memory().available / 1e9
        print(f"  [{tag}] process {rss:.1f} GB, system free {free:.1f} GB", flush=True)
        if free < 8:
            raise SystemExit(f"System free memory fell to {free:.1f} GB; stopping to protect the PC")

    paths = cache_paths(args)
    if not all(p.exists() for p in paths.values()):
        import multiprocessing as mp

        args.cache.mkdir(parents=True, exist_ok=True)
        status_file = args.cache / "prepare_status.txt"
        status_file.unlink(missing_ok=True)
        p = mp.get_context("spawn").Process(target=_prepare_inputs, args=(args, str(status_file)))
        p.start()
        while not status_file.exists() and p.is_alive():
            time.sleep(1)
        time.sleep(1)
        status = int(status_file.read_text() or 1) if status_file.exists() else (p.exitcode or 1)
        if p.is_alive():
            p.terminate()
        status_file.unlink(missing_ok=True)
        if status != 0:
            raise SystemExit("Data preparation failed; see the traceback above.")
    stack, names, cdl, clear_pack, grid, meta = get_inputs(args)
    clear_by_month = clear_pack[0]
    t_data = time.time() - t0
    print(f"Inputs ready: {stack.shape[0]} features on a {grid.shape[0]} x {grid.shape[1]} grid ({t_data:.0f} s)")
    mem("inputs loaded")

    # ---- labels and masks
    labels, label_report = train.cdl_to_classes(cdl)
    s1_cols = features.select(names, ("S1",))
    valid = np.isfinite(stack[s1_cols]).all(axis=0) & (labels > 0)
    labels = np.where(valid, labels, 0).astype("uint8")
    interior = train.interior_mask(labels) & valid
    classes = [train.RICE, train.CORN, train.OTHER]
    class_names = [train.CLASS_NAMES[c] for c in classes]
    print("Class shares (CDL):", label_report["class_fractions"], "| excluded codes present:",
          label_report["excluded_present"])

    # ---- spatial split
    blocks = train.block_ids(grid.shape, grid.transform, args.block_km * 1000)
    test_mask = train.split_test_blocks(blocks, valid, args.test_fraction, args.seed)
    buffer = train.buffer_zone(test_mask, int(round(args.buffer_m / args.resolution)))
    train_area = interior & ~test_mask & ~buffer
    test_pop = valid & test_mask
    X_all = stack.reshape(stack.shape[0], -1).T  # view: pixels x features
    y_all = labels.ravel()

    tr_idx = train.stratified_sample(labels, train_area, args.n_train, rng)
    cl_idx, cl_seed = train.clustered_sample(labels, train_area, args.n_train, args.cluster_px,
                                             np.random.default_rng(args.seed))
    print(f"Training: {tr_idx.size} scattered pixels; {cl_idx.size} pixels in {np.unique(cl_seed).size} "
          f"{args.cluster_px}x{args.cluster_px} patches")
    test_flat = np.flatnonzero(test_pop.ravel())
    ref_test = y_all[test_flat]
    interior_test = interior.ravel()[test_flat]
    # reference-stratified evaluation sample (equal per CDL class) for in-season curves and importance
    ev_idx = train.stratified_sample(labels, test_pop, args.n_eval, np.random.default_rng(args.seed + 2))

    cols = {k: features.select(names, s) for k, s in SETS.items()}
    models = train.make_models(args.seed)
    from sklearn.base import clone
    from sklearn.metrics import balanced_accuracy_score

    cv, results, fitted = {}, {}, {}
    px_km2 = (args.resolution / 1000) ** 2
    for fs, c in cols.items():
        designs = {"scattered": (tr_idx, tr_idx), "clustered": (cl_idx, cl_seed)}
        for design, (idx, gpx) in designs.items():
            Xd, yd = X_all[np.ix_(idx, c)], y_all[idx]
            g = blocks.ravel()[gpx]
            for name in (models if design == "scattered" else ["random_forest"]):
                key = f"{fs}|{name}|{design}"
                cv[key] = {"random_cv": train.cross_validate(models[name], Xd, yd, None, seed=args.seed),
                           "spatial_cv": train.cross_validate(models[name], Xd, yd, g, seed=args.seed)}
                if design == "clustered":
                    m = clone(models[name]).fit(Xd, yd)
                    cv[key]["test"] = assess.sample_metrics(
                        assess.error_matrix(m.predict(X_all[np.ix_(test_flat, c)]), ref_test, classes))
                print(f"{key}: random CV BA {cv[key]['random_cv']['balanced_accuracy']:.3f} | spatial CV BA "
                      f"{cv[key]['spatial_cv']['balanced_accuracy']:.3f}", flush=True)

        Xtr = X_all[np.ix_(tr_idx, c)]
        Xte = X_all[np.ix_(test_flat, c)]
        for name, model in models.items():
            m = clone(model).fit(Xtr, y_all[tr_idx])
            fitted[(fs, name)] = m
            pred = m.predict(Xte)
            key = f"{fs}|{name}|scattered"
            census = assess.sample_metrics(assess.error_matrix(pred, ref_test, classes))
            census_int = assess.sample_metrics(
                assess.error_matrix(pred[interior_test], ref_test[interior_test], classes))
            cv[key]["test"] = census
            cv[key]["test_interior"] = census_int
            # Olofsson et al. (2014): stratified random sample by MAP class inside the test blocks
            test_map = np.zeros(labels.size, dtype="uint8")
            test_map[test_flat] = pred
            s_idx = train.stratified_sample(test_map.reshape(labels.shape), test_pop, args.n_test,
                                            np.random.default_rng(args.seed + 1))
            n = assess.error_matrix(test_map[s_idx], y_all[s_idx], classes)
            olof = assess.olofsson(n, [int((pred == k).sum()) for k in classes], pixel_area=px_km2)
            olof["cdl_area_census"] = [int((ref_test == k).sum()) * px_km2 for k in classes]
            olof["ci_covers_census"] = [bool(abs(e - cc) <= h) for e, cc, h in
                                        zip(olof["estimated_area"], olof["cdl_area_census"], olof["estimated_area_ci95"])]
            results[key] = {"census": census, "census_interior": census_int, "olofsson": olof,
                            "sample_counts": n.tolist(), "sample_size": int(n.sum()), "pred_test": pred}
            print(f"  {key}: held-out BA {census['balanced_accuracy']:.3f} (interior {census_int['balanced_accuracy']:.3f}), "
                  f"Olofsson OA {olof['overall_accuracy']:.3f} +/- {olof['overall_accuracy_ci95']:.3f}", flush=True)
        mem(f"{fs} models done")

    # ---- does radar help where optical data are scarce? accuracy by S2 clear-observation count
    n_clear_total = clear_by_month.sum(axis=0).ravel()[test_flat]
    edges = np.quantile(n_clear_total, [0, 1 / 3, 2 / 3, 1])
    bins = np.clip(np.searchsorted(edges, n_clear_total, side="right") - 1, 0, 2)
    by_clear = {"edges": edges.tolist(), "bins": []}
    for b in range(3):
        sel = bins == b
        row = {"n_pixels": int(sel.sum()), "clear_obs_range": [int(n_clear_total[sel].min()), int(n_clear_total[sel].max())]}
        for fs in SETS:
            pr = results[f"{fs}|random_forest|scattered"]["pred_test"][sel]
            row[fs] = float(balanced_accuracy_score(ref_test[sel], pr))
        by_clear["bins"].append(row)
    print("BA by S2 clear-observation tercile:", [(r["clear_obs_range"], round(r["S1"], 3), round(r["S2"], 3),
                                                   round(r["S1+S2"], 3)) for r in by_clear["bins"]])

    # ---- what are the "other" pixels mapped as rice or corn? (CDL code and field-edge share)
    cdl_test = cdl.ravel()[test_flat]
    commission = {}
    for fs in ("S1", "S1+S2"):
        pr = results[f"{fs}|random_forest|scattered"]["pred_test"]
        commission[fs] = {}
        for k in (train.RICE, train.CORN):
            sel = (pr == k) & (ref_test == train.OTHER)
            codes, counts = np.unique(cdl_test[sel], return_counts=True)
            top = sorted(zip(codes.tolist(), counts.tolist()), key=lambda t: -t[1])[:6]
            commission[fs][train.CLASS_NAMES[k]] = {
                "pixels_mapped_as_class": int((pr == k).sum()), "of_which_cdl_other": int(sel.sum()),
                "share_of_those_at_field_edges": round(float((~interior_test[sel]).mean()), 4) if sel.any() else None,
                "top_cdl_codes": [{"code": c, "share": round(n / max(sel.sum(), 1), 4)} for c, n in top],
            }
    print("Commission of 'other' into rice (S1+S2):", commission["S1+S2"]["Rice"])

    # ---- cloud-gap test: remove Sentinel-2 months (as if they were cloudy) and refit
    s1_full_ev = None
    gap_scenarios = {"none": [], "Apr-May": [4, 5], "Apr-Jun": [4, 5, 6], "Apr-Jul": [4, 5, 6, 7],
                     "Apr-Aug": [4, 5, 6, 7, 8]}
    cloud_gap = []
    X_tr_all, y_tr_all = X_all[tr_idx], y_all[tr_idx]
    X_ev_all, y_ev_all = X_all[ev_idx], y_all[ev_idx]
    m1 = clone(models["random_forest"]).fit(X_tr_all[:, cols["S1"]], y_tr_all)
    s1_full_ev = float(balanced_accuracy_score(y_ev_all, m1.predict(X_ev_all[:, cols["S1"]])))
    for label, gap in gap_scenarios.items():
        row = {"gap": label, "months_removed": len(gap), "S1": s1_full_ev}
        for fs, sensors in (("S2", ("S2",)), ("S1+S2", ("S1", "S2"))):
            c = [k for k in features.select(names, sensors)
                 if not (names[k].startswith("S2_") and int(names[k][3:5]) in gap)]
            m = clone(models["random_forest"]).fit(X_tr_all[:, c], y_tr_all)
            row[fs] = float(balanced_accuracy_score(y_ev_all, m.predict(X_ev_all[:, c])))
        cloud_gap.append(row)
        print(f"  S2 gap {label}: S1 {row['S1']:.3f}, S2 {row['S2']:.3f}, S1+S2 {row['S1+S2']:.3f}", flush=True)

    # ---- in-season: accuracy with only the data available by the end of each month
    cut_months = [m for m in range(int(args.start[:2]) + 1, int(args.end[:2]) + 1)]
    inseason = {fs: [] for fs in SETS}
    Xtr_full, ytr = X_all[tr_idx], y_all[tr_idx]
    Xev, yev = X_all[ev_idx], y_all[ev_idx]
    for mth in cut_months:
        for fs, s in SETS.items():
            c = features.select(names, s, until=f"{mth:02d}31")
            m = clone(models["random_forest"]).fit(Xtr_full[:, c], ytr)
            pred = m.predict(Xev[:, c])
            pa = assess.sample_metrics(assess.error_matrix(pred, yev, classes))["producers_accuracy"]
            inseason[fs].append({"until_month": mth, "n_features": len(c),
                                 "balanced_accuracy": float(balanced_accuracy_score(yev, pred)),
                                 "producers_accuracy": dict(zip(class_names, pa))})
        print(f"  in-season to end of {features.MONTHS[mth - 1]}: " +
              ", ".join(f"{fs} {inseason[fs][-1]['balanced_accuracy']:.3f}" for fs in SETS), flush=True)

    # ---- grouped permutation importance by time window
    pi = {}
    for fs, by in (("S1", "window"), ("S1+S2", "month")):
        c = cols[fs]
        grp = assess.feature_groups([names[k] for k in c], by=by)
        _, imp = assess.grouped_permutation_importance(fitted[(fs, "random_forest")], Xev[:, c], yev, grp,
                                                       balanced_accuracy_score, n_repeats=5, seed=args.seed)
        pi[fs] = imp
    mem("importance done")

    # ---- full maps (Random Forest) for the figures and the GeoTIFF outputs
    maps = {}
    for fs, c in cols.items():
        sub = stack[c]
        maps[fs] = train.predict_raster(fitted[(fs, "random_forest")], sub, valid)
        del sub
    mem("maps predicted")
    for fs in ("S1", "S1+S2"):
        data.save_stack(args.cache / f"rf_{fs.replace('+', '')}_classification.tif", maps[fs], [f"rf_{fs}"], grid, nodata=0)

    # ---- figures
    figs = ROOT / "figures"
    colors = plots.class_colors(meta["cdl"]["colors"])
    res = args.resolution / 1000
    extent = [0, grid.shape[1] * res, 0, grid.shape[0] * res]
    plots.class_curves(meta["curves"], meta["s2"]["months"], colors, figs / "class_curves.png", args.place, args.year)
    clear_total = clear_by_month.sum(axis=0)
    plots.clear_observations(clear_total, clear_by_month, meta["s2"], meta["s1"], extent, figs / "s2_clear_observations.png",
                             args.year)
    plots.model_comparison(cv, inseason, cloud_gap, figs / "model_comparison.png")
    plots.confusion_figure({fs: results[f"{fs}|random_forest|scattered"] for fs in SETS}, class_names,
                           figs / "confusion_matrix.png")
    plots.importance_figure(pi, figs / "feature_importance.png")
    plots.map_comparison(maps, labels, colors, extent, test_mask, figs / "map_comparison.png",
                         f"{args.place}, {args.year}. Black outlines: {args.block_km:g} km blocks held out for testing.")
    ba = {fs: results[f"{fs}|random_forest|scattered"]["census"]["balanced_accuracy"] for fs in SETS}
    plots.hero_figure(maps["S1"], labels, colors, extent, meta["curves"], ba, figs / "hero.png", args.place, args.year)

    # ---- metrics.json
    def r(x, k=4):
        if isinstance(x, (list, tuple)):
            return [r(v, k) for v in x]
        if isinstance(x, dict):
            return {kk: r(vv, k) for kk, vv in x.items()}
        return round(float(x), k) if isinstance(x, (float, np.floating)) else x

    s1m = meta["s1"]
    out = {
        "run": {
            "bbox_wgs84": args.bbox, "place": args.place, "year": args.year, "season": meta["season"],
            "resolution_m": args.resolution, "seed": args.seed, "crs": grid.crs, "grid_shape": list(grid.shape),
            "area_km2": round(grid.shape[0] * grid.shape[1] * px_km2, 1),
            "block_km": args.block_km, "test_fraction_of_blocks": args.test_fraction, "buffer_m": args.buffer_m,
            "n_train_per_class": args.n_train, "n_test_per_class": args.n_test, "n_eval_per_class": args.n_eval,
            "runtime_seconds": round(time.time() - t0, 1), "data_seconds": round(t_data, 1),
            "peak_rss_gb_analysis_process": round(peak[0], 2),
            "peak_rss_gb_data_process": meta.get("peak_rss_gb_data_process"),
        },
        "sentinel1": {k: s1m[k] for k in ("orbit_state", "relative_orbit", "n_acquisitions", "dates", "platforms",
                                          "windows", "acquisitions_per_window", "first_date_median_linear",
                                          "valid_share_per_date", "dates_reread", "items")},
        "sentinel2": {k: meta["s2"][k] for k in ("n_items", "n_dates", "tiles", "processing_baselines", "dates_loaded",
                                                  "months", "s1_acquisitions_per_month", "clear_fraction_per_date",
                                                  "dates_reread_after_incomplete_threaded_read",
                                                  "dates")},
        "cdl": {"items": meta["cdl"]["items"], **{k: v for k, v in label_report.items()}},
        "features": {"n_total": len(names), "n_S1": len(cols["S1"]), "n_S2": len(cols["S2"]),
                     "share_of_S2_feature_values_missing_in_valid_pixels": None},
        "samples": {"train": int(tr_idx.size), "train_clustered": int(cl_idx.size), "cluster_px": args.cluster_px,
                    "test_blocks": int(np.unique(blocks[test_mask]).size), "test_population_pixels": int(test_pop.sum()),
                    "test_interior_share": r(interior_test.mean()),
                    "valid_pixels": int(valid.sum()), "interior_share_of_valid": r(interior.sum() / valid.sum())},
        "cross_validation": {k: {kk: {m: r(v) for m, v in vv.items() if not isinstance(v, list)} for kk, vv in d.items()}
                             for k, d in cv.items()},
        "test": {},
        "accuracy_by_s2_clear_observations": r(by_clear),
        "cloud_gap_test": r(cloud_gap),
        "commission_of_other": commission,
        "in_season": r(inseason),
        "grouped_permutation_importance": {fs: [{"group": g, "mean_drop_balanced_accuracy": r(v[0]), "sd": r(v[1])}
                                                for g, v in sorted(d.items(), key=lambda kv: -kv[1][0])]
                                           for fs, d in pi.items()},
        "s2_clear_observations": {
            "median_total_per_pixel": float(np.median(clear_total[valid])),
            "min_total_per_pixel": int(clear_total[valid].min()), "max_total_per_pixel": int(clear_total[valid].max()),
            "median_per_month": dict(zip(meta["s2"]["months"],
                                         [float(np.median(cm[valid])) for cm in clear_by_month])),
            "share_pixels_zero_clear_per_month": dict(zip(meta["s2"]["months"],
                                                          [r(float((cm[valid] == 0).mean())) for cm in clear_by_month])),
        },
    }
    s2c = cols["S2"]
    v_idx = np.flatnonzero(valid.ravel())[:: max(1, int(valid.sum() // 200_000))]
    out["features"]["share_of_S2_feature_values_missing_in_valid_pixels"] = r(float(np.isnan(X_all[np.ix_(v_idx, s2c)]).mean()))
    for key, d in results.items():
        o, c = d["olofsson"], d["census"]
        out["test"][key] = {
            "census_of_test_blocks": {
                "overall_accuracy": r(c["overall_accuracy"]), "balanced_accuracy": r(c["balanced_accuracy"]),
                "macro_f1": r(c["macro_f1"]), "kappa": r(c["kappa"]),
                "balanced_accuracy_interior_pixels": r(d["census_interior"]["balanced_accuracy"]),
                "per_class": {nm: {"users_accuracy": r(u), "producers_accuracy": r(p), "f1": r(f)}
                              for nm, u, p, f in zip(class_names, c["users_accuracy"], c["producers_accuracy"], c["f1"])},
            },
            "olofsson_2014": {
                "sample_size": d["sample_size"], "sample_counts_rows_map_cols_reference": d["sample_counts"],
                "overall_accuracy": r(o["overall_accuracy"]), "overall_accuracy_ci95": r(o["overall_accuracy_ci95"]),
                "per_class": {nm: {
                    "users_accuracy": r(o["users_accuracy"][k]), "users_accuracy_ci95": r(o["users_accuracy_ci95"][k]),
                    "producers_accuracy": r(o["producers_accuracy"][k]),
                    "producers_accuracy_ci95": r(o["producers_accuracy_ci95"][k]),
                    "mapped_area_km2": r(o["mapped_area"][k], 2), "estimated_area_km2": r(o["estimated_area"][k], 2),
                    "estimated_area_ci95_km2": r(o["estimated_area_ci95"][k], 2),
                    "cdl_census_area_km2": r(o["cdl_area_census"][k], 2), "ci_covers_census": o["ci_covers_census"][k],
                } for k, nm in enumerate(class_names)},
            },
        }
    (ROOT / "results" / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    curves_out = {"dates": meta["curves"]["dates"], **{p: meta["curves"][p] for p in ("VV", "VH", "VHVV")},
                  "NDVI_monthly": meta["curves"]["NDVI_monthly"], "note": "per class: [mean (linear power -> dB) or "
                  "median for NDVI, 25th percentile, 75th percentile], interior CDL pixels"}
    (ROOT / "results" / "class_curves.json").write_text(json.dumps(r(curves_out, 3), indent=1, default=float))
    print(f"Done in {time.time() - t0:.0f} s (peak analysis RSS {peak[0]:.1f} GB). Metrics: results/metrics.json")


if __name__ == "__main__":
    main()
