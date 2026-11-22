"""Figures. One color per class and one per sensor set, used the same way in every figure."""
from __future__ import annotations

from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

INK = "#1f2328"
MUTED = "#57606a"
GRID = "#d0d7de"
# Classes: rice blue, corn yellow, "other" a neutral gray (a background class, not a series).
CLASS_COLORS = {"Rice": "#2a78d6", "Corn": "#eda100", "Other": "#a3a29c"}
# Sensor sets (validated as a 3-slot categorical palette, all pairs, light surface).
SET_COLORS = {"S1": "#6250d6", "S2": "#1baf7a", "S1+S2": "#eb6834"}
SET_LABELS = {"S1": "Sentinel-1 only (radar)", "S2": "Sentinel-2 only (optical)", "S1+S2": "Sentinel-1 + Sentinel-2"}
SCHEMES = {"random_cv": ("Random 5-fold CV", "#b4b4b4", "o"), "spatial_cv": ("Spatial block CV", "#2f5d8a", "s"),
           "test_interior": ("Held-out blocks, field interiors", "#7a9cc0", "D"),
           "test": ("Held-out blocks, all pixels", "#16324f", "D")}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white",
})

CLASS_ORDER = ["Rice", "Corn", "Other"]


def class_colors(cdl_colors=None) -> dict:
    """Class colors (``cdl_colors`` from the CDL file are recorded in metadata but not used for
    the figures, because CDL corn yellow is unreadable as a line on white)."""
    return dict(CLASS_COLORS)


def _cmap(colors):
    cols = ["#ffffff"] + [colors[c] for c in CLASS_ORDER]
    return ListedColormap(cols), BoundaryNorm(np.arange(5) - 0.5, 4)


def _draw_map(ax, img, colors, title, extent, outline=None, ticks=True):
    cmap, norm = _cmap(colors)
    ax.imshow(img, cmap=cmap, norm=norm, interpolation="nearest", extent=extent)
    if outline is not None:
        ax.contour(outline.astype(float), levels=[0.5], colors="black", linewidths=0.7, extent=extent, origin="upper")
    ax.set_title(title, loc="left", fontsize=12)
    ax.set_aspect("equal")
    if ticks:
        ax.set_xlabel("km east")
        ax.set_ylabel("km north")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(GRID)


def _legend(colors):
    return [Patch(facecolor=colors[c], edgecolor="#888888", label=c) for c in CLASS_ORDER]


def _dates(strs):
    return [date.fromisoformat(s) for s in strs]


def _curve(ax, x, rows, color, label):
    """Mean line with a 25-75 % band; returns (x, y) of the last point for a direct label."""
    a = np.asarray(rows, dtype=float)
    ax.fill_between(x, a[:, 1], a[:, 2], color=color, alpha=0.18, linewidth=0)
    ax.plot(x, a[:, 0], color=color, linewidth=2, marker="o", markersize=3.5, label=label)
    k = np.flatnonzero(np.isfinite(a[:, 0]))[-1]
    return x[k], a[k, 0]


def _direct_labels(ax, ends: dict, min_gap_frac=0.06):
    """Label line ends, nudging labels apart vertically so they never overlap."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    order = sorted(ends.items(), key=lambda kv: kv[1][1])
    ys = []
    for _, (_, y) in order:
        ys.append(max(y, ys[-1] + gap) if ys else y)
    excess = ys[-1] - (hi - gap / 2) if ys else 0
    if excess > 0:  # keep the top label inside the axes
        ys = [y - excess for y in ys]
    for (label, (x, _)), y in zip(order, ys):
        ax.annotate(label, (x, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=10,
                    color=INK, weight="bold", annotation_clip=False)


def _date_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def class_curves(curves, months, colors, path, place, year):
    """VH, VV, VH/VV per acquisition date and monthly NDVI, by CDL class (interior pixels)."""
    x = _dates(curves["dates"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, constrained_layout=True)
    spec = [("VH", "VH backscatter (dB)", "Cross-polarized VH: rises as the canopy grows"),
            ("VV", "VV backscatter (dB)", "Co-polarized VV"),
            ("VHVV", "VH/VV ratio (dB)", "VH/VV ratio: low over bare or flooded soil, higher over canopy")]
    for ax, (key, ylab, title) in zip(axes.ravel()[:3], spec):
        ends = {c: _curve(ax, x, curves[key][c], colors[c], c) for c in CLASS_ORDER}
        _direct_labels(ax, ends)
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=12)
        _date_axis(ax)
    ax = axes[1, 1]
    xm = [date(year, int(m[:2]), 15) for m in months]
    ends = {c: _curve(ax, xm, curves["NDVI_monthly"][c], colors[c], c) for c in CLASS_ORDER}
    _direct_labels(ax, ends)
    ax.set_ylabel("NDVI (monthly median, clear pixels)")
    ax.set_title("Sentinel-2 NDVI, for comparison", loc="left", fontsize=12)
    _date_axis(ax)
    for a in axes.ravel():
        a.set_xlim(date(year, 1, 25), date(year, 12, 20))
    fig.suptitle(f"How rice, corn and other land look through the {year} season, {place}\n"
                 "Line: class mean (radar averaged in linear power, then dB; NDVI median). "
                 "Band: 25th to 75th percentile of field-interior pixels.", color=MUTED, fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def clear_observations(clear_total, clear_by_month, s2meta, s1meta, extent, path, year):
    """Map of clear Sentinel-2 observations per pixel, and monthly S1 vs S2 counts."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), constrained_layout=True,
                                  gridspec_kw={"width_ratios": [1, 1.25]})
    im = ax.imshow(clear_total, cmap="Blues", extent=extent, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("clear Sentinel-2 observations, Feb to Nov")
    ax.set_title(f"Clear Sentinel-2 views per pixel ({s2meta['n_dates']} acquisition dates)", loc="left", fontsize=12)
    ax.set_xlabel("km east")
    ax.set_ylabel("km north")
    months = s2meta["months"]
    labels = [date(year, int(m[:2]), 1).strftime("%b") for m in months]
    s1 = np.asarray(s2meta["s1_acquisitions_per_month"])
    s2_dates = [date.fromisoformat(d) for d in s2meta["dates"]]
    s2n = np.array([sum(d.month == int(m[:2]) for d in s2_dates) for m in months])
    valid = clear_total >= 0
    med = np.array([np.median(cm[valid]) for cm in clear_by_month])
    xs = np.arange(len(months))
    w = 0.27
    bars = [(s1, SET_COLORS["S1"], "Sentinel-1 acquisitions (one orbit; clouds do not matter)"),
            (s2n, "#a8dcc8", "Sentinel-2 acquisition dates"),
            (med, SET_COLORS["S2"], "Sentinel-2 clear views of the median pixel")]
    for k, (vals, col, lab) in enumerate(bars):
        off = (k - 1) * (w + 0.02)
        ax2.bar(xs + off, vals, w, color=col, label=lab)
        if k == 2:  # label only the clear views, the point of the chart
            for j in range(len(xs)):
                ax2.text(xs[j] + off, vals[j] + 0.1, f"{vals[j]:g}", ha="center", va="bottom", fontsize=9,
                         weight="bold", color=INK)
    ax2.set_xticks(xs, labels)
    ax2.set_ylim(0, max(s1.max(), s2n.max()) + 2.5)
    ax2.set_ylabel("observations per month")
    total_s2, total_clear = int(s2n.sum()), float(np.median(clear_total[valid]))
    ax2.set_title(f"Radar sees every pass; the median pixel had {total_clear:g} clear optical views "
                  f"in {total_s2} dates", loc="left", fontsize=12)
    ax2.legend(frameon=False, loc="upper left", fontsize=9)
    ax2.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax2.set_axisbelow(True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def model_comparison(cv, inseason, cloud_gap, path):
    """Balanced accuracy by feature set, model and validation; in-season accuracy; cloud-gap test."""
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6.6), constrained_layout=True,
                                       gridspec_kw={"width_ratios": [1.2, 1, 1]})
    keys = [f"{fs}|{m}|scattered" for fs in ("S1", "S2", "S1+S2") for m in ("random_forest", "hist_gradient_boosting")]
    offsets = {"random_cv": -0.3, "spatial_cv": -0.1, "test_interior": 0.1, "test": 0.3}
    allv = [cv[k][s]["balanced_accuracy"] for k in keys for s in offsets]
    lo, hi = min(allv), max(allv)
    span = max(hi - lo, 0.02)
    for i, k in enumerate(keys):
        for s, off in offsets.items():
            v = cv[k][s]["balanced_accuracy"]
            name, col, mk = SCHEMES[s]
            ax.plot([lo - 0.5 * span, v], [i + off] * 2, color=GRID, linewidth=1, zorder=1)
            ax.plot(v, i + off, marker=mk, color=col, markersize=7, linestyle="none", zorder=3,
                    label=name if i == 0 else None)
            if s == "test":  # label only the headline score; the rest are read from the axis
                ax.text(v + 0.03 * span, i + off, f"{v:.3f}", va="center", fontsize=9, weight="bold", color=INK)
    labels = []
    for k in keys:
        fs, m, _ = k.split("|")
        labels.append(f"{SET_LABELS[fs]}\n{'Random Forest' if m == 'random_forest' else 'Gradient boosting'}")
    ax.set_yticks(range(len(keys)), labels, fontsize=9)
    ax.set_ylim(len(keys) - 0.5, -0.6)
    ax.set_xlim(lo - 0.5 * span, hi + 0.3 * span)
    ax.set_xlabel("Balanced accuracy (mean producer's accuracy); zoomed axis, dots not bars")
    ax.set_title("Which data, which model, which validation", loc="left", fontsize=12)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.1), ncol=2, fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)

    ends = {}
    for fs, rows in inseason.items():
        x = [r["until_month"] for r in rows]
        y = [r["balanced_accuracy"] for r in rows]
        ax2.plot(x, y, color=SET_COLORS[fs], linewidth=2, marker="o", markersize=6, label=SET_LABELS[fs])
        ends[fs] = (x[-1], y[-1])
    ax2.set_xticks(x, [date(2001, m, 1).strftime("%b") for m in x])
    ax2.set_xlim(x[0] - 0.3, x[-1] + 1.2)
    _direct_labels(ax2, ends, 0.05)
    ax2.set_xlabel("Data used: from February to the end of this month")
    ax2.set_ylabel("Balanced accuracy, held-out sample")
    ax2.set_title("How early can each sensor tell the crops apart?", loc="left", fontsize=12)
    ax2.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, loc="lower right", fontsize=9)

    xs = np.arange(len(cloud_gap))
    ends = {}
    for fs in ("S1", "S2", "S1+S2"):
        y = [r[fs] for r in cloud_gap]
        ax3.plot(xs, y, color=SET_COLORS[fs], linewidth=2, marker="o", markersize=6, label=SET_LABELS[fs])
        ends[fs] = (xs[-1], y[-1])
    ax3.set_xticks(xs, ["no gap" if r["gap"] == "none" else r["gap"] for r in cloud_gap])
    ax3.set_xlim(-0.3, xs[-1] + 0.8)
    _direct_labels(ax3, ends, 0.05)
    ax3.set_xlabel("Sentinel-2 months removed (as if fully cloudy); full Sentinel-1 season kept")
    ax3.set_ylabel("Balanced accuracy, held-out sample")
    ax3.set_title("Cloud-gap test: when does radar beat optical?", loc="left", fontsize=12)
    ax3.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax3.set_axisbelow(True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def confusion_figure(res_by_set, class_names, path):
    """Olofsson sample count matrices (rows = map, columns = CDL) for the three Random Forest models."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), constrained_layout=True)
    for ax, (fs, d) in zip(axes, res_by_set.items()):
        n = np.asarray(d["sample_counts"])
        o = d["olofsson"]
        share = n / np.maximum(n.sum(1, keepdims=True), 1)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(n.shape[0]):
            for j in range(n.shape[1]):
                ax.text(j, i, f"{n[i, j]}", ha="center", va="center", fontsize=11,
                        color="white" if share[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(class_names)), [f"{c}\nPA {p:.2f}" for c, p in zip(class_names, o["producers_accuracy"])])
        ax.set_yticks(range(len(class_names)), [f"{c}  UA {u:.2f}" for c, u in zip(class_names, o["users_accuracy"])])
        ax.set_xlabel("Reference (CDL)")
        ax.set_ylabel("Map (Random Forest)")
        ax.set_title(f"{SET_LABELS[fs]}\nOA {o['overall_accuracy']:.3f} ± {o['overall_accuracy_ci95']:.3f}",
                     loc="left", fontsize=11)
        ax.spines[:].set_visible(False)
    fig.suptitle("Held-out test sample, stratified by map class (UA, PA and OA area-weighted, Olofsson et al. 2014)",
                 color=MUTED, fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def importance_figure(pi, path):
    """Grouped permutation importance in time order: S1 by half-month, S1+S2 by month and sensor."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True,
                                  gridspec_kw={"width_ratios": [1.1, 1]})
    s1 = pi["S1"]
    names = list(s1)
    mean = [s1[n][0] for n in names]
    sd = [s1[n][1] for n in names]
    xs = np.arange(len(names))
    ax.bar(xs, mean, yerr=sd, color=SET_COLORS["S1"], width=0.75, error_kw={"ecolor": MUTED, "capsize": 2})
    ax.set_xticks(xs, [n.replace("S1 ", "") for n in names], rotation=60, ha="right", fontsize=9)
    ax.set_ylabel("Drop in balanced accuracy when shuffled")
    ax.set_title("Sentinel-1 model: which half-months matter (VV, VH, VH/VV shuffled together)", loc="left",
                 fontsize=12)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    both = pi["S1+S2"]
    months = []
    for n in both:
        m = n.split(" ", 1)[1]
        if m not in months:
            months.append(m)
    xs = np.arange(len(months))
    w = 0.38
    for k, (sensor, off) in enumerate((("S1", -w / 2 - 0.01), ("S2", w / 2 + 0.01))):
        v = [both.get(f"{sensor} {m}", (np.nan, 0))[0] for m in months]
        e = [both.get(f"{sensor} {m}", (np.nan, 0))[1] for m in months]
        ax2.bar(xs + off, v, w, yerr=e, color=SET_COLORS[sensor], label=f"{sensor} features of that month",
                error_kw={"ecolor": MUTED, "capsize": 2})
    ax2.set_xticks(xs, months)
    ax2.set_title("Combined model: radar month vs optical month", loc="left", fontsize=12)
    ax2.set_ylabel("Drop in balanced accuracy when shuffled")
    ax2.legend(frameon=False, fontsize=9)
    ax2.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax2.set_axisbelow(True)
    for a in (ax, ax2):
        a.axhline(0, color=MUTED, linewidth=0.8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def map_comparison(maps, ref, colors, extent, test_mask, path, subtitle=""):
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), constrained_layout=True)
    show = lambda m: np.where(ref > 0, m, 0)  # noqa: E731
    for ax, fs in zip(axes.ravel()[:3], ("S1", "S2", "S1+S2")):
        _draw_map(ax, show(maps[fs]), colors, f"Random Forest, {SET_LABELS[fs]}", extent, test_mask)
    _draw_map(axes[1, 1], ref, colors, "USDA Cropland Data Layer (reference labels)", extent, test_mask)
    handles = _legend(colors) + [Patch(facecolor="none", edgecolor="black", label="Held-out test blocks")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    if subtitle:
        fig.suptitle(subtitle, color=MUTED, fontsize=11)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def hero_figure(map_s1, ref, colors, extent, curves, ba, path, place, year):
    """16:9 card: radar-only map, CDL, and the VH curves that make it possible."""
    fig = plt.figure(figsize=(16, 7), dpi=100)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.1], height_ratios=[0.2, 1], left=0.025, right=0.95,
                          top=0.96, bottom=0.11, wspace=0.2, hspace=0.22)
    head = fig.add_subplot(gs[0, :])
    head.axis("off")
    head.text(0, 0.75, "Rice and corn from Sentinel-1 radar", fontsize=24, weight="bold", color=INK, va="center")
    head.text(0, 0.05, f"{place}  |  {year} season  |  labels: USDA Cropland Data Layer {year}", fontsize=14,
              color=MUTED, va="center")
    head.text(1.0, 0.75, "Balanced accuracy on held-out blocks", fontsize=13, color=MUTED, ha="right", va="center")
    head.text(1.0, 0.05, f"radar {ba['S1']:.2f}   optical {ba['S2']:.2f}   both {ba['S1+S2']:.2f}", fontsize=17,
              weight="bold", color=INK, ha="right", va="center")
    show = np.where(ref > 0, map_s1, 0)
    _draw_map(fig.add_subplot(gs[1, 0]), show, colors, "Random Forest, Sentinel-1 only", extent, ticks=False)
    _draw_map(fig.add_subplot(gs[1, 1]), ref, colors, f"USDA CDL {year}", extent, ticks=False)
    ax = fig.add_subplot(gs[1, 2])
    x = _dates(curves["dates"])
    ends = {c: _curve(ax, x, curves["VH"][c], colors[c], c) for c in CLASS_ORDER}
    _direct_labels(ax, ends)
    _date_axis(ax)
    ax.set_xlim(date(year, 1, 25), date(year, 12, 20))
    ax.set_ylabel("VH backscatter (dB)")
    ax.set_title("VH through the season (mean, 25-75 % band)", loc="left", fontsize=13)
    fig.legend(handles=_legend(colors), loc="lower left", ncol=3, frameon=False, bbox_to_anchor=(0.03, 0.01),
               fontsize=12)
    fig.savefig(path, dpi=100)
    plt.close(fig)
