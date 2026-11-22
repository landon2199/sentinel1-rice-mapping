"""Radar and optical features: dB handling, time windows, composites and spectral indices.

Sentinel-1 RTC values on Planetary Computer are backscatter in **linear power** units
(terrain-flattened gamma-nought). Averaging must happen in linear units; decibels (dB)
are only taken afterwards. The mean of dB values is not the dB of the mean power: it is
biased low, because speckle makes single-pixel power values strongly right-skewed.
"""
from __future__ import annotations

import warnings
from datetime import date

import numpy as np

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# --------------------------------------------------------------------------- dB / linear
def to_db(linear):
    """10 * log10(power). Zero, negative or missing power gives NaN."""
    x = np.asarray(linear, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 10.0 * np.log10(np.where(x > 0, x, np.nan))
    return out.astype("float32")


def from_db(db):
    """Inverse of :func:`to_db`."""
    return (10.0 ** (np.asarray(db, dtype="float64") / 10.0)).astype("float32")


def mean_power_db(stack, axis=0):
    """Average power values in linear units along ``axis``, then convert to dB.

    NaNs (no data) are ignored; an all-NaN position gives NaN.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return to_db(np.nanmean(np.asarray(stack, dtype="float32"), axis=axis))


def check_linear_power(values, name="backscatter"):
    """Guard against loading dB (or amplitude in DN) where linear power is expected.

    Linear gamma-nought over land is positive and almost always below 1 (0 dB);
    a median outside (0, 1) means the values are not linear power.
    """
    v = np.asarray(values, dtype="float32")
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError(f"{name}: no finite values")
    med = float(np.median(v))
    if not 0 < med < 1:
        raise ValueError(f"{name}: median {med:.4g} is not plausible linear power (expected 0 < median < 1)")
    return med


# --------------------------------------------------------------------------- time windows
def periods(start: str, end: str, kind: str = "semimonthly"):
    """List of (label, first_day, last_day) windows covering ``start``..``end`` (ISO dates).

    ``semimonthly``: days 1-15 and 16-end of each month. ``monthly``: calendar months.
    Labels look like ``"0401"`` (window starting 1 April) and sort in time order.
    """
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    y, m = s.year, s.month
    while date(y, m, 1) <= e:
        nxt = date(y + (m == 12), m % 12 + 1, 1)
        last = date.fromordinal(nxt.toordinal() - 1)
        windows = [(date(y, m, 1), last)] if kind == "monthly" else [(date(y, m, 1), date(y, m, 15)), (date(y, m, 16), last)]
        for a, b in windows:
            if b >= s and a <= e:
                out.append((f"{a.month:02d}{a.day:02d}", max(a, s), min(b, e)))
        y, m = nxt.year, nxt.month
    return out


def period_label(code: str) -> str:
    """'0416' -> 'Apr 16', '0401' -> 'Apr 1'."""
    return f"{MONTHS[int(code[:2]) - 1]} {int(code[2:])}"


def assign_periods(dates, windows) -> np.ndarray:
    """Index of the window containing each date (-1 if none)."""
    out = np.full(len(dates), -1, dtype=int)
    for k, d in enumerate(dates):
        for j, (_, a, b) in enumerate(windows):
            if a <= d <= b:
                out[k] = j
                break
    return out


def s1_composites(vv, vh, dates, windows):
    """Per-window VV, VH and VH/VV in dB from linear-power stacks [time, rows, cols].

    VH/VV (dB) = VH(dB) - VV(dB) of the window means, i.e. the ratio of mean powers.
    Returns (dict name -> 2-D float32 array, acquisitions per window).
    """
    idx = assign_periods(dates, windows)
    layers, counts = {}, []
    for j, (code, _, _) in enumerate(windows):
        sel = np.flatnonzero(idx == j)
        counts.append(int(sel.size))
        if sel.size == 0:
            shape = vv.shape[1:]
            vv_db = vh_db = np.full(shape, np.nan, dtype="float32")
        else:
            vv_db = mean_power_db(vv[sel])
            vh_db = mean_power_db(vh[sel])
        layers[f"S1_{code}_VV"] = vv_db
        layers[f"S1_{code}_VH"] = vh_db
        layers[f"S1_{code}_VHVV"] = (vh_db - vv_db).astype("float32")
    return layers, counts


# --------------------------------------------------------------------------- optical
def normalized_difference(a, b):
    """(a - b) / (a + b), NaN where the denominator is zero or an input is missing."""
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    den = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / den
    return np.where(den == 0, np.nan, out).astype("float32")


def spectral_indices(bands: dict) -> dict:
    """Indices from a dict of Sentinel-2 reflectance arrays.

    NDVI  = (B08 - B04) / (B08 + B04)      Rouse et al. (1974)
    LSWI  = (B8A - B11) / (B8A + B11)      land surface water index (Xiao et al. 2005); same formula as NDMI
    NDWI  = (B03 - B08) / (B03 + B08)      McFeeters (1996), open water
    MNDWI = (B03 - B11) / (B03 + B11)      Xu (2006), open water
    """
    return {
        "NDVI": normalized_difference(bands["B08"], bands["B04"]),
        "LSWI": normalized_difference(bands["B8A"], bands["B11"]),
        "NDWI": normalized_difference(bands["B03"], bands["B08"]),
        "MNDWI": normalized_difference(bands["B03"], bands["B11"]),
    }


def s2_layers(monthly: dict[str, dict]) -> dict:
    """Bands + indices per monthly composite, named ``S2_<MMDD>_<band>``."""
    out = {}
    for code, bands in monthly.items():
        for name, arr in list(bands.items()) + list(spectral_indices(bands).items()):
            out[f"S2_{code}_{name}"] = np.asarray(arr, dtype="float32")
    return out


def parse_name(name: str):
    """'S1_0401_VH' -> ('S1', '0401', 'VH')."""
    sensor, code, var = name.split("_", 2)
    return sensor, code, var


def select(names: list[str], sensors=("S1", "S2"), until: str | None = None) -> list[int]:
    """Indices of features from ``sensors`` whose window starts on or before ``until`` (MMDD)."""
    out = []
    for k, n in enumerate(names):
        s, code, _ = parse_name(n)
        if s in sensors and (until is None or code <= until):
            out.append(k)
    return out
