"""Data access from the Microsoft Planetary Computer STAC API (no account or key needed).

* ``sentinel-1-rtc``: Sentinel-1 radiometrically terrain-corrected backscatter, VV and VH,
  linear power, 10 m pixel spacing. One relative orbit is used so the viewing geometry is
  the same on every date.
* ``sentinel-2-l2a``: surface reflectance with the Scene Classification Layer (SCL) cloud mask.
* ``usda-cdl``: USDA NASS Cropland Data Layer, 30 m, used as labels.

Rasters are cached as GeoTIFFs in ``data/`` so reruns are fast.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

# SCL classes kept as clear: 4 vegetation, 5 bare soil, 6 water, 7 unclassified, 2 dark area.
# Dropped: 0 no data, 1 saturated/defective, 3 cloud shadow, 8/9 cloud, 10 thin cirrus, 11 snow.
CLEAR_SCL = (2, 4, 5, 6, 7)

LOAD_THREADS = int(os.environ.get("RICEMAP_LOAD_THREADS", "8"))
_POOL = None


def load_pool():
    """One thread pool shared by every load, with all its threads started up front.

    On Windows, starting a *new* Python thread after GDAL has made network reads was seen to
    hang forever inside ``Thread.start()`` (odc-stac creates a fresh pool for each load call).
    Reusing threads that already exist avoids creating any thread after the first read.
    Returns None (sequential reads) when RICEMAP_LOAD_THREADS is 0 or 1.
    """
    global _POOL
    if LOAD_THREADS <= 1:
        return None
    if _POOL is None:
        import threading
        from concurrent.futures import ThreadPoolExecutor

        class _KeepAlivePool(ThreadPoolExecutor):
            def __exit__(self, *exc):  # odc-stac runs ``with pool:``; keep the threads for the next load
                return False

        _POOL = _KeepAlivePool(max_workers=LOAD_THREADS, thread_name_prefix="cog-read")
        barrier = threading.Barrier(LOAD_THREADS)
        list(_POOL.map(lambda _: barrier.wait(), range(LOAD_THREADS)))  # forces all threads to exist
    return _POOL


@dataclass
class Grid:
    """Pixel grid shared by every raster in a run."""

    crs: str
    transform: tuple  # affine coefficients (a, b, c, d, e, f)
    shape: tuple  # (rows, cols)
    resolution: float


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS 84 / UTM zone containing a point."""
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _catalog():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def _grid_of(ds, epsg, resolution) -> Grid:
    gb = ds.odc.geobox
    t = gb.affine
    return Grid(crs=f"EPSG:{epsg}", transform=(t.a, t.b, t.c, t.d, t.e, t.f), shape=tuple(gb.shape),
                resolution=resolution)


def _geobox(grid: Grid):
    from affine import Affine
    from odc.geo.geobox import GeoBox

    return GeoBox(grid.shape, Affine(*grid.transform), grid.crs)


# --------------------------------------------------------------------------- Sentinel-1
def search_sentinel1(bbox, datetime: str, orbit_state: str = "descending", relative_orbit: int | None = None):
    """Sentinel-1 RTC items over ``bbox``, restricted to one orbit direction (and relative orbit).

    If ``relative_orbit`` is None, the relative orbit with the most acquisitions is chosen.
    """
    items = list(
        _catalog()
        .search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=datetime,
                query={"sat:orbit_state": {"eq": orbit_state}})
        .items()
    )
    if not items:
        raise RuntimeError(f"No {orbit_state} Sentinel-1 RTC items for {datetime}")
    orbits = [it.properties["sat:relative_orbit"] for it in items]
    if relative_orbit is None:
        relative_orbit = max(set(orbits), key=orbits.count)
    items = [it for it, o in zip(items, orbits) if o == relative_orbit]
    items.sort(key=lambda it: it.datetime)
    return items, relative_orbit


def load_sentinel1(items, bbox, epsg: int, resolution: float, min_valid: float = 0.98):
    """VV and VH linear power stacks [time, rows, cols] on a UTM grid.

    Pixels are averaged (in linear power) from 10 m to ``resolution``. Acquisitions from the
    same day (adjacent slices) are merged. No-data and non-positive values become NaN.
    Any date whose valid-pixel share is below ``min_valid`` (a failed or garbled network
    read) is read again without threads; the shares before and after are returned.
    Returns (vv, vh, acquisition dates, platforms, Grid, valid share per date, reloaded dates).
    """
    import odc.stac

    kw = dict(bands=["vv", "vh"], bbox=bbox, crs=f"EPSG:{epsg}", resolution=resolution, groupby="solar_day",
              resampling="average", dtype="float32", nodata=np.nan, progress=None)
    ds = odc.stac.load(items, pool=load_pool(), **kw)
    vv = ds["vv"].values
    vh = ds["vh"].values
    for a in (vv, vh):
        a[~(a > 0)] = np.nan  # -32768 fill, zeros and NaN all become missing
    dates = [np.datetime64(t, "D").astype(object) for t in ds.time.values]
    share = lambda t: float(np.mean(np.isfinite(vv[t]) & np.isfinite(vh[t])))  # noqa: E731
    reloaded = []
    for t, d in enumerate(dates):
        if share(t) < min_valid:
            day = [it for it in items if it.datetime.date() == d]
            again = odc.stac.load(day, pool=None, **kw)
            for a, b in ((vv, "vv"), (vh, "vh")):
                x = again[b].values[0]
                x[~(x > 0)] = np.nan
                a[t] = x
            reloaded.append(d.isoformat())
    valid_share = [round(share(t), 4) for t in range(len(dates))]
    plat = {it.datetime.date(): it.properties.get("platform", "") for it in items}
    platforms = [plat.get(d, "") for d in dates]
    return vv, vh, dates, platforms, _grid_of(ds, epsg, resolution), valid_share, reloaded


# --------------------------------------------------------------------------- Sentinel-2
def search_sentinel2(bbox, datetime: str, max_cloud: float | None = None):
    """All Sentinel-2 L2A items over ``bbox`` (optionally under a scene cloud limit)."""
    q = {"eo:cloud_cover": {"lt": max_cloud}} if max_cloud is not None else None
    items = list(_catalog().search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=datetime, query=q).items())
    items.sort(key=lambda it: it.datetime)
    return items


def boa_offset(baseline: str) -> float:
    """Processing baseline 04.00+ (from 25 January 2022) adds a -1000 DN offset to L2A reflectance."""
    return -1000.0 if float(baseline) >= 4.0 else 0.0


def _reread_day(items, day, grid, **kw):
    """Load one acquisition day again without threads (used when a threaded read came back incomplete)."""
    import odc.stac

    return odc.stac.load([it for it in items if it.datetime.date() == day], geobox=_geobox(grid),
                         groupby="solar_day", pool=None, progress=None, **kw)


def load_scl(items, grid: Grid, max_nodata: float = 0.01):
    """Clear-sky mask [time, rows, cols] from the SCL band, plus the acquisition dates.

    The box lies inside one tile, so SCL should have data everywhere on every date. A date with
    more than ``max_nodata`` missing pixels is treated as a failed read and read again without
    threads. Returns (clear mask, dates, re-read dates).
    """
    import odc.stac

    kw = dict(bands=["SCL"], resampling="nearest", dtype="uint8", nodata=0)
    ds = odc.stac.load(items, geobox=_geobox(grid), groupby="solar_day", pool=load_pool(), progress=None, **kw)
    scl = ds["SCL"].values
    dates = [np.datetime64(t, "D").astype(object) for t in ds.time.values]
    reread = []
    for t, d in enumerate(dates):
        if (scl[t] == 0).mean() > max_nodata:
            scl[t] = _reread_day(items, d, grid, **kw)["SCL"].values[0]
            reread.append(d.isoformat())
    return np.isin(scl, CLEAR_SCL), dates, reread


def load_s2_median(items, grid: Grid, max_missing: float = 0.001):
    """Cloud-masked per-pixel median reflectance of ``items`` (one composite).

    A date where SCL says "clear" but a band has no data on more than ``max_missing`` of the
    clear pixels is treated as a failed read and read again without threads.
    Returns (dict band -> float32 array, clear-observation count per pixel, re-read dates).
    """
    import odc.stac

    kw = dict(bands=S2_BANDS + ["SCL"], resampling={"SCL": "nearest", "*": "average"}, dtype="uint16", nodata=0)
    ds = odc.stac.load(items, geobox=_geobox(grid), groupby="solar_day", pool=load_pool(), progress=None, **kw)
    arr = {b: ds[b].values for b in S2_BANDS + ["SCL"]}
    times = ds.time.values
    reread = []
    for t, tt in enumerate(times):
        clear_t = np.isin(arr["SCL"][t], CLEAR_SCL)
        if not clear_t.any():
            continue
        missing = max(float(((arr[b][t] == 0) & clear_t).sum()) / clear_t.sum() for b in S2_BANDS)
        if missing > max_missing:
            d = np.datetime64(tt, "D").astype(object)
            again = _reread_day(items, d, grid, **kw)
            for b in arr:
                arr[b][t] = again[b].values[0]
            reread.append(d.isoformat())
    offsets = {np.datetime64(it.datetime.date()): boa_offset(it.properties.get("s2:processing_baseline", "0"))
               for it in items}
    day_off = np.array([offsets.get(np.datetime64(t, "D"), 0.0) for t in times], dtype="float32")
    clear = np.isin(arr["SCL"], CLEAR_SCL)
    bands = {}
    for b in S2_BANDS:
        dn = arr[b].astype("float32")
        refl = (dn + day_off[:, None, None]) / 10000.0
        refl[~(clear & (dn > 0))] = np.nan
        with warnings.catch_warnings():  # all-NaN pixels (never clear) are expected
            warnings.simplefilter("ignore", RuntimeWarning)
            bands[b] = np.nanmedian(refl, axis=0).astype("float32")
    return bands, clear.sum(axis=0).astype("int16"), reread


# --------------------------------------------------------------------------- CDL
def search_cdl(bbox, year: int):
    """CDL 'cropland' items for ``year``."""
    items = [
        it for it in _catalog().search(collections=["usda-cdl"], bbox=bbox, datetime=f"{year}-01-01/{year}-12-31").items()
        if it.properties.get("usda_cdl:type") == "cropland"
    ]
    if not items:
        raise RuntimeError(f"No USDA CDL 'cropland' layer for {year} in the Planetary Computer catalog")
    return items


def load_cdl(items, grid: Grid) -> np.ndarray:
    """CDL codes resampled (nearest neighbour) from 30 m onto ``grid``."""
    import odc.stac

    ds = odc.stac.load(items, bands=["cropland"], geobox=_geobox(grid), resampling="nearest", dtype="uint8",
                       nodata=0, pool=load_pool())
    return ds["cropland"].values[0]


def cdl_colors(item, codes) -> dict:
    """Legend colors (hex) of CDL ``codes``, read from the color table stored in the CDL GeoTIFF."""
    import rasterio

    with rasterio.open(item.assets["cropland"].href) as src:
        cmap = src.colormap(1)
    return {int(c): "#%02x%02x%02x" % cmap[int(c)][:3] for c in codes if int(c) in cmap}


# --------------------------------------------------------------------------- caching
def save_stack(path: Path, arrays, names: list[str], grid: Grid, nodata=np.nan):
    """Write a multi-band GeoTIFF with band descriptions."""
    import rasterio
    from affine import Affine

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = np.asarray(arrays)
    if arrays.ndim == 2:
        arrays = arrays[None]
    profile = dict(driver="GTiff", height=grid.shape[0], width=grid.shape[1], count=arrays.shape[0],
                   dtype=arrays.dtype.name, crs=grid.crs, transform=Affine(*grid.transform), compress="deflate",
                   tiled=True, nodata=nodata, BIGTIFF="IF_SAFER",
                   predictor=3 if arrays.dtype.kind == "f" else 2)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arrays)
        for i, n in enumerate(names, start=1):
            dst.set_band_description(i, n)


def read_stack(path: Path):
    """Inverse of :func:`save_stack`; returns (array, names, Grid)."""
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read()
        names = list(src.descriptions)
        t = src.transform
        grid = Grid(crs=src.crs.to_string(), transform=(t.a, t.b, t.c, t.d, t.e, t.f), shape=(src.height, src.width),
                    resolution=abs(t.a))
    return arr, names, grid


def configure_gdal():
    """GDAL data path when the env is not activated, and HTTP timeouts/retries for COG reads."""
    for key, value in {"GDAL_HTTP_TIMEOUT": "60", "GDAL_HTTP_MAX_RETRY": "5", "GDAL_HTTP_RETRY_DELAY": "2"}.items():
        os.environ.setdefault(key, value)
    if "GDAL_DATA" not in os.environ:
        import sys

        for cand in (Path(sys.prefix) / "Library" / "share" / "gdal", Path(sys.prefix) / "share" / "gdal"):
            if cand.exists():
                os.environ["GDAL_DATA"] = str(cand)
                break
