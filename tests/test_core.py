"""Fast unit tests (no network)."""
from datetime import date

import numpy as np
import pytest

from ricemap.assess import error_matrix, feature_groups, olofsson, sample_metrics
from ricemap.features import (assign_periods, check_linear_power, from_db, mean_power_db, normalized_difference,
                              periods, s1_composites, select, spectral_indices, to_db)
from ricemap.train import (CORN, OTHER, RICE, block_ids, buffer_zone, cdl_to_classes, clustered_sample,
                           interior_mask, stratified_sample)


# ---------------------------------------------------------------- dB / linear power
def test_db_round_trip_and_known_values():
    assert to_db(1.0) == pytest.approx(0.0)
    assert to_db(0.1) == pytest.approx(-10.0)
    assert to_db(0.01) == pytest.approx(-20.0)
    x = np.array([0.082, 0.025, 0.5])
    assert from_db(to_db(x)) == pytest.approx(x, rel=1e-5)
    # Gulf Coast check value: VV 0.082 linear is about -10.9 dB
    assert to_db(0.082) == pytest.approx(-10.86, abs=0.01)


def test_to_db_rejects_non_positive():
    out = to_db(np.array([0.0, -0.1, np.nan, 0.1]))
    assert np.isnan(out[:3]).all() and out[3] == pytest.approx(-10.0)


def test_average_in_linear_not_db():
    # Two speckled looks, 1.0 and 0.01 linear: mean power 0.505 -> -2.97 dB.
    # Averaging the dB values instead (0 and -20 -> -10 dB) would be 7 dB too low.
    stack = np.array([[[1.0]], [[0.01]]], dtype="float32")
    assert mean_power_db(stack)[0, 0] == pytest.approx(10 * np.log10(0.505), abs=1e-4)
    assert mean_power_db(stack)[0, 0] > to_db(stack).mean() + 6
    # missing looks are ignored, all-missing gives NaN
    s = np.array([[[0.1, np.nan]], [[np.nan, np.nan]]], dtype="float32")
    out = mean_power_db(s)
    assert out[0, 0] == pytest.approx(-10.0) and np.isnan(out[0, 1])


def test_check_linear_power():
    assert check_linear_power(np.array([0.08, 0.1, 0.02])) == pytest.approx(0.08)
    with pytest.raises(ValueError):
        check_linear_power(np.array([-10.9, -16.1, -12.0]))  # values already in dB
    with pytest.raises(ValueError):
        check_linear_power(np.array([np.nan]))


# ---------------------------------------------------------------- time windows and composites
def test_semimonthly_and_monthly_periods():
    p = periods("2021-02-01", "2021-03-31", "semimonthly")
    assert [c for c, _, _ in p] == ["0201", "0216", "0301", "0316"]
    assert p[1][2] == date(2021, 2, 28)
    m = periods("2021-02-01", "2021-11-30", "monthly")
    assert len(m) == 10 and m[-1][0] == "1101" and m[-1][2] == date(2021, 11, 30)


def test_s1_composites_window_assignment():
    win = periods("2021-04-01", "2021-04-30", "semimonthly")
    dates = [date(2021, 4, 3), date(2021, 4, 9), date(2021, 4, 21)]
    assert assign_periods(dates, win).tolist() == [0, 0, 1]
    vv = np.array([0.1, 0.3, 0.05], dtype="float32")[:, None, None] * np.ones((3, 2, 2), "float32")
    vh = vv / 10
    layers, counts = s1_composites(vv, vh, dates, win)
    assert counts == [2, 1]
    assert layers["S1_0401_VV"][0, 0] == pytest.approx(to_db(0.2))
    assert layers["S1_0416_VH"][0, 0] == pytest.approx(to_db(0.005))
    assert layers["S1_0401_VHVV"][0, 0] == pytest.approx(-10.0, abs=1e-4)  # VH/VV = 0.1


def test_feature_selection_until_cutoff():
    names = ["S1_0401_VV", "S1_0416_VH", "S1_0501_VV", "S2_0401_NDVI", "S2_0501_B04"]
    assert select(names, ("S1",)) == [0, 1, 2]
    assert select(names, ("S1", "S2"), until="0431") == [0, 1, 3]


def test_feature_groups_by_window_and_month():
    names = ["S1_0401_VV", "S1_0401_VH", "S1_0416_VV", "S2_0401_NDVI"]
    g = feature_groups(names, by="window")
    assert g["S1 Apr 1"] == [0, 1] and g["S1 Apr 16"] == [2] and g["S2 Apr 1"] == [3]
    g = feature_groups(names, by="month")
    assert g["S1 Apr"] == [0, 1, 2] and g["S2 Apr"] == [3]


# ---------------------------------------------------------------- optical
def test_normalized_difference():
    out = normalized_difference(np.array([0.4, 0.2, 0.0]), np.array([0.1, 0.2, 0.0]))
    assert out[0] == pytest.approx(0.6) and out[1] == pytest.approx(0.0) and np.isnan(out[2])


def test_indices_flooded_field_vs_canopy():
    flooded = {"B03": 0.07, "B04": 0.05, "B08": 0.06, "B8A": 0.06, "B11": 0.03}
    canopy = {"B03": 0.08, "B04": 0.04, "B08": 0.45, "B8A": 0.45, "B11": 0.20}
    f = spectral_indices({k: np.array(v) for k, v in flooded.items()})
    c = spectral_indices({k: np.array(v) for k, v in canopy.items()})
    assert c["NDVI"] > 0.7 > f["NDVI"]
    assert f["LSWI"] > f["NDVI"]  # the flooding signal used in phenology-based rice mapping
    assert f["NDWI"] > 0 > c["NDWI"]


# ---------------------------------------------------------------- labels, blocks, sampling
def test_cdl_to_classes():
    cdl = np.array([[3, 1, 176], [12, 0, 111], [3, 241, 2]], dtype=np.uint8)
    labels, rep = cdl_to_classes(cdl)
    assert labels.tolist() == [[RICE, CORN, OTHER], [0, 0, OTHER], [RICE, 0, OTHER]]
    assert set(rep["excluded_present"]) == {12, 241}


def test_interior_mask_drops_edges():
    lab = np.full((6, 6), OTHER, dtype=np.uint8)
    lab[:, :3] = RICE
    m = interior_mask(lab)
    assert m[3, 0] and m[3, 1] and not m[3, 2] and not m[3, 3] and m[3, 4]


def test_block_ids():
    transform = (20.0, 0.0, 500000.0, 0.0, -20.0, 3400000.0)
    b = block_ids((10, 10), transform, 100.0)
    assert len(np.unique(b)) == 4 and b[0, 0] == b[4, 4] != b[0, 5]


def test_buffer_zone():
    m = np.zeros((7, 7), bool)
    m[3, 3] = True
    z = buffer_zone(m, 1)
    assert z.sum() == 4 and not z[3, 3]


def test_stratified_sample_caps_per_class():
    strata = np.array([1] * 100 + [2] * 5).reshape(-1, 1)
    idx = stratified_sample(strata, np.ones_like(strata, bool), 10, np.random.default_rng(0))
    vals = strata.ravel()[idx]
    assert (vals == 1).sum() == 10 and (vals == 2).sum() == 5


def test_clustered_sample_makes_patches():
    labels = np.ones((50, 50), dtype=np.uint8)
    idx, seeds = clustered_sample(labels, labels > 0, 50, 5, np.random.default_rng(1))
    assert seeds.shape == idx.shape and (seeds >= 0).all()
    assert len(np.unique(seeds)) < idx.size and 50 <= idx.size <= 74


# ---------------------------------------------------------------- Olofsson et al. (2014)
# Worked example (Tables 8-9 of Olofsson et al. 2014, Remote Sensing of Environment 148: 42-57):
# strata deforestation, forest gain, stable forest, stable non-forest; 30 m Landsat pixels.
OLOFSSON_N = np.array([[66, 0, 5, 4], [0, 55, 8, 12], [1, 0, 153, 11], [2, 1, 9, 313]])
OLOFSSON_MAPPED_PX = np.array([200_000, 150_000, 3_200_000, 6_450_000])


def test_olofsson_worked_example():
    r = olofsson(OLOFSSON_N, OLOFSSON_MAPPED_PX, pixel_area=0.09)  # hectares
    assert r["estimated_area"][0] == pytest.approx(21_158, abs=5)  # published 21,158 ha
    assert r["estimated_area_ci95"][0] == pytest.approx(6_158, abs=5)  # +/- 6,158 ha
    assert r["estimated_area"][1] == pytest.approx(11_686, abs=5)
    assert r["estimated_area_ci95"][1] == pytest.approx(3_756, abs=5)
    assert r["overall_accuracy"] == pytest.approx(0.95, abs=0.005)
    assert r["users_accuracy"][0] == pytest.approx(0.88, abs=0.005)
    assert r["producers_accuracy"][0] == pytest.approx(0.75, abs=0.005)
    assert sum(r["estimated_area"]) == pytest.approx(OLOFSSON_MAPPED_PX.sum() * 0.09)


def test_olofsson_guards_small_strata():
    with pytest.warns(UserWarning):
        r = olofsson(np.array([[5, 0], [0, 1]]), [100, 50])
    assert np.isnan(r["overall_accuracy_ci95"])
    with pytest.raises(ValueError):
        olofsson(np.array([[5, 0], [0, 0]]), [100, 50])


def test_sample_metrics_perfect():
    m = sample_metrics(error_matrix([1, 1, 2, 2], [1, 1, 2, 2], [1, 2]))
    assert m["overall_accuracy"] == 1.0 and m["balanced_accuracy"] == 1.0
