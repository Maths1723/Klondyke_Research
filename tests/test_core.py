"""Core tests. Run:  python -m pytest -q tests   (or python tests/test_core.py)"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from klondyke import baselines, cv, metrics, models, qc, units, valley  # noqa: E402


def test_grade_conversion():
    # 1 cent/yd3 at $20.67/oz ~ 19.68 mg/m3
    assert abs(units.cents_per_yd3_to_mg_per_m3(1.0) - 19.68) < 0.01
    # $20.67 per yd3 (2067 cents) is exactly one troy ounce per yd3
    assert abs(units.cents_per_yd3_to_mg_per_m3(2067.0) - units.TROY_OZ_MG / units.YD3_M3) < 1e-6


def test_hole_id_parsing():
    p = qc.parse_hole_id("87B-12")
    assert p["line_id"] == "87B" and p["claim_no"] == 87 and p["hole_num"] == 12
    p = qc.parse_hole_id("141-a")
    assert p["line_id"] == "141" and p["hole_no"] == "a" and np.isnan(p["hole_num"])
    assert qc.parse_hole_id("garbage")["line_id"] is None


def test_feet_datum_detected():
    # rows from van Loon (2016) Appendix 1, Dominion Creek
    rows = [("99-9", 859, 839, 5.49, 1.52), ("96-14", 947, 895, 14.94, 1.22),
            ("125-26", 948, 867, 21.34, 4.27), ("137-28", 872, 835, 2.44, 9.14),
            ("100-2", 858, 832, 6.10, 3.05), ("128-13", 862, 805, 13.41, 3.66)]
    df = pd.DataFrame(rows, columns=["hole_id", "surface_elev_raw", "bedrock_elev_raw",
                                     "overburden_m", "dredge_section_m"])
    rep = qc.infer_elevation_unit(df)
    assert rep.inferred_unit == "ft", rep


def test_projection_straight_line():
    poly = np.column_stack([np.linspace(0, 1000, 201), np.zeros(201)])
    s, n = valley.project_to_centreline(np.array([100.0, 500.0]), np.array([20.0, -30.0]), poly)
    assert np.allclose(s, [100, 500]) and np.allclose(n, [20, -30])  # + = left of downstream (+x)


def test_cv_splits_disjoint():
    lines = np.repeat(np.arange(20), 5)
    for tr, te in cv.leave_one_line_out(lines):
        assert set(lines[tr]).isdisjoint(set(lines[te]))
    s = np.linspace(0, 5000, 100)
    for tr, te in cv.along_valley_blocks(s, 1000, 150):
        assert np.min(np.abs(s[tr][:, None] - s[te][None, :])) >= 150 - 1e-9


def test_crps_reduces_to_mae():
    y, mu = np.array([1.0, 2.0]), np.array([1.5, 1.0])
    assert np.allclose(metrics.crps_gaussian(y, mu, None), np.abs(y - mu))
    # Gaussian CRPS of a perfect-mean forecast is sd*(sqrt(2)-1)/sqrt(pi)
    c = metrics.crps_gaussian(np.array([0.0]), np.array([0.0]), np.array([1.0]))[0]
    assert abs(c - (np.sqrt(2) - 1) / np.sqrt(np.pi)) < 1e-9


def test_kriging_exact_interpolation_without_nugget_reasonable():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1000, (150, 2))
    y = np.sin(X[:, 0] / 150) + 0.05 * rng.normal(size=150)
    m = baselines.OrdinaryKriging().fit(X, y)
    mu, sd = m.predict(X[:5] + 1.0)
    assert np.all(np.abs(mu - y[:5]) < 0.3) and np.all(sd > 0)


def test_gp_recovers_anisotropy():
    rng = np.random.default_rng(1)
    s = rng.uniform(0, 3000, 300); n = rng.uniform(-200, 200, 300)
    f = np.sin(s / 300) * np.exp(-0.5 * (n / 40) ** 2)
    y = f + 0.05 * rng.normal(size=300)
    gp = models.AnisotropicGP(n_restarts=1).fit(np.column_stack([s, n]), y)
    ls = gp.length_scales_m()
    assert ls[0] > 2 * ls[1], ls  # longer correlation along the valley than across




def test_grain_likelihood_normalised_and_derivatives():
    from scipy.integrate import quad
    from klondyke import sampling
    mbar, alpha = 0.8, 0.7
    for lam in (0.5, 3.0, 25.0):
        llt, _, _ = sampling.count_posterior(np.array([lam]), np.array([1]), np.array([np.nan]),
                                             np.array([0.3]), mbar, alpha)
        f = lambda s: np.exp(sampling.count_posterior(np.array([lam]), np.array([2]), np.array([s]),
                                                      np.array([0.3]), mbar, alpha)[0][0])
        mu = lam * mbar
        total = np.exp(-lam) + np.exp(llt[0]) + quad(f, 0.3, mu * 8 + 20, limit=400, points=[mu])[0]
        assert abs(total - 1) < 1e-4, (lam, total)
    lam = np.array([0.7, 4.0, 30.0, 30.0]); kind = np.array([0, 2, 2, 1])
    s = np.array([np.nan, 2.5, 30.0, np.nan]); st = np.full(4, 0.4); h = 1e-4
    ll, EN, VN = sampling.count_posterior(lam, kind, s, st, mbar, alpha)
    lp, _, _ = sampling.count_posterior(lam * np.exp(h), kind, s, st, mbar, alpha)
    lm, _, _ = sampling.count_posterior(lam * np.exp(-h), kind, s, st, mbar, alpha)
    assert np.allclose((lp - lm) / (2 * h), EN - lam, atol=1e-5)


def test_price_basis_inference():
    from klondyke import ycgc_service as Y
    h = pd.DataFrame({"map_id": ["A"] * 4 + ["B"] * 4, "grade_cents_raw": [10, 50, 120, 7, 16.9, 84.5, 202.8, 11.8],
                      "x": [0, 100, 200, 300, 2, 101, 199, 303], "y": [0, 0, 0, 0, 1, 2, 1, 0]})
    inf = Y.infer_price_basis(h)
    assert len(inf) == 1 and inf.iloc[0].map_id == "B" and inf.iloc[0].inferred_price == 35.0


def test_line_graph_recovers_lines():
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(3)
    xs, ys, lab = [], [], []
    for k in range(12):                       # lines 150 m apart, holes 60 m apart, tilted 25 deg
        for j in range(8):
            t = np.deg2rad(25)
            xs.append(k * 150 + j * 60 * np.sin(t) + rng.normal(0, 3)); ys.append(j * 60 * np.cos(t) + rng.normal(0, 3))
            lab.append(k)
    poly = valley.centreline_from_points(np.array(xs), np.array(ys))
    got = valley.cluster_lines_graph(np.array(xs), np.array(ys), poly, radius=75, max_angle_deg=90)
    assert adjusted_rand_score(lab, got) > 0.99

def test_grain_information_limits():
    """Zero-channel information peaks near lam ~ 1.6 and vanishes for many grains;
    dispersion information vanishes once grain variance is small next to v0."""
    from klondyke import sampling
    lam = np.array([0.1, 1.6, 20.0])
    Iz, Id = sampling.grain_information(lam, 1.0, 1.0, 1.0, v0=0.5)
    assert Iz[1] > Iz[0] and Iz[1] > Iz[2] and Iz[2] < 1e-6
    assert abs(Iz[1] - 1.6 ** 2 * np.exp(-1.6) / (1 - np.exp(-1.6))) < 1e-12
    assert Id[1] > Id[2]
    _, Id_many = sampling.grain_information(1e4, 1.0, 1.0, 1.0, v0=0.5)
    assert Id_many < 1e-6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
