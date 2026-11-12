"""Scientific contracts and bounded export smoke tests; never overwrite previews."""

from dataclasses import replace
import json
from pathlib import Path
import re
import sys

import numpy as np
import pytest
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from day48_visual_core import (LabConfig, METHODS, LIBRARY_METHODS, adjusted_matrix,
                               dependence_study, discovery_counts, equicorrelation,
                               independent_fwer, normal_families, reference_check,
                               sequential_study, tradeoff_study)
from from_scratch import adjust_pvalues


@pytest.mark.parametrize("m,alpha", [(1, .05), (20, .05), (100, .01), (20, 0), (5, 1)])
def test_theoretical_fwer(m, alpha):
    assert independent_fwer(m, alpha) == pytest.approx(1-(1-alpha)**m)


def test_fwer_surface_broadcast_and_monotonicity():
    m = np.arange(1, 30)
    alpha = np.linspace(.001, .1, 15)
    values = independent_fwer(m[None, :], alpha[:, None])
    assert values.shape == (15, 29)
    assert np.all(np.diff(values, axis=0) > 0)
    assert np.all(np.diff(values, axis=1) > 0)
    np.testing.assert_allclose(values, 1-(1-alpha[:, None])**m[None, :])


@pytest.mark.parametrize("method", METHODS)
def test_vectorized_adjustments_match_statsmodels_and_educational_core(method):
    rng = np.random.default_rng(48)
    p = np.vstack([rng.uniform(size=(15, 5)), [0, 1, .03, .04, .03], [.001, .012, .021, .049, .2]])
    original = p.copy()
    actual = adjusted_matrix(p, method)
    expected = p if method == "uncorrected" else np.stack([
        multipletests(row, method=LIBRARY_METHODS[method])[1] for row in p])
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_array_equal(actual <= .05, expected <= .05)
    np.testing.assert_array_equal(p, original)
    if method != "by":
        np.testing.assert_allclose(actual, np.stack([adjust_pvalues(row, method) for row in p]))


def test_last_passing_rank_and_holm_dominates_bonferroni():
    np.testing.assert_array_equal(adjusted_matrix([[.03, .04]], "bh") <= .05, [[True, True]])
    p = np.random.default_rng(489).uniform(size=(100, 30))
    p[:20, :3] *= .002
    assert np.all((adjusted_matrix(p, "bonferroni") > .05) | (adjusted_matrix(p, "holm") <= .05))
    assert reference_check(p, .05)["families_checked"] == 32


def test_mean_fdp_includes_zero_discovery_runs_and_known_counts():
    rejected = np.array([[True, False], [True, True], [False, False]])
    counts, summary = discovery_counts(rejected, np.array([True, False]))
    np.testing.assert_array_equal(counts["tp"], [0, 1, 0])
    np.testing.assert_array_equal(counts["fp"], [1, 1, 0])
    np.testing.assert_array_equal(counts["fn"], [1, 0, 1])
    assert summary["fdr"]["estimate"] == .5
    assert summary["fwer"]["estimate"] == pytest.approx(2/3)
    assert summary["power"]["estimate"] == pytest.approx(1/3)


@pytest.mark.parametrize("rho", [0., .5, 1., -1/19])
def test_equicorrelation_is_psd_and_has_unit_diagonal(rho):
    covariance = equicorrelation(20, rho)
    np.testing.assert_allclose(np.diag(covariance), 1)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-10


@pytest.mark.parametrize("rho", [-.1, 1.1, np.nan])
def test_invalid_correlation_fails(rho):
    with pytest.raises(ValueError):
        equicorrelation(20, rho)


def test_normal_generator_valid_pvalues_covariance_and_reproducibility():
    config = LabConfig(repetitions=10000, hypotheses=8)
    z, p, truth = normal_families(config, 6, rho=.5, signals=0)
    assert ((p >= 0) & (p <= 1)).all()
    assert truth.all()
    np.testing.assert_allclose(np.cov(z, rowvar=False), equicorrelation(8, .5), atol=.06)
    np.testing.assert_array_equal(p, normal_families(config, 6, .5, signals=0)[1])


def test_zero_effect_is_global_null_and_power_is_undefined():
    config = LabConfig(repetitions=100, hypotheses=10, effect=0)
    _, p, truth = normal_families(config, 3)
    assert truth.all()
    sweep, _, _ = tradeoff_study(config)
    assert sweep[0]["effect"] == 0 and sweep[0]["actual_signals"] == 0
    assert sweep[1]["actual_signals"] == 2
    for metrics in sweep[0]["metrics"].values():
        assert metrics["fdr"] == metrics["fwer"]
        assert metrics["power"] is None


def test_peeking_shares_the_actual_paths_and_discloses_selection():
    config = LabConfig(repetitions=1000)
    looks, z, p, decisions, details = sequential_study(config)
    np.testing.assert_allclose(p, 2*norm.sf(np.abs(z)))
    np.testing.assert_array_equal(decisions["fixed_horizon"], p[:, -1] <= config.alpha)
    np.testing.assert_array_equal(decisions["naive_peeking"], (p <= config.alpha).any(axis=1))
    np.testing.assert_array_equal(decisions["bonferroni_looks"], (p <= config.alpha/config.looks).any(axis=1))
    assert np.all(decisions["naive_peeking"] | ~decisions["fixed_horizon"])
    index = details["selected_index_zero_based"]
    assert decisions["naive_peeking"][index] and not decisions["fixed_horizon"][index]
    assert index == np.flatnonzero(decisions["naive_peeking"] & ~decisions["fixed_horizon"])[0]
    assert len(looks) == config.looks and looks[-1] == config.final_n
    single = sequential_study(replace(config, looks=1))[3]
    np.testing.assert_array_equal(single["fixed_horizon"], single["naive_peeking"])
    np.testing.assert_array_equal(single["fixed_horizon"], single["bonferroni_looks"])


def test_dependence_global_null_fdr_equals_fwer():
    results = dependence_study(LabConfig(repetitions=100))
    assert len(results) == 5
    for record in results:
        assert record["min_eigenvalue"] >= -1e-10
        for metrics in record["global_null"].values():
            assert metrics["fdr"] == metrics["fwer"]
            assert metrics["power"] is None


@pytest.mark.parametrize("kwargs", [{"hypotheses": 1}, {"frames": 2}, {"repetitions": 10},
                                    {"nonnull_proportion": 1.1}, {"effect": np.nan},
                                    {"alpha": 0}, {"looks": 10, "final_n": 5},
                                    {"hypotheses": 500, "repetitions": 50000}])
def test_invalid_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        LabConfig(**kwargs)


def test_small_all_export_smoke_and_real_gif_frames(tmp_path, capsys):
    from visualize_day48 import main, surface_figure
    report = main(["--output-dir", str(tmp_path), "--repetitions", "100", "--frames", "8",
                   "--hypotheses", "8", "--null-tests", "100", "--looks", "4", "--final-n", "20"])
    assert len(report["experiments"]) == 6
    assets = [asset for record in report["experiments"].values() for asset in record["validation"]]
    expected = {"01_false_positives.gif", "01_false_positives.png", "02_fwer_growth.png",
                "02_fwer_surface.html", "03_correction_comparison.png", "03_bh_step_up.gif",
                "04_error_control_tradeoffs.png", "04_discovery_simulation.gif",
                "05_peeking_problem.gif", "05_peeking_error_rates.png", "06_correlation_effects.png"}
    assert {asset["file"] for asset in assets} == expected
    gifs = [asset for asset in assets if asset["file"].endswith(".gif")]
    assert len(gifs) == 4
    assert all(asset["distinct_frames"] >= 2 for asset in gifs)
    assert all(asset["bytes"] > 0 for asset in assets)
    page = (tmp_path / "02_fwer_surface.html").read_text(encoding="utf-8")
    assert not re.search(r'<script[^>]+src=["\']https?://', page, re.I)
    assert 'id="m"' in page and 'id="alpha"' in page
    assert json.loads((tmp_path/"run.json").read_text()) == report
    figure = surface_figure()
    np.testing.assert_allclose(np.array(figure.data[0].z),
                               independent_fwer(np.array(figure.data[0].x)[None, :],
                                                np.array(figure.data[0].y)[:, None]))
    with pytest.raises(SystemExit) as error:
        main(["--output-dir", str(tmp_path)])
    assert error.value.code == 2  # Preserve existing historical outputs by default.
    capsys.readouterr()
