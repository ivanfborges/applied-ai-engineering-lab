"""Numerical contracts and bounded export checks for the offline visual lab."""

from html.parser import HTMLParser
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import average_precision_score


TOPIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOPIC / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load("anomaly_visual_core", "anomaly_visual_core.py")
scenes = load("anomaly_visuals", "anomaly_visuals.py")
cli = load("day41_visual_cli", "visual_lab.py")


def test_contamination_estimates_and_zero_mad():
    states = core.contamination_states(12)
    a, b = states[0], states[-1]
    np.testing.assert_array_equal(a["values"], b["values"][:100])
    assert b["n_outliers"] == 10
    assert abs(b["mean"] - a["mean"]) > abs(b["median"] - a["median"])
    assert b["std"] - a["std"] > b["robust_scale"] - a["robust_scale"]
    with pytest.raises(ValueError, match="zero MAD"):
        core.robust_summary(np.ones(5))


@pytest.mark.parametrize("profile", ["quick", "full"])
def test_data_roles_and_determinism(profile):
    train, cal, test, groups = core.telemetry_data(profile)
    cfg = core.PROFILES[profile]
    assert train.shape == (cfg["train"], 2)
    assert cal.shape == (cfg["calibration"], 2)
    assert test.shape == (cfg["normal_test"] + cfg["relationship"] + cfg["extreme"], 2)
    assert np.sum(groups == "normal") == cfg["normal_test"]
    assert not np.shares_memory(train, cal)
    for a, b in zip((train, cal, test, groups), core.telemetry_data(profile)):
        np.testing.assert_array_equal(a, b)


def test_covariance_distance_matches_manual_inverse_and_detects_joint_deviation():
    train, _, _, _ = core.telemetry_data("quick")
    mad = core.RobustMAD().fit(train)
    point = mad.center_ + [1.5, -1.5] * mad.scale_
    delta = point - train.mean(axis=0)
    expected = delta @ np.linalg.inv(np.cov(train, rowvar=False)) @ delta
    assert core.covariance_distance(point[None], train)[0] == pytest.approx(expected)
    assert np.max(np.abs((point - mad.center_) / mad.scale_)) < 3
    assert expected > 9.21034
    with pytest.raises(ValueError, match="singular"):
        core.covariance_distance([[1, 2]], np.ones((8, 2)))


def test_actual_tree_trace_partitions_bounds_and_paths():
    X, selected = core.isolation_data()
    tree, events = core.tree_trace(X)
    for event in events:
        feature, threshold = event["feature"], event["threshold"]
        indices = event["indices"]
        assert np.ptp(X[indices, feature]) > 0
        assert X[indices, feature].min() <= threshold < X[indices, feature].max()
        assert len(event["left"]) > 0 and len(event["right"]) > 0
        np.testing.assert_array_equal(event["left"], indices[X[indices, feature] <= threshold])
        np.testing.assert_array_equal(event["right"], indices[X[indices, feature] > threshold])
        x0, x1, y0, y1 = event["bounds"]
        assert np.all((X[indices, 0] >= x0) & (X[indices, 0] <= x1))
        assert np.all((X[indices, 1] >= y0) & (X[indices, 1] <= y1))
        assert event["bounds"][2 * feature] <= threshold <= event["bounds"][2 * feature + 1]
    for i in selected:
        depth = sum(i in event["indices"] for event in events)
        assert core.path_length(X[i], tree.trees_[0]) >= depth
    paths, means, scores = core.ensemble_paths(X, selected, 12)
    np.testing.assert_allclose(means[-1], paths.mean(axis=0))
    np.testing.assert_allclose(scores[-1], 2 ** (-paths.mean(axis=0) / core.average_path_length(len(X))))
    np.testing.assert_allclose(paths[0], [core.path_length(X[i], tree.trees_[0]) for i in selected])


def test_svm_scaler_supports_and_real_boundary_fields():
    train, anomalies = core.ring_data(True)
    configs = core.svm_configurations(train)
    assert len(configs) == 18
    for config in configs:
        model = config["model"]
        svc = model.steps[-1][1] if hasattr(model, "steps") else model
        np.testing.assert_array_equal(config["support"], train[svc.support_])
        if hasattr(model, "steps"):
            scaler = model.steps[0][1]
            np.testing.assert_allclose(scaler.center_, np.median(train, axis=0))
            expected = svc.decision_function(scaler.transform(anomalies))
        else:
            expected = svc.decision_function(anomalies)
        np.testing.assert_allclose(model.decision_function(anomalies), expected)
        assert np.isfinite(expected).all()


def test_frozen_thresholds_and_identical_ap_different_builtin_alerts():
    b = scenes.bundle("quick")
    center = b["models"]["OneClassSVM"].steps[0][1].center_.copy()
    policies = core.threshold_policies(b["models"], b["train"], b["cal"], b["test"], b["groups"], "quick")
    assert len(policies) == 11
    for i in (0, 3, 6):
        a, b1, c = policies[i:i + 3]
        np.testing.assert_array_equal(a["scores"], c["scores"])
        assert a["ap"] == b1["ap"] == c["ap"]
        assert a["alerts"] >= b1["alerts"] >= c["alerts"]
        assert a["threshold"] <= b1["threshold"] <= c["threshold"]
        expected = core.calibrate_threshold(a["cal_scores"], a["alpha"])
        assert a["threshold"] == expected
    first, second = policies[-2:]
    np.testing.assert_allclose(first["scores"] - second["scores"], first["scores"][0] - second["scores"][0], atol=1e-14)
    assert first["ap"] == second["ap"]
    assert first["alerts"] != second["alerts"]
    for policy in policies:
        assert policy["ap"] == average_precision_score(b["groups"] != "normal", policy["scores"])
        assert policy["tp"] + policy["fn"] == 20
        assert policy["tn"] + policy["fp"] == 100
    np.testing.assert_array_equal(center, b["models"]["OneClassSVM"].steps[0][1].center_)


def test_monitoring_prefix_invariance_and_approved_window_timing():
    _, latency, _, _ = core.monitoring_data()
    full = core.monitoring_policies(latency)
    prefix = core.monitoring_policies(latency[:315])
    future_changed = latency.copy()
    future_changed[315:] += 10000
    future = core.monitoring_policies(future_changed)
    for key in ("fixed_scores", "fixed_threshold", "adaptive_scores", "adaptive_threshold"):
        np.testing.assert_array_equal(full[key][:315], prefix[key])
        np.testing.assert_array_equal(full[key][:315], future[key][:315])
    for update in full["updates"]:
        assert update["fit_end"] <= update["cal_start"] < update["cal_end"] <= update["activate"] - 12
    incident_changed = latency.copy()
    incident_changed[180:188] += 10000
    revised = core.monitoring_policies(incident_changed)
    np.testing.assert_array_equal(revised["adaptive_scores"][300:], full["adaptive_scores"][300:])
    with pytest.raises(ValueError, match="windows"):
        core.monitoring_policies(latency, [dict(activate=200, fit_start=0, fit_end=80, cal_start=80, cal_end=201)])


def test_relationship_export_is_self_contained_and_uses_temp_directory(tmp_path):
    result = scenes.relationships(tmp_path, "quick")
    content = (tmp_path / result["files"][0]).read_text(encoding="utf-8")
    scripts = []

    class Scripts(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == "script":
                scripts.append(dict(attrs))

    Scripts().feed(content)
    assert scripts and all("src" not in attrs for attrs in scripts)
    assert "plotly.js" in content
    assert "Synthetic" in content
    assert "Covariance contour" in content
    assert result["observations"]["marked_d2"] > 9.21034


@pytest.mark.parametrize("argv", [[], ["--all", "--quick"], ["--section", "invalid"], ["--all", "--section", "monitoring"]])
def test_cli_invalid_options_are_explicit(argv):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
