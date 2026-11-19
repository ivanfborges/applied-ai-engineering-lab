"""Numerical and real export coverage; all rendering writes to temporary paths."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments
from experiments import (COLORS, Config, SCALES, SignalNet, cached_bundle, diagnostic,
                         jacobian_experiment, normalization_experiment, relu_experiment,
                         stability_surface, supervised_data, train_one)
from visualizations import export_all, positive_log, surface_figure


@pytest.mark.parametrize("strategy", list(SCALES))
def test_weight_variance_dimensions_and_matching(strategy):
    c = Config(width=128, depth=2, batch=4)
    net = SignalNet(c, strategy)
    expected = 2 / c.width * SCALES[strategy]**2
    assert float(net.linears[0].weight.detach().var(unbiased=False)) == pytest.approx(expected, rel=.06)
    assert net(torch.ones(4, 128, dtype=torch.float64)).shape == (4, 128)
    he = SignalNet(c)
    torch.testing.assert_close(net.linears[0].weight / SCALES[strategy], he.linears[0].weight,
                               rtol=1e-13, atol=1e-14)


def test_symmetry_relu_moments_not_variance():
    r = relu_experiment(n=100000)
    assert r["q_after"] / r["q_before"] == pytest.approx(.5, abs=1e-12)
    assert r["var_after"] / r["var_before"] == pytest.approx(.5 - 1/(2*np.pi), abs=.007)
    assert r["var_after"] != pytest.approx(r["q_after"])


@pytest.mark.parametrize("activation,normalization", [("relu","none"),("tanh","none"),("relu","layer"),("relu","batch")])
def test_diagnostic_finite_objectives_and_reproducibility(activation, normalization):
    c = Config(width=12, depth=5, batch=8, activation=activation, normalization=normalization)
    a, b = diagnostic(c), diagnostic(c)
    assert a == b
    assert len(a["layers"]) == c.depth+1
    assert a["layers"][-1]["probe_rms"] == pytest.approx(1)
    for r in a["layers"]:
        assert r["q"] == pytest.approx(r["variance"] + r["mean"]**2)
        assert all(r[key] is not None for key in ("q","probe_rms","loss_rms"))
    q = a["layers"][-1]["q"]
    assert a["layers"][-1]["loss_rms"] == pytest.approx(2*np.sqrt(q)/(c.width*c.batch))


def test_surface_actual_measurements_and_scale_homogeneity():
    c = Config(width=8, depth=5, batch=8)
    s = stability_surface(c)
    i = s["scales"].index(1.)
    reference = diagnostic(replace(c, depth=s["depths"][-1], batch=16))
    assert s["forward"][i][-1] == pytest.approx(reference["layers"][-1]["q"])
    assert s["backward"][i][-1] == pytest.approx(reference["layers"][0]["probe_rms"])
    j = s["scales"].index(1 / np.sqrt(2))
    for k, depth in enumerate(s["depths"]):
        assert s["forward"][j][k] == pytest.approx(s["forward"][i][k]*.5**depth, rel=1e-10)
    fig = surface_figure({"surface":s})
    assert len(fig.data) == 6
    assert list(fig.layout.updatemenus[0].buttons[1].args[0]["visible"]) == [False]*3+[True]*3


def test_normalization_axes_companions_populated_buffers_and_eval_formula():
    n = normalization_experiment(repetitions=80)
    x, bn, ln = (np.array(n[key]) for key in ("before","batchnorm","layernorm"))
    np.testing.assert_allclose(bn.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(ln.mean(1), 0, atol=1e-12)
    np.testing.assert_allclose(bn.var(0), x.var(0)/(x.var(0)+1e-5), atol=1e-12)
    np.testing.assert_allclose(n["anchor_ln_a"], n["anchor_ln_b"], atol=1e-12)
    assert not np.allclose(n["anchor_bn_a"], n["anchor_bn_b"])
    assert n["training_batches_for_eval"] == 101
    expected = (np.array(n["eval_input"])-n["running_mean"])/np.sqrt(np.array(n["running_var"])+n["eps"])
    np.testing.assert_allclose(n["eval_test"], expected, atol=1e-12)
    assert n["sensitivity"][0]["feature_mean_sd"][0] > n["sensitivity"][-1]["feature_mean_sd"][0]


def test_actual_sgd_update_clipping_and_fixed_data():
    c = dict(seed=4, strategy="large", lr=.15, epochs=3, width=8, depth=3)
    run = train_one(**c, clip=1)
    assert run == train_one(**c, clip=1)
    for row in run["history"][:-1]:
        assert row["clipped_gradient_norm"] <= 1+1e-10
        assert row["update_norm"] == pytest.approx(.15*row["clipped_gradient_norm"], abs=1e-12)
    other = train_one(**{**c,"strategy":"he"})
    assert run["data_sha256"] == other["data_sha256"]
    assert run["train_n"] == 288 and run["validation_n"] == 96
    for a,b in zip(supervised_data(), supervised_data()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_failures_are_recorded_and_stop_without_fake_metrics():
    # This fixed realization was observed to fail in the executed quick experiment.
    # A large scale/rate alone does not guarantee overflow for every finite network.
    r = train_one(strategy="large", lr=.15, epochs=30, width=32, depth=6)
    assert r["nonfinite"] is not None
    assert r["history"][-1]["epoch"] == r["nonfinite"]["epoch"]
    assert r["history"][-1]["update_norm"] is None
    json.dumps(r, allow_nan=False)


def test_jacobian_svd_manual_relu_product_and_directional_control():
    c = Config(width=12, depth=5)
    net = SignalNet(c)
    x = torch.randn(12, dtype=torch.float64, generator=torch.Generator().manual_seed(c.seed))
    acts = net.trace(x[None])
    manual = torch.eye(12, dtype=torch.float64)
    for layer, h in zip(net.linears, acts[1:]):
        manual = torch.diag((h[0]>0).double()) @ layer.weight @ manual
    auto = torch.autograd.functional.jacobian(lambda v:net(v[None])[0], x)
    torch.testing.assert_close(auto, manual)
    r = jacobian_experiment()
    assert len(r["records"]) == 12
    for spectrum in r["controlled_spectra"].values():
        assert np.mean(np.square(spectrum)) == pytest.approx(1)
    assert np.count_nonzero(r["controlled_spectra"]["rank_one"]) == 1


def test_log_representation_keeps_zero_and_nonfinite_as_gaps():
    output = positive_log([1., 1e-310, 0., None, np.inf])
    assert output[0] == 0 and output[1] == pytest.approx(-310)
    assert np.isnan(output[2:]).all()


@pytest.mark.parametrize("kwargs", [{"depth":0},{"depth":51},{"width":1},{"batch":1},{"seed":-1},
                                    {"activation":"sigmoid"},{"normalization":"unknown"}])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_cache_reuses_measurements_without_recomputation(tmp_path, monkeypatch):
    calls=[]
    monkeypatch.setattr(experiments,"build_bundle",lambda c,full: calls.append(c) or {"result":3})
    first,path,reused = cached_bundle(Config(),tmp_path)
    second,other,reused_again = cached_bundle(Config(),tmp_path)
    assert first == second and path == other
    assert not reused and reused_again and len(calls)==1


@pytest.fixture
def tiny_bundle():
    c = Config(width=8,depth=3,batch=4)
    baseline = {s:diagnostic(c,s) for s in SCALES}
    baseline["he_layer"] = diagnostic(replace(c,normalization="layer"))
    return {"config":asdict(c),"seeds":[58],"baseline":baseline,
            "grid":[{"config":asdict(c),"results":{s:baseline[s] for s in SCALES}}],
            "repeats":{s:[baseline[s]] for s in ("xavier","he")},
            "surface":stability_surface(c),"relu":relu_experiment(n=10000),
            "normalization":normalization_experiment(repetitions=8),
            "training":[train_one(strategy=s,lr=lr,epochs=3,width=8,depth=2) for lr in (.01,.15) for s in SCALES],
            "clipping":[train_one(strategy="large",lr=.15,clip=1,epochs=3,width=8,depth=2)],
            "jacobian":jacobian_experiment(),"versions":{"python":"test","numpy":np.__version__,"torch":torch.__version__}}


@pytest.mark.rendering
def test_all_real_exports_standalone_html_multiframe_gifs(tmp_path, tiny_bundle):
    paths = export_all(tiny_bundle,tmp_path)
    assert len(paths)==13  # 2 HTML + 6 PNG + 5 GIF, including full-mode SVD.
    for path in paths:
        assert path.stat().st_size > 1000
        if path.suffix==".html":
            html=path.read_text(encoding="utf-8")
            assert "plotly.js" in html.lower()
            assert not re.search(r'<script[^>]+src\s*=',html,re.I)
            assert "<html" in html and "</html>" in html
        else:
            with Image.open(path) as im:
                assert im.width >= 800 and im.height >= 400
                if path.suffix==".gif":
                    assert im.n_frames > 1
                    im.seek(im.n_frames-1)
                    im.load()
