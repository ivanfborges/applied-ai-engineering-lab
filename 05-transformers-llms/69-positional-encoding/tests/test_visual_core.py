"""Visual computations checked against analytical identities and PyTorch."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from from_scratch import apply_rope, sinusoidal_encoding
from visual_core import Config, METHODS, addition_data, alibi_data, attention, fixture, json_safe, multifrequency_data, numerical_report, permutation_data, precision_data, projected_geometry, rope_planes, softmax
from visual_html import payload


@pytest.mark.parametrize("dim", [4,8,16,32,128])
def test_visual_sinusoidal_identities_and_equivariance(dim):
    config=Config(dim=dim)
    p=permutation_data(config)
    assert p["none"]["error"] < 1e-12
    assert p["joint_error"] < 1e-12
    assert p["sinusoidal"]["error"] > 1e-3
    pe=sinusoidal_encoding(np.arange(32),dim,config.base)
    np.testing.assert_allclose(np.linalg.norm(pe,axis=-1)**2,dim/2,atol=1e-12)
    expected=sum(np.cos(5*config.base**(-2*i/dim)) for i in range(dim//2))
    assert pe[3]@pe[8] == pytest.approx(expected,abs=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_visual_attention_matches_independent_pytorch_forward(method):
    config=Config(dim=8)
    x,w,table=fixture(config,12)
    actual=attention(x,w,method=method,table=table,causal=True)
    z=torch.tensor(x,dtype=torch.float64)
    if method=="sinusoidal":
        positions=torch.arange(12,dtype=torch.float64)[:,None]
        omega=10000.**(-torch.arange(0,8,2,dtype=torch.float64)/8)
        pe=torch.empty_like(z)
        pe[:,0::2]=torch.sin(positions*omega);pe[:,1::2]=torch.cos(positions*omega)
        z=z+pe
    elif method=="absolute":
        z=z+torch.tensor(table[:12])
    q,k,v=[z@torch.tensor(weight) for weight in w]
    if method=="rope":
        angles=torch.arange(12,dtype=torch.float64)[:,None]*10000.**(-torch.arange(0,8,2,dtype=torch.float64)/8)
        def rotate(tensor):
            # Independent complex-number rotation of adjacent real/imaginary pairs.
            complex_values=torch.view_as_complex(tensor.reshape(12,4,2).contiguous())
            rotated=complex_values*torch.polar(torch.ones_like(angles),angles)
            return torch.view_as_real(rotated).reshape(12,8)
        q,k=rotate(q),rotate(k)
    logits=q@k.T/np.sqrt(8)
    positions=torch.arange(12,dtype=torch.float64)
    distance=positions[:,None]-positions[None,:]
    if method=="relative":
        logits=logits-.08*distance.abs()
    elif method=="alibi":
        logits=logits-.1*distance
    logits=logits.masked_fill(distance<0,-torch.inf)
    probs=torch.softmax(logits,dim=-1)
    np.testing.assert_allclose(actual["probabilities"],probs.numpy(),atol=1e-13,rtol=1e-13)
    np.testing.assert_allclose(actual["output"],(probs@v).numpy(),atol=1e-12,rtol=1e-13)
    np.testing.assert_allclose(actual["probabilities"].sum(-1),1,atol=1e-14)
    np.testing.assert_array_equal(np.triu(actual["probabilities"],1),0)


def test_fixed_pca_transform_and_exact_coordinate_hover_metrics():
    config=Config(seq_len=48)
    g=projected_geometry(config)
    pe=sinusoidal_encoding(np.arange(128),config.dim,config.base)
    np.testing.assert_allclose(g["coordinates"],(pe-g["mean"])@g["components"].T,atol=1e-13)
    np.testing.assert_array_equal(g["exact"],pe[:,:3])
    assert np.sum(g["explained_variance_ratio"]) <= 1+1e-12
    np.testing.assert_allclose(g["previous_distance"][1:],np.linalg.norm(np.diff(pe,axis=0),axis=-1))
    assert g["fit_positions"]==128


def test_shared_uncentered_vector_projection_preserves_addition():
    a=addition_data(Config())
    np.testing.assert_allclose(a["projected_content"]+a["projected_position"],a["projected_combined"],atol=1e-13)
    np.testing.assert_array_equal(a["content"][0],a["content"][3])
    assert not np.allclose(a["combined"][0],a["combined"][3])


def test_rope_inverse_norm_and_relative_shift_geometry():
    rng=np.random.default_rng(33)
    x=rng.normal(size=(4,8));positions=np.array([0,1,5,100])
    actual=apply_rope(x,positions)
    angles=positions[:,None]/10000.**(np.arange(0,8,2)/8)
    complex_values=actual[:,0::2]+1j*actual[:,1::2]
    restored=complex_values*np.exp(-1j*angles)
    np.testing.assert_allclose(restored.real,x[:,0::2],atol=1e-13)
    np.testing.assert_allclose(restored.imag,x[:,1::2],atol=1e-13)
    np.testing.assert_allclose(np.linalg.norm(actual,axis=-1),np.linalg.norm(x,axis=-1),atol=1e-13)
    a,b=rope_planes(2,5),rope_planes(102,105)
    assert a["dot_after"]==pytest.approx(a["relative_dot"],abs=1e-14)
    assert a["dot_after"]==pytest.approx(b["dot_after"],abs=1e-13)


def test_all_rotary_pair_contributions_sum_to_dot():
    record=multifrequency_data(Config(dim=32))
    assert np.sum(record["contributions"])==pytest.approx(record["dot"],abs=1e-13)


def test_attention_rejects_position_broadcast_and_alibi_without_causality():
    x,w,table=fixture(Config())
    with pytest.raises(ValueError,match="One position"):
        attention(x,w,positions=[0],method="sinusoidal")
    with pytest.raises(ValueError,match="causal"):
        attention(x,w,method="alibi")


def test_alibi_content_competition_and_masked_future():
    flat=alibi_data(32,24,.1)
    strong=alibi_data(32,24,.1,12,2)
    assert flat["probabilities"][24]>flat["probabilities"][2]
    assert strong["probabilities"][2]>strong["probabilities"][24]
    np.testing.assert_array_equal(strong["probabilities"][25:],0)
    assert strong["bias"][24]==0
    assert strong["bias"][2]==pytest.approx(-2.2)
    assert strong["probabilities"].sum()==pytest.approx(1)


def test_precision_experiment_records_actual_integer_aliasing_not_model_quality():
    data=precision_data(Config())
    assert data["rows"][0]["float64_error"]<1e-12
    assert data["rows"][-1]["float32_represented_offset"]==0
    assert data["rows"][-1]["float64_represented_offset"]==5
    assert np.isfinite(data["rows"][-1]["float32_error"])


@pytest.mark.parametrize("kwargs", [{"dim":3},{"dim":0},{"dim":True},{"seq_len":4096},{"seq_len":0},{"frames":2},{"fps":20},{"seed":-1},{"base":np.inf},{"dpi":20}])
def test_invalid_export_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs).validate()


@pytest.mark.parametrize("logits", [[[np.nan,0]],[[np.inf,0]],[[-np.inf,-np.inf]],[]])
def test_softmax_rejects_invalid_rows(logits):
    with pytest.raises(ValueError):
        softmax(logits)


def test_page_payload_json_has_no_nan_or_infinite_literals():
    data=payload("permutation",Config())
    rendered=json.dumps(data,allow_nan=False)
    assert json.loads(rendered)["permutation"]["none"]["error"]<1e-12
    assert "NaN" not in rendered and "Infinity" not in rendered
    report=json_safe(numerical_report(Config()))
    assert report["linear_projection_addition_error"]<1e-12
