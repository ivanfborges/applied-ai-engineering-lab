"""Mechanism oracles, real gradients, checkpoint/cache integrity and real exports."""

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import experiments
from example import SequenceClassifier, evaluate, make_data
from experiments import configuration, load_model, load_or_run, matched_hidden, parameter_count
from visual_core import (causal_attention, computed_gru, computed_lstm, gru_control,
                         input_gradients, lstm_control, pca_states, rnn_trace, scalar_gradients)
from visualizations import generate, lstm_gate_schedule


@pytest.fixture(autouse=True)
def single_thread():
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_rnn_first_step_and_repeated_zero_input():
    states=rnn_trace([1,0,0],strength=0)
    np.testing.assert_allclose(states[1],np.tanh([.8,-.5]))
    np.testing.assert_array_equal(states[0],np.zeros(2))
    np.testing.assert_array_equal(states[2:],np.zeros((2,2)))


def test_scalar_derivative_products_cover_three_regimes():
    assert scalar_gradients(.8)[-1]<1e-4
    np.testing.assert_array_equal(scalar_gradients(-1),np.ones(51))
    assert scalar_gradients(1.2)[-1]>1000
    np.testing.assert_array_equal(scalar_gradients(0,2),[1,0,0])


def test_lstm_gates_separate_cell_memory_from_exposure():
    closed=lstm_control(2,1,0,-1,0)
    opened=lstm_control(2,1,0,-1,1)
    assert closed["cell"]==opened["cell"]==2
    assert closed["hidden"]==0
    np.testing.assert_allclose(opened["hidden"],np.tanh(2))
    assert lstm_control(2,0,1,-.7,1)["cell"]==-.7
    for row in computed_lstm():
        assert all(0<=row[k]<=1 for k in ("i","f","o"))
        np.testing.assert_allclose(row["cell"],row["f"]*row["c_prev"]+row["i"]*row["g"])


def test_lstm_animation_schedule_is_explicit_retention_reset_and_exposure():
    rows=lstm_gate_schedule()
    assert rows[0]["cell"]==.8
    assert rows[1]["hidden"]==0 and rows[1]["cell"]>0
    assert rows[3]["cell"]==0
    np.testing.assert_allclose(rows[4]["cell"],.81)


def test_gru_limits_and_candidate_convention():
    assert gru_control(.7,.2,1,.5)["hidden"]==.7
    result=gru_control(.7,.2,0,.5)
    assert result["hidden"]==result["candidate"]
    assert gru_control(.7,.2,.8,0,"pytorch")["candidate"]!=gru_control(.7,.2,.8,0,"before")["candidate"]


def test_computed_gru_uses_pytorch_order_and_reset_after_affine():
    reference=torch.nn.GRUCell(1,1,dtype=torch.float64)
    with torch.no_grad():
        reference.weight_ih.copy_(torch.tensor([[-.4],[.6],[.8]],dtype=torch.float64))
        reference.weight_hh.copy_(torch.tensor([[.3],[.4],[.9]],dtype=torch.float64))
        reference.bias_ih.copy_(torch.tensor([0,.5,.1],dtype=torch.float64))
        reference.bias_hh.copy_(torch.tensor([0,0,.3],dtype=torch.float64))
        hidden=torch.zeros(1,dtype=torch.float64)
        for row in computed_gru():
            hidden=reference(torch.tensor([row["x"]],dtype=torch.float64),hidden)
            np.testing.assert_allclose(row["hidden"],hidden.item(),atol=1e-14)


def test_empirical_gradient_norm_is_mean_of_individual_loss_gradients():
    model=SequenceClassifier("rnn",hidden=3)
    x,y=make_data(4,5,160)
    expected=[]
    for index in range(len(y)):
        value=x[index:index+1].clone().requires_grad_(True)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(value),y[index:index+1])
        gradient,=torch.autograd.grad(loss,value)
        expected.append(gradient[0].norm(dim=-1).detach().numpy())
    np.testing.assert_allclose(input_gradients(model,x,y),np.mean(expected,axis=0),rtol=2e-6,atol=1e-9)
    assert all(p.grad is None for p in model.parameters())


def test_pca_preserves_distances_when_hidden_dimension_is_three():
    values=np.random.default_rng(160).normal(size=(4,6,3))
    projected,ratios=pca_states(values)
    assert projected.shape==(4,6,3)
    np.testing.assert_allclose(ratios.sum(),1)
    np.testing.assert_allclose(np.linalg.norm(values[0,0]-values[1,3]),np.linalg.norm(projected[0,0]-projected[1,3]))
    _,zero=pca_states(np.zeros((2,3,4)))
    np.testing.assert_array_equal(zero,[0,0,0])


def test_causal_attention_masks_future_and_normalizes_rows():
    weights,output,values=causal_attention(query=2)
    np.testing.assert_array_equal(weights[np.triu_indices(6,1)],0)
    np.testing.assert_allclose(weights.sum(1),1)
    np.testing.assert_allclose(output,weights[2]@values)
    scores=torch.tensor(values@values.T/np.sqrt(3))
    scores[torch.triu(torch.ones(6,6,dtype=torch.bool),diagonal=1)]=-torch.inf
    np.testing.assert_allclose(weights,torch.softmax(scores,1).numpy(),atol=1e-14)


@pytest.mark.parametrize("kind",["rnn","lstm","gru"])
def test_parameter_budget_matches_actual_models(kind):
    hidden=matched_hidden(kind,1297)
    model=SequenceClassifier(kind,hidden)
    assert sum(p.numel() for p in model.parameters())==parameter_count(kind,hidden)
    assert abs(parameter_count(kind,hidden)-1297)/1297<.06


@pytest.mark.parametrize("operation",[
    lambda:rnn_trace([np.nan]), lambda:scalar_gradients(3),
    lambda:lstm_control(1,1.1,0,0,1),lambda:gru_control(0,0,.5,-1),
    lambda:pca_states(np.zeros((2,2,2))),lambda:causal_attention(2,2),
])
def test_mechanisms_reject_invalid_inputs(operation):
    with pytest.raises(ValueError):
        operation()


@pytest.mark.integration
def test_cache_restores_actual_checkpoints_and_skips_training(tmp_path,monkeypatch):
    config=configuration()
    config.update(seeds=[160],lengths=[4],epochs=2,hidden=4,train_size=32,validation_size=16,test_size=16,probe_size=8)
    report,folder=load_or_run(tmp_path,config,lambda _:None)
    assert len(report["results"])==3
    test=make_data(16,4,1602)
    for row in report["results"]:
        model=load_model(folder,row)
        assert evaluate(model,*test)==row["test"]
        assert np.asarray(row["projection"]).shape==(4,5,3)
        assert row["training_seconds"]>0
    monkeypatch.setattr(experiments,"run_comparison",lambda *args:pytest.fail("cache hit retrained models"))
    cached,again=load_or_run(tmp_path,config,lambda _:None)
    assert cached==report and again==folder
    with (folder/"weights.npz").open("ab") as handle:
        handle.write(b"altered")
    with pytest.raises(ValueError,match="checksum"):
        load_or_run(tmp_path,config,lambda _:None)


@pytest.mark.rendering
def test_real_lstm_gif_and_offline_html_export(tmp_path):
    output,report=generate(tmp_path/"export",only="lstm",progress=lambda _:None)
    assert report is None
    with Image.open(output/"03_lstm_memory_flow.gif") as image:
        assert image.n_frames==8
        assert image.size==(1050,680)
        assert image.info["loop"]==0
        for index in range(image.n_frames):
            image.seek(index)
            assert image.info["duration"]==600
    page=(output/"03_lstm_gates_interactive.html").read_text(encoding="utf-8")
    assert 'id="f"' in page and 'id="mode"' in page
    assert '<script src=' not in page and 'window.LAB_DATA=' in page
    assert json.loads((output/"manifest.json").read_text())["groups"]==["lstm"]
    with pytest.raises(ValueError,match="new or empty"):
        generate(output,only="lstm")
