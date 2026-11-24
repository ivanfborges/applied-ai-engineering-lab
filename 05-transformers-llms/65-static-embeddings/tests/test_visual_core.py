"""Independent numerical checks for visual traces, sampling and evaluation."""

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from embedding_lab import (
    association_matrices, expected_sgns_loss_grad, fixed_projection, forward_trace,
    make_corpus, mean_vectors, negative_trace, noise_distribution, pmi_experiment,
    ranking_metrics, retrieval_experiment, sgns_loss_grad, sigmoid, snapshot_vectors,
    train_model,
)
from from_scratch import EmbeddingFit, cosine, predictive_loss_grad, training_data


def finite_difference(objective,parameters):
    analytical=objective(*parameters)[1]
    for param,gradient in zip(parameters,analytical):
        numerical=np.zeros_like(param)
        for index in np.ndindex(param.shape):
            value=param[index]
            param[index]=value+1e-6; plus=objective(*parameters)[0]
            param[index]=value-1e-6; minus=objective(*parameters)[0]
            param[index]=value
            numerical[index]=(plus-minus)/2e-6
        np.testing.assert_allclose(gradient,numerical,atol=2e-9,rtol=2e-5)


def test_corpus_reproducibility_rare_entries_and_frequency_control():
    a=make_corpus(size=20,seed=2)
    assert a==make_corpus(size=20,seed=2)
    assert a!=make_corpus(size=20,seed=3)
    freq=Counter(w for sentence in a for w in sentence)
    assert freq["surgeon"]==freq["architect"]==1
    assert Counter(w for sentence in make_corpus(size=20,seed=2,repetitions=2) for w in sentence)["surgeon"]==2
    finance=make_corpus(size=120,finance_ratio=1)
    river=make_corpus(size=120,finance_ratio=0)
    assert sum("loan" in row for row in finance)>sum("loan" in row for row in river)
    assert sum("river" in row for row in river)>sum("river" in row for row in finance)


@pytest.mark.parametrize("invalid",[0,-1,True,1.5])
def test_invalid_corpus_size(invalid):
    with pytest.raises(ValueError): make_corpus(size=invalid)


@pytest.mark.parametrize("exponent",[1.,.75])
def test_noise_distribution_normalized_and_smoothed(exponent):
    q=noise_distribution([100,10,1,0],exponent)
    assert q.sum()==pytest.approx(1)
    assert q[-1]==0
    assert q[0]/q[2]==pytest.approx(100**exponent)


@pytest.mark.parametrize("frequency",[[0,0],[-1,2],[1,np.inf],[[1,2]],[]])
def test_invalid_noise(frequency):
    with pytest.raises(ValueError): noise_distribution(frequency)


def test_sgns_gradients_repeated_ids_and_positive_noise_collision():
    rng=np.random.default_rng(2)
    v,u=rng.normal(0,.3,(3,2)),rng.normal(0,.3,(3,2))
    centers=np.array([0,0,1]);contexts=np.array([1,1,2]);noise=np.array([[1,2],[2,2],[1,1]])
    finite_difference(lambda v,u:sgns_loss_grad(v,u,centers,contexts,noise),[v,u])


def test_sgns_stable_extreme_scores_and_signs():
    v=np.array([[1000.],[1.]])
    u=np.array([[1.],[-1.]])
    loss,gradients=sgns_loss_grad(v,u,np.array([0]),np.array([1]),np.array([[0]]))
    assert loss==pytest.approx(2000)
    assert all(np.isfinite(g).all() for g in gradients)
    assert sigmoid(1000)==1
    assert sigmoid(-1000)==0


def test_expected_sgns_gradients_and_stationary_theoretical_target():
    counts=np.array([[3.,1],[2,4]])
    q=np.array([.4,.6])
    rng=np.random.default_rng(5)
    v,u=rng.normal(0,.2,(2,2)),rng.normal(0,.2,(2,2))
    finite_difference(lambda v,u:expected_sgns_loss_grad(v,u,counts,q,5),[v,u])
    target=np.log((counts/counts.sum(axis=1,keepdims=True))/(5*q[None,:]))
    loss,gradients=expected_sgns_loss_grad(np.eye(2),target.T,counts,q,5)
    assert np.isfinite(loss)
    for g in gradients: np.testing.assert_allclose(g,0,atol=1e-15)


def test_pmi_zero_policy_and_generalized_noise_target():
    x=np.array([[2.,0],[1,3]])
    matrices=association_matrices(x)
    assert np.isnan(matrices["pmi"][0,1])
    assert np.isnan(matrices["log_counts"][0,1])
    assert matrices["ppmi"][0,1]==0
    assert matrices["pmi"][0,0]==pytest.approx(np.log(2*6/(2*3)))
    np.testing.assert_allclose(matrices["conditional"].sum(axis=1),1)
    unigram,smoothed=pmi_experiment(steps=3)
    assert not np.allclose(unigram["target"],smoothed["target"])


@pytest.mark.parametrize("mode",["cbow","skipgram","glove","sgns"])
def test_snapshots_are_real_independent_and_reproducible(mode):
    corpus=[["doctor","treats","patient"],["physician","treats","patient"]]
    a=train_model(corpus,mode,dim=2,epochs=4)
    b=train_model(corpus,mode,dim=2,epochs=4)
    assert [s.epoch for s in a.snapshots]==list(range(5))
    np.testing.assert_array_equal(a.fit.vectors,b.fit.vectors)
    np.testing.assert_allclose(snapshot_vectors(a)[-1],a.fit.vectors)
    assert not np.shares_memory(a.snapshots[0].parameters[0],a.snapshots[-1].parameters[0])
    assert np.isfinite(a.fit.losses).all()


def test_fixed_projection_reuses_one_affine_map():
    first=np.array([[0.,1,2],[2,1,0],[3,2,1]])
    second=first*1.5+1
    coords,evr,pca=fixed_projection([first,second],3)
    np.testing.assert_allclose(coords[0],pca.transform(first))
    np.testing.assert_allclose(coords[1],pca.transform(second))
    assert evr.sum()==pytest.approx(1)


@pytest.mark.parametrize("mode",["cbow","skipgram"])
def test_forward_trace_probs_loss_and_update(mode):
    row=forward_trace(mode)
    np.testing.assert_allclose(row["probabilities"].sum(axis=1),1)
    expected=-np.log(row["probabilities"][np.arange(len(row["targets"])),row["targets"]]).mean()
    assert row["loss"]==pytest.approx(expected)
    np.testing.assert_allclose(row["after_inputs"],row["inputs"]-.4*row["dv"])
    assert row["after_loss"]<row["loss"]


def test_negative_trace_is_simultaneous_sgd_and_reports_actual_dots():
    for row in negative_trace(steps=3):
        for old,new,grad in zip(row["before"],row["after"],row["gradients"]):
            np.testing.assert_allclose(new,old-row["lr"]*grad)
        assert row["dot_after"]==pytest.approx(row["after"][0][0]@row["after"][1][1])


def test_rank_metrics_hand_calculated_and_empty_policy():
    result=ranking_metrics([2,0,1],{0:2,1:1},k=2)
    assert result["recall_at2"]==.5
    assert result["rr"]==.5
    assert result["ndcg_at2"]==pytest.approx((3/np.log2(3))/(3+1/np.log2(3)))
    assert ranking_metrics([],{},k=3)==dict(recall_at3=0.,rr=0.,ndcg_at3=0.)
    with pytest.raises(ValueError): ranking_metrics([1,1],{1:1})


def test_mean_pooling_order_negation_and_oov_policy():
    fit=EmbeddingFit(("a","b","not"),np.array([[1.,0],[0,1],[-1,1]]),np.array([]))
    vectors,coverage=mean_vectors(["a b","b a","unseen","a not b"],fit)
    np.testing.assert_array_equal(vectors[0],vectors[1])
    np.testing.assert_array_equal(vectors[2],[0,0])
    assert coverage[2]==0
    assert not np.array_equal(vectors[0],vectors[3])


def test_retrieval_protocol_and_oov_abstention():
    corpus=make_corpus(size=5)
    models={mode:train_model(corpus,mode,epochs=2) for mode in ("skipgram","glove")}
    result=retrieval_experiment(models)
    for rows in result["details"].values():
        assert rows[-1]["ranking"]==[]
        assert rows[-1]["rr"]==0
        assert len(rows)==6
