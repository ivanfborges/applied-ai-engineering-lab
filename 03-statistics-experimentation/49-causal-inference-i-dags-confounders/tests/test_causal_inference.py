"""Graphical counterexamples, hand calculations and numerical agreement."""

from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from from_scratch import DAG, fwl_coefficient, standardized_difference
from example import estimate_specifications, main, simulate_scm, study_graph


@pytest.mark.parametrize("edges", [
    [("A", "B"), ("B", "Y")],
    [("B", "A"), ("B", "Y")],
])
def test_chain_and_fork_block_when_middle_is_conditioned(edges):
    graph = DAG("ABY", edges)
    assert not graph.is_d_separated("A", "Y")
    assert graph.is_d_separated("A", "Y", {"B"})


def test_collider_and_its_descendant_activate_path():
    graph = DAG("ASUDY", [("A", "S"), ("U", "S"), ("S", "D"), ("U", "Y")])
    assert graph.is_d_separated("A", "Y")
    assert not graph.is_d_separated("A", "Y", {"S"})
    assert not graph.is_d_separated("A", "Y", {"D"})
    assert graph.is_d_separated("A", "Y", {"D", "U"})
    assert graph.descendants("S") == {"D"}


def test_every_path_must_be_blocked_and_direct_edge_cannot_be_blocked():
    graph = DAG("ACDY", [("C", "A"), ("C", "Y"), ("D", "A"), ("D", "Y")])
    assert not graph.satisfies_backdoor("A", "Y", {"C"})
    assert graph.satisfies_backdoor("A", "Y", {"C", "D"})
    assert not DAG("AY", [("A", "Y")]).is_d_separated("A", "Y")
    assert DAG("AY", []).is_d_separated("A", "Y")


def test_original_backdoor_rejects_descendants_even_if_no_backdoor_path():
    graph = DAG("AMY", [("A", "M"), ("M", "Y")])
    assert graph.satisfies_backdoor("A", "Y")
    assert not graph.satisfies_backdoor("A", "Y", {"M"})


def test_pretreatment_collider_is_not_a_safe_control():
    graph = DAG("APZQY", [("P", "A"), ("P", "Z"), ("Q", "Z"), ("Q", "Y")])
    assert graph.satisfies_backdoor("A", "Y")
    assert not graph.satisfies_backdoor("A", "Y", {"Z"})
    assert graph.satisfies_backdoor("A", "Y", {"Z", "P"})


def test_study_graph_and_hidden_confounding_change_identification():
    graph = study_graph()
    assert graph.satisfies_backdoor("A", "Y", {"C"})
    assert not graph.satisfies_backdoor("A", "Y")
    assert not graph.satisfies_backdoor("A", "Y", {"C", "M"})
    assert not graph.satisfies_backdoor("A", "Y", {"C", "S"})
    # Ordinary d-separation in the original DAG would also block causal paths.
    assert not graph.is_d_separated("A", "Y", {"C"})
    hidden = DAG(graph.nodes, [*graph.edges, ("U", "A")])
    assert not hidden.satisfies_backdoor("A", "Y", {"C"})
    assert hidden.satisfies_backdoor("A", "Y", {"C", "U"})


@pytest.mark.parametrize("nodes,edges", [
    ([], []), (["A", "A"], []), ([""], []),
    ("AY", [("A", "A")]), ("AY", [("A", "B")]),
    ("AY", [("A", "Y"), ("Y", "A")]), ("AY", [("A",)]),
])
def test_invalid_graphs_raise(nodes, edges):
    with pytest.raises(ValueError):
        DAG(nodes, edges)


@pytest.mark.parametrize("x,y,z", [("A", "A", ()), ("A", "Y", ("A",)),
                                    ("X", "Y", ()), ("A", "Y", ("X",))])
def test_invalid_queries_raise(x, y, z):
    with pytest.raises(ValueError):
        study_graph().is_d_separated(x, y, z)
    with pytest.raises(ValueError):
        study_graph().satisfies_backdoor(x, y, z)


@pytest.mark.parametrize("k", [0, 1, 3])
def test_fwl_matches_joint_ols_and_is_invariant_to_offsets(k):
    rng = np.random.default_rng(490)
    controls = rng.normal(size=(200, k))
    a = rng.normal(size=200) + controls.sum(axis=1)
    y = 8 + 2 * a + 3 * controls.sum(axis=1) + rng.normal(size=200)
    expected = LinearRegression().fit(np.column_stack([a, controls]), y).coef_[0]
    assert fwl_coefficient(a, y, controls) == pytest.approx(expected, abs=1e-12)
    assert fwl_coefficient(a + 20, y - 10, controls) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("a,y,c", [
    ([0, 1], [0, 1], [[0], [1]]),
    ([1, 1, 1, 1], [0, 1, 2, 3], np.empty((4, 0))),
    ([0, 1, 0, 1], [0, 1, 2, 3], np.ones((4, 1))),
    ([0, 1, np.nan, 1], [0, 1, 2, 3], np.empty((4, 0))),
    ([0, 1, 0, 1], [0, np.inf, 2, 3], np.empty((4, 0))),
    ([0, 1, 0, 1], [0, 1, 2], np.empty((4, 0))),
    ([0, 1, 0, 1], [0, 1, 2, 3], [0, 1, 0, 1]),
    ([[0], [1]], [0, 1], np.empty((2, 0))),
])
def test_fwl_invalid_or_nonidentified_design_raises(a, y, c):
    with pytest.raises(ValueError):
        fwl_coefficient(a, y, c)


def test_standardization_uses_target_frequencies_with_heterogeneous_effects():
    # C=0 has four units and effect 2; C=1 has two units and effect 5.
    a, c, y = [0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 1], [0, 0, 0, 2, 10, 15]
    assert standardized_difference(a, y, c) == pytest.approx(3.0)
    naive = np.mean([2, 15]) - np.mean([0, 0, 0, 10])
    assert naive == 6.0


@pytest.mark.parametrize("a,y,c", [
    ([0, 1], [0, 1], [0, 1]), ([0, 2], [0, 1], [0, 0]),
    ([0, 1], [0, 1], [0, np.nan]), ([0, 1], [0, 1, 2], [0, 0]),
])
def test_standardization_invalid_inputs_and_missing_overlap_raise(a, y, c):
    with pytest.raises(ValueError):
        standardized_difference(a, y, c)


def test_scm_consistency_paired_interventions_reproducibility_and_estimates():
    data = simulate_scm(30_000, 49)
    repeated = simulate_scm(30_000, 49)
    for name in data:
        np.testing.assert_array_equal(data[name], repeated[name])
    np.testing.assert_allclose(data["Y1"] - data["Y0"], 2.0, atol=1e-12)
    np.testing.assert_allclose(data["Y"], np.where(data["A"] == 1, data["Y1"], data["Y0"]), atol=1e-12)
    estimates = estimate_specifications(data)
    for name, expected in [("Naive", 2.9), ("C", 2), ("C + M", 1.3),
                           ("C + S", 1.04), ("Standardized C", 2)]:
        assert estimates[name] == pytest.approx(expected, abs=0.08)


@pytest.mark.parametrize("n,seed", [(10, 49), (100.5, 49), (True, 49),
                                   (100, -1), (100, 1.5), (100, False)])
def test_invalid_simulation_configuration(n, seed):
    with pytest.raises(ValueError):
        simulate_scm(n, seed)


def test_cli_small_run(capsys):
    assert main(["--n-samples", "1000", "--seed", "49", "--repetitions", "2"]) == 0
    assert "Author review required" in capsys.readouterr().out


@pytest.mark.parametrize("args", [["--repetitions", "1"], ["--n-samples", "3"], ["--seed", "-1"]])
def test_cli_rejects_invalid_configuration(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
