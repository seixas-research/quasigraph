import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule

from quasigraph import QuasiGraph, GEOMETRIC_FEATURES, geometry


def _chain(spacing):
    """Three atoms of covalent radius 0.5 A on a line (ideal bond length 1.0 A)."""
    positions = np.array([[0.0, 0, 0], [spacing, 0, 0], [2 * spacing, 0, 0]])
    radii = np.full(3, 0.5)
    threshold = geometry.bond_threshold(radii, tolerance=0.4)
    distances = geometry.distance_matrix(positions)
    bonds = geometry.bonds_from_distance_matrix(distances, threshold)
    return distances, bonds, threshold, radii


def test_smooth_cn_is_a_sigmoid_centred_on_the_bond_threshold():
    distances, bonds, threshold, radii = _chain(1.0)  # bonds 0.4 A inside the 1.4 A threshold
    smooth = geometry.smooth_coordination_numbers(distances, threshold, temperature=0.1)
    hard = geometry.coordination_numbers(bonds)
    assert hard.tolist() == [1, 2, 1]
    np.testing.assert_allclose(smooth, hard, atol=0.04)   # ~0.98 per bond, plus a tiny 2nd-neighbour tail
    assert (smooth < hard).all() or np.allclose(smooth, hard)
    np.testing.assert_allclose(geometry.bond_strain(distances, bonds, radii), [0, 0, 0])

    distances, _, threshold, _ = _chain(1.4)  # bonds exactly at the threshold: weight 0.5 each
    smooth = geometry.smooth_coordination_numbers(distances, threshold, temperature=0.1)
    np.testing.assert_allclose(smooth, [0.5, 1.0, 0.5], atol=1e-6)

    # monotonic decrease with stretching, higher temperature gives a softer curve
    values = [geometry.smooth_coordination_numbers(*_chain(a)[:1], _chain(a)[2], temperature=0.1)[1] for a in (1.0, 1.2, 1.4, 1.6)]
    assert values == sorted(values, reverse=True)
    soft = geometry.smooth_coordination_numbers(*_chain(1.6)[:1], _chain(1.6)[2], temperature=0.5)[1]
    assert soft > values[-1]


def test_smooth_cn_zero_temperature_reproduces_standard_cn():
    for spacing in (1.0, 1.3, 1.5):
        distances, bonds, threshold, _ = _chain(spacing)
        smooth = geometry.smooth_coordination_numbers(distances, threshold, temperature=0.0)
        np.testing.assert_array_equal(smooth, geometry.coordination_numbers(bonds))
    with pytest.raises(ValueError):
        geometry.smooth_coordination_numbers(distances, threshold, temperature=-1.0)


def test_smooth_cn_temperature_is_a_constructor_argument():
    atoms = bulk("Au", cubic=True).repeat([2, 2, 2])
    cold = QuasiGraph(atoms, pbc=[True, True, True], temperature=0.0, geometric_features=["CN", "CN_smooth"]).get_dataframe()
    np.testing.assert_array_equal(cold["CN_smooth"], cold["CN"])
    warm = QuasiGraph(atoms, pbc=[True, True, True], temperature=0.1, geometric_features=["CN", "CN_smooth"]).get_dataframe()
    np.testing.assert_allclose(warm["CN_smooth"], 12.0, atol=0.05)
    assert not np.allclose(warm["CN_smooth"], 12.0, atol=1e-6)


def test_bond_length_statistics():
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1.2, 0], [10.0, 0, 0]])
    radii = np.full(4, 0.5)
    distances = geometry.distance_matrix(positions)
    bonds = geometry.bonds_from_distance_matrix(distances, geometry.bond_threshold(radii, 0.4))
    stats = geometry.bond_length_statistics(distances, bonds)
    np.testing.assert_allclose(stats['bond_mean'], [1.1, 1.0, 1.2, 0.0])
    np.testing.assert_allclose(stats['bond_min'], [1.0, 1.0, 1.2, 0.0])
    np.testing.assert_allclose(stats['bond_max'], [1.2, 1.0, 1.2, 0.0])
    np.testing.assert_allclose(stats['bond_std'], [0.1, 0.0, 0.0, 0.0])


def test_geometric_features_are_opt_in_and_validated():
    atoms = molecule("H2O")
    assert list(QuasiGraph(atoms).get_dataframe().columns) == [
        "VEC", "atomic_radius", "en_pauling", "electron_affinity", "CN", "GCN"]
    df = QuasiGraph(atoms, geometric_features=GEOMETRIC_FEATURES).get_dataframe()
    assert list(df.columns[4:]) == GEOMETRIC_FEATURES
    np.testing.assert_allclose(df["CN_smooth"], df["CN"], atol=0.1)
    assert df["bond_min"][0] == pytest.approx(df["bond_max"][0])  # both O-H bonds have equal length
    assert df["bond_std"][0] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="Geometric feature 'XYZ' is not recognized"):
        QuasiGraph(atoms, geometric_features=["CN", "XYZ"]).get_dataframe()


def test_geometric_features_for_periodic_crystal():
    descriptor = QuasiGraph(bulk("Au", cubic=True).repeat([2, 2, 2]), pbc=[True, True, True],
                            geometric_features=GEOMETRIC_FEATURES)
    df = descriptor.get_dataframe()
    nearest = 4.08 / np.sqrt(2)
    np.testing.assert_allclose(df["bond_mean"], nearest, rtol=1e-6)
    np.testing.assert_allclose(df["bond_std"], 0.0, atol=1e-12)
    np.testing.assert_allclose(descriptor.pair_distances.diagonal(), 0.0)
    assert descriptor.pair_distances.max() <= 8.16 * np.sqrt(3) / 2 + 1e-9  # half the supercell body diagonal


def test_neighbor_shells_on_a_path_graph():
    adjacency = np.zeros((5, 5), dtype=bool)
    for i in range(4):
        adjacency[i, i + 1] = adjacency[i + 1, i] = True
    shells = geometry.neighbor_shells(adjacency, 0, n_shells=3)
    assert [s.tolist() for s in shells] == [[1], [2], [3]]
    shells = geometry.neighbor_shells(adjacency, 2, n_shells=3)
    assert [s.tolist() for s in shells] == [[1, 3], [0, 4], []]


def test_site_environment_of_methanol_oxygen():
    descriptor = QuasiGraph(molecule("CH3OH"))
    env = descriptor.get_site_environment("O")
    assert env["site_CN"] == 2
    assert env["shell1_n"] == 2 and env["shell1_n_C"] == 1 and env["shell1_n_H"] == 1
    assert env["shell2_n"] == 3 and env["shell2_n_H"] == 3 and env["shell2_n_C"] == 0
    assert env["shell1_dist_min"] < env["shell1_dist_max"] < env["shell2_dist_min"]
    assert env["shell1_en_pauling_mean"] == pytest.approx((2.55 + 2.20) / 2)
    # same site addressed by (negative) index
    index = descriptor.chemical_symbols.index("O")
    assert env.equals(descriptor.get_site_environment(index - len(descriptor.atoms)))


def test_site_environment_fixed_element_list_and_empty_shells():
    env = QuasiGraph(molecule("H2O")).get_site_environment("O", shells=3, elements=["H", "O", "Pt"])
    assert env["shell1_n_Pt"] == 0 and env["shell1_n_H"] == 2
    assert env["shell3_n"] == 0 and env["shell3_CN_mean"] == 0.0 and env["shell3_dist_mean"] == 0.0
    vector = QuasiGraph(molecule("H2O")).get_site_vector("O", shells=3, elements=["H", "O", "Pt"])
    assert vector.shape == (len(env),) and vector.dtype == float


def test_site_symbol_must_be_unique():
    with pytest.raises(ValueError, match="occurs 2 times"):
        QuasiGraph(molecule("H2O")).get_site_environment("H")


def test_site_vector_length_is_independent_of_structure_size():
    kwargs = dict(geometric_features=GEOMETRIC_FEATURES)
    elements = ["Ag", "Au", "Cu", "Pd", "Pt"]
    small = Atoms("Pt2H", positions=[[0, 0, 0], [2.7, 0, 0], [1.35, 1.6, 0]])
    large = Atoms("Pt4H", positions=[[0, 0, 0], [2.7, 0, 0], [0, 2.7, 0], [2.7, 2.7, 0], [1.35, 1.35, 1.2]])
    v_small = QuasiGraph(small, **kwargs).get_site_vector("H", elements=elements)
    v_large = QuasiGraph(large, **kwargs).get_site_vector("H", elements=elements)
    assert v_small.shape == v_large.shape
