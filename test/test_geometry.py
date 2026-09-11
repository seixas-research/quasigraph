import numpy as np
import pytest
from ase.build import bulk, fcc111, molecule

from quasigraph import QuasiGraph, geometry
from quasigraph.elements import get_element, get_feature, validate_features


def _reference_distance_tensor(positions, cell, offsets_list):
    """Straightforward triple loop, as in the original implementation."""
    n = len(positions)
    tensor = np.zeros((len(offsets_list), n, n))
    for k, offset in enumerate(offsets_list):
        shift = offset @ cell
        for i in range(n):
            for j in range(n):
                tensor[k, i, j] = np.linalg.norm(positions[j] + shift - positions[i])
    return tensor


def test_periodic_offsets_enumerates_only_periodic_axes():
    assert geometry.periodic_offsets([0, 0, 0]) == [[0, 0, 0]]
    offsets = geometry.periodic_offsets([1, 1, 0])
    assert len(offsets) == 9
    assert all(c == 0 for _, _, c in offsets)
    assert len(geometry.periodic_offsets([1, 1, 1])) == 27


def test_distance_tensor_matches_explicit_loop():
    atoms = fcc111("Cu", size=(2, 2, 2), vacuum=4.0)
    offsets = geometry.periodic_offsets([1, 1, 0])
    tensor = geometry.distance_tensor(atoms.get_positions(), atoms.cell, offsets)
    reference = _reference_distance_tensor(atoms.get_positions(), np.asarray(atoms.cell), offsets)
    assert tensor.shape == (9, len(atoms), len(atoms))
    np.testing.assert_array_equal(tensor, reference)


def test_distance_matrix_matches_ase():
    atoms = molecule("CH3OH")
    np.testing.assert_array_equal(geometry.distance_matrix(atoms.get_positions()), atoms.get_all_distances())


def test_bonds_and_coordination_numbers_for_a_chain():
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [10.0, 0, 0]])
    radii = np.full(4, 0.5)
    threshold = geometry.bond_threshold(radii, tolerance=0.1)
    bonds = geometry.bonds_from_distance_matrix(geometry.distance_matrix(positions), threshold)
    np.testing.assert_array_equal(geometry.coordination_numbers(bonds), [1, 2, 1, 0])
    assert geometry.neighbor_lists(geometry.adjacency(bonds)) == [[1], [0, 2], [1], []]
    gcn = geometry.generalized_coordination_numbers(bonds, geometry.coordination_numbers(bonds))
    np.testing.assert_allclose(gcn, [1.0, 1.0, 1.0, 0.0])
    gcn_raw = geometry.generalized_coordination_numbers(bonds, geometry.coordination_numbers(bonds), normalization=False)
    np.testing.assert_allclose(gcn_raw, [2.0, 2.0, 2.0, 0.0])


def test_periodic_images_count_towards_cn_but_not_towards_neighbor_list():
    primitive = bulk("Au")
    descriptor = QuasiGraph(primitive, pbc=[True, True, True])
    assert descriptor.cn.tolist() == [12]
    assert descriptor.bonded_atoms == [[0]]
    assert descriptor.distances_tensor.shape == (27, 1, 1)
    assert len(descriptor.distances_list) == 27
    assert descriptor.distances_list[0][:3] == [0, 0, [-1, -1, -1]]


def test_slab_with_mixed_periodicity():
    slab = fcc111("Cu", size=(2, 2, 3), vacuum=6.0)
    descriptor = QuasiGraph(slab, pbc=[True, True, False])
    assert descriptor.offsets == [1, 1, 0]
    assert descriptor.distances_tensor.shape == (9, 12, 12)
    # surface atoms have 9 neighbours, the middle layer has 12
    assert sorted(set(descriptor.cn.tolist())) == [9, 12]


def test_distances_list_is_only_defined_for_periodic_structures():
    descriptor = QuasiGraph(molecule("H2O"))
    with pytest.raises(AttributeError):
        descriptor.distances_list


def test_element_lookup_is_cached():
    get_element.cache_clear()
    assert get_element("Pt") is get_element("Pt")
    assert get_element.cache_info().hits >= 1
    assert get_feature("Pt", "covalent_radius") == get_element("Pt").covalent_radius / 100
    assert get_feature("Pt", "VEC") == 10


def test_validate_features_rejects_unknown_names():
    validate_features(["VEC", "en_pauling"])
    with pytest.raises(ValueError, match="Feature 'ABC' is not recognized"):
        validate_features(["ABC"])
