import warnings

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, fcc111
from ase.neighborlist import neighbor_list

from quasigraph import QuasiGraph, GEOMETRIC_FEATURES, MissingFeatureWarning, geometry
from quasigraph.elements import get_feature


def _ase_cn(atoms, tolerance=0.4):
    """Reference CN from ASE's neighbour list for a single-element structure."""
    cutoff = 2 * (1 + tolerance) * get_feature(atoms[0].symbol, "covalent_radius")
    return np.bincount(neighbor_list("i", atoms, cutoff), minlength=len(atoms))


def test_pbc_defaults_to_the_atoms_object():
    atoms = bulk("Au", cubic=True).repeat([2, 2, 2])
    assert QuasiGraph(atoms).cn.tolist() == 32 * [12]
    assert min(QuasiGraph(atoms, pbc=False).cn) < 12  # explicit pbc still overrides


def test_gcn_is_independent_of_the_supercell():
    primitive = QuasiGraph(bulk("Au"))
    supercell = QuasiGraph(bulk("Au").repeat(3))
    np.testing.assert_allclose(primitive.gcn, [12.0])
    np.testing.assert_allclose(supercell.gcn, 27 * [12.0])


def test_site_environment_is_independent_of_the_supercell():
    kwargs = dict(geometric_features=GEOMETRIC_FEATURES)
    primitive = QuasiGraph(bulk("Au"), **kwargs).get_site_environment(0, elements=["Au"])
    supercell = QuasiGraph(bulk("Au").repeat(3), **kwargs).get_site_environment(0, elements=["Au"])
    assert primitive["shell1_n"] == 12 and primitive["shell2_n"] == 42  # 1 + 12 + 42 = 55 (cuboctahedron)
    assert primitive["shell1_dist_mean"] == pytest.approx(4.08 / np.sqrt(2))
    np.testing.assert_allclose(primitive.to_numpy(dtype=float), supercell.to_numpy(dtype=float), atol=1e-12)


def test_image_neighbor_shells_matches_neighbor_shells_without_pbc():
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [10.0, 0, 0]])
    threshold = geometry.bond_threshold(np.full(4, 0.5), tolerance=0.1)
    bonds = geometry.bonds_from_distance_matrix(geometry.distance_matrix(positions), threshold)
    plain = geometry.neighbor_shells(geometry.adjacency(bonds), 0, 3)
    with_images = geometry.image_neighbor_shells(bonds, None, 0, 3)
    for atoms_k, (image_atoms_k, images_k) in zip(plain, with_images):
        np.testing.assert_array_equal(atoms_k, image_atoms_k)
        assert not images_k.any()


def test_skewed_cell_finds_all_neighbors():
    atoms = Atoms("Pt", cell=[[2.77, 0, 0], [27.7, 2.4, 0], [0, 0, 2.77]], pbc=True)
    descriptor = QuasiGraph(atoms)
    assert descriptor.offsets[0] > 1
    np.testing.assert_array_equal(descriptor.cn, _ase_cn(atoms))


def test_random_cells_with_unwrapped_positions_match_ase():
    rng = np.random.default_rng(0)
    for _ in range(20):
        cell = rng.normal(size=(3, 3)) * 1.5 + np.eye(3) * 3
        if abs(np.linalg.det(cell)) < 2:
            continue
        atoms = Atoms("Pt2", scaled_positions=rng.random((2, 3)) * 1.6 - 0.3, cell=cell, pbc=True)
        np.testing.assert_array_equal(QuasiGraph(atoms).cn, _ase_cn(atoms))


def test_image_range_keeps_one_translation_on_periodic_axes():
    slab = fcc111("Cu", size=(2, 2, 3), vacuum=6.0)
    assert geometry.periodic_image_range(slab.cell.complete(), slab.positions, [1, 1, 0], 3.7) == [1, 1, 0]


def test_periodic_axis_without_a_cell_vector_is_rejected():
    atoms = Atoms("Pt2", positions=[[0, 0, 0], [2.7, 0, 0]], pbc=True)
    with pytest.raises(ValueError, match="non-degenerate cell"):
        QuasiGraph(atoms)
    assert QuasiGraph(atoms, pbc=False).cn.tolist() == [1, 1]


def test_gcn_without_bonds_is_zero_not_nan():
    atoms = Atoms("Pt2", positions=[[0, 0, 0], [9, 0, 0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_array_equal(QuasiGraph(atoms).gcn, [0.0, 0.0])


def test_missing_chemical_values_are_nan_with_a_warning():
    get_feature.cache_clear()
    atoms = Atoms("ArKr", positions=[[0, 0, 0], [4, 0, 0]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vector = QuasiGraph(atoms).get_vector()
    messages = [str(w.message) for w in caught if issubclass(w.category, MissingFeatureWarning)]
    assert "Feature 'atomic_radius' is not available for Kr; using NaN." in messages
    assert vector.dtype == float
    assert np.isnan(vector[2])  # en_pauling of Ar
    assert get_feature("Ar", "VEC") == 8.0
