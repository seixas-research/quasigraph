from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk, molecule
from ase.io import read

from quasigraph import QuasiGraph, GEOMETRIC_FEATURES, DEFAULT_CHEMICAL_FEATURES, geometry

EXAMPLES = Path(__file__).parent.parent / "examples"
HEA = ["Ag", "Au", "Cu", "Pd", "Pt"]


def _hea_cluster(seed=1):
    atoms = read(EXAMPLES / "M55.xyz")
    atoms.set_chemical_symbols(np.random.default_rng(seed).choice(HEA, len(atoms)))
    return atoms


def test_cn_max_fixes_the_gcn_reference():
    atoms = read(EXAMPLES / "M13.xyz")  # icosahedron: centre CN 12, surface CN 6
    default = QuasiGraph(atoms)
    fixed = QuasiGraph(atoms, cn_max=12)
    assert max(default.cn) == 12
    np.testing.assert_allclose(fixed.gcn, default.gcn)
    # without a 12-coordinated atom the default divisor changes, a fixed cn_max does not
    surface = atoms[[i for i in range(len(atoms)) if default.cn[i] < 12]]
    raw = QuasiGraph(surface, normalization=False).gcn
    np.testing.assert_allclose(QuasiGraph(surface, cn_max=12).gcn, np.asarray(raw) / 12)
    np.testing.assert_allclose(QuasiGraph(surface, cn_max=12, backend="vesin").gcn, np.asarray(raw) / 12)
    with pytest.raises(ValueError, match="cn_max"):
        geometry.generalized_coordination_numbers(np.ones((2, 2)), [1, 1], cn_max=0)


def test_structure_features_are_named_and_size_independent():
    kwargs = dict(geometric_features=GEOMETRIC_FEATURES)
    small = QuasiGraph(read(EXAMPLES / "M13.xyz"), **kwargs).get_structure_features(elements=HEA)
    large = QuasiGraph(_hea_cluster(), **kwargs).get_structure_features(elements=HEA)
    assert list(small.index) == list(large.index)
    assert small["n_atoms"] == 13 and large["n_atoms"] == 55
    assert small["frac_Ag"] == 1.0 and small["frac_Pt"] == 0.0
    assert large[[f"frac_{el}" for el in HEA]].sum() == pytest.approx(1.0)
    columns = list(DEFAULT_CHEMICAL_FEATURES) + GEOMETRIC_FEATURES
    assert list(small.index[1 + len(HEA):]) == [f"{c}_{s}" for c in columns for s in ("mean", "min", "max", "std")]
    df = QuasiGraph(_hea_cluster(), **kwargs).get_dataframe()
    assert large["GCN_mean"] == pytest.approx(df["GCN"].mean())
    assert large["CN_std"] == pytest.approx(df["CN"].std(ddof=0))


def test_structure_vector_options_and_single_atom():
    single = QuasiGraph(molecule("H")).get_structure_features(statistics=("mean", "std"))
    assert single["CN_std"] == 0.0 and single["frac_H"] == 1.0
    vector = QuasiGraph(molecule("H2O")).get_structure_vector(statistics=("max",))
    assert vector.dtype == float and vector.shape == (1 + 2 + 6,)
    with pytest.raises(ValueError, match="Unknown statistics"):
        QuasiGraph(molecule("H2O")).get_structure_features(statistics=("median",))


@pytest.mark.parametrize("backend", ["dense", "vesin"])
def test_descriptors_are_invariant_to_atom_order(backend):
    atoms = _hea_cluster()
    atoms += molecule("H")
    atoms.positions[-1] = atoms.positions[atoms.positions[:, 2].argmax()] + [0, 0, 1.6]
    permutation = np.random.default_rng(3).permutation(len(atoms))
    shuffled = atoms[permutation]
    kwargs = dict(geometric_features=GEOMETRIC_FEATURES, backend=backend)
    a, b = QuasiGraph(atoms, **kwargs), QuasiGraph(shuffled, **kwargs)
    np.testing.assert_allclose(a.get_structure_vector(elements=HEA + ["H"]),
                               b.get_structure_vector(elements=HEA + ["H"]), rtol=1e-12)
    np.testing.assert_allclose(a.get_site_vector("H", shells=3, elements=HEA),
                               b.get_site_vector("H", shells=3, elements=HEA), rtol=1e-12)
    # the per-atom table follows the atoms
    np.testing.assert_allclose(a.get_dataframe().to_numpy()[permutation], b.get_dataframe().to_numpy())


def test_supercell_gives_the_same_structure_features():
    primitive = QuasiGraph(bulk("Au"), geometric_features=GEOMETRIC_FEATURES).get_structure_features()
    supercell = QuasiGraph(bulk("Au").repeat(2), geometric_features=GEOMETRIC_FEATURES).get_structure_features()
    np.testing.assert_allclose(primitive.drop("n_atoms").to_numpy(), supercell.drop("n_atoms").to_numpy(), atol=1e-12)


def test_feature_names_match_the_vector():
    descriptor = QuasiGraph(molecule("H2O"), nmax=4)
    names = descriptor.get_feature_names()
    assert len(names) == len(descriptor.get_vector()) == 4 * 6
    assert names[:2] == ["atom0_VEC", "atom0_atomic_radius"] and names[-1] == "atom3_GCN"


def test_default_feature_lists_are_not_shared():
    first = QuasiGraph(molecule("H2O"))
    first.chemical_features.append("group")
    assert QuasiGraph(molecule("H2O")).chemical_features == list(DEFAULT_CHEMICAL_FEATURES)
    assert "backend='dense'" in repr(first)


def test_quasigraph_wraps_atoms_instead_of_subclassing():
    from ase import Atoms

    atoms = molecule("CH3OH")
    descriptor = QuasiGraph(atoms)
    assert not isinstance(descriptor, Atoms) and descriptor.atoms is atoms
    assert len(descriptor) == 6
    assert descriptor.pbc.tolist() == [False, False, False]
    assert QuasiGraph(atoms, pbc=False).pbc.dtype == bool
    with pytest.warns(DeprecationWarning, match="use QuasiGraph.atoms.get_positions"):
        np.testing.assert_array_equal(descriptor.get_positions(), atoms.get_positions())
    with pytest.raises(AttributeError, match="no attribute 'distances_tensor'"):
        descriptor.distances_tensor
    with pytest.raises(AttributeError, match="no attribute 'not_an_atoms_method'"):
        descriptor.not_an_atoms_method
