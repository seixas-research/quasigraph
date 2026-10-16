from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, fcc111, molecule
from ase.io import read

from quasigraph import QuasiGraph, GEOMETRIC_FEATURES, neighbors

EXAMPLES = Path(__file__).parent.parent / "examples"


def _random_cell(seed):
    """Strongly skewed cell, some axes non-periodic, positions not wrapped into the cell."""
    rng = np.random.default_rng(seed)
    cell = rng.normal(size=(3, 3)) * 1.5 + np.eye(3) * 3.5
    pbc = [True, True, bool(seed % 2)]
    return Atoms("Pt3", scaled_positions=rng.random((3, 3)) * 1.6 - 0.3, cell=cell, pbc=pbc)


def _hea_cluster():
    """Pt55 cluster with a random AgAuCuPdPt decoration and an H atom above an atom."""
    atoms = read(EXAMPLES / "M55.xyz")
    atoms.set_chemical_symbols(np.random.default_rng(1).choice(["Ag", "Au", "Cu", "Pd", "Pt"], len(atoms)))
    top = atoms.positions[:, 2].argmax()
    return atoms + Atoms("H", positions=[atoms.positions[top] + [0, 0, 1.6]])


STRUCTURES = {
    "methanol": molecule("CH3OH"),
    "acetic_acid_xyz": read(Path(__file__).with_name("acetic_acid.xyz")),
    "M13": read(EXAMPLES / "M13.xyz"),
    "hea_cluster_with_H": _hea_cluster(),
    "isolated_atoms": Atoms("Pt2", positions=[[0, 0, 0], [9, 0, 0]]),
    "Au_primitive": bulk("Au"),
    "Pt_bulk_xyz": read(EXAMPLES / "Pt_bulk.xyz"),
    "Au_supercell": bulk("Au", cubic=True).repeat(2),
    "Cu_slab": fcc111("Cu", size=(2, 2, 3), vacuum=6.0),
    "skewed_Pt": Atoms("Pt", cell=[[2.77, 0, 0], [27.7, 2.4, 0], [0, 0, 2.77]], pbc=True),
    **{f"random_cell_{seed}": _random_cell(seed) for seed in range(8)},
}


@pytest.fixture(params=list(STRUCTURES), ids=list(STRUCTURES))
def both_backends(request):
    atoms = STRUCTURES[request.param]
    kwargs = dict(geometric_features=GEOMETRIC_FEATURES)
    return QuasiGraph(atoms, backend="dense", **kwargs), QuasiGraph(atoms, backend="vesin", **kwargs)


def test_bonds_cn_and_gcn_agree(both_backends):
    dense, vesin = both_backends
    np.testing.assert_array_equal(vesin.cn, dense.cn)
    np.testing.assert_allclose(vesin.gcn, dense.gcn, rtol=1e-12)
    assert vesin.bonded_atoms == dense.bonded_atoms
    np.testing.assert_array_equal(vesin.adjacency, dense.adjacency)


def test_geometric_features_agree(both_backends):
    dense, vesin = both_backends
    df_dense, df_vesin = dense.get_dataframe(), vesin.get_dataframe()
    assert list(df_vesin.columns) == list(df_dense.columns)
    exact = [c for c in df_dense.columns if c != "CN_smooth"]
    np.testing.assert_allclose(df_vesin[exact].to_numpy(dtype=float), df_dense[exact].to_numpy(dtype=float),
                               rtol=1e-12, atol=1e-12)
    # the vesin sigmoid sum of CN_smooth stops 20 * temperature past the bond threshold (error ~1e-8)
    np.testing.assert_allclose(df_vesin["CN_smooth"], df_dense["CN_smooth"], atol=1e-7)


def test_site_environments_agree(both_backends):
    dense, vesin = both_backends
    for site in {0, len(dense.atoms) - 1}:
        env_dense = dense.get_site_environment(site, shells=3)
        env_vesin = vesin.get_site_environment(site, shells=3)
        assert list(env_vesin.index) == list(env_dense.index)
        np.testing.assert_allclose(env_vesin.to_numpy(dtype=float), env_dense.to_numpy(dtype=float), atol=1e-7)


def test_pair_distances_agree(both_backends):
    dense, vesin = both_backends
    np.testing.assert_allclose(vesin.pair_distances, dense.pair_distances, rtol=1e-12)


def test_neighbor_pairs_keeps_pairs_exactly_at_the_cutoff():
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]])
    i, j, shifts, distances = neighbors.neighbor_pairs(positions, np.eye(3) * 10, [False] * 3, cutoff=1.0)
    assert i.dtype == np.int64 and shifts.shape == (2, 3)
    assert sorted(zip(i.tolist(), j.tolist())) == [(0, 1), (1, 0)]
    np.testing.assert_allclose(distances, 1.0)


def test_vesin_backend_does_not_build_dense_arrays():
    descriptor = QuasiGraph(bulk("Au", cubic=True).repeat(2), backend="vesin")
    for name in ("bonds", "distances_tensor", "bond_threshold"):
        assert not hasattr(descriptor, name)
    with pytest.raises(AttributeError, match="backend='dense'"):
        descriptor.distances_list
    assert len(descriptor.pairs["i"]) == 32 * 12


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="backend must be"):
        QuasiGraph(molecule("H2O"), backend="kdtree")
