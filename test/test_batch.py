from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import read

from quasigraph import QuasiGraph, GEOMETRIC_FEATURES, featurize

EXAMPLES = Path(__file__).parent.parent / "examples"


def _dataset():
    """Three HEA-like clusters of different sizes, each with one H adsorbed on top."""
    structures = []
    for seed, name in enumerate(["M13.xyz", "M55.xyz", "M55.xyz"]):
        atoms = read(EXAMPLES / name)
        atoms.set_chemical_symbols(np.random.default_rng(seed).choice(["Ag", "Au", "Pt"], len(atoms)))
        top = atoms.positions[atoms.positions[:, 2].argmax()]
        structures.append(atoms + Atoms("H", positions=[top + [0, 0, 1.6]]))
    return structures


def test_structure_features_of_a_dataset():
    structures = _dataset()
    df = featurize(structures, geometric_features=GEOMETRIC_FEATURES)
    assert df.shape[0] == 3
    assert [c for c in df.columns if c.startswith("frac_")] == ["frac_Ag", "frac_Au", "frac_H", "frac_Pt"]
    expected = QuasiGraph(structures[1], geometric_features=GEOMETRIC_FEATURES).get_structure_vector(
        elements=["Ag", "Au", "H", "Pt"])
    np.testing.assert_allclose(df.iloc[1].to_numpy(), expected)


def test_site_features_of_a_dataset_in_parallel():
    structures = _dataset()
    serial = featurize(structures, kind="site", site="H", shells=2, backend="vesin")
    parallel = featurize(structures, kind="site", site="H", shells=2, backend="vesin", n_jobs=2)
    assert list(serial.columns) == list(parallel.columns)
    np.testing.assert_array_equal(serial.to_numpy(), parallel.to_numpy())
    assert serial["shell1_n"].tolist() == [1.0, 1.0, 1.0]


def test_per_atom_vectors_are_padded_to_the_largest_structure():
    df = featurize([molecule("H2O"), molecule("CH3OH")], kind="atoms")
    assert df.shape == (2, 6 * 6)
    assert df.columns[-1] == "atom5_GCN"
    np.testing.assert_array_equal(df.iloc[0, 3 * 6:].to_numpy(), 0.0)


def test_featurize_validates_its_arguments():
    with pytest.raises(ValueError, match="kind must be"):
        featurize([molecule("H2O")], kind="graph")
    with pytest.raises(ValueError, match="needs a site"):
        featurize([molecule("H2O")], kind="site")
    assert featurize([]).empty
