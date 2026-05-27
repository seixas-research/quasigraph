import numpy as np
import pytest
from pathlib import Path
from ase.build import bulk, molecule
from ase.io import read

from quasigraph import QuasiGraph


@pytest.fixture
def atoms_molecule():
    atoms = molecule("H2O")
    atoms.center(vacuum=5.0)
    atoms.pbc = False
    return atoms


@pytest.fixture
def atoms_crystal():
    atoms = bulk("Au", cubic=True).repeat([2,2,2])
    return atoms


def test_descriptor_dataframe_matches_expected_values(atoms_molecule):
    descriptor = QuasiGraph(atoms_molecule)

    dataframe = descriptor.get_dataframe()

    assert list(dataframe.columns) == [
        "VEC",
        "atomic_radius",
        "en_pauling",
        "electron_affinity",
        "CN",
        "GCN",
    ]
    assert dataframe.shape == (3, 6)

    np.testing.assert_allclose(
        dataframe.to_numpy(),
        np.array(
            [
                [6.0, 0.60, 3.44, 1.4611135, 2.0, 1.0],
                [1.0, 0.25, 2.20, 0.7541950, 1.0, 1.0],
                [1.0, 0.25, 2.20, 0.7541950, 1.0, 1.0],
            ]
        ),
        rtol=1e-7,
        atol=1e-7,
    )


def test_descriptor_vector_is_flattened_dataframe(atoms_molecule):
    descriptor = QuasiGraph(atoms_molecule)

    dataframe = descriptor.get_dataframe()
    vector = descriptor.get_vector()

    assert vector.shape == (18,)
    np.testing.assert_allclose(vector, dataframe.to_numpy().flatten())


def test_descriptor_can_be_zero_padded_with_nmax(atoms_molecule):
    descriptor = QuasiGraph(atoms_molecule, nmax=5)

    dataframe = descriptor.get_dataframe()

    assert dataframe.shape == (5, 6)
    np.testing.assert_allclose(dataframe.iloc[:3].to_numpy(), QuasiGraph(atoms_molecule).get_dataframe().to_numpy())
    np.testing.assert_allclose(dataframe.iloc[3:].to_numpy(), np.zeros((2, 6)))


def test_descriptor_from_xyz_file():
    atoms = read(Path(__file__).with_name("acetic_acid.xyz"))

    descriptor = QuasiGraph(atoms)
    dataframe = descriptor.get_dataframe()

    assert len(atoms) == 8
    assert dataframe.shape == (8, 6)
    np.testing.assert_array_equal(dataframe["CN"].to_numpy(), np.array([3, 1, 2, 1, 4, 1, 1, 1]))
    np.testing.assert_allclose(dataframe["GCN"].to_numpy(), np.array([1.75, 0.75, 1.0, 0.5, 1.5, 1.0, 1.0, 1.0]))


def test_descriptor_with_invalid_chemical_features(atoms_molecule):
    descriptor = QuasiGraph(atoms_molecule, chemical_features=["ABC"])

    with pytest.raises(ValueError, match="Feature 'ABC' is not recognized"):
        descriptor.get_dataframe()


def test_descriptor_for_crystal_with_pbc(atoms_crystal):
    descriptor=QuasiGraph(atoms=atoms_crystal, pbc=[True, True, True])
    dataframe = descriptor.get_dataframe()

    assert dataframe.shape == (32, 6)
    np.testing.assert_array_equal(descriptor.cn, np.array(32*[12]))
    np.testing.assert_allclose(descriptor.gcn, np.array(32*[12.0]))
    np.testing.assert_allclose(dataframe["CN"].to_numpy(), np.array(32*[12]))
    np.testing.assert_allclose(dataframe["GCN"].to_numpy(), np.array(32*[12.0]))
