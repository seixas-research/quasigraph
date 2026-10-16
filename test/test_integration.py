from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.build import bulk, fcc111, molecule
from ase.io import read

from quasigraph import QuasiGraph, GEOMETRIC_FEATURES, to_atoms

EXAMPLES = Path(__file__).parent.parent / "examples"


def _adsorption_dataset(n=12):
    """Small Ag/Au/Pt clusters with one H on top and a made-up target (H-site GCN plus noise)."""
    rng = np.random.default_rng(0)
    structures, targets = [], []
    base = read(EXAMPLES / "M13.xyz")
    for _ in range(n):
        atoms = base.copy()
        atoms.set_chemical_symbols(rng.choice(["Ag", "Au", "Pt"], len(atoms)))
        top = atoms.positions[rng.integers(len(atoms))]
        atoms += Atoms("H", positions=[top + 1.6 * top / max(np.linalg.norm(top), 1e-9)])
        structures.append(atoms)
        targets.append(rng.normal())
    return structures, np.array(targets)


@pytest.mark.parametrize("structure", [bulk("Au", cubic=True), molecule("CH3OH")], ids=["Structure", "Molecule"])
def test_pymatgen_objects_are_accepted(structure):
    pytest.importorskip("pymatgen")
    from pymatgen.io.ase import AseAtomsAdaptor
    converted = (AseAtomsAdaptor.get_structure(structure) if structure.pbc.any()
                 else AseAtomsAdaptor.get_molecule(structure))
    np.testing.assert_allclose(QuasiGraph(converted).get_vector(), QuasiGraph(structure).get_vector())
    assert isinstance(to_atoms(converted), Atoms)
    with pytest.raises(TypeError, match="Expected an ase.Atoms"):
        to_atoms("CH3OH")


@pytest.mark.parametrize("atoms", [molecule("CH3OH"), bulk("Au"), fcc111("Cu", size=(2, 2, 3), vacuum=6.0)],
                         ids=["molecule", "primitive", "slab"])
def test_bond_table_is_the_same_for_both_backends(atoms):
    dense, vesin = QuasiGraph(atoms).get_bonds(), QuasiGraph(atoms, backend="vesin").get_bonds()
    assert list(dense.columns) == ["i", "j", "shift_a", "shift_b", "shift_c", "distance"]
    pd.testing.assert_frame_equal(dense[["i", "j", "shift_a", "shift_b", "shift_c"]],
                                  vesin[["i", "j", "shift_a", "shift_b", "shift_c"]])
    np.testing.assert_allclose(dense["distance"], vesin["distance"], rtol=1e-12)
    assert len(dense) == QuasiGraph(atoms).cn.sum()


def test_networkx_graph():
    pytest.importorskip("networkx")
    methanol = QuasiGraph(molecule("CH3OH")).to_networkx()
    assert methanol.number_of_nodes() == 6 and methanol.number_of_edges() == 5
    assert methanol.nodes[0]["symbol"] == "C" and methanol.nodes[0]["CN"] == 4
    primitive = QuasiGraph(bulk("Au")).to_networkx()  # 12 bonds to own images = 6 self-loops
    assert primitive.number_of_edges() == 6 and all(u == v for u, v in primitive.edges())


def test_sklearn_pipeline_with_cross_validation():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    from quasigraph import QuasiGraphFeaturizer

    structures, y = _adsorption_dataset()
    featurizer = QuasiGraphFeaturizer(kind="site", site="H", geometric_features=GEOMETRIC_FEATURES)
    X = featurizer.fit_transform(structures)
    assert X.shape == (len(structures), len(featurizer.get_feature_names_out()))
    assert featurizer.elements_ == ["Ag", "Au", "H", "Pt"]
    np.testing.assert_allclose(X[0], QuasiGraph(structures[0], geometric_features=GEOMETRIC_FEATURES)
                               .get_site_vector("H", elements=["Ag", "Au", "H", "Pt"]))

    model = make_pipeline(QuasiGraphFeaturizer(kind="site", site="H"), RandomForestRegressor(n_estimators=10))
    scores = cross_val_score(model, structures, y, cv=3)
    assert scores.shape == (3,)

    frame = QuasiGraphFeaturizer().set_output(transform="pandas").fit_transform(structures)
    assert isinstance(frame, pd.DataFrame) and "GCN_mean" in frame.columns


def test_featurizer_atoms_kind_and_errors():
    pytest.importorskip("sklearn")
    from quasigraph import QuasiGraphFeaturizer

    featurizer = QuasiGraphFeaturizer(kind="atoms").fit([molecule("H2O"), molecule("CH3OH")])
    assert featurizer.transform([molecule("H2O")]).shape == (1, 6 * 6)
    with pytest.raises(ValueError, match="more atoms"):
        featurizer.transform([molecule("C6H6")])
    with pytest.raises(AttributeError, match="not fitted"):
        QuasiGraphFeaturizer().transform([molecule("H2O")])
