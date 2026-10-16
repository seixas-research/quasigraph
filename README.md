<h1 align="center" style="margin-top:20px; margin-bottom:50px;">
<img src="https://raw.githubusercontent.com/seixas-research/quasigraph/refs/heads/main/resources/logo.png" style="height: 70px"></h1>


[![PyPI - License](https://img.shields.io/pypi/l/quasigraph?color=green&style=for-the-badge)](LICENSE.txt)    [![PyPI](https://img.shields.io/pypi/v/quasigraph?color=red&label=version&style=for-the-badge)](https://pypi.org/project/quasigraph/) [![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.14963121-blue?style=for-the-badge)](https://doi.org/10.5281/zenodo.14963121)

**Quasigraph** is an open-source toolkit designed for generating chemical and geometric descriptors to be used in machine learning models.

# Installation

The easiest method to install quasigraph is by utilizing pip (Python 3.10 or newer):
```bash
$ pip install quasigraph
```

Optional extras: `quasigraph[ml]` (scikit-learn transformer), `quasigraph[pymatgen]` (pymatgen input), `quasigraph[graph]` (networkx export), `quasigraph[mendeleev]` (`get_element`), `quasigraph[dev]` (tests, ruff, mypy).

# Getting started

```python
from ase.build import molecule
from quasigraph import QuasiGraph

# Initialize an Atoms object for methanol (CH3OH) using ASE's molecule function
atoms = molecule('CH3OH')

# Instantiate a QuasiGraph object containing chemical and coordination numbers
qgr = QuasiGraph(atoms)

# Convert the QuasiGraph object into a pandas DataFrame
df = qgr.get_dataframe()

# Convert the QuasiGraph object into a vector
vector = qgr.get_vector()

# Descriptor of the whole structure, independent of the order and number of atoms
features = qgr.get_structure_features()
```

Structures written by DFT codes (VASP, GPAW, Quantum ESPRESSO, SIESTA, ...) are read with `ase.io.read`; pymatgen `Structure` and `Molecule` objects are accepted as well.

# Descriptor

The descriptor can be separated into two parts, a chemical part and a geometric part.

## Chemical part

The chemical part of the descriptor uses element properties from the [Mendeleev library](https://github.com/lmmentel/mendeleev), bundled with quasigraph as a table. By default it holds the valence electron concentration (`VEC`), atomic radius (Å), Pauling electronegativity and electron affinity (eV) of every atom; `chemical_features` selects others among `AVAILABLE_FEATURES` (`group`, `period`, `atomic_weight`, `covalent_radius`, `atomic_radius`, `vdw_radius`, `en_pauling`, `electron_affinity`, `dipole_polarizability`, `VEC`).

For example, for methanol (CH<sub>3</sub>OH) we have the table:

|    |   VEC |   atomic_radius |   en_pauling |   electron_affinity |
|---:|------:|----------------:|-------------:|--------------------:|
|  0 |     4 |            0.7  |         2.55 |            1.26212  |
|  1 |     6 |            0.6  |         3.44 |            1.46111  |
|  2 |     1 |            0.25 |         2.2  |            0.754195 |
|  3 |     1 |            0.25 |         2.2  |            0.754195 |
|  4 |     1 |            0.25 |         2.2  |            0.754195 |
|  5 |     1 |            0.25 |         2.2  |            0.754195 |

## Geometric part

The geometric part involves identifying all bonds and computing the coordination numbers for each atom, indicated as CN. Additionally, the generalized coordination number (GCN)[^1] is determined by summing the coordination numbers of the neighboring ligands for each atom and normalizing this sum by the highest coordination number found in the molecule. To compare GCN values between structures, fix the normalization instead with `cn_max`, e.g. `QuasiGraph(atoms, cn_max=12)` for fcc metals as in the original definition.

<p align="center">
<img src="https://raw.githubusercontent.com/seixas-research/quasigraph/refs/heads/main/resources/methanol.png" style="height: 150px"></p>

<p align="center"><a name="fig1">Figure 1</a> - Schematic representation of the methanol molecule, indicating the chemical symbol and coordination number (CN) for every atom.</p>

For example, for methanol (CH<sub>3</sub>OH) we have the geometric data, as shown in [Fig. 1](#fig1).

|   CN  |  GCN  |
|:-----:|:-----:|
|     4 |  1.25 |
|     2 |  1.25 |
|     1 |  1.00 |
|     1 |  0.50 |
|     1 |  1.00 |
|     1 |  1.00 |

## Periodic structures

Periodic boundary conditions are taken from `atoms.pbc` unless the `pbc` argument is given (e.g. `pbc=False` treats a periodic cell as an isolated cluster). Along each periodic axis, as many periodic images are included as needed to find every bond, also for small or strongly skewed cells. Periodic images of a neighbour count separately in `CN`, `GCN` and the neighbour shells of `get_site_environment`, so a primitive cell and any supercell of it give the same descriptor.

Chemical properties missing from the Mendeleev database (e.g. the Pauling electronegativity of noble gases) are returned as `NaN`, with a `MissingFeatureWarning`.

## Large structures: `backend="vesin"`

By default, bonds are found from the full distance matrix (a tensor over periodic images for a periodic structure), whose memory grows with the square of the number of atoms. With `backend="vesin"` they are found with the sparse neighbour list of the [vesin](https://github.com/Luthaf/vesin) library instead, which scales linearly:

```python
qgr = QuasiGraph(atoms, backend="vesin")
```

Both backends give the same descriptor (`CN_smooth` agrees to ~1e-8, since the vesin sum of its sigmoid stops 20 `temperature` beyond the bond threshold). With `"vesin"` the dense attributes `distances`, `distances_tensor`, `bonds` and `bond_threshold` are not built; the bonds are in `pairs` instead. For a periodic 2916-atom Pt supercell with all geometric features, `"dense"` takes 24 s and 9.5 GB, `"vesin"` 0.07 s; for a ~55-atom nanoparticle both take a few milliseconds (`"dense"` is slightly faster).

## Continuous bond information (optional)

By default the geometric part is `CN` and `GCN`. The `geometric_features` argument selects any of the per-atom features below; the chemical columns come first, then the geometric ones in the order given.

```python
from quasigraph import QuasiGraph, GEOMETRIC_FEATURES

qgr = QuasiGraph(atoms, geometric_features=GEOMETRIC_FEATURES)
qgr.get_dataframe()   # ... CN, GCN, CN_smooth, bond_mean, bond_min, bond_max, bond_std, bond_strain
```

| Feature | Meaning |
|---|---|
| `CN` | number of bonded atoms (a pair is bonded when its distance is at most (1 + `tolerance`) times the sum of covalent radii) |
| `GCN` | generalized coordination number |
| `CN_smooth` | continuous CN: every pair contributes a sigmoid weight 1 / (1 + exp((d − r<sub>c</sub>) / T)) centred on the bond threshold r<sub>c</sub> = (1 + `tolerance`)(r<sub>i</sub> + r<sub>j</sub>), with `temperature` T in Å (default 0.1). T → 0 gives a step function and `CN_smooth` = `CN` |
| `bond_mean`, `bond_min`, `bond_max`, `bond_std` | statistics of the atom's bond lengths (0 for an atom without bonds) |
| `bond_strain` | mean of (bond length / sum of covalent radii) − 1: positive for stretched bonds |

Covalent radii are the Pyykkö values from Mendeleev. Metallic bonds are typically 10–20 % longer than the sum of these radii, so in a perfect fcc metal `bond_strain` ≈ 0.15; the useful information is the variation between sites. With the default `temperature` of 0.1 Å a bond 0.3 Å inside the threshold has weight 0.95 and one 0.3 Å outside it 0.05, so `CN_smooth` stays close to `CN` while varying continuously with bond length; a larger `temperature` smooths more.

## Adsorption-site environment

For a target that depends on one atom, such as the adsorption free energy of a hydrogen atom, `get_site_environment` aggregates the descriptor over the neighbour shells of that atom on the bond graph and returns a fixed-length, named vector that does not depend on the number of atoms or their order:

```python
qgr = QuasiGraph(atoms, geometric_features=GEOMETRIC_FEATURES)
env = qgr.get_site_environment("H", shells=2, elements=["Ag", "Au", "Cu", "Pd", "Pt"])  # pandas Series
vector = qgr.get_site_vector("H", shells=2, elements=["Ag", "Au", "Cu", "Pd", "Pt"])   # numpy array
```

`site` is an atom index or a chemical symbol that occurs once in the structure. The Series contains `site_<feature>` for the site atom itself and, for each shell *k* (atoms *k* bonds away from the site): `shell<k>_n`, `shell<k>_n_<El>` (counts per element in `elements`), `shell<k>_dist_mean/min/max` (distance from the site), `shell<k>_<feature>_mean` for chemical features and `shell<k>_<feature>_mean/min/max` for geometric features. Pass the same `elements` list for every structure of a dataset so all vectors have the same length; an empty shell contributes zeros.

## Whole-structure descriptor

`get_structure_features` returns a named pandas Series that depends neither on the order nor on the number of atoms: `n_atoms`, the composition fractions `frac_<El>` and the `mean`, `min`, `max` and `std` of every per-atom feature (`get_structure_vector` gives the same as a numpy array). Pass the same `elements` list for every structure of a dataset.

```python
qgr = QuasiGraph(atoms, geometric_features=GEOMETRIC_FEATURES)
qgr.get_structure_features(elements=["Ag", "Au", "Cu", "Pd", "Pt"])   # n_atoms, frac_Ag, ..., GCN_mean, GCN_min, ...
```

The flattened per-atom vector `get_vector()` depends on the order of the atoms and, unless `nmax` pads it, on their number; `get_feature_names()` names its entries.

# Machine learning workflows

`featurize` computes a descriptor for every structure of a dataset and returns one DataFrame, with the same columns for every row (`elements` defaults to all elements of the dataset):

```python
from ase.io import read
from quasigraph import featurize, GEOMETRIC_FEATURES

structures = [read(f) for f in files]
X = featurize(structures, kind="site", site="H", geometric_features=GEOMETRIC_FEATURES)   # or kind="structure" / "atoms"
```

`n_jobs` spreads the work over processes; it pays off for large structures, while for small ones (e.g. ~55-atom nanoparticles) starting the processes costs more than it saves. On macOS and Windows, call it under `if __name__ == "__main__":`.

`QuasiGraphFeaturizer` (`pip install quasigraph[ml]`) is a scikit-learn transformer, so quasigraph can be the first step of a pipeline and be cross-validated or tuned with the model:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from quasigraph import QuasiGraphFeaturizer

model = make_pipeline(QuasiGraphFeaturizer(kind="site", site="H", cn_max=12), RandomForestRegressor())
scores = cross_val_score(model, structures, y, cv=5)
```

`fit` fixes the element list from the training structures; `get_feature_names_out()` names the columns and `set_output(transform="pandas")` returns DataFrames.

## Bond graph

`get_bonds()` returns the bonds as an edge list (`i`, `j`, lattice translation `shift_a/b/c`, `distance`; every bond in both directions), and `to_networkx()` the bond graph as a `networkx.MultiGraph` with the per-atom features as node attributes (`pip install quasigraph[graph]`).

# Package structure

- `quasigraph.quasigraph.QuasiGraph` – user-facing class; holds an ASE `Atoms` object as `atoms` (it is no longer an `Atoms` subclass) and exposes `get_dataframe()` / `get_vector()`, `get_structure_features()`, `get_site_environment()`, `get_bonds()` plus the attributes `cn`, `gcn`, `bonded_atoms`, `adjacency` and, with the dense backend, `bonds`, `distances` (non-periodic) and `distances_tensor` (periodic).
- `quasigraph.geometry` – pure NumPy functions for distances, bonds, CN and GCN (`distance_matrix`, `distance_tensor`, `bonds_from_distance_matrix`, `coordination_numbers`, `generalized_coordination_numbers`, ...). They take plain arrays and can be reused without ASE.
- `quasigraph.neighbors` – the same features computed from a sparse pair list found with vesin (`neighbor_pairs`, `coordination_numbers`, `bond_length_statistics`, ...), used by `backend="vesin"`.
- `quasigraph.elements` – chemical features (`get_feature`) from the bundled table `quasigraph/data/elements.json`, built from the [Mendeleev library](https://github.com/lmmentel/mendeleev) by `scripts/build_element_table.py`, so results do not depend on the installed Mendeleev version.
- `quasigraph.ptable` – hand-curated tables (`VEC`, `CONFIG`) not available in Mendeleev.
- `quasigraph.batch` – `featurize`, descriptors of a dataset.
- `quasigraph.estimators` – `QuasiGraphFeaturizer`, the scikit-learn transformer.

# Development

```bash
$ pip install -e ".[dev]"
$ pytest          # tests
$ ruff check .    # lint
$ mypy            # type check
```

Changes are listed in [CHANGELOG.md](CHANGELOG.md).

# License

This is an open source code under [MIT License](LICENSE.txt).

# Acknowledgements

We thank financial support from FAPESP (Grant No. 2022/14549-3), INCT Materials Informatics (Grant No. 406447/2022-5), and CNPq (Grant No. 311324/2020-7).

[^1]: Calle-Vallejo, F., Martínez, J. I., García-Lastra, J. M., Sautet, P. & Loffreda, D. [Fast Prediction of Adsorption Properties for Platinum Nanocatalysts with Generalized Coordination Numbers](https://doi.org/10.1002/anie.201402958), *Angew. Chem. Int. Ed.* **53**, 8316-8319 (2014).
