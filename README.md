<h1 align="center" style="margin-top:20px; margin-bottom:50px;">
<img src="https://raw.githubusercontent.com/leseixas/quasigraph/refs/heads/main/resources/logo.png" style="height: 70px"></h1>


[![PyPI - License](https://img.shields.io/pypi/l/quasigraph?color=green&style=for-the-badge)](LICENSE.txt)    [![PyPI](https://img.shields.io/pypi/v/quasigraph?color=red&label=version&style=for-the-badge)](https://pypi.org/project/quasigraph/) [![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.14963121-blue?style=for-the-badge)](https://doi.org/10.5281/zenodo.14963121)

**Quasigraph** is an open-source toolkit designed for generating chemical and geometric descriptors to be used in machine learning models.

# Installation

The easiest method to install quasigraph is by utilizing pip:
```bash
$ pip install quasigraph
```

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
```

# Descriptor

The descriptor can be separated into two parts, a chemical part and a geometric part.

## Chemical part

The chemical part of the descriptor employs the [Mendeleev library](https://github.com/lmmentel/mendeleev), incorporating atomic details like the valence electron concentration, covalent radius, atomic radius, Pauling electronegativity and electron affinitity for every element within the object.

For example, for methanol (CH<sub>3</sub>OH) we have the table:

|    |   VEC  |   covalent_radius |   en_pauling |
|---:|:--------:|:-----------------:|:------------:|
|  0 |          4 |              0.75 |         2.55 |
|  1 |          6 |              0.63 |         3.44 |
|  2 |          1 |              0.32 |         2.2  |
|  3 |          1 |              0.32 |         2.2  |
|  4 |          1 |              0.32 |         2.2  |
|  5 |          1 |              0.32 |         2.2  |

## Geometric part

The geometric part involves identifying all bonds and computing the coordination numbers for each atom, indicated as CN. Additionally, the generalized coordination number (GCN)[^1] is determined by summing the coordination numbers of the neighboring ligands for each atom and normalizing this sum by the highest coordination number found in the molecule.

<p align="center">
<img src="https://raw.githubusercontent.com/leseixas/quasigraph/master/resources/methanol.png" style="height: 150px"></p>

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

# Package structure

- `quasigraph.quasigraph.QuasiGraph` – user-facing class; wraps an ASE `Atoms` object and exposes `get_dataframe()` / `get_vector()` plus the attributes `cn`, `gcn`, `bonded_atoms`, `bonds`, `adjacency`, `distances` (non-periodic) and `distances_tensor` (periodic).
- `quasigraph.geometry` – pure NumPy functions for distances, bonds, CN and GCN (`distance_matrix`, `distance_tensor`, `bonds_from_distance_matrix`, `coordination_numbers`, `generalized_coordination_numbers`, ...). They take plain arrays and can be reused without ASE.
- `quasigraph.elements` – chemical features taken from the [Mendeleev library](https://github.com/lmmentel/mendeleev). Database lookups are cached per element symbol (`get_element`, `get_feature`), so the cost of a Mendeleev query is paid once per element per process instead of once per atom.
- `quasigraph.ptable` – hand-curated tables (`VEC`, `CONFIG`) not available in Mendeleev.

# License

This is an open source code under [MIT License](LICENSE.txt).

# Acknowledgements

We thank financial support from FAPESP (Grant No. 2022/14549-3), INCT Materials Informatics (Grant No. 406447/2022-5), and CNPq (Grant No. 311324/2020-7).

[^1]: Calle-Vallejo, F., Martínez, J. I., García-Lastra, J. M., Sautet, P. & Loffreda, D. [Fast Prediction of Adsorption Properties for Platinum Nanocatalysts with Generalized Coordination Numbers](https://doi.org/10.1002/anie.201402958), *Angew. Chem. Int. Ed.* **53**, 8316-8319 (2014).
