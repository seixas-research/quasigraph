# Changelog

## Unreleased

### Changed (may change results or break code)

- `QuasiGraph` no longer subclasses `ase.Atoms`; the structure is the `atoms` attribute. `isinstance(qgr, ase.Atoms)` is now `False`; `ase.Atoms` attributes and methods are forwarded to `atoms` with a `DeprecationWarning`, and `len(qgr)` is the number of atoms (was 0).
- `pbc` defaults to `atoms.pbc` (it was `[False, False, False]`, so periodic structures were treated as clusters unless `pbc` was given). Pass `pbc=False` for the old behaviour.
- Chemical feature columns are floats (`VEC`, `group` and `period` were integers); `CN` stays an integer.
- `CN_smooth` of periodic structures with the dense backend now includes the periodic images reaching 20 `temperature` past the bond cutoff. Before, its sigmoid tail was cut at the images needed for bonds, an error of up to ~4e-3 in small or skewed cells. Non-periodic structures are unchanged.
- `adjacency` is a property computed on access instead of an attribute set in `__init__`.
- Chemical properties come from a bundled table (`quasigraph/data/elements.json`, built from mendeleev 1.1.0) instead of live mendeleev queries; values are identical. `mendeleev` is now an optional dependency (`pip install quasigraph[mendeleev]`), needed only by `get_element`.
- Requires Python 3.10 or newer (needed by vesin). `pytest` is no longer installed as a runtime dependency (`pip install quasigraph[test]`).

### Fixed

- GCN and the neighbour shells of `get_site_environment` count every periodic image of a neighbour, as the CN does: a primitive cell and any supercell of it give the same descriptor (GCN of primitive fcc Au was 1 instead of 12).
- Periodic structures include as many lattice translations as needed to find every bond; small or skewed cells used to miss neighbours (fixed ±1 translation).
- GCN is 0 instead of NaN when no atom has a bond.
- A chemical property missing from the database is NaN with a `MissingFeatureWarning`, instead of `None` (object-typed vectors) or a crash (e.g. `atomic_radius` of Kr).
- A periodic axis without a cell vector raises `ValueError` instead of giving meaningless coordination numbers.

### Added

- `backend="vesin"`: bonds from the sparse neighbour list of [vesin](https://github.com/Luthaf/vesin), linear in the number of atoms (2916-atom periodic cell: 0.07 s instead of 24 s and 9.5 GB). Module `quasigraph.neighbors`.
- `get_structure_features` / `get_structure_vector`: descriptor of the whole structure that depends neither on the order nor on the number of atoms (composition fractions and mean/min/max/std of every per-atom feature).
- `featurize(structures, kind=..., n_jobs=...)`: descriptors of a dataset as one DataFrame, optionally in parallel processes.
- `QuasiGraphFeaturizer`: scikit-learn transformer for pipelines (`pip install quasigraph[ml]`).
- `cn_max`: fixed reference coordination number for the GCN (e.g. 12 for fcc metals).
- `get_feature_names` (names of the `get_vector` entries), `get_bonds` (edge list) and `to_networkx` (bond graph; `pip install quasigraph[graph]`).
- pymatgen `Structure` and `Molecule` objects are accepted wherever an `ase.Atoms` is (`to_atoms`).
- Continuous integration (tests on Python 3.10–3.13, ruff, mypy), type hints on the public API.

## 26.9.3 and earlier

See the git history.
