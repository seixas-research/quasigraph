#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: quasigraph.py

# This code is part of quasigraph.
# MIT License
#
# Copyright (c) 2023 Leandro Seixas Rocha <leandro.fisica@gmail.com>


from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from ase import Atoms

from . import elements, geometry, neighbors


GEOMETRIC_FEATURES = ['CN', 'GCN', 'CN_smooth', 'bond_mean', 'bond_min', 'bond_max', 'bond_std', 'bond_strain']
DEFAULT_CHEMICAL_FEATURES = ('VEC', 'atomic_radius', 'en_pauling', 'electron_affinity')
DEFAULT_GEOMETRIC_FEATURES = ('CN', 'GCN')
STRUCTURE_STATISTICS = ('mean', 'min', 'max', 'std')


def to_atoms(structure: Any) -> Atoms:
    """
    ``structure`` as an ``ase.Atoms`` object: Atoms objects are returned as
    they are; pymatgen ``Structure`` and ``Molecule`` objects are converted with
    ``pymatgen.io.ase.AseAtomsAdaptor`` (periodic and non-periodic,
    respectively).  Files of DFT codes (VASP, Quantum ESPRESSO, GPAW, SIESTA,
    ...) are read with ``ase.io.read``.
    """
    if isinstance(structure, Atoms):
        return structure
    if type(structure).__module__.startswith('pymatgen'):
        from pymatgen.io.ase import AseAtomsAdaptor
        return AseAtomsAdaptor.get_atoms(structure)
    raise TypeError(f"Expected an ase.Atoms or a pymatgen Structure/Molecule, got {type(structure).__name__}")


class QuasiGraph:
    """
    Chemical and geometric descriptor of an atomic structure.

    The structure itself is the ``atoms`` attribute.  (QuasiGraph used to be a
    subclass of ``ase.Atoms``; ``ase.Atoms`` attributes and methods are still
    forwarded to ``atoms``, with a ``DeprecationWarning``.)
    """

    def __init__(self,
                 atoms: Any,
                 pbc: bool | Sequence[bool] | None = None,
                 tolerance: float = 0.4,
                 temperature: float = 0.1,
                 normalization: bool = True,
                 show_bonded_atoms: bool = False,
                 nmax: int | None = None,
                 chemical_features: Sequence[str] | None = None,
                 geometric_features: Sequence[str] | None = None,
                 backend: str = 'dense',
                 cn_max: float | None = None) -> None:
        """
        Initialize the QuasiGraph object.

        Parameters:
        -----------
        atoms : ase.Atoms or pymatgen Structure/Molecule
            The atomic structure (see ``to_atoms``).
        pbc : bool or list of bool, optional
            Periodic boundary conditions along the three axes.  Default (None)
            takes them from ``atoms.pbc``; pass e.g. ``False`` to treat a
            periodic cell as an isolated cluster.  For a periodic axis, as many
            lattice translations are included as needed to find every bond.
        tolerance : float, optional
            Tolerance value for determining bonded atoms (default is 0.4).
        temperature : float, optional
            Width (in angstrom) of the sigmoid used by the ``CN_smooth`` feature,
            centred on the bond threshold (1 + tolerance) * (r_i + r_j).  0 gives
            a step function and reproduces the standard CN (default is 0.1).
        normalization : bool, optional
            Whether to divide the GCN by a reference coordination number:
            ``cn_max`` if given, otherwise the largest CN in the structure
            (default is True).
        show_bonded_atoms : bool, optional
            Whether to add a ``bonded_atoms`` column to ``get_dataframe`` (default is False).
        nmax : int or None, optional
            Number of rows to which ``get_dataframe`` (and ``get_vector``) is
            zero-padded, so that structures of different sizes give vectors of
            equal length (default is None: no padding).
        chemical_features : list of str, optional
            List of chemical features to consider (default is
            ``DEFAULT_CHEMICAL_FEATURES``: 'VEC', 'atomic_radius', 'en_pauling',
            'electron_affinity'; see ``AVAILABLE_FEATURES``).
        geometric_features : list of str, optional
            List of per-atom geometric features to consider (default is
            ``DEFAULT_GEOMETRIC_FEATURES``: 'CN', 'GCN').
            Available: 'CN', 'GCN', 'CN_smooth', 'bond_mean', 'bond_min', 'bond_max',
            'bond_std', 'bond_strain' (see ``GEOMETRIC_FEATURES``).
        backend : {'dense', 'vesin'}, optional
            How bonded pairs are found.  'dense' (default) builds the full
            distance matrix (a tensor over periodic images for a periodic
            structure), whose memory grows as n_atoms**2.  'vesin' uses the
            sparse neighbour list of the vesin library, linear in n_atoms, and
            does not build ``distances``, ``distances_tensor``, ``bonds`` or
            ``bond_threshold``.  Both give the same descriptor, except for the
            ``CN_smooth``, whose sigmoid sum 'vesin' stops 20 * ``temperature``
            beyond the bond threshold (differences ~1e-8).
        cn_max : float or None, optional
            Reference coordination number by which the GCN is divided, e.g. 12
            for fcc metals as in Calle-Vallejo et al.  A fixed value keeps the
            GCN comparable between structures; the default (None) uses the
            largest CN of each structure.

        Attributes:
        -----------
        atoms : object
            The atomic structure object.
        pbc : (3,) array of bool
            Periodic boundary conditions.
        tolerance : float
            Tolerance value for determining bonded atoms.
        normalization : bool
            Whether the GCN is normalized.
        show_bonded_atoms : bool
            Whether to show bonded atoms.
        nmax : int or None
            Maximum number of neighbors to consider.
        chemical_features : list of str
            List of chemical features to consider.
        offsets : list of int
            Number of lattice translations included along each axis (0 for a
            non-periodic axis; periodic case only).
        distances_list : list
            List of ``[i, j, offset, distance]`` entries considering periodic boundary
            conditions (periodic case only, built on first access).
        distances_tensor : tensor
            Tensor of distances considering periodic boundary conditions (periodic case only).
        distances : array
            Array of distances between atoms (non-periodic case only).
        bonds : array
            Boolean bond matrix (non-periodic) or bond tensor over offsets (periodic).
        pairs : dict of arrays
            Bonds as a pair list (``backend='vesin'`` only): ``i``, ``j``,
            ``shifts`` (lattice translation of atom j) and ``distances``, with
            every bond listed in both directions.
        cn : array
            Coordination numbers of atoms.
        bonded_atoms : list
            List of bonded atoms.
        gcn : array
            Generalized coordination numbers of atoms.
        """
        if backend not in ('dense', 'vesin'):
            raise ValueError(f"backend must be 'dense' or 'vesin', not {backend!r}")
        self.backend = backend
        atoms = to_atoms(atoms)
        self.atoms = atoms
        self.pbc = np.broadcast_to(np.asarray(atoms.pbc if pbc is None else pbc, dtype=bool), 3).copy()
        self.tolerance: float = tolerance
        self.temperature: float = temperature
        self.normalization: bool = normalization
        self.show_bonded_atoms: bool = show_bonded_atoms
        self.nmax = nmax
        self.cn_max = cn_max
        self.chemical_features = list(DEFAULT_CHEMICAL_FEATURES if chemical_features is None else chemical_features)
        self.geometric_features = list(DEFAULT_GEOMETRIC_FEATURES if geometric_features is None else geometric_features)
        self._geometric_table: dict[str, np.ndarray] | None = None

        positions = self.atoms.get_positions()
        radii = self.covalent_radii
        cutoff = self._bond_cutoff(radii)

        if any(self.pbc):
            lengths = self.atoms.cell.lengths()
            if (lengths[self.pbc] == 0).any() or abs(np.linalg.det(self.atoms.cell.complete())) < 1e-12:
                raise ValueError(f"pbc={self.pbc.tolist()} needs a non-degenerate cell vector along every "
                                 f"periodic axis (cell lengths {lengths.tolist()}); set the cell or pass pbc=False.")
            self.offsets = geometry.periodic_image_range(self.atoms.cell.complete(), positions, self.pbc, cutoff)

        if self.backend == 'vesin':
            i, j, shifts, distances = neighbors.neighbor_pairs(positions, self.atoms.cell.complete(), self.pbc, cutoff)
            bonded = distances <= neighbors.pair_bond_threshold(radii, self.tolerance, i, j)
            self.pairs = {'i': i[bonded], 'j': j[bonded], 'shifts': shifts[bonded], 'distances': distances[bonded]}
            self.cn = neighbors.coordination_numbers(self.pairs['i'], len(self.atoms))
            self.bonded_atoms = neighbors.neighbor_lists(self.pairs['i'], self.pairs['j'], len(self.atoms))
        else:
            self.bond_threshold = geometry.bond_threshold(radii, self.tolerance)
            threshold = self.bond_threshold
            if any(self.pbc):
                self.distances_tensor = geometry.distance_tensor(positions, self.atoms.cell, self.get_offsets())
                self.bonds = geometry.bonds_from_distance_tensor(self.distances_tensor, threshold)
            else:
                self.distances = geometry.distance_matrix(positions)
                self.bonds = geometry.bonds_from_distance_matrix(self.distances, threshold)
            self.cn = geometry.coordination_numbers(self.bonds)
            self.bonded_atoms = geometry.neighbor_lists(self.adjacency)
        self.gcn = self.get_gcn()

    def _bond_cutoff(self, radii):
        """Longest possible bond length, (1 + tolerance) * 2 * the largest covalent radius."""
        return (1 + self.tolerance) * 2 * float(np.max(radii)) if len(radii) else 0.0

    @property
    def adjacency(self) -> np.ndarray:
        """Boolean (n_atoms, n_atoms) matrix telling whether atoms i and j share at least one bond."""
        if self.backend == 'vesin':
            return neighbors.adjacency(self.pairs['i'], self.pairs['j'], len(self.atoms))
        return geometry.adjacency(self.bonds)

    @property
    def chemical_symbols(self) -> list[str]:
        """Chemical symbols of the wrapped Atoms object."""
        return self.atoms.get_chemical_symbols()

    @property
    def covalent_radii(self) -> np.ndarray:
        """Covalent radius (angstrom) of every atom."""
        return elements.covalent_radii(self.chemical_symbols)

    def get_offsets(self) -> list[list[int]]:
        return geometry.periodic_offsets(self.offsets)

    @property
    def distances_list(self) -> list[list[Any]]:
        """
        Flat list of ``[i, j, offset, distance]`` for every atom pair and periodic
        offset, in the same order as ``distances_tensor``.  Only available for
        periodic structures; built lazily because it is large and rarely needed.
        """
        try:
            tensor = self.distances_tensor
        except AttributeError:
            raise AttributeError("'distances_list' is only available for periodic structures with backend='dense'")
        if not hasattr(self, '_distances_list'):
            offsets_list = self.get_offsets()
            n_atoms = len(self.atoms)
            self._distances_list = [[i, j, offsets_list[n], tensor[n, i, j]]
                                    for n in range(len(offsets_list))
                                    for i in range(n_atoms)
                                    for j in range(n_atoms)]
        return self._distances_list

    def get_gcn(self) -> list[float]:
        if self.backend == 'vesin':
            return list(neighbors.generalized_coordination_numbers(self.pairs['i'], self.pairs['j'], self.cn,
                                                                   self.normalization, self.cn_max))
        multiplicity = geometry.bond_multiplicity(self.bonds)
        gcn = geometry.generalized_coordination_numbers(multiplicity, self.cn, self.normalization, self.cn_max)
        return list(gcn)

    @property
    def pair_distances(self) -> np.ndarray:
        """
        (n_atoms, n_atoms) distance matrix.  For periodic structures this is the
        minimum-image distance (smallest distance over all lattice offsets).
        Built on request with ``backend='vesin'``.
        """
        if self.backend == 'vesin':
            positions = self.atoms.get_positions()
            if any(self.pbc):
                return geometry.distance_tensor(positions, self.atoms.cell, self.get_offsets()).min(axis=0)
            return geometry.distance_matrix(positions)
        if any(self.pbc):
            return self.distances_tensor.min(axis=0)
        return self.distances

    @property
    def _bond_distances(self):
        """Distances with the same shape as ``self.bonds``."""
        return self.distances_tensor if any(self.pbc) else self.distances

    def get_geometric_features(self) -> dict[str, np.ndarray]:
        """
        Per-atom geometric features as a dict of name -> (n_atoms,) array, for
        every name in ``GEOMETRIC_FEATURES`` (computed once, on first use).
        """
        if self._geometric_table is None and self.backend == 'vesin':
            self._geometric_table = self._geometric_features_from_pairs()
        if self._geometric_table is None:
            distances, bonds, radii = self._bond_distances, self.bonds, self.covalent_radii
            table = {'CN': self.cn, 'GCN': self.gcn, 'CN_smooth': self._dense_smooth_cn()}
            table.update(geometry.bond_length_statistics(distances, bonds))
            table['bond_strain'] = geometry.bond_strain(distances, bonds, radii)
            self._geometric_table = table
        return self._geometric_table

    def _dense_smooth_cn(self):
        """
        ``CN_smooth`` from dense distances.  For a periodic structure, the
        lattice translations reach 20 * ``temperature`` beyond the longest bond,
        so that the sigmoid tail is not cut off in small or skewed cells.
        """
        if not any(self.pbc):
            return geometry.smooth_coordination_numbers(self.distances, self.bond_threshold, self.temperature)
        positions = self.atoms.get_positions()
        cutoff = self._bond_cutoff(self.covalent_radii) + 20 * self.temperature
        offsets_list = geometry.periodic_offsets(
            geometry.periodic_image_range(self.atoms.cell.complete(), positions, self.pbc, cutoff))
        if offsets_list == self.get_offsets():
            distances = self.distances_tensor
        else:
            distances = geometry.distance_tensor(positions, self.atoms.cell, offsets_list)
        return geometry.smooth_coordination_numbers(distances, self.bond_threshold, self.temperature)

    def _geometric_features_from_pairs(self):
        """``get_geometric_features`` from the vesin pair list."""
        n_atoms, radii = len(self.atoms), self.covalent_radii
        i, j, distances = self.pairs['i'], self.pairs['j'], self.pairs['distances']
        # CN_smooth also needs the pairs just beyond the bond threshold, where the sigmoid weight is not yet negligible
        si, sj, _, sd = neighbors.neighbor_pairs(self.atoms.get_positions(), self.atoms.cell.complete(), self.pbc,
                                                 self._bond_cutoff(radii) + 20 * self.temperature)
        threshold = neighbors.pair_bond_threshold(radii, self.tolerance, si, sj)
        table = {'CN': self.cn, 'GCN': self.gcn,
                 'CN_smooth': neighbors.smooth_coordination_numbers(si, sd, threshold, self.temperature, n_atoms)}
        table.update(neighbors.bond_length_statistics(i, distances, n_atoms))
        table['bond_strain'] = neighbors.bond_strain(i, j, distances, radii, n_atoms)
        return table

    @staticmethod
    def validate_geometric_features(features: Sequence[str]) -> None:
        for feature in features:
            if feature not in GEOMETRIC_FEATURES:
                raise ValueError(f"Geometric feature '{feature}' is not recognized. "
                                 f"Available features: {GEOMETRIC_FEATURES}")

    def _atom_table(self):
        """One row per atom: the selected chemical features followed by the selected geometric features."""
        elements.validate_features(self.chemical_features)
        self.validate_geometric_features(self.geometric_features)

        # Chemical data: one row per atom, one column per feature (cached per element symbol)
        atoms_data = [{feature: elements.get_feature(symbol, feature) for feature in self.chemical_features}
                      for symbol in self.chemical_symbols]
        df = pd.DataFrame(atoms_data)

        # Geometric data
        geometric = self.get_geometric_features()
        for feature in self.geometric_features:
            df[feature] = geometric[feature]
        return df

    def _resolve_site(self, site):
        """Turn ``site`` (atom index or a chemical symbol occurring exactly once) into an atom index."""
        symbols = self.chemical_symbols
        if isinstance(site, str):
            matches = [i for i, symbol in enumerate(symbols) if symbol == site]
            if len(matches) != 1:
                raise ValueError(f"Site symbol '{site}' occurs {len(matches)} times; pass an atom index instead.")
            return matches[0]
        return range(len(symbols))[site]

    def get_site_environment(self, site: int | str, shells: int = 2,
                             elements: Sequence[str] | None = None) -> pd.Series:
        """
        Local-environment descriptor of one atom (e.g. an adsorbate) as a
        pandas Series of named features.

        Parameters:
        -----------
        site : int or str
            Atom index (negative indices allowed) or a chemical symbol that
            occurs exactly once in the structure (e.g. ``'H'``).
        shells : int, optional
            Number of neighbour shells to aggregate (default 2).  Shell k holds
            the atoms k bonds away from the site on the bond graph; in a
            periodic structure every image of an atom is a separate node.
        elements : list of str, optional
            Element symbols for the per-element neighbour counts.  Pass the
            same list for every structure of a dataset to get vectors of equal
            length; default is the sorted set of symbols in this structure.

        Returns:
        --------
            pandas.Series with, for the site, ``site_<feature>`` for every
            selected chemical and geometric feature, and for every shell k:
            ``shell<k>_n`` (atom count), ``shell<k>_n_<El>`` (count per element),
            ``shell<k>_dist_mean/min/max`` (distance from the site),
            ``shell<k>_<feature>_mean`` for chemical features and
            ``shell<k>_<feature>_mean/min/max`` for geometric features.
            Statistics of an empty shell are 0.
        """
        index = self._resolve_site(site)
        table = self._atom_table()
        symbols = np.asarray(self.chemical_symbols)
        if elements is None:
            elements = sorted(set(symbols.tolist()))
        positions = self.atoms.get_positions()
        if any(self.pbc):
            offsets_list, cell = self.get_offsets(), np.asarray(self.atoms.cell)
        else:
            offsets_list, cell = [[0, 0, 0]], np.zeros((3, 3))

        features = {f'site_{name}': value for name, value in table.iloc[index].items()}
        if self.backend == 'vesin':
            image_shells = geometry.pair_neighbor_shells(self.pairs['i'], self.pairs['j'], self.pairs['shifts'],
                                                         len(self.atoms), index, shells)
        else:
            image_shells = geometry.image_neighbor_shells(self.bonds, offsets_list, index, shells)
        for k, (shell, images) in enumerate(image_shells, start=1):
            prefix = f'shell{k}_'
            has_atoms = len(shell) > 0
            features[prefix + 'n'] = len(shell)
            for element in elements:
                features[prefix + f'n_{element}'] = int(np.count_nonzero(symbols[shell] == element))
            d = np.linalg.norm(positions[shell] + images @ cell - positions[index], axis=1)
            features[prefix + 'dist_mean'] = d.mean() if has_atoms else 0.0
            features[prefix + 'dist_min'] = d.min() if has_atoms else 0.0
            features[prefix + 'dist_max'] = d.max() if has_atoms else 0.0
            subset = table.iloc[shell]
            for name in self.chemical_features:
                column = pd.to_numeric(subset[name], errors='coerce')
                features[prefix + f'{name}_mean'] = column.mean() if has_atoms else 0.0
            for name in self.geometric_features:
                column = subset[name].to_numpy(dtype=float)
                features[prefix + f'{name}_mean'] = column.mean() if has_atoms else 0.0
                features[prefix + f'{name}_min'] = column.min() if has_atoms else 0.0
                features[prefix + f'{name}_max'] = column.max() if has_atoms else 0.0
        return pd.Series(features)

    def get_site_vector(self, site: int | str, shells: int = 2,
                        elements: Sequence[str] | None = None) -> np.ndarray:
        """``get_site_environment`` as a flat float vector."""
        return self.get_site_environment(site, shells=shells, elements=elements).to_numpy(dtype=float)

    def get_structure_features(self, elements: Sequence[str] | None = None,
                               statistics: Sequence[str] = STRUCTURE_STATISTICS) -> pd.Series:
        """
        Descriptor of the whole structure as a pandas Series of named features
        that depends neither on the order nor on the number of atoms.

        Parameters:
        -----------
        elements : list of str, optional
            Element symbols for the composition fractions.  Pass the same list
            for every structure of a dataset to get vectors of equal length;
            default is the sorted set of symbols in this structure.
        statistics : sequence of str, optional
            Statistics of every per-atom feature over the atoms, among 'mean',
            'min', 'max' and 'std' (population standard deviation; default all).

        Returns:
        --------
            pandas.Series with ``n_atoms``, ``frac_<El>`` (fraction of atoms of
            each element in ``elements``) and ``<feature>_<statistic>`` for
            every selected chemical and geometric feature.  A feature missing
            for some atom (NaN, see ``MissingFeatureWarning``) gives NaN
            statistics.
        """
        functions: dict[str, Callable[[np.ndarray], Any]] = {'mean': np.mean, 'min': np.min, 'max': np.max,
                                                              'std': np.std}
        unknown = [name for name in statistics if name not in functions]
        if unknown:
            raise ValueError(f"Unknown statistics {unknown}; available: {list(functions)}")
        table = self._atom_table()
        symbols = np.asarray(self.chemical_symbols)
        if elements is None:
            elements = sorted(set(symbols.tolist()))
        n_atoms = len(symbols)

        features: dict[str, float] = {'n_atoms': n_atoms}
        for element in elements:
            features[f'frac_{element}'] = np.count_nonzero(symbols == element) / n_atoms if n_atoms else 0.0
        for name in table.columns:
            column = table[name].to_numpy(dtype=float)
            for statistic in statistics:
                features[f'{name}_{statistic}'] = float(functions[statistic](column)) if n_atoms else 0.0
        return pd.Series(features)

    def get_structure_vector(self, elements: Sequence[str] | None = None,
                             statistics: Sequence[str] = STRUCTURE_STATISTICS) -> np.ndarray:
        """``get_structure_features`` as a flat float vector."""
        return self.get_structure_features(elements=elements, statistics=statistics).to_numpy(dtype=float)

    def get_dataframe(self) -> pd.DataFrame:
        """
        Per-atom descriptor as a pandas DataFrame with one row per atom (in the
        order of ``atoms``) and one column for each selected chemical feature
        followed by each selected geometric feature, plus ``bonded_atoms`` if
        ``show_bonded_atoms`` is set.  With ``nmax``, zero rows are appended up
        to ``nmax`` rows.
        """
        df = self._atom_table()
        if self.show_bonded_atoms:
            df['bonded_atoms'] = self.bonded_atoms

        if self.nmax:
            if len(self.atoms) > self.nmax:
                raise ValueError("The Atoms object has more atoms than the nmax value. Increase the value of nmax.")
            lines_to_fill = self.nmax - len(self.atoms)
            columns = list(df.keys())
            num_columns = len(columns)
            values_zeros = np.zeros([lines_to_fill, num_columns])
            df_zeros = pd.DataFrame(values_zeros)
            df_zeros.columns = columns
            df_filled = pd.concat([df, df_zeros], ignore_index=True)
            return df_filled
        else:
            return df


    def get_vector(self) -> np.ndarray:
        """
        Per-atom descriptor (``get_dataframe``) flattened row by row; see
        ``get_feature_names`` for the name of every entry.
        """
        df = self.get_dataframe()
        return df.values.flatten()

    def get_feature_names(self) -> list[str]:
        """Names of the entries of ``get_vector``: ``atom<k>_<feature>``, row by row."""
        df = self.get_dataframe()
        return [f'atom{k}_{name}' for k in range(len(df)) for name in df.columns]

    def get_bonds(self) -> pd.DataFrame:
        """
        Bonds as a pandas DataFrame with columns ``i``, ``j``, ``shift_a``,
        ``shift_b``, ``shift_c`` (lattice translation of atom j) and
        ``distance``, one row per bond and direction, sorted by ``i``, ``j``
        and translation.  The same for both backends; a ready-made edge list
        for graph models.
        """
        if self.backend == 'vesin':
            i, j, shifts, distances = (self.pairs[key] for key in ('i', 'j', 'shifts', 'distances'))
        else:
            bonds = self.bonds if self.bonds.ndim == 3 else self.bonds[np.newaxis]
            distances = self._bond_distances if self.bonds.ndim == 3 else self.distances[np.newaxis]
            k, i, j = np.nonzero(bonds)
            offsets = np.asarray(self.get_offsets() if any(self.pbc) else [[0, 0, 0]], dtype=int).reshape(-1, 3)
            shifts, distances = offsets[k], distances[k, i, j]
        shifts = np.asarray(shifts, dtype=int).reshape(-1, 3)
        order = np.lexsort((shifts[:, 2], shifts[:, 1], shifts[:, 0], j, i))
        return pd.DataFrame({'i': np.asarray(i, dtype=int)[order], 'j': np.asarray(j, dtype=int)[order],
                             'shift_a': shifts[order, 0], 'shift_b': shifts[order, 1], 'shift_c': shifts[order, 2],
                             'distance': np.asarray(distances, dtype=float)[order]})

    def to_networkx(self) -> Any:
        """
        Bond graph as a ``networkx.MultiGraph`` (requires networkx).  Nodes are
        atom indices with the attribute ``symbol`` and the columns of
        ``get_dataframe`` (without padding); edges carry ``distance`` and
        ``shift``.  In a periodic structure, bonds to different images of the
        same atom are parallel edges (or self-loops for the atom's own images).
        """
        import networkx as nx

        graph = nx.MultiGraph()
        table = self._atom_table()
        for index, (symbol, row) in enumerate(zip(self.chemical_symbols, table.to_dict('records'))):
            graph.add_node(index, symbol=symbol, **row)
        for bond in self.get_bonds().itertuples(index=False):
            shift = (bond.shift_a, bond.shift_b, bond.shift_c)
            # every bond is listed in both directions; keep one of them
            if bond.i < bond.j or (bond.i == bond.j and shift > (0, 0, 0)):
                graph.add_edge(bond.i, bond.j, distance=bond.distance, shift=shift)
        return graph

    def __len__(self) -> int:
        return len(self.atoms)

    def __getattr__(self, name: str) -> Any:
        # Only called for attributes QuasiGraph does not have.  Forward the
        # public ase.Atoms API to the wrapped structure (deprecated).
        if hasattr(type(self), name):
            # a QuasiGraph property raised AttributeError: report its own message
            return object.__getattribute__(self, name)
        if name.startswith('_') or name == 'atoms' or not hasattr(Atoms, name):
            raise AttributeError(f"'QuasiGraph' object has no attribute '{name}'")
        warnings.warn(f"QuasiGraph is no longer an ase.Atoms object; use QuasiGraph.atoms.{name} instead of "
                      f"QuasiGraph.{name}.", DeprecationWarning, stacklevel=2)
        return getattr(self.atoms, name)

    def __repr__(self) -> str:
        return (f"QuasiGraph(number_of_atoms={len(self.atoms)}, pbc={self.pbc.tolist()}, nmax={self.nmax}, "
                f"chemical_features={self.chemical_features}, geometric_features={self.geometric_features}, "
                f"backend={self.backend!r})")
