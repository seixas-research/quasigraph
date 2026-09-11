#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: quasigraph.py

# This code is part of quasigraph.
# MIT License
#
# Copyright (c) 2023 Leandro Seixas Rocha <leandro.fisica@gmail.com>


import numpy as np
import pandas as pd
from ase import Atoms

from . import elements, geometry


GEOMETRIC_FEATURES = ['CN', 'GCN', 'CN_smooth', 'bond_mean', 'bond_min', 'bond_max', 'bond_std', 'bond_strain']


class QuasiGraph(Atoms):
    def __init__(self,
                 atoms,
                 pbc=[False, False, False],
                 tolerance=0.4,
                 normalization=True,
                 show_bonded_atoms=False,
                 nmax=None,
                 chemical_features=['VEC', 'atomic_radius', 'en_pauling', 'electron_affinity'],
                 geometric_features=['CN', 'GCN']):
        """
        Initialize the QuasiGraph object.

        Parameters:
        -----------
        atoms : object
            The atomic structure object containing atom positions and other properties.
        pbc : list of bool, optional
            Periodic boundary conditions along the three axes (default is [False, False, False]).
        tolerance : float, optional
            Tolerance value for determining bonded atoms (default is 0.4).
        normalization : bool, optional
            Whether to normalize the chemical features (default is True).
        show_bonded_atoms : bool, optional
            Whether to show bonded atoms (default is False).
        nmax : int or None, optional
            Maximum number of neighbors to consider (default is None).
        chemical_features : list of str, optional
            List of chemical features to consider (default is ['VEC', 'atomic_radius', 'en_pauling', 'electron_affinity']).
        geometric_features : list of str, optional
            List of per-atom geometric features to consider (default is ['CN', 'GCN']).
            Available: 'CN', 'GCN', 'CN_smooth', 'bond_mean', 'bond_min', 'bond_max',
            'bond_std', 'bond_strain' (see ``GEOMETRIC_FEATURES``).

        Attributes:
        -----------
        atoms : object
            The atomic structure object.
        pbc : list of bool
            Periodic boundary conditions.
        tolerance : float
            Tolerance value for determining bonded atoms.
        normalization : bool
            Whether to normalize the chemical features.
        show_bonded_atoms : bool
            Whether to show bonded atoms.
        nmax : int or None
            Maximum number of neighbors to consider.
        chemical_features : list of str
            List of chemical features to consider.
        offsets : list of int
            Offsets for periodic boundary conditions (periodic case only).
        distances_list : list
            List of ``[i, j, offset, distance]`` entries considering periodic boundary
            conditions (periodic case only, built on first access).
        distances_tensor : tensor
            Tensor of distances considering periodic boundary conditions (periodic case only).
        distances : array
            Array of distances between atoms (non-periodic case only).
        bonds : array
            Boolean bond matrix (non-periodic) or bond tensor over offsets (periodic).
        cn : array
            Coordination numbers of atoms.
        bonded_atoms : list
            List of bonded atoms.
        gcn : array
            Generalized coordination numbers of atoms.
        """
        super().__init__()
        self.atoms = atoms
        self.pbc = pbc
        self.tolerance: float = tolerance
        self.normalization: bool = normalization
        self.show_bonded_atoms: bool = show_bonded_atoms
        self.nmax = nmax
        self.chemical_features = chemical_features
        self.geometric_features = geometric_features
        self._geometric_table = None

        positions = self.atoms.get_positions()
        self.bond_threshold = geometry.bond_threshold(self.covalent_radii, self.tolerance)
        threshold = self.bond_threshold

        if any(self.pbc):
            self.offsets = [int(offset) for offset in self.pbc]
            self.distances_tensor = geometry.distance_tensor(positions, self.atoms.cell, self.get_offsets())
            self.bonds = geometry.bonds_from_distance_tensor(self.distances_tensor, threshold)
        else:
            self.distances = geometry.distance_matrix(positions)
            self.bonds = geometry.bonds_from_distance_matrix(self.distances, threshold)

        self.cn = geometry.coordination_numbers(self.bonds)
        self.adjacency = geometry.adjacency(self.bonds)
        self.bonded_atoms = geometry.neighbor_lists(self.adjacency)
        self.gcn = self.get_gcn()

    @property
    def chemical_symbols(self):
        """Chemical symbols of the wrapped Atoms object."""
        return self.atoms.get_chemical_symbols()

    @property
    def covalent_radii(self):
        """Covalent radius (angstrom) of every atom."""
        return elements.covalent_radii(self.chemical_symbols)

    def get_offsets(self):
        return geometry.periodic_offsets(self.offsets)

    @property
    def distances_list(self):
        """
        Flat list of ``[i, j, offset, distance]`` for every atom pair and periodic
        offset, in the same order as ``distances_tensor``.  Only available for
        periodic structures; built lazily because it is large and rarely needed.
        """
        try:
            tensor = self.distances_tensor
        except AttributeError:
            raise AttributeError("'distances_list' is only available for periodic structures")
        if not hasattr(self, '_distances_list'):
            offsets_list = self.get_offsets()
            n_atoms = len(self.atoms)
            self._distances_list = [[i, j, offsets_list[n], tensor[n, i, j]]
                                    for n in range(len(offsets_list))
                                    for i in range(n_atoms)
                                    for j in range(n_atoms)]
        return self._distances_list

    def get_gcn(self):
        gcn = geometry.generalized_coordination_numbers(self.adjacency, self.cn, self.normalization)
        return list(gcn)

    @property
    def pair_distances(self):
        """
        (n_atoms, n_atoms) distance matrix.  For periodic structures this is the
        minimum-image distance (smallest distance over all lattice offsets).
        """
        if any(self.pbc):
            return self.distances_tensor.min(axis=0)
        return self.distances

    @property
    def _bond_distances(self):
        """Distances with the same shape as ``self.bonds``."""
        return self.distances_tensor if any(self.pbc) else self.distances

    def get_geometric_features(self):
        """
        Per-atom geometric features as a dict of name -> (n_atoms,) array, for
        every name in ``GEOMETRIC_FEATURES`` (computed once, on first use).
        """
        if self._geometric_table is None:
            distances, bonds, radii = self._bond_distances, self.bonds, self.covalent_radii
            table = {'CN': self.cn, 'GCN': self.gcn,
                     'CN_smooth': geometry.smooth_coordination_numbers(distances, bonds, self.bond_threshold, radii)}
            table.update(geometry.bond_length_statistics(distances, bonds))
            table['bond_strain'] = geometry.bond_strain(distances, bonds, radii)
            self._geometric_table = table
        return self._geometric_table

    @staticmethod
    def validate_geometric_features(features):
        for feature in features:
            if feature not in GEOMETRIC_FEATURES:
                raise ValueError(f"Geometric feature '{feature}' is not recognized. Available features: {GEOMETRIC_FEATURES}")

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

    def get_site_environment(self, site, shells=2, elements=None):
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
            the atoms k bonds away from the site on the bond graph.
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
        distances_from_site = self.pair_distances[index]

        features = {f'site_{name}': value for name, value in table.iloc[index].items()}
        for k, shell in enumerate(geometry.neighbor_shells(self.adjacency, index, shells), start=1):
            prefix = f'shell{k}_'
            has_atoms = len(shell) > 0
            features[prefix + 'n'] = len(shell)
            for element in elements:
                features[prefix + f'n_{element}'] = int(np.count_nonzero(symbols[shell] == element))
            d = distances_from_site[shell]
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

    def get_site_vector(self, site, shells=2, elements=None):
        """``get_site_environment`` as a flat float vector."""
        return self.get_site_environment(site, shells=shells, elements=elements).to_numpy(dtype=float)

    def get_dataframe(self):
        """
        Constructs a pandas DataFrame of element properties for the atoms.
        
        Parameters:
        -----------
            features (list of str, optional): List of property names to include.
                Available features are:
                    - 'group'
                    - 'period'
                    - 'atomic_weight'
                    - 'covalent_radius'
                    - 'atomic_radius'
                    - 'vdw_radius'
                    - 'en_pauling'
                    - 'electron_affinity'
                    - 'dipole_polarizability'
                    - 'VEC'
        
        Returns:
        --------
            pandas.DataFrame: A DataFrame with one row per atom and columns for each selected chemical
            feature followed by each selected geometric feature (default ``CN`` and ``GCN``).
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


    def get_vector(self):
        """
        QuasiGraph descriptor as a flatten vector.
        """
        df = self.get_dataframe()
        return df.values.flatten()
    
    
    def __repr__(self):
        return f"QuasiGraph(number_of_atoms={len(self.atoms)}, pbc={self.pbc}, nmax={self.nmax}, chemical_features={self.chemical_features})"
