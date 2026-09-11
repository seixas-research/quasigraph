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


class QuasiGraph(Atoms):
    def __init__(self,
                 atoms,
                 pbc=[False, False, False],
                 tolerance=0.4,
                 normalization=True,
                 show_bonded_atoms=False,
                 nmax=None,
                 chemical_features=['VEC', 'atomic_radius', 'en_pauling', 'electron_affinity']):
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

        positions = self.atoms.get_positions()
        threshold = geometry.bond_threshold(self.covalent_radii, self.tolerance)

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
            pandas.DataFrame: A DataFrame with one row per atom and columns for each selected feature and geometric data.
        """
        elements.validate_features(self.chemical_features)

        # Chemical data: one row per atom, one column per feature (cached per element symbol)
        atoms_data = [{feature: elements.get_feature(symbol, feature) for feature in self.chemical_features}
                      for symbol in self.chemical_symbols]
        df = pd.DataFrame(atoms_data)

        # Geometric data
        df['CN'] = self.cn
        df['GCN'] = self.gcn
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
