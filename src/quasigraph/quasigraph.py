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
from mendeleev import element
from itertools import product
from .ptable import VEC

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
            Offsets for periodic boundary conditions.
        distances_list : list
            List of distances considering periodic boundary conditions.
        distances_tensor : tensor
            Tensor of distances considering periodic boundary conditions.
        cn : array
            Coordination numbers of atoms.
        bonded_atoms : list
            List of bonded atoms.
        distances : array
            Array of distances between atoms.
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
        if any(self.pbc):
            self.offsets = [int(offset) for offset in self.pbc]
            self.distances_list, self.distances_tensor = self.get_distances_pbc()
            self.cn, self.bonded_atoms = self.get_cn_pbc_vectorized()
        else:
            self.distances = self.atoms.get_all_distances()
            covalent_radii, positions = self.prepare_cn_data()
            self.cn, self.bonded_atoms = self.get_cn_nopbc_vectorized(covalent_radii, positions)
        self.gcn = self.get_gcn()
 

    def get_offsets(self):
        offset_basis = [list(range(-offset,offset+1)) for offset in self.offsets]
        combinations = list(product(*offset_basis))
        offsets_list = [list(comb) for comb in combinations]
        return offsets_list

    def get_distances_pbc(self):
        offsets_list = self.get_offsets()
        offsets_vec = [offsets_list[i]@self.atoms.cell for i in range(len(offsets_list))]
        distances_list = []
        distances_tensor = np.zeros([len(offsets_list), len(self.atoms), len(self.atoms)])
        for n, offset in enumerate(offsets_vec):
            for i, atom_i in enumerate(self.atoms):
                for j, atom_j in enumerate(self.atoms):
                    distance = np.linalg.norm( atom_j.position+offset - atom_i.position )
                    distances_list.append([i, j, offsets_list[n], distance])
                    distances_tensor[n,i,j] = distance

        return distances_list, distances_tensor 

    def get_distances_pbc_vectorized(self):
        offsets_list = self.get_offsets()
        # Assuming self.atoms.cell is a 3x3 matrix and offsets_list is a list of 3D vectors
        offsets_vec = np.dot(offsets_list, self.atoms.cell)  # Shape: (n_offsets, 3)

        # Assuming atom positions are stored in a structured array or similar
        positions = np.array([atom.position for atom in self.atoms])  # Shape: (n_atoms, 3)

        # Reshape positions for broadcasting: (1, n_atoms, 1, 3) and (n_offsets, 1, n_atoms, 3)
        pos_i = positions[np.newaxis, :, np.newaxis, :]  # Add axes for offsets and for j atoms
        pos_j = positions[np.newaxis, np.newaxis, :, :]  # Add axes for offsets and for i atoms
    
        # Reshape offsets for broadcasting: (n_offsets, 1, 1, 3)
        offsets_reshaped = offsets_vec[:, np.newaxis, np.newaxis, :]  # Prepare for broadcasting with positions

        # Calculate all distances using broadcasting. The new axis alignment allows for the calculation
        # of distances between all pairs of atoms, considering all offsets at once.
        # Resulting shape: (n_offsets, n_atoms, n_atoms, 3)
        distances = np.linalg.norm(pos_j + offsets_reshaped - pos_i, axis=-1)

        # Prepare distances_list in the expected format
        # This operation is inherently more complex to vectorize directly into the desired list format,
        # but we can efficiently create a similar structure
        n_offsets, n_atoms, _ = distances.shape
        offsets_expanded = np.repeat(offsets_list, n_atoms**2, axis=0).reshape(n_offsets, n_atoms, n_atoms, 3)
        i_indices, j_indices = np.indices((n_atoms, n_atoms))
        distances_list = np.stack((i_indices.ravel(), j_indices.ravel(), offsets_expanded.ravel(), distances.ravel()), axis=1)

        # distances_tensor is already in the correct shape: (n_offsets, n_atoms, n_atoms)
        distances_tensor = distances

        return distances_list.reshape(-1, 4), distances_tensor

    def prepare_cn_data(self):
        #Store Mendeleev data in memory
        atomic_symbols = set(self.atoms.get_chemical_symbols())
        cvr = {sym: element(sym).covalent_radius for sym in atomic_symbols}

        # Compute covalent radii array
        covalent_radii = np.array([cvr[atom.symbol] / 100 for atom in self.atoms])

        # Compute positions array
        positions = np.array([atom.position for atom in self.atoms])

        return covalent_radii, positions

    def get_cn_nopbc_vectorized(self, covalent_radii, positions):
        # Calculate distance matrix
        dist_matrix = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1)

        # Create threshold matrix
        threshold_matrix = (1 + self.tolerance) * (covalent_radii[:, np.newaxis] + covalent_radii)

        # Compare distances with thresholds
        bonding_matrix = (dist_matrix <= threshold_matrix)

        # Set diagonal to False to ignore self-bonding
        np.fill_diagonal(bonding_matrix,False)

        # Count bonded atoms
        cn = bonding_matrix.sum(axis=1)

        # Determine bonded atoms
        bonded_atoms = [list(np.where(row)[0]) for row in bonding_matrix]

        return cn, bonded_atoms


    def get_cn_pbc_vectorized(self):
        covalent_radii = np.array([element(atom.symbol).covalent_radius / 100 for atom in self.atoms])
        n_atoms = len(self.atoms)
    
        # Calculate the sum of covalent radii for all pairs (broadcasting to create a matrix of shape (n_atoms, n_atoms))
        sum_radii = covalent_radii[:, np.newaxis] + covalent_radii  # Shape: (n_atoms, n_atoms)
        sum_radii_with_tolerance = (1 + self.tolerance) * sum_radii
    
        # Now, compare each distance to the sum of covalent radii with tolerance, across all offsets
        # Using broadcasting to compare distances with sum_radii_with_tolerance
        bonded_matrix = (0 < self.distances_tensor) & (self.distances_tensor <= sum_radii_with_tolerance[np.newaxis, :, :])
    
        # Sum over the first axis (n_offsets) to count bonded instances, then over j to get the total count for each atom i
        cn = bonded_matrix.sum(axis=(0, 2))
    
        # To get bonded atoms, we need a bit more work since it's a list of lists. This part is inherently not fully vectorizable
        # due to the variable number of bonded atoms for each atom, but we can still avoid explicit Python loops over atoms
        bonded_atoms = []
        for i in range(n_atoms):
            # Flatten the matrix for atom i across all offsets and find indices (atoms) where bonding occurs
            bonded_indices = np.where(bonded_matrix[:, i, :].any(axis=0))[0]
            bonded_atoms.append(bonded_indices.tolist())
    
        return cn, bonded_atoms

    def get_gcn(self):
        cn, bonded_atoms = self.cn, self.bonded_atoms
        if self.normalization:
            norm_cn = max(cn, default=1)
        else:
            norm_cn = 1
        gcn = [sum(cn[j] for j in bonded_atoms[i]) / norm_cn for i in range(len(self.atoms))]
        return gcn

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
        
        # Mapping from feature name to a function that returns the property given an element symbol.
        feature_funcs = {
            'group': lambda sym: element(sym).group_id,
            'period': lambda sym: element(sym).period,
            'atomic_weight': lambda sym: element(sym).atomic_weight,
            'covalent_radius': lambda sym: element(sym).covalent_radius / 100,
            'atomic_radius': lambda sym: element(sym).atomic_radius / 100,
            'vdw_radius': lambda sym: element(sym).vdw_radius / 100,
            'en_pauling': lambda sym: element(sym).en_pauling,
            'electron_affinity': lambda sym: element(sym).electron_affinity,
            'dipole_polarizability': lambda sym: element(sym).dipole_polarizability,
            'VEC': lambda sym: VEC[sym] 
        }
        
        # Build the data for each atom
        atoms_data = []
        for atom in self.atoms:
            symbol = atom.symbol
            atom_props = {}
            for feature in self.chemical_features:
                if feature in feature_funcs:
                    atom_props[feature] = feature_funcs[feature](symbol)
                else:
                    raise ValueError(f"Feature '{feature}' is not recognized. Available features: {list(feature_funcs.keys())}")
            atoms_data.append(atom_props)

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
    
    # atoms={self.atoms.get_chemical_formula()}

