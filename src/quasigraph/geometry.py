# -*- coding: utf-8 -*-
# file: geometry.py

# This code is part of quasigraph.
# MIT License
#
# Copyright (c) 2023 Leandro Seixas Rocha <leandro.fisica@gmail.com>

"""
Geometric part of the descriptor: interatomic distances, bonds, coordination
numbers (CN) and generalized coordination numbers (GCN).

All functions are pure NumPy and operate on plain arrays, so they can be
tested and reused independently of ASE.
"""

from itertools import product

import numpy as np


def periodic_offsets(offsets):
    """
    All integer lattice translations to consider for periodic images.

    ``offsets`` holds one non-negative integer per cell axis (0 for a
    non-periodic axis, 1 for a periodic one).  Returns a list of ``[a, b, c]``
    lists in ``itertools.product`` order.
    """
    offset_basis = [list(range(-offset, offset + 1)) for offset in offsets]
    return [list(comb) for comb in product(*offset_basis)]


def distance_matrix(positions):
    """(n_atoms, n_atoms) matrix of distances ``|r_i - r_j|``."""
    positions = np.asarray(positions, dtype=float)
    return np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1)


def distance_tensor(positions, cell, offsets_list):
    """
    (n_offsets, n_atoms, n_atoms) tensor of distances ``|r_j + T - r_i|`` for
    every lattice translation ``T`` built from ``offsets_list`` and ``cell``.
    """
    positions = np.asarray(positions, dtype=float)
    offsets_vec = np.array([offset @ cell for offset in offsets_list], dtype=float).reshape(-1, 3)
    displacements = (positions[np.newaxis, np.newaxis, :, :]
                     + offsets_vec[:, np.newaxis, np.newaxis, :]
                     - positions[np.newaxis, :, np.newaxis, :])
    return np.linalg.norm(displacements, axis=-1)


def bond_threshold(covalent_radii, tolerance):
    """(n_atoms, n_atoms) matrix of maximum bond lengths ``(1 + tol) * (r_i + r_j)``."""
    covalent_radii = np.asarray(covalent_radii, dtype=float)
    return (1 + tolerance) * (covalent_radii[:, np.newaxis] + covalent_radii)


def bonds_from_distance_matrix(distances, threshold):
    """Boolean (n_atoms, n_atoms) bond matrix; an atom is never bonded to itself."""
    bonds = distances <= threshold
    np.fill_diagonal(bonds, False)
    return bonds


def bonds_from_distance_tensor(distances, threshold):
    """
    Boolean (n_offsets, n_atoms, n_atoms) bond tensor.  Zero distances (an atom
    and its own image in the home cell) are never bonds.
    """
    return (0 < distances) & (distances <= threshold[np.newaxis, :, :])


def coordination_numbers(bonds):
    """
    Number of bonds per atom.  ``bonds`` is either a (n, n) matrix or a
    (n_offsets, n, n) tensor; periodic images of the same neighbour count
    separately.
    """
    if bonds.ndim == 2:
        return bonds.sum(axis=1)
    return bonds.sum(axis=(0, 2))


def adjacency(bonds):
    """Boolean (n, n) matrix telling whether atoms i and j share at least one bond."""
    if bonds.ndim == 2:
        return bonds
    return bonds.any(axis=0)


def neighbor_lists(adjacency_matrix):
    """List (one entry per atom) of the indices of the atoms it is bonded to."""
    return [np.flatnonzero(row).tolist() for row in adjacency_matrix]


def generalized_coordination_numbers(adjacency_matrix, cn, normalization=True):
    """
    GCN of every atom: the sum of the coordination numbers of its neighbours,
    divided by the largest coordination number in the structure when
    ``normalization`` is true.
    """
    cn = np.asarray(cn)
    if normalization:
        norm_cn = cn.max() if cn.size else 1
    else:
        norm_cn = 1
    neighbor_cn_sum = np.asarray(adjacency_matrix, dtype=cn.dtype) @ cn
    return neighbor_cn_sum / norm_cn
