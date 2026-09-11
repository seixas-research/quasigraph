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


# ---------------------------------------------------------------------------
# Continuous bond information
# ---------------------------------------------------------------------------

def _neighbor_axes(bonds):
    """Axes of a bond matrix/tensor that enumerate the neighbours of each atom."""
    return (1,) if bonds.ndim == 2 else (0, 2)


def _per_atom(values, bonds):
    """Reshape a per-atom vector so it broadcasts along the atom-i axis of ``bonds``."""
    values = np.asarray(values)
    return values[:, np.newaxis] if bonds.ndim == 2 else values[np.newaxis, :, np.newaxis]


def smooth_coordination_numbers(distances, bonds, threshold, covalent_radii):
    """
    Coordination number with a continuous bond weight instead of a hard count.

    A bond of length ``d`` between atoms i and j contributes 1 when
    ``d <= r_i + r_j`` (sum of covalent radii, the ideal bond length) and then
    decays with a cosine switch to 0 at the bond threshold used for the hard
    CN.  Only pairs flagged in ``bonds`` contribute, so ``CN_smooth <= CN``.
    """
    covalent_radii = np.asarray(covalent_radii, dtype=float)
    ideal = covalent_radii[:, np.newaxis] + covalent_radii
    with np.errstate(divide='ignore', invalid='ignore'):
        x = (distances - ideal) / (threshold - ideal)
        switch = 0.5 * (1 + np.cos(np.pi * x))
    weights = np.where(distances <= ideal, 1.0, np.where(distances >= threshold, 0.0, switch))
    return (weights * bonds).sum(axis=_neighbor_axes(bonds))


def bond_length_statistics(distances, bonds):
    """
    Mean, minimum, maximum and standard deviation of the bond lengths of every
    atom, as a dict of (n_atoms,) arrays.  Atoms without bonds get 0 for all.
    """
    axes = _neighbor_axes(bonds)
    count = bonds.sum(axis=axes)
    has_bonds = count > 0
    safe_count = np.where(has_bonds, count, 1)

    mean = np.where(bonds, distances, 0.0).sum(axis=axes) / safe_count
    deviation = np.where(bonds, (distances - _per_atom(mean, bonds)) ** 2, 0.0)
    std = np.sqrt(deviation.sum(axis=axes) / safe_count)
    minimum = np.where(bonds, distances, np.inf).min(axis=axes)
    maximum = np.where(bonds, distances, -np.inf).max(axis=axes)

    zero = np.zeros_like(mean)
    return {
        'bond_mean': np.where(has_bonds, mean, zero),
        'bond_min': np.where(has_bonds, minimum, zero),
        'bond_max': np.where(has_bonds, maximum, zero),
        'bond_std': np.where(has_bonds, std, zero),
    }


def bond_strain(distances, bonds, covalent_radii):
    """
    Mean relative deviation of an atom's bond lengths from the ideal length
    ``r_i + r_j``: positive for stretched bonds, negative for compressed ones,
    0 for atoms without bonds.
    """
    covalent_radii = np.asarray(covalent_radii, dtype=float)
    ideal = covalent_radii[:, np.newaxis] + covalent_radii
    axes = _neighbor_axes(bonds)
    count = bonds.sum(axis=axes)
    ratio = np.where(bonds, distances / ideal, 0.0).sum(axis=axes)
    return np.where(count > 0, ratio / np.where(count > 0, count, 1) - 1, 0.0)


# ---------------------------------------------------------------------------
# Neighbour shells around a site (e.g. an adsorbate)
# ---------------------------------------------------------------------------

def neighbor_shells(adjacency_matrix, index, n_shells=2):
    """
    Indices of the atoms in the 1st, 2nd, ... ``n_shells``-th neighbour shell
    of atom ``index``, by breadth-first search on the bond graph.  Shell k
    holds the atoms whose shortest path to the site has exactly k bonds; the
    site itself never appears.
    """
    adjacency_matrix = np.asarray(adjacency_matrix, dtype=bool)
    visited = np.zeros(len(adjacency_matrix), dtype=bool)
    visited[index] = True
    frontier = np.array([index])
    shells = []
    for _ in range(n_shells):
        reachable = adjacency_matrix[frontier].any(axis=0) & ~visited
        frontier = np.flatnonzero(reachable)
        visited |= reachable
        shells.append(frontier)
    return shells
