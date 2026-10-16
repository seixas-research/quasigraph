# -*- coding: utf-8 -*-
# file: neighbors.py

# This code is part of quasigraph.
# MIT License
#
# Copyright (c) 2023 Leandro Seixas Rocha <leandro.fisica@gmail.com>

"""
Geometric part of the descriptor from a sparse neighbour list.

Instead of the dense (n_offsets, n_atoms, n_atoms) distance tensor used by
``geometry``, the pairs of atoms within a cutoff are found with the vesin
library (https://github.com/Luthaf/vesin), whose cost and memory grow
linearly with the number of atoms.  A pair list holds every pair in both
directions as arrays ``i``, ``j``, ``shifts`` and ``distances``, with
``distances = |r_j + shifts @ cell - r_i|`` as in ``geometry``.  The
functions below compute the same per-atom features as their dense
counterparts in ``geometry`` from such a list.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from vesin import NeighborList

from .geometry import _gcn_norm


def neighbor_pairs(positions: ArrayLike, cell: ArrayLike, pbc: ArrayLike,
                   cutoff: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    All pairs of atoms at most ``cutoff`` apart, periodic images included,
    as ``(i, j, shifts, distances)`` arrays with integer (int64) indices and
    (n_pairs, 3) integer lattice translations.  An atom is never paired with
    itself, but it is with its own periodic images.  ``cell`` must be
    non-singular (complete the vectors of non-periodic axes, e.g.
    ``atoms.cell.complete()``).
    """
    # vesin keeps pairs strictly closer than the cutoff; widen it slightly so
    # that a pair exactly at ``cutoff`` is kept, as with the dense d <= threshold
    calculator = NeighborList(cutoff=float(cutoff) * (1 + 1e-9) + 1e-12, full_list=True, sorted=True)
    i, j, shifts, distances = calculator.compute(points=np.asarray(positions, dtype=float),
                                                 box=np.asarray(cell, dtype=float),
                                                 periodic=[bool(p) for p in np.asarray(pbc)],
                                                 quantities="ijSd")
    keep = distances <= cutoff
    return i[keep].astype(np.int64), j[keep].astype(np.int64), shifts[keep].astype(np.int64), distances[keep]


def pair_bond_threshold(covalent_radii: ArrayLike, tolerance: float, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Maximum bond length ``(1 + tol) * (r_i + r_j)`` of every pair."""
    covalent_radii = np.asarray(covalent_radii, dtype=float)
    return (1 + tolerance) * (covalent_radii[i] + covalent_radii[j])


def coordination_numbers(i: np.ndarray, n_atoms: int) -> np.ndarray:
    """Number of bonds per atom from the ``i`` indices of a list of bonds."""
    return np.bincount(i, minlength=n_atoms)


def generalized_coordination_numbers(i: np.ndarray, j: np.ndarray, cn: ArrayLike, normalization: bool = True,
                                     cn_max: float | None = None) -> np.ndarray:
    """
    GCN from a list of bonds: the sum of the coordination numbers of the
    neighbours of every atom (each periodic image counted), normalized as in
    ``geometry.generalized_coordination_numbers``.
    """
    cn = np.asarray(cn)
    norm_cn = _gcn_norm(cn, normalization, cn_max)
    return np.bincount(i, weights=cn[j], minlength=len(cn)) / norm_cn


def neighbor_lists(i: np.ndarray, j: np.ndarray, n_atoms: int) -> list[list[int]]:
    """List (one entry per atom) of the sorted, distinct indices of the atoms it is bonded to."""
    pairs = np.unique(np.stack([i, j], axis=1), axis=0).reshape(-1, 2)
    splits = np.cumsum(np.bincount(pairs[:, 0], minlength=n_atoms))[:-1]
    return [chunk.tolist() for chunk in np.split(pairs[:, 1], splits)]


def adjacency(i: np.ndarray, j: np.ndarray, n_atoms: int) -> np.ndarray:
    """Boolean (n_atoms, n_atoms) matrix telling whether atoms i and j share at least one bond."""
    matrix = np.zeros((n_atoms, n_atoms), dtype=bool)
    matrix[i, j] = True
    return matrix


def smooth_coordination_numbers(i: np.ndarray, distances: ArrayLike, threshold: ArrayLike, temperature: float,
                                n_atoms: int) -> np.ndarray:
    """
    Continuous CN from a list of pairs, with the sigmoid weight of
    ``geometry.smooth_coordination_numbers`` for every pair.  The list must
    reach far enough past ``threshold`` for the neglected weights to be
    negligible (e.g. 20 * ``temperature``, weight ~2e-9).
    """
    distances = np.asarray(distances, dtype=float)
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if temperature == 0:
        weights = (distances <= threshold).astype(float)
    else:
        weights = 0.5 * (1 + np.tanh(0.5 * (threshold - distances) / temperature))
    return np.bincount(i, weights=weights, minlength=n_atoms)


def bond_length_statistics(i: np.ndarray, distances: np.ndarray, n_atoms: int) -> dict[str, np.ndarray]:
    """
    Mean, minimum, maximum and standard deviation of the bond lengths of every
    atom from a list of bonds, as a dict of (n_atoms,) arrays.  Atoms without
    bonds get 0 for all.
    """
    count = np.bincount(i, minlength=n_atoms)
    has_bonds = count > 0
    safe_count = np.where(has_bonds, count, 1)

    mean = np.bincount(i, weights=distances, minlength=n_atoms) / safe_count
    std = np.sqrt(np.bincount(i, weights=(distances - mean[i]) ** 2, minlength=n_atoms) / safe_count)
    minimum = np.full(n_atoms, np.inf)
    np.minimum.at(minimum, i, distances)
    maximum = np.full(n_atoms, -np.inf)
    np.maximum.at(maximum, i, distances)

    zero = np.zeros(n_atoms)
    return {
        'bond_mean': np.where(has_bonds, mean, zero),
        'bond_min': np.where(has_bonds, minimum, zero),
        'bond_max': np.where(has_bonds, maximum, zero),
        'bond_std': np.where(has_bonds, std, zero),
    }


def bond_strain(i: np.ndarray, j: np.ndarray, distances: np.ndarray, covalent_radii: ArrayLike,
                n_atoms: int) -> np.ndarray:
    """
    Mean relative deviation of an atom's bond lengths from the ideal length
    ``r_i + r_j`` from a list of bonds; 0 for atoms without bonds.
    """
    covalent_radii = np.asarray(covalent_radii, dtype=float)
    count = np.bincount(i, minlength=n_atoms)
    ratio = np.bincount(i, weights=distances / (covalent_radii[i] + covalent_radii[j]), minlength=n_atoms)
    return np.where(count > 0, ratio / np.where(count > 0, count, 1) - 1, 0.0)
