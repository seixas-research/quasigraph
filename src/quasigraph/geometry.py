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


def periodic_image_range(cell, positions, pbc, cutoff):
    """
    Number of lattice translations to enumerate along each cell axis so that
    every pair of atoms closer than ``cutoff`` is found (input for
    ``periodic_offsets``).

    Along axis i, with lattice-plane spacing d_i, an image n cells away is at
    least ``|ds + n| * d_i`` from an atom, where ``ds`` is the difference of the
    fractional coordinates of the two atoms.  So ``|n| <= cutoff / d_i + spread``
    suffices, with ``spread`` the extent of the fractional coordinates along
    that axis (below 1 for wrapped positions; unwrapped positions are handled
    too).  Non-periodic axes get 0; periodic axes at least 1, so that
    minimum-image distances stay available.  ``cell`` must be non-singular
    (complete the vectors of non-periodic axes, e.g. ``atoms.cell.complete()``).
    """
    cell = np.asarray(cell, dtype=float)
    pbc = np.asarray(pbc, dtype=bool)
    reciprocal = np.linalg.inv(cell)  # column i is the reciprocal vector b_i (without 2 pi)
    spacing = 1 / np.linalg.norm(reciprocal, axis=0)
    scaled = np.asarray(positions, dtype=float).reshape(-1, 3) @ reciprocal
    spread = scaled.max(axis=0) - scaled.min(axis=0) if len(scaled) else np.zeros(3)
    n_images = np.maximum(1, np.floor(cutoff / spacing + spread + 1e-9).astype(int))
    return np.where(pbc, n_images, 0).tolist()


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


def bond_multiplicity(bonds):
    """
    Integer (n, n) matrix with the number of bonds between atoms i and j.
    Periodic images of the same neighbour count separately, so in a small
    cell an atom can be bonded several times to j (or to itself).
    """
    if bonds.ndim == 2:
        return bonds.astype(int)
    return bonds.sum(axis=0)


def generalized_coordination_numbers(adjacency_matrix, cn, normalization=True, cn_max=None):
    """
    GCN of every atom: the sum of the coordination numbers of its neighbours,
    divided, when ``normalization`` is true, by ``cn_max`` or by default by
    the largest coordination number in the structure (no division when no
    atom has a bond).

    ``adjacency_matrix`` gives the number of bonds between atoms i and j; pass
    ``bond_multiplicity(bonds)`` for a periodic structure so that every
    periodic image of a neighbour is counted, as in the CN.
    """
    cn = np.asarray(cn)
    norm_cn = _gcn_norm(cn, normalization, cn_max)
    neighbor_cn_sum = np.asarray(adjacency_matrix, dtype=cn.dtype) @ cn
    return neighbor_cn_sum / norm_cn


def _gcn_norm(cn, normalization, cn_max):
    """Divisor of the GCN (see ``generalized_coordination_numbers``)."""
    if not normalization:
        return 1
    if cn_max is not None:
        if cn_max <= 0:
            raise ValueError("cn_max must be positive")
        return cn_max
    norm_cn = cn.max() if cn.size else 1
    return norm_cn if norm_cn != 0 else 1


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


def smooth_coordination_numbers(distances, threshold, temperature):
    """
    Coordination number with a continuous bond weight instead of a hard count.

    Every pair (i, j) contributes a Fermi-type sigmoid weight centred on the
    bond threshold ``r_c = (1 + tolerance) * (r_i + r_j)`` used by the hard CN::

        w_ij = 1 / (1 + exp((d_ij - r_c) / temperature))

    ``temperature`` (in the same length unit as the distances) sets the width
    of the transition: w = 0.5 at the threshold, ~0.95 one ``3 * temperature``
    inside it and ~0.05 one ``3 * temperature`` outside it.  As
    ``temperature -> 0`` the weight becomes the step ``d_ij <= r_c`` and the
    smooth CN equals the standard CN; ``temperature = 0`` returns that limit
    exactly.  An atom never contributes to itself (zero distances are skipped,
    as in the hard CN), and periodic images count separately.
    """
    distances = np.asarray(distances, dtype=float)
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if temperature == 0:
        weights = (distances <= threshold).astype(float)
    else:
        # sigmoid(x) written with tanh to avoid overflow in exp for large |x|
        x = (threshold - distances) / temperature
        weights = 0.5 * (1 + np.tanh(0.5 * x))
    if weights.ndim == 2:
        np.fill_diagonal(weights, 0.0)
    else:
        weights = np.where(distances > 0, weights, 0.0)
    return weights.sum(axis=_neighbor_axes(weights))


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


def image_neighbor_shells(bonds, offsets_list, index, n_shells=2):
    """
    Neighbour shells of atom ``index`` (in the home cell) on the bond graph in
    which every periodic image of an atom is a separate node.  This differs
    from ``neighbor_shells`` in small cells, where several images of one atom
    (or of the site itself) can be neighbours.

    ``bonds`` is a (n_offsets, n, n) bond tensor over ``offsets_list``, or a
    (n, n) bond matrix for a non-periodic structure (``offsets_list`` is then
    ignored).  Returns one ``(atoms, images)`` pair per shell, as
    ``pair_neighbor_shells``.
    """
    if bonds.ndim == 2:
        bonds, offsets_list = bonds[np.newaxis], [[0, 0, 0]]
    offsets = np.asarray(offsets_list, dtype=int).reshape(-1, 3)
    k_idx, i_idx, j_idx = np.nonzero(bonds)
    return pair_neighbor_shells(i_idx, j_idx, offsets[k_idx], bonds.shape[1], index, n_shells)


def pair_neighbor_shells(i, j, shifts, n_atoms, index, n_shells=2):
    """
    Neighbour shells of atom ``index`` (in the home cell) on the periodic bond
    graph given as a list of bonds: atom ``j`` translated by the integer
    lattice vector ``shifts[b]`` is bonded to atom ``i`` for every bond ``b``.

    Returns one ``(atoms, images)`` pair per shell: ``atoms`` holds the (n_k,)
    atom indices and ``images`` the (n_k, 3) integer lattice translations of
    the atoms whose shortest path to the site has exactly k bonds, sorted by
    atom index and translation.
    """
    neighbors = [[] for _ in range(n_atoms)]
    for a, b, shift in zip(np.asarray(i).tolist(), np.asarray(j).tolist(), np.asarray(shifts, dtype=int).tolist()):
        neighbors[a].append((b, tuple(shift)))

    origin = (0, 0, 0)
    visited = {(index, origin)}
    frontier = [(index, origin)]
    shells = []
    for _ in range(n_shells):
        reached = set()
        for atom, image in frontier:
            for b, shift in neighbors[atom]:
                node = (b, tuple(x + y for x, y in zip(image, shift)))
                if node not in visited:
                    visited.add(node)
                    reached.add(node)
        frontier = sorted(reached)
        shells.append((np.array([atom for atom, _ in frontier], dtype=int),
                       np.array([image for _, image in frontier], dtype=int).reshape(-1, 3)))
    return shells
