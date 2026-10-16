# -*- coding: utf-8 -*-
# file: batch.py

# This code is part of quasigraph.
# MIT License
#
# Copyright (c) 2023 Leandro Seixas Rocha <leandro.fisica@gmail.com>

"""
Descriptors of a whole dataset of structures as one pandas DataFrame, with
optional parallelism over processes.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from .quasigraph import QuasiGraph, STRUCTURE_STATISTICS, to_atoms

KINDS = ('structure', 'site', 'atoms')


def featurize(structures: Iterable[Any], kind: str = 'structure', n_jobs: int = 1,
              elements: Sequence[str] | None = None, site: int | str | None = None, shells: int = 2,
              statistics: Sequence[str] = STRUCTURE_STATISTICS, **kwargs: Any) -> pd.DataFrame:
    """
    Descriptor of every structure in ``structures`` as one row of a DataFrame.

    Parameters:
    -----------
    structures : iterable of ase.Atoms (or pymatgen Structure/Molecule)
        The dataset, e.g. ``ase.io.read('relax.extxyz', index=':')``.
    kind : {'structure', 'site', 'atoms'}, optional
        'structure' (default): ``QuasiGraph.get_structure_features``;
        'site': ``QuasiGraph.get_site_environment`` of ``site``;
        'atoms': the per-atom ``QuasiGraph.get_vector``, zero-padded to the
        largest structure unless ``nmax`` is given.
    n_jobs : int, optional
        Number of worker processes (default 1: no parallelism; -1: one per
        CPU).  On macOS and Windows, worker processes import the main script,
        so a script calling ``featurize`` with ``n_jobs != 1`` must do so
        under ``if __name__ == '__main__':``.  Parallelism pays off for large
        structures; for small ones the start-up cost of the workers dominates.
    elements : list of str, optional
        Element symbols of the composition / neighbour-count columns ('structure'
        and 'site'); default is the sorted set of symbols in the whole
        dataset, so that every row has the same columns.
    site : int or str
        Site of every structure (required for ``kind='site'``).
    shells : int, optional
        Neighbour shells of the site (``kind='site'``, default 2).
    statistics : sequence of str, optional
        Statistics over the atoms (``kind='structure'``).
    **kwargs
        Passed to ``QuasiGraph`` (e.g. ``geometric_features``, ``backend``,
        ``pbc``, ``cn_max``).

    Returns:
    --------
        pandas.DataFrame with one row per structure, in the order given.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    if kind == 'site' and site is None:
        raise ValueError("kind='site' needs a site (atom index or unique chemical symbol)")
    structures = [to_atoms(structure) for structure in structures]
    if elements is None and kind != 'atoms':
        elements = sorted({symbol for atoms in structures for symbol in atoms.get_chemical_symbols()})
    if kind == 'atoms' and kwargs.get('nmax') is None:
        kwargs['nmax'] = max((len(atoms) for atoms in structures), default=0)
    options = dict(kind=kind, elements=elements, site=site, shells=shells, statistics=statistics, kwargs=kwargs)
    tasks = [(atoms, options) for atoms in structures]

    if n_jobs == 1 or len(tasks) < 2:
        rows = [_featurize_one(task) for task in tasks]
    else:
        workers = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_featurize_one, tasks, chunksize=max(1, len(tasks) // (4 * workers))))

    if not rows:
        return pd.DataFrame()
    names = rows[0][0]
    return pd.DataFrame(np.vstack([values for _, values in rows]), columns=names)


def _featurize_one(task):
    """Feature names and values of one structure (module level, so that it can be sent to worker processes)."""
    atoms, options = task
    descriptor = QuasiGraph(atoms, **options['kwargs'])
    kind = options['kind']
    if kind == 'structure':
        series = descriptor.get_structure_features(elements=options['elements'], statistics=options['statistics'])
    elif kind == 'site':
        series = descriptor.get_site_environment(options['site'], shells=options['shells'],
                                                 elements=options['elements'])
    else:
        return descriptor.get_feature_names(), descriptor.get_vector().astype(float)
    return list(series.index), series.to_numpy(dtype=float)
