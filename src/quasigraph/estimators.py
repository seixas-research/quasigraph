# -*- coding: utf-8 -*-
# file: estimators.py

# This code is part of quasigraph.
# MIT License
#
# Copyright (c) 2023 Leandro Seixas Rocha <leandro.fisica@gmail.com>

"""
scikit-learn transformer that turns a list of structures into a feature
matrix, so that quasigraph can be the first step of a ``Pipeline`` (requires
scikit-learn).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .batch import featurize
from .quasigraph import STRUCTURE_STATISTICS, to_atoms


class QuasiGraphFeaturizer(TransformerMixin, BaseEstimator):
    """
    Featurize structures (``ase.Atoms`` or pymatgen objects) with quasigraph.

    ``fit`` fixes the columns from the training structures (the element list
    and, for ``kind='atoms'``, the padding length); ``transform`` returns a
    float array with one row per structure, named by ``get_feature_names_out``.
    Use ``set_output(transform='pandas')`` to get a DataFrame instead.

    Parameters:
    -----------
    kind : {'structure', 'site', 'atoms'}
        Descriptor of every structure, as in ``quasigraph.featurize``.
    site, shells, statistics :
        As in ``quasigraph.featurize``.
    elements : list of str or None
        Element symbols of the composition / neighbour-count columns; None
        learns the sorted set of symbols of the training structures.
    n_jobs : int
        Worker processes of ``featurize``.
    pbc, tolerance, temperature, normalization, cn_max, chemical_features, geometric_features, backend :
        Passed to ``QuasiGraph``.
    """

    def __init__(self, kind='structure', site=None, shells=2, statistics=STRUCTURE_STATISTICS, elements=None,
                 n_jobs=1, pbc=None, tolerance=0.4, temperature=0.1, normalization=True, cn_max=None,
                 chemical_features=None, geometric_features=None, backend='dense'):
        self.kind = kind
        self.site = site
        self.shells = shells
        self.statistics = statistics
        self.elements = elements
        self.n_jobs = n_jobs
        self.pbc = pbc
        self.tolerance = tolerance
        self.temperature = temperature
        self.normalization = normalization
        self.cn_max = cn_max
        self.chemical_features = chemical_features
        self.geometric_features = geometric_features
        self.backend = backend

    def _featurize(self, structures, n_jobs):
        options = dict(pbc=self.pbc, tolerance=self.tolerance, temperature=self.temperature,
                       normalization=self.normalization, cn_max=self.cn_max,
                       chemical_features=self.chemical_features, geometric_features=self.geometric_features,
                       backend=self.backend)
        if self.kind == 'atoms':
            options['nmax'] = self.nmax_
        return featurize(structures, kind=self.kind, n_jobs=n_jobs, elements=self.elements_, site=self.site,
                         shells=self.shells, statistics=self.statistics, **options)

    def fit(self, X: Iterable[Any], y: Any = None) -> QuasiGraphFeaturizer:
        """Learn the element list (and padding length) from the structures ``X``."""
        structures = [to_atoms(structure) for structure in X]
        if not structures:
            raise ValueError("QuasiGraphFeaturizer needs at least one structure to fit")
        if self.elements is None:
            self.elements_ = sorted({s for atoms in structures for s in atoms.get_chemical_symbols()})
        else:
            self.elements_ = list(self.elements)
        self.nmax_ = max(len(atoms) for atoms in structures)
        self.feature_names_out_ = list(self._featurize(structures[:1], n_jobs=1).columns)
        self.n_features_out_ = len(self.feature_names_out_)
        return self

    def transform(self, X: Iterable[Any]) -> np.ndarray:
        """Feature matrix (n_structures, n_features) of the structures ``X``."""
        if not hasattr(self, 'feature_names_out_'):
            raise AttributeError("QuasiGraphFeaturizer is not fitted yet; call fit first")
        structures = [to_atoms(structure) for structure in X]
        if self.kind == 'atoms' and max((len(atoms) for atoms in structures), default=0) > self.nmax_:
            raise ValueError(f"A structure has more atoms than the {self.nmax_} seen in fit")
        df = self._featurize(structures, n_jobs=self.n_jobs)
        if list(df.columns) != self.feature_names_out_:
            raise ValueError("The structures give other features than in fit")
        return df.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Names of the columns of ``transform``."""
        return np.asarray(self.feature_names_out_, dtype=object)
