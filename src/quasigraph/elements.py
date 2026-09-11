# -*- coding: utf-8 -*-
# file: elements.py

# This code is part of quasigraph.
# MIT License
#
# Copyright (c) 2023 Leandro Seixas Rocha <leandro.fisica@gmail.com>

"""
Per-element chemical properties.

Every property is fetched from the ``mendeleev`` database, which is an
SQLAlchemy ORM.  A single ``mendeleev.element(symbol)`` call costs roughly
0.2 s because it loads the element together with all of its related tables,
so lookups are cached per chemical symbol for the lifetime of the process.
"""

from functools import lru_cache

import numpy as np
from mendeleev import element as _mendeleev_element

from .ptable import VEC


@lru_cache(maxsize=None)
def get_element(symbol):
    """Return the ``mendeleev`` Element for ``symbol``, loading it from the database only once."""
    return _mendeleev_element(symbol)


# Mapping from feature name to a function that returns the property given an element symbol.
FEATURE_FUNCTIONS = {
    'group': lambda sym: get_element(sym).group_id,
    'period': lambda sym: get_element(sym).period,
    'atomic_weight': lambda sym: get_element(sym).atomic_weight,
    'covalent_radius': lambda sym: get_element(sym).covalent_radius / 100,
    'atomic_radius': lambda sym: get_element(sym).atomic_radius / 100,
    'vdw_radius': lambda sym: get_element(sym).vdw_radius / 100,
    'en_pauling': lambda sym: get_element(sym).en_pauling,
    'electron_affinity': lambda sym: get_element(sym).electron_affinity,
    'dipole_polarizability': lambda sym: get_element(sym).dipole_polarizability,
    'VEC': lambda sym: VEC[sym],
}

AVAILABLE_FEATURES = list(FEATURE_FUNCTIONS.keys())


def validate_features(features):
    """Raise ``ValueError`` if any name in ``features`` is not a known chemical feature."""
    for feature in features:
        if feature not in FEATURE_FUNCTIONS:
            raise ValueError(f"Feature '{feature}' is not recognized. Available features: {AVAILABLE_FEATURES}")


@lru_cache(maxsize=None)
def get_feature(symbol, feature):
    """Return the value of chemical ``feature`` for the element ``symbol`` (cached)."""
    validate_features([feature])
    return FEATURE_FUNCTIONS[feature](symbol)


def covalent_radii(symbols):
    """Covalent radii in angstrom, one entry per symbol in ``symbols``."""
    return np.array([get_feature(symbol, 'covalent_radius') for symbol in symbols])
