# -*- coding: utf-8 -*-
# file: elements.py

# This code is part of quasigraph.
# MIT License
#
# Copyright (c) 2023 Leandro Seixas Rocha <leandro.fisica@gmail.com>

"""
Per-element chemical properties.

The properties come from the ``mendeleev`` database, bundled with quasigraph
as ``data/elements.json`` (regenerate it with ``scripts/build_element_table.py``).
Reading the bundled table avoids the slow database queries (~0.2 s per
element) and makes the results independent of the installed ``mendeleev``
version; ``mendeleev`` itself is only needed by ``get_element``.

Values are returned as floats; a property missing from the database (e.g. the
Pauling electronegativity of Ne) becomes NaN, with a warning.
"""

import json
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

from .ptable import VEC

_TABLE_PATH = Path(__file__).resolve().parent / 'data' / 'elements.json'


class MissingFeatureWarning(UserWarning):
    """A chemical property is missing from the database and was replaced by NaN."""


@lru_cache(maxsize=None)
def element_table():
    """Bundled per-element properties: symbol -> {mendeleev attribute: value in database units}."""
    return json.loads(_TABLE_PATH.read_text())['elements']


@lru_cache(maxsize=None)
def get_element(symbol):
    """
    Return the ``mendeleev`` Element for ``symbol``, loading it from the
    database only once.  Requires the optional ``mendeleev`` package; the
    descriptor itself uses the bundled table instead.
    """
    try:
        from mendeleev import element
    except ImportError as error:
        raise ImportError("get_element needs mendeleev: pip install mendeleev") from error
    return element(symbol)


def _table_property(name, divisor=1):
    """Function returning property ``name`` of an element divided by ``divisor`` (None if missing)."""
    def lookup(symbol):
        try:
            value = element_table()[symbol][name]
        except KeyError:
            raise ValueError(f"Unknown chemical element '{symbol}'") from None
        return None if value is None else value / divisor
    return lookup


# Mapping from feature name to a function that returns the property (None if
# unknown) given an element symbol.  Radii are converted from pm to angstrom.
FEATURE_FUNCTIONS = {
    'group': _table_property('group_id'),
    'period': _table_property('period'),
    'atomic_weight': _table_property('atomic_weight'),
    'covalent_radius': _table_property('covalent_radius', 100),
    'atomic_radius': _table_property('atomic_radius', 100),
    'vdw_radius': _table_property('vdw_radius', 100),
    'en_pauling': _table_property('en_pauling'),
    'electron_affinity': _table_property('electron_affinity'),
    'dipole_polarizability': _table_property('dipole_polarizability'),
    'VEC': lambda sym: VEC.get(sym),
}

AVAILABLE_FEATURES = list(FEATURE_FUNCTIONS.keys())


def validate_features(features):
    """Raise ``ValueError`` if any name in ``features`` is not a known chemical feature."""
    for feature in features:
        if feature not in FEATURE_FUNCTIONS:
            raise ValueError(f"Feature '{feature}' is not recognized. Available features: {AVAILABLE_FEATURES}")


@lru_cache(maxsize=None)
def get_feature(symbol, feature):
    """
    Return the value of chemical ``feature`` for the element ``symbol`` as a
    float (cached).  A value missing from the database is NaN, with a warning
    issued once per element and feature.
    """
    validate_features([feature])
    value = FEATURE_FUNCTIONS[feature](symbol)
    if value is None:
        warnings.warn(f"Feature '{feature}' is not available for {symbol}; using NaN.", MissingFeatureWarning)
        return np.nan
    return float(value)


def covalent_radii(symbols):
    """Covalent radii in angstrom, one entry per symbol in ``symbols``."""
    radii = np.array([get_feature(symbol, 'covalent_radius') for symbol in symbols], dtype=float)
    missing = sorted({symbol for symbol, radius in zip(symbols, radii) if np.isnan(radius)})
    if missing:
        raise ValueError(f"No covalent radius for {missing}; bonds cannot be determined.")
    return radii
