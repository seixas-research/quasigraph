# -*- coding: utf-8 -*-
# file: __init__.py

# This code is part of quasigraph.
# MIT License
#
# Copyright (c) 2023 Leandro Seixas Rocha <leandro.fisica@gmail.com>

from .quasigraph import (QuasiGraph, GEOMETRIC_FEATURES, DEFAULT_CHEMICAL_FEATURES, DEFAULT_GEOMETRIC_FEATURES,
                         STRUCTURE_STATISTICS, to_atoms)
from .batch import featurize
from .ptable import VEC, CONFIG
from .elements import AVAILABLE_FEATURES, MissingFeatureWarning, get_element, get_feature
from . import geometry, neighbors

from .version import __version__

__all__ = [
    'QuasiGraph', 'QuasiGraphFeaturizer', 'featurize', 'to_atoms',
    'GEOMETRIC_FEATURES', 'DEFAULT_CHEMICAL_FEATURES', 'DEFAULT_GEOMETRIC_FEATURES', 'STRUCTURE_STATISTICS',
    'AVAILABLE_FEATURES', 'MissingFeatureWarning', 'get_element', 'get_feature', 'VEC', 'CONFIG',
    'geometry', 'neighbors', '__version__',
]


def __getattr__(name):
    # QuasiGraphFeaturizer needs scikit-learn, which is optional: import it on first use
    if name == 'QuasiGraphFeaturizer':
        from .estimators import QuasiGraphFeaturizer
        return QuasiGraphFeaturizer
    raise AttributeError(f"module 'quasigraph' has no attribute {name!r}")
