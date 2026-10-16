import warnings

import numpy as np
import pytest

from quasigraph.elements import AVAILABLE_FEATURES, FEATURE_FUNCTIONS, element_table, get_feature


def test_bundled_table_matches_mendeleev():
    mendeleev = pytest.importorskip("mendeleev")
    table = element_table()
    assert len(table) == 118
    for atomic_number in range(1, 119):
        el = mendeleev.element(atomic_number)
        for name, value in table[el.symbol].items():
            assert getattr(el, name) == value, (el.symbol, name)


def test_features_come_from_the_bundled_table():
    assert get_feature("Pt", "covalent_radius") == element_table()["Pt"]["covalent_radius"] / 100
    assert get_feature("O", "en_pauling") == 3.44
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for feature in AVAILABLE_FEATURES:
            assert isinstance(get_feature("Pt", feature), float)
        assert np.isnan(get_feature("Ne", "en_pauling"))


def test_unknown_element_is_rejected():
    with pytest.raises(ValueError, match="Unknown chemical element 'Xx'"):
        FEATURE_FUNCTIONS["en_pauling"]("Xx")
