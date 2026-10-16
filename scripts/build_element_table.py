"""
Regenerate ``src/quasigraph/data/elements.json`` from the mendeleev database.

quasigraph reads its per-element properties from this bundled table, so that
results do not depend on the installed mendeleev version and no database is
queried at run time.  Run after upgrading mendeleev (``pip install mendeleev``)
and review the diff of the JSON file before committing it:

    python scripts/build_element_table.py
"""

import json
from pathlib import Path

import mendeleev
from mendeleev import element

# mendeleev attributes stored in the table, in their database units (radii in pm)
PROPERTIES = ['group_id', 'period', 'atomic_weight', 'covalent_radius', 'atomic_radius', 'vdw_radius',
              'en_pauling', 'electron_affinity', 'dipole_polarizability']
OUTPUT = Path(__file__).resolve().parent.parent / 'src' / 'quasigraph' / 'data' / 'elements.json'


def main():
    table = {}
    for atomic_number in range(1, 119):
        el = element(atomic_number)
        table[el.symbol] = {name: getattr(el, name) for name in PROPERTIES}
    data = {'source': f'mendeleev {mendeleev.__version__}', 'properties': PROPERTIES, 'elements': table}
    OUTPUT.write_text(json.dumps(data, indent=1) + '\n')
    print(f'wrote {len(table)} elements to {OUTPUT}')


if __name__ == '__main__':
    main()
