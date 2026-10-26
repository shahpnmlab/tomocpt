"""Shared test helpers (written at Gate 1 by the Test Author, frozen afterwards).

Import as ``utils.<module>``. ``tests/`` is on ``sys.path`` once ``tests/conftest.py`` is
loaded (pytest "prepend" import mode); gate conftests also insert it explicitly. Do not
import ``tests.utils``: a third-party package named ``tests`` in site-packages shadows the
repository's ``tests/``.

Give every ``tests/gateN/`` directory an ``__init__.py``: otherwise its ``conftest.py`` is
imported as the top-level module ``conftest`` and breaks the legacy tests, which do
``from conftest import CHUNK_SHAPE``.

Modules: ``wedge``, ``phantoms``, ``tiny``, ``legacy``.
"""
