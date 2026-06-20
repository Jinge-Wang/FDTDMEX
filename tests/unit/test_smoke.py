"""Smoke test: the package imports and exposes a version.

Keeps the test suite collectable while the engine is still a scaffold.
"""

import pytest


@pytest.mark.unit
def test_import_and_version():
    import fdtdmex

    assert isinstance(fdtdmex.__version__, str)
    assert fdtdmex.__version__
