"""Phase 0 smoke tests: verify that sarimax_rs builds and imports correctly."""


def test_import():
    """sarimax_rs module can be imported."""
    import sarimax_rs
    assert sarimax_rs is not None


def test_version():
    """version() returns the package version string."""
    import sarimax_rs
    v = sarimax_rs.version()
    assert isinstance(v, str)
    assert v == "0.1.0"
