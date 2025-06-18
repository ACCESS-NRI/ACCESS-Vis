def test_import_accessvis():
    try:
        import accessvis  # noqa: F401
    except ImportError:
        assert False, "Failed to import the 'accessvis' package"
