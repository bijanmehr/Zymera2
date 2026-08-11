def test_import_and_version():
    import zymera2
    assert isinstance(zymera2.__version__, str)
    assert zymera2.__version__.startswith("0.1.0")
