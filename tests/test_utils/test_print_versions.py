import io

import spectrochempy as scp


def test_show_versions() -> None:
    f = io.StringIO()
    scp.show_versions(file=f)
    assert "INSTALLED VERSIONS" in f.getvalue()
    assert "python:" in f.getvalue()
