# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
import io
import subprocess


def test_show_versions() -> None:
    import spectrochempy as scp

    print(scp.show_versions)
    f = io.StringIO()
    scp.show_versions(file=f)
    assert "INSTALLED PACKAGES" in f.getvalue()
    assert "python" in f.getvalue()
    assert "numpy" in f.getvalue()
    assert "pint" in f.getvalue()
    assert "matplotlib" in f.getvalue()
    assert "pytest" in f.getvalue()
    assert "SPECTROCHEMPY" in f.getvalue()


# def test_show_versions_script() -> None:
#     result = subprocess.run(
#         ["python", "-m", "spectrochempy", "show_versions"],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         text=True,
#     )
#     output = result.stdout
#     assert "INSTALLED PACKAGES" in output
#     assert "python" in output
#     assert "numpy" in output
#     assert "pint" in output
#     assert "matplotlib" in output
#     assert "pytest" in output
#     assert "SPECTROCHEMPY" in output


if __name__ == "__main__":
    test_show_versions()
    test_show_versions_script()
