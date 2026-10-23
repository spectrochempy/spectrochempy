"""Regression tests for directory imports across file extensions (issue #1780)."""

import spectrochempy as scp
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils.objects import ScpObjectList

JDX = """##TITLE=generic
##JCAMP-DX=5.01
##DATA TYPE=INFRARED SPECTRUM
##XUNITS=1/CM
##YUNITS=ABSORBANCE
##FIRSTX=4000.0
##LASTX=3997.0
##XFACTOR=1.0
##YFACTOR=1.0
##NPOINTS=4
##XYDATA=(X++(Y..Y))
4000.0 1.0 0.9 0.8 0.7
##END
"""


def test_jcamp_directory_merge_across_extensions(tmp_path):
    first = tmp_path / "a.jdx"
    second = tmp_path / "b.dx"
    first.write_text(JDX)
    second.write_text(JDX)

    explicit = scp.read_jcamp(first, second, merge=True)
    discovered = scp.read_jcamp(directory=tmp_path, merge=True)
    assert isinstance(explicit, NDDataset)
    assert isinstance(discovered, NDDataset)
    assert discovered.shape == explicit.shape == (2, 4)


def test_jcamp_directory_merge_false_keeps_datasets(tmp_path):
    (tmp_path / "a.jdx").write_text(JDX)
    (tmp_path / "b.dx").write_text(JDX)

    discovered = scp.read_jcamp(directory=tmp_path, merge=False)
    assert isinstance(discovered, ScpObjectList)
    assert len(discovered) == 2
