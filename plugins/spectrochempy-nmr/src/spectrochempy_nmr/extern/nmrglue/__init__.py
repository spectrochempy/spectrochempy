# ======================================================================================
# Vendored NMRGlue reading functions (BSD 3-Clause, Jonathan J. Helmus).
#
# This package contains read-only subsets of nmrglue.fileio.bruker and
# nmrglue.fileio.varian, vendored to avoid a runtime dependency on the
# full nmrglue package.  See individual submodules for attribution.
# ======================================================================================

from ._bruker import create_blank_udic
from ._bruker import guess_udic
from ._bruker import read_fid
from ._bruker import read_pdata
from ._bruker import uc_from_udic
from ._varian import find_varian_shape
from ._varian import find_varian_torder
from ._varian import read_varian
from ._varian import read_varian_procpar

__all__ = [
    "create_blank_udic",
    "find_varian_shape",
    "find_varian_torder",
    "guess_udic",
    "read_fid",
    "read_pdata",
    "read_varian",
    "read_varian_procpar",
    "uc_from_udic",
]
