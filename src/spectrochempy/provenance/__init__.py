# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

# lazy_stub: skip

"""
Experimental, opt-in structured provenance substrate.

This namespace contains passive immutable values and a context-local ledger.
The current bounded runtime slices capture selected dataset and CenterTransformer
operations only. They provide no manifest or replay.
"""

from spectrochempy.provenance._capture import ProvenanceCapture
from spectrochempy.provenance._capture import ProvenanceLedger
from spectrochempy.provenance._models import OPERATION_RECORD_SCHEMA_ID
from spectrochempy.provenance._models import OPERATION_RECORD_SCHEMA_VERSION
from spectrochempy.provenance._models import ObjectRef
from spectrochempy.provenance._models import OperationRecord
from spectrochempy.provenance._models import OperationRef
from spectrochempy.provenance._models import ReferenceLink
from spectrochempy.provenance._models import ResultRef
from spectrochempy.provenance._models import StateRef
from spectrochempy.provenance._normalization import ProvenanceValidationError

__all__ = [
    "OPERATION_RECORD_SCHEMA_ID",
    "OPERATION_RECORD_SCHEMA_VERSION",
    "ObjectRef",
    "OperationRecord",
    "OperationRef",
    "ProvenanceCapture",
    "ProvenanceLedger",
    "ProvenanceValidationError",
    "ReferenceLink",
    "ResultRef",
    "StateRef",
]
