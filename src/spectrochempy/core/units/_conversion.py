# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Internal conversion of quantities between Pint registries."""

from pint.errors import DimensionalityError
from pint.errors import UndefinedUnitError

from spectrochempy.core.units import Quantity


def _equivalent_registry_unit(foreign, local):
    """Check that the registries use the same dimension and root conversion."""
    if foreign.dimensionality != local.dimensionality:
        return False

    foreign_type = type(foreign)
    _, foreign_spec = foreign.to_tuple()
    _, local_spec = local.to_tuple()
    for reference in (0, 1, 2):
        foreign_root, foreign_units = (
            foreign_type.from_tuple((reference, foreign_spec))
            .to_root_units()
            .to_tuple()
        )
        local_root, local_units = (
            Quantity.from_tuple((reference, local_spec)).to_root_units().to_tuple()
        )
        if foreign_root != local_root or dict(foreign_units) != dict(local_units):
            return False
    return True


def _import_quantity(quantity, units):
    """Transfer a Pint scalar without an avoidable conversion through root units."""
    try:
        local_source = Quantity.from_tuple(quantity.to_tuple())
        if _equivalent_registry_unit(quantity, local_source):
            return local_source
    except UndefinedUnitError:
        pass

    local_target = Quantity(1, units)
    try:
        foreign_target = type(quantity).from_tuple(local_target.to_tuple())
        if _equivalent_registry_unit(foreign_target, local_target):
            return Quantity(quantity.to(foreign_target.units).magnitude, units)
    except (UndefinedUnitError, DimensionalityError):
        pass

    imported = Quantity.from_tuple(quantity.to_root_units().to_tuple())
    if imported.dimensionality != quantity.dimensionality:
        raise DimensionalityError(
            quantity.units,
            imported.units,
            quantity.dimensionality,
            imported.dimensionality,
        )
    return imported
