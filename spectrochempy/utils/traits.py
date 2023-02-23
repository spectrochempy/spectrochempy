# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import inspect
from pathlib import Path

import numpy as np
import traitlets as tr
from matplotlib import cycler
from traitlets.config.configurable import Configurable
from traitlets.config.loader import LazyConfigValue

from spectrochempy.extern.traittypes import Empty, SciType

__all__ = ["MetaConfigurable", "Range"]


class MetaConfigurable(Configurable):
    """
    A subclass of Configurable that stores configuration changes in a json file.

    Saving the configuration changes allows to retrieve them between different
    executions of the main application.
    """

    def __init__(self, section, **kwargs):  # lgtm[py/missing-call-to-init]

        super().__init__(**kwargs)

        self.cfg = self.parent.config_manager
        self.section = section

    def to_dict(self):
        """
        Return config value in a dict form.

        Returns
        -------
        dict
            A regular dictionary.
        """
        d = {}
        for k, v in self.traits(config=True).items():
            d[k] = v.default_value
        return d

    def trait_defaults(self, *names, **metadata):
        # override traitlets trait default to take into accound changes in the config file
        defaults = super().trait_defaults(*names, **metadata)
        # modify with the loaded external config
        if not names:  # full dictionary
            config = self.config[self.section]
            if "shape" in config and isinstance(config["shape"], LazyConfigValue):
                del config["shape"]  # remove the lazy configurable object
            defaults.update(config)
        return defaults

    @tr.observe(tr.All)
    def _anytrait_changed(self, change):
        # update configuration after any change

        if not hasattr(self, "cfg"):
            # not yet initialized
            return

        if change.name in self.traits(config=True):

            value = change.new
            # replace non serializable value by an equivalent
            if isinstance(value, (type(cycler), Path)):
                value = str(value)
            if isinstance(value, np.ndarray):
                # we need to transform it to a list of elements, bUT with python built-in
                # types, which is not the case e.g., for int64
                value = value.tolist()

            self.cfg.update(
                self.section,
                {
                    self.__class__.__name__: {
                        change.name: value,
                    }
                },
            )

            self.updated = True


# ======================================================================================
# Range trait type
# ======================================================================================
class Range(tr.List):
    """
    The trait-type Range.

    Create a trait with two values defining an ordered range of values.

    Parameters
    ----------
    default_value : SequenceType [ optional ]
        The default value for the Trait.  Must be list/tuple/set, and
        will be cast to the container type.
    """

    # Describe the trait type
    info_text = "An ordered interval trait."

    def length_error(self, obj, value):
        e = (
            "The '%s' trait of '%s' instance must be of length 2 exactly,"
            " but a value of %s was specified." % (self.name, type(obj), value)
        )
        raise tr.TraitError(e)

    def validate_elements(self, obj, value):
        if value is None or len(value) == 0:
            return
        length = len(value)
        if length != 2:
            self.length_error(obj, value)
        value.sort()
        value = super().validate_elements(obj, value)
        return value

    def validate(self, obj, value):

        value = super().validate(object, value)
        value = self.validate_elements(obj, value)

        return value


class SpectroChemPyType(SciType):
    """
    A SpectroChemPy trait type.
    """

    info_text = "a NDDataset"

    klass = None

    def validate(self, obj, value):
        if value is None and not self.allow_none:
            self.error(obj, value)
        if value is None or value is tr.Undefined:
            return super().validate(obj, value)
        try:
            value = self.klass(value)
        except (ValueError, TypeError) as e:
            raise tr.TraitError(e)
        return super().validate(obj, value)

    def set(self, obj, value):
        new_value = self._validate(obj, value)
        old_value = obj._trait_values.get(self.name, self.default_value)
        obj._trait_values[self.name] = new_value
        if (
            (old_value is None and new_value is not None)
            or (old_value is tr.Undefined and new_value is not tr.Undefined)
            or not (old_value == new_value)
        ):
            obj._notify_trait(self.name, old_value, new_value)

    def __init__(self, default_value=Empty, allow_none=False, klass=None, **kwargs):
        if klass is None:
            klass = self.klass
        if (klass is not None) and inspect.isclass(klass):
            self.klass = klass
        else:
            raise tr.TraitError(
                "The klass attribute must be a class" " not: %r" % klass
            )
        if default_value is Empty:
            default_value = klass()
        elif default_value is not None and default_value is not tr.Undefined:
            default_value = klass(default_value)
        super().__init__(default_value=default_value, allow_none=allow_none, **kwargs)

    def make_dynamic_default(self):
        if self.default_value is None or self.default_value is tr.Undefined:
            return self.default_value
        else:
            return self.default_value.copy()


class NDDatasetType(SpectroChemPyType):
    """
    A NDDataset trait type.
    """

    info_text = "a SpectroChemPy NDDataset"

    def __init__(self, default_value=Empty, allow_none=False, dtype=None, **kwargs):
        if "klass" not in kwargs and self.klass is None:
            from spectrochempy.core.dataset.nddataset import NDDataset

            kwargs["klass"] = NDDataset
        super().__init__(
            default_value=default_value, allow_none=allow_none, dtype=dtype, **kwargs
        )


class ProjectType(SpectroChemPyType):
    """
    A NDDataset trait type.
    """

    info_text = "a SpectroChemPy Project"

    def __init__(self, default_value=Empty, allow_none=False, dtype=None, **kwargs):
        if "klass" not in kwargs and self.klass is None:
            from spectrochempy.core.project.project import Project

            kwargs["klass"] = Project
        super().__init__(
            default_value=default_value, allow_none=allow_none, dtype=dtype, **kwargs
        )


# ======================================================================================
if __name__ == "__main__":
    pass
