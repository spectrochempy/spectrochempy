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
from traitlets.config import Config
from traitlets.config.configurable import Configurable
from traitlets.config.loader import LazyConfigValue

from spectrochempy.extern.traittypes import Empty, SciType
from spectrochempy.utils.docstrings import _docstring
from spectrochempy.utils.objects import Adict


class MetaConfigurable(Configurable):
    """
    A subclass of Configurable that stores configuration changes in a json file.

    Saving the configuration changes allows to retrieve them between different
    executions of the main application.
    """

    name = tr.Unicode()

    def __init__(self, **kwargs):
        # keep only the current config section
        reset = kwargs.pop("reset", False)
        parent = kwargs.get("parent")
        parent_config = parent.config
        config = Config()
        if self.name in parent_config and not reset:
            config = Config({self.name: parent_config[self.name]})
        # call the superclass __init__ is required
        super().__init__(parent=parent, config=config)
        # get the config manager
        self.cfg = self.parent.config_manager

    @tr.default("name")
    def _name_default(self):
        # this ensures a name has been defined for the subclassed model estimators
        # or an error is returned
        raise NameError(
            "The name of the object was not defined. It is necessary"
            "for writing configuration files"
        )

    def to_dict(self):
        """
        Return config value in a dict form.

        Returns
        -------
        `dict`
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
            config = self.config[self.name]
            if "shape" in config and isinstance(config["shape"], LazyConfigValue):
                del config["shape"]  # remove the lazy configurable object
            defaults.update(config)
        return defaults

    @_docstring.get_docstring(base="metaconfigurable.parameters")
    @_docstring.dedent
    def parameters(self, default=False):
        """
        Current or default configuration values.

        Parameters
        ----------
        default : `bool`, optional, default: `False`
            If `default` is `True`, the default parameters are returned,
            else the current values.

        Returns
        -------
        `dict`
            Current or default configuration values.
        """
        d = Adict()
        if not default:
            d.update(self.trait_values(config=True))
        else:
            d.update(self.trait_defaults(config=True))
        return d

    def reset(self):
        """
        Reset configuration parameters to their default values
        """
        # for this we need to remove the section corresponding
        # to the current configurable (i.e., self.name)
        if self.name in self.config:
            # remove this entry in config
            del self.config[self.name]
            # also delete the current JSON config file
            f = (Path(self.cfg.config_dir) / self.name).with_suffix(".json")
            f.unlink()

        # then set the default parameters
        for k, v in self.parameters(default=True).items():
            if getattr(self, k) != v:
                setattr(self, k, v)

    @tr.observe(tr.All)
    def _anytrait_changed(self, change):
        # update configuration after any change

        if not hasattr(self, "cfg"):
            # not yet initialized
            return

        if change.name in self.trait_names(config=True):

            value = change.new

            # Serialization of callable functions
            # (avoid recursive functions, though!)
            # for this we use the dill library
            # (see
            # https://medium.com/@greyboi/serialising-all-the-functions-in-python-cd880a63b591)
            if callable(value):
                import dill

                value = dill.dumps(value)
                # bytes are however not JSON serialisable: make an encoded string
                import base64

                value = base64.b64encode(value).decode()

            # replace other serializable value by an equivalent
            elif isinstance(value, (type(cycler), Path)):
                value = str(value)
            if isinstance(value, np.ndarray):
                # we need to transform it to a list of elements, bUT with python built-in
                # types, which is not the case e.g., for int64
                value = value.tolist()

            self.cfg.update(
                self.name,
                {
                    self.__class__.__name__: {
                        change.name: value,
                    }
                },
            )

            self.updated = True


class SpectroChemPyType(SciType):
    """
    A SpectroChemPy trait type.
    """

    info_text = "a Spectrochempy object"

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
            default_value=default_value,
            allow_none=allow_none,
            **kwargs,
        )
        self.metadata.update({"dtype": dtype})


class CoordType(SpectroChemPyType):
    """
    A NDDataset trait type.
    """

    info_text = "a SpectroChemPy coordinates object"

    def __init__(self, default_value=Empty, allow_none=False, dtype=None, **kwargs):
        if "klass" not in kwargs and self.klass is None:
            from spectrochempy.core.dataset.coord import Coord

            kwargs["klass"] = Coord
        super().__init__(default_value=default_value, allow_none=allow_none, **kwargs)
        self.metadata.update({"dtype": dtype})


# ======================================================================================
if __name__ == "__main__":
    pass
