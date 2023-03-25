# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Apigen is used to generate API reference api.rst automatically.


Usage
-----
    $ python apigen.py
    $ python make.py --apigen
"""

import inspect
import pathlib

from traitlets import import_item
import spectrochempy
from spectrochempy.utils.packages import list_packages

__all__ = []

REFERENCE = pathlib.Path(__file__).parent / "userguide" / "reference"


# ======================================================================================
# Class Apigen
# ======================================================================================
class Apigen:
    """
    Generate api.rst
    """

    header = """
.. Generate API reference pages, but don't display these pages in tables.
.. Do not modify directly because this file is automatically generated
.. when the documentation is built.
.. Only classes and methods appearing in __all__ statements are scanned.

:orphan:

.. currentmodule:: spectrochempy
.. autosummary::
   :toctree: generated/

"""

    def __init__(self):
        entries = self.list_entries()
        self.write_api_rst(entries)

    @staticmethod
    def get_packages():
        pkgs = list_packages(spectrochempy)
        for pkg in pkgs[:]:
            if pkg.endswith(".api"):
                pkgs.remove(pkg)
        return pkgs

    def get_members(self, obj, objname, alls=None):
        res = []
        members = inspect.getmembers(obj)
        for member in members:
            _name, _type = member
            if (
                (alls is not None and _name not in alls)
                or str(_name).startswith("_")
                or not str(_type).startswith("<")
                or "HasTraits" in str(_type)
                or "cross_validation_lock" in str(_name)
                or not (
                    str(_type).startswith("<class")
                    or str(_type).startswith("<function")
                    or str(_type).startswith("<property")
                )
            ):
                # we keep only the members in __all__ but the constants
                # print(f">>>>>>>>>>>>>>>>   {_name}\t\t{_type}")
                continue
            else:

                if objname != "spectrochempy" and objname.split(".")[1:][0] in [
                    "core",
                    "analysis",
                    "utils",
                    "widgets",
                ]:

                    continue

                module = ".".join(objname.split(".")[1:])
                module = module + "." if module else ""
                print(f"{module}{_name}\t\t{_type}")

                res.append(f"{module}{_name}")

        return res

    def list_entries(self):

        pkgs = self.get_packages()

        results = []
        for pkg_name in pkgs:

            print(pkg_name)
            if "mcrals" in pkg_name:
                pass
            pkg = import_item(pkg_name)
            try:
                alls = getattr(pkg, "__all__")
                print(f"\t__all__ : {alls}")

            except AttributeError:
                # warn("This module has no __all__ attribute")
                continue

            if alls == []:
                continue

            res = self.get_members(pkg, pkg_name, alls)
            results.extend(res)

        return results

    def write_api_rst(self, items):
        with open(REFERENCE / "api.rst", "w") as f:
            f.write(self.header)
            for item in items:
                f.write(f"    {item}\n")


# ======================================================================================
if __name__ == "__main__":
    api = Apigen()
