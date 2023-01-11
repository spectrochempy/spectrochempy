import inspect
from warnings import warn

from traitlets import import_item

import spectrochempy
from spectrochempy.utils.packages import list_packages

header = """.. Generate API reference pages, but don't display these in tables.

:orphan:

.. currentmodule:: spectrochempy
.. autosummary::
   :toctree: generated/

"""


def get_packages():
    pkgs = list_packages(spectrochempy)
    for pkg in pkgs[:]:
        if pkg.endswith(".api"):
            pkgs.remove(pkg)
    return pkgs


def get_members(obj, objname, alls=None):
    res = []
    members = inspect.getmembers(obj)
    for member in members:
        _name, _type = member

        if (
            (alls is not None and _name not in alls)
            or not str(_type).startswith("<")
            or str(_name).startswith("_")
            or "HasTraits" in str(_type)
            or "cross_validation_lock" in str(_name)
            or not (
                str(_type).startswith("<class")
                or str(_type).startswith("<function")
                or str(_type).startswith("<property")
            )
        ):
            # we keep only the members in __all__ but the constants
            continue
        else:
            module = ".".join(objname.split(".")[1:])
            if module == "core":
                continue

            module = module + "." if module else ""
            print(f"{module}{_name}\t\t{_type}")
            #          o = import_item(f"{objname}")

            res.append(f"{module}{_name}")

            if str(_type).startswith("<class"):
                # find also members in class
                klass = getattr(obj, _name)

                subres = get_members(klass, objname + "." + _name)
                res.extend(subres)

    return res


def list_entries():

    pkgs = get_packages()

    results = []
    for pkg_name in pkgs:

        print()
        print("*" * 88)
        print(pkg_name)
        print("-" * len(pkg_name))

        pkg = import_item(pkg_name)
        try:
            alls = getattr(pkg, "__all__")
            print(alls)

        except AttributeError:
            warn("This module has no __all__ attribute")
            continue

        if alls == []:
            continue

        res = get_members(pkg, pkg_name, alls)
        results.extend(res)

    return results


def write_api_rst(items):
    with open("../docs/userguide/reference/api.rst", "w") as f:
        f.write(header)
        for item in items:
            f.write(f"    {item}\n")


if __name__ == "__main__":
    entries = list_entries()
    write_api_rst(entries)
