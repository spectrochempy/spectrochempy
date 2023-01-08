import inspect
from warnings import warn

from traitlets import import_item

import spectrochempy
from spectrochempy.utils.packages import list_packages


def get_packages():
    pkgs = list_packages(spectrochempy)
    for pkg in pkgs[:]:
        if pkg.endswith(".api"):
            pkgs.remove(pkg)

    return pkgs


def write_api_rst(items):
    header = """
        .. Generate API reference pages, but don't display these in tables.

        :orphan:

        .. currentmodule:: spectrochempy
        .. autosummary::
           :toctree: generated/

        """

    with open("~temp/api.rst", "w") as f:

        f.write(header)

        for item in items:
            f.write(f"    {item}\n")


# if __name__ == "__main__":

pkgs = get_packages()

lmembers = []

for pkg in pkgs:
    print("*" * 88)
    print(pkg)
    print("*" * 88)

    objects = pkg.split(".")

    parent = ".".join(objects[:-1])
    module = objects[-1]

    pkg = import_item(pkg)
    try:
        all = getattr(pkg, "__all__")
        print(all)

    except AttributeError:
        warn("This module has no __all__ attribute")
        continue

    if all == []:
        continue

    members = inspect.getmembers(pkg)

    for member in members:
        _name, _type = member
        # if "HasTraits" in str(_type) \
        #     or "traitlets" in str(_type) \
        #     or "module" in str(type) \
        #     or  _name.startswith('_')  \
        #     or _name in ['cross_validation_lock',]:
        if _name not in all:
            continue
        else:
            print(f"{parent} . {_name} >>>  {str(_type)}")


pass
# str(type(getattr(getattr(sys.modules[pkgs[n]], "EFA"), "_f_ev")))
