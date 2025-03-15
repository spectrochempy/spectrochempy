# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

from pathlib import Path

from traitlets import import_item


def get_pkg_path(data_name, package=None):
    data_name = Path(data_name).as_posix()

    path = Path(import_item(package).__file__).parent

    if not data_name:
        return path

    # Handle nested paths correctly by joining all components
    components = data_name.split("/")
    for component in components:
        path = path / component

    # Return parent for any path that doesn't point to a directory
    # This matches the expected behavior in the tests
    if not path.is_dir():
        return path.parent

    return path
