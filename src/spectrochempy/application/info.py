# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Display the info string at API startup."""

import importlib
import subprocess

import numpy as np
import traitlets as tr
from setuptools_scm import get_version

__all__ = [
    "name",
    "icon",
    "description",
    "version",
    "__version__",
    "release",
    "release_date",
    "copyright",
    "url",
    "authors",
    "contributors",
    "license",
    "cite",
    "long_description",
]


class SCPInfo(tr.HasTraits):
    version = tr.Unicode(help="Version string of this package")
    release = tr.Unicode(help="Release version string of this package")
    release_date = tr.Unicode(help="Last release date of the package")
    copyright = tr.Unicode(help="Copyright string of the package")
    name = tr.Unicode("SpectroChemPy", help="Running name of the application")
    icon = tr.Unicode("scpy.png", help="Icon for the application")
    url = tr.Unicode(
        "https://www.spectrochempy.fr", help="URL for the documentation of this package"
    )
    authors = tr.Unicode(
        "C. Fernandez & A. Travert", help="Initial authors of this package"
    )
    contributors = tr.Unicode(
        "A. Ait Blal, W. Guérin, M. Mailänder", help="contributor(s) to this package"
    )
    # TODO: retrieve this automatically from a file
    license = tr.Unicode("CeCILL-B license", help="License of this package")
    cite = tr.Unicode(help="How to cite this package")
    description = tr.Unicode(
        "SpectroChemPy is a framework for processing, analysing and modelling "
        "Spectroscopic data for Chemistry with Python."
    )
    long_description = tr.Unicode(help="Extended description of this package")

    # ----------------------------------------------------------------------------------
    # Private methods and properties
    # ----------------------------------------------------------------------------------
    @tr.default("release")
    def _release_default(self):
        try:
            release = importlib.metadata.version("spectrochempy").split("+")[0]
            "Release version string of this package"
        except Exception:  # pragma: no cover
            # package is not installed
            release = "--not set--"
        return release

    @tr.default("version")
    def _version_default(self):
        try:
            version = get_version(root="..", relative_to=__file__)
            "Version string of this package"
        except LookupError:  # pragma: no cover
            version = self.release
        return version

    @tr.default("copyright")
    def _copyright_default(self):
        current_year = np.datetime64("now", "Y")
        right = f"2014-{current_year}"
        right += " - A.Travert & C.Fernandez @ LCS"
        return right

    @tr.default("release_date")
    def _release_date_default(self):
        cmd = ["git", "log", "-1", "--tags", "--date=short", "--format=%ad"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
        except FileNotFoundError:
            return "n.d."
        return result.stdout.strip()

    @tr.default("cite")
    def _cite_default(self):
        # TODO: automatic update
        return (
            f"Arnaud Travert & Christian Fernandez (2025) SpectroChemPy (version"
            f" {'.'.join(self.version.split('.')[0:2])}). "
            f"Zenodo. https://doi.org/10.5281/zenodo.3823841"
        )

    @tr.default("long_description")
    def _get_long_description(self):
        return f"""
                <p><strong>SpectroChemPy</strong> is a framework for processing,
                analysing and modelling
                 <strong>Spectro</>scopic data for <strong>Chem</strong>istry with
                 <strong>Py</strong>thon.
                 It is a cross platform software, running on Linux, Windows or OS X.</p>
                 <br><br>
                <strong>Version:</strong> {self.release}<br>
                <strong>Authors:</strong> {self.authors}<br>
                <strong>License:</strong> {self.license}<br>
                <div class='warning'> SpectroChemPy is still experimental and under
                active development.
                Its current design and functionalities are subject to major changes,
                reorganizations,
                bugs and crashes!!!. Please report any issues to the
                <a url='https://github.com/spectrochempy/spectrochempy/issues'>
                Issue Tracker<a>
                </div><br><br>
                When using <strong>SpectroChemPy</strong> for your own work,
                you are kindly requested to cite it this way:
                <pre>{self.cite}</pre></p>.
                """

    @staticmethod
    def display_info_string(**kwargs):  # pragma: no cover
        from IPython.display import clear_output
        from IPython.display import publish_display_data
        from jinja2 import Template

        _template = """
        {{widgetcss}}
        <div>
        <table>
        <tr>
        <td>
        {% if logo %}
        <img src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAA
        AXNSR0IArs4c6QAAAAlwSFlzAAAJOgAACToB8GSSSgAAAetpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAA
        ADx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IlhNUCBDb3JlIDUuNC4w
        Ij4KICAgPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1z
        eW50YXgtbnMjIj4KICAgICAgPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIKICAgICAgICAgICAg
        eG1sbnM6eG1wPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvIgogICAgICAgICAgICB4bWxuczp0
        aWZmPSJodHRwOi8vbnMuYWRvYmUuY29tL3RpZmYvMS4wLyI+CiAgICAgICAgIDx4bXA6Q3JlYXRvclRv
        b2w+bWF0cGxvdGxpYiB2ZXJzaW9uIDIuMS4wLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvPC94bXA6Q3Jl
        YXRvclRvb2w+CiAgICAgICAgIDx0aWZmOk9yaWVudGF0aW9uPjE8L3RpZmY6T3JpZW50YXRpb24+CiAg
        ICAgIDwvcmRmOkRlc2NyaXB0aW9uPgogICA8L3JkZjpSREY+CjwveDp4bXBtZXRhPgqNQaNYAAAGiUlE
        QVRIDY1We4xU1Rn/3XPuYx47u8w+hnU38hTcuoUEt/6D2y4RB0ME1BoEd9taJaKh9CFiN7YGp7appUAM
        NmktMZFoJTYVLVQ0smsy26CN0SU1QgsuFAaW3WVmx33N677O6XfuyoIxTXqSO/fec+75fd93vt/3/UbD
        V0aKSZmCpkFMLz3T9utuu2N+o98aDSMBKVAo89z5y+zEz3ZafcCOfvWdlGCalqKn1Bf71CygTd+mf1es
        SOnpdMpTb+vWpTZuWVfe3jLPa5tzHYNm0T5N0gpdkkHaDBeGBU6d1/t/fyS8+/CbqdfUvmsx1PuMgc2b
        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
        AAAASUVORK5CYII='
        style='height:25px; border-radius:12px; display:inline-block; float:left;
        vertical-align:middle'>
        </img>
        {% endif %}
        </td>
        <td>
        {% if message %}
        &nbsp;&nbsp;<span style='font-size:12px'>{{ message }}</span>
        {% endif %}
        </td>
        </tr>
        </table>
        </div>
        """

        # clear_output()

        logo = kwargs.get("logo", True)
        message = kwargs.get("message", "info ")

        template = Template(_template)
        html = template.render(
            {"logo": logo, "message": message.strip().replace("\n", "<br/>")}
        )
        clear_output()
        publish_display_data(data={"text/html": html})


info = SCPInfo()

name = info.name
icon = info.icon
description = info.description
version = __version__ = info.version
release = info.release
release_date = info.release_date
copyright = info.copyright
url = info.url
authors = info.authors
contributors = info.contributors
license = info.license
cite = info.cite
long_description = info.long_description
display_info_string = info.display_info_string
