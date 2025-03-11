# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import re
import textwrap

from traitlets import TraitError

from spectrochempy.application.application import app
from spectrochempy.application.application import error_
from spectrochempy.utils.file import pathclean
from spectrochempy.utils.meta import Meta

__all__ = ["preferences"]


# ======================================================================================
# Management of the preferences
# ======================================================================================
class PreferencesSet(Meta):
    """Preferences setting."""

    def __getitem__(self, key):
        # search on the preferences
        if self.parent is not None:
            res = getattr(self.parent, f"{self.name}_{key}")
        elif hasattr(app.plot_preferences, key):
            res = getattr(app.plot_preferences, key)
        elif hasattr(app.general_preferences, key):
            res = getattr(app.general_preferences, key)
        else:
            alias = self._get_alias(key)
            if alias:
                if isinstance(alias, list):
                    res = PreferencesSet(
                        parent=self,
                        name=key,
                        **{n: getattr(self, f"{key}_{n}") for n in alias},
                    )
                else:
                    res = getattr(self, alias)
            else:
                res = super().__getitem__(key)
                if res is None:
                    error_(
                        f"not found {key}"
                    )  # key = key.replace('_','.').replace('...', '_').replace('..',
                    # '-')  #  # res = mpl.rcParams[key]

        return res

    def __setitem__(self, key, value):
        # also change the corresponding preferences
        if hasattr(app.plot_preferences, key):
            try:
                setattr(app.plot_preferences, key, value)
            except TraitError:
                value = type(app.plot_preferences.traits()[key].default_value)(value)
                setattr(app.plot_preferences, key, value)
        elif hasattr(app.general_preferences, key):
            setattr(app.general_preferences, key, value)
        elif key in self.keys():
            newkey = f"{self.name}_{key}"
            setattr(app.plot_preferences, newkey, value)
            self.parent[newkey] = value
            return
        else:
            # try to find an alias for matplotlib values
            alias = self._get_alias(key)
            if alias:
                newkey = f"{alias}_{key}"
                setattr(app.plot_preferences, newkey, value)
                self.parent[newkey] = value
            else:
                error_(KeyError, f"`{key}` not found in the preferences")
            return

        super().__setitem__(key, value)

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------
    def _get_alias(self, key):
        alias = []
        lkeyp = len(key) + 1

        regex = r"[a-zA-Z0-9_]*(?:\b|_)" + key + "(?:\b|_)[a-zA-Z0-9_]*"
        for item in app.plot_preferences.trait_names():
            matches = re.match(regex, item)
            if matches is not None:
                alias.append(item)

        if alias:
            starts = any(par.startswith(key) for par in alias)
            # ends = any([par.endswith(key) for par in alias])

            if len(alias) > 1:
                if alias[0].endswith(key) and (not starts and self.parent is not None):
                    # it is a member of a group but we don't know which one:
                    raise KeyError(
                        f"Found several keys for {key}: {alias}, so it is ambiguous. "
                        f"Please choose on one of them"
                    )
                if any(par.startswith(key) for par in alias):
                    # we return the group of parameters
                    pars = []
                    for par in alias:
                        if par.startswith(key):
                            pars.append(par[lkeyp:])
                    return pars
            else:
                return alias[0][:-lkeyp]

        raise KeyError(f"{key} not found in preferences")

    # ----------------------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------------------
    def reset(self, include_all=False):
        """Remove the matplotlib_user json file to reset to defaults."""
        config_dir = pathclean(app.general_preferences.cfg.config_dir)

        f = config_dir / "PlotPreferences.json"
        if f.exists():
            f.unlink()

        if include_all:
            # by default we do not delete general preferences
            f = config_dir / "GeneralPreferences.json"
            if f.exists():
                f.unlink()

        app.plot_preferences._apply_style("scpy")
        self.style = "scpy"

        # reset also non-matplolib preferences
        nonmplpars = [
            "method_1D",
            "method_2D",
            "method_3D",
            "colorbar",
            "show_projections",
            "show_projection_x",
            "show_projection_y",
            "colormap",
            "max_lines_in_stack",
            "simplify",
            "number_of_x_labels",
            "number_of_y_labels",
            "number_of_z_labels",
            "number_of_contours",
            "contour_alpha",
            "contour_start",
            "antialiased",
            "rcount",
            "ccount",
        ]
        for par in nonmplpars:
            setattr(self, par, app.plot_preferences.traits()[par].default_value)

        self._data = {}

    def list_all(self):
        """
        List all plot parameters with their current and default value.

        ..versionadded:: 1.0.0
        """
        for key in app.plot_preferences.trait_names(config=True):
            self.help(key)

    def all(self):
        """
        List all plot parameters with their current and default value.

        ..deprecated:: 1.0.0, use `list_all` instead. To be removed in 1.1.0.
        """
        self.list_all()

    def help(self, key):
        """
        Display information on a given parameter.

        Parameters
        ----------
        key : str
            Name of the parameter for which we want information.
        """
        from spectrochempy.utils.print import TBold
        from spectrochempy.utils.print import colored

        value = self[key]
        trait = app.plot_preferences.traits()[key]
        default = trait.default_value
        thelp = trait.help.replace("\n", " ").replace("\t", " ")
        sav = ""
        while thelp != sav:
            sav = thelp
            thelp = thelp.replace("  ", " ")
        help = "\n".join(
            textwrap.wrap(
                thelp, 100, initial_indent=" " * 20, subsequent_indent=" " * 20
            )
        )

        value = colored(value, "GREEN")
        default = colored(default, "BLUE")

        print(TBold(f"{key} = {value} \t[default: {default}]"))  # noqa: T201
        print(f"{help}\n")  # noqa: T201

    def makestyle(self, stylename="mydefault", to_mpl=False):
        """
        Create Matplotlib Style files.

        Parameters
        ----------
        stylename :
        to_mpl :

        Returns
        -------
        stylename
            Name of the style

        """
        import matplotlib as mpl

        if stylename.startswith("scpy"):
            error_(
                "Style name starting with `scpy` are READ-ONLY. Please use an another "
                "name."
            )
            return None

        txt = ""
        sline = ""

        for key in mpl.rcParams:
            if key in [
                "animation.avconv_args",
                "animation.avconv_path",
                "animation.html_args",
                "keymap.all_axes",
                "mathtext.fallback_to_cm",
                "validate_bool_maybe_none",
                "savefig.jpeg_quality",
                "text.latex.preview",
                "backend",
                "backend_fallback",
                "date.epoch",
                "docstring.hardcopy",
                "figure.max_open_warning",
                "figure.raise_window",
                "interactive",
                "savefig.directory",
                "timezone",
                "tk.window_focus",
                "toolbar",
                "webagg.address",
                "webagg.open_in_browser",
                "webagg.port",
                "webagg.port_retries",
            ]:
                continue

            val = str(mpl.rcParams[key])
            if val.startswith("CapStyle") or val.startswith("JoinStyle"):
                val = val.split(".")[-1]

            sav = ""
            while val != sav:
                sav = val
                val = val.replace("  ", " ")
            line = f"{key:40s} : {val}\n"
            if line[0] != sline:
                txt += "\n"
                sline = line[0]
            if key not in ["axes.prop_cycle"]:
                line = (
                    line.replace("[", "")
                    .replace("]", "")
                    .replace("'", "")
                    .replace('"', "")
                )
            if key == "savefig.bbox":
                line = f"{key:40s} : standard\n"
            txt += line.replace("#", "")

        # Non matplotlib parameters,
        # some parameters are not saved in matplotlib style sheets so we willa dd them
        # here
        nonmplpars = [
            "method_1D",
            "method_2D",
            "method_3D",
            "colorbar",
            "show_projections",
            "show_projection_x",
            "show_projection_y",
            "colormap",
            "max_lines_in_stack",
            "simplify",
            "number_of_x_labels",
            "number_of_y_labels",
            "number_of_z_labels",
            "number_of_contours",
            "contour_alpha",
            "contour_start",
            "antialiased",
            "rcount",
            "ccount",
        ]
        txt += "\n\n##\n## ADDITIONAL PARAMETERS FOR SPECTROCHEMPY\n##\n"
        for par in nonmplpars:
            txt += f"##@{par:37s} : {getattr(self, par)}\n"

        stylesheet = (pathclean(self.stylesheets) / stylename).with_suffix(".mplstyle")
        stylesheet.write_text(txt)

        if to_mpl:
            # make it also accessible to pyplot
            stylelib = (
                pathclean(mpl.get_configdir()) / "stylelib" / stylename
            ).with_suffix(".mplstyle")
            stylelib.write_text()

        return stylename


preferences = PreferencesSet()
