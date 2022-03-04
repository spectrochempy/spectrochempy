import ipywidgets as widgets

from IPython.display import display

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.processors.concatenate import concatenate
from spectrochempy.core.processors.baseline import BaselineCorrection
from spectrochempy.core.readers.importer import read
from spectrochempy.core.plotters.multiplot import multiplot
from spectrochempy.core import error_


__all__ = ["BaselineCorrector"]


class BaselineCorrector:
    """
    Launch a GUI for baseline corrections.

    Wrapper of scp.BaselineCorrection(X), with widgets for dataset slicing,
    input parameters and graphical output.
    Should be run in jupyter notebook (not jupyter lab) with the widget backend
    (magic `%matplotlib widget`).

    Parameters
    ----------
    X : NDDataset, default: None
        The NDDataset to process. If None, a FileUpload widget is enabled.
    initial_ranges : list, optional, default: None
        The initial regions where to compute the baseline. If not given, 10% on each
        side of the spectra will be taken as a starting range's list.


    Attributes
    ----------
    original : NDDataset
        The (sliced) NDDataset, without baseline correction
    corrected : NDDataset
        The baseline corrected NDDataset
    baseline : NDDataset
        The baselines

    See Also
    --------
    BaselineCorrection : Baseline correction method.

    Notes
    -----
    The `BaselineCorrector()` widget can be used in jupyter notebooks.
    - The GUI buttons are as follows:
        - `upload`: upload files (disabled if a NDDataset is passed as parameter).
          Uploading file will not trigger the reading
          and processing. To do so, the user is expected to click the `process` button.
        - `process`: triggers baseline correct and plotting of original
          dataset + baseline and corrected datasets
        - `save as`: save the baseline corrected NDDataset

    - The `x slice` and `y slice` textboxes can be used to slice the initial
      dataset with the usual `[start:stop:stride]`
      format. In the `x` dimension, coordinates or indexes can be used
      (e.g. `[3000.0:2000.0:1]` or `[0:100:1]` are valid
      entries). In the `y` dimension only indexes can be used (e.g. `[0:10:1]`).
      Note also that currently none of the
      `start`, `stop`, `stride` parameters can be omitted, e.g. `[3000.0:2000.0]`
      or `[:,:]` are not valid entries.
    - Method and Interpolation are self explaining, see BaselineCorrection() for
      details.
    - Ranges should be entered as a tuple of intervals or wavenumbers, e.g.
      ```
      (
      [5900.0, 5400.0],
      2000.0,
      [1550.0, 1555.0],
      )
      ```

    Examples
    --------
    # runs only in Jupyter notebook/ipython environment.
    # See : User's Guide > Processing > Baseline corrections for an example.
    [In 2] X = scp.read_omnic("irdata/nh4y-activation.spg")
           out = scp.BaselineCorrector(X)
    """

    def __init__(self, X=None, initial_ranges=None):

        if not isinstance(X, (NDDataset, type(None))):
            raise ValueError("X must be None or a valid NDDataset")

        self._X = X
        self._fig = None
        self._initial_ranges = initial_ranges
        self._done = False

        self._output = widgets.Output()

        if X is not None:
            disabled = True
        else:
            disabled = False
        self._uploader = widgets.FileUpload(
            accept="",  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
            multiple=True,  # True to accept multiple files upload else False
            description="upload",
            disabled=disabled,
        )

        self._processbutton = widgets.Button(description="process", icon="fa-play")
        self._processbutton.on_click(self.process_clicked)

        self._savebutton = widgets.Button(description="save as", icon="fa-save")
        self._savebutton.on_click(self.save_clicked)

        self._methodselector = widgets.Dropdown(
            description="Method",
            options=["sequential", "multivariate"],
            value="sequential",
        )

        self._npcslider = widgets.IntSlider(description="npc", value=1, min=1, max=5)

        self._method_control = widgets.HBox(children=[self._methodselector])

        def toggle_npc(change):
            if self._methodselector.value == "multivariate":
                self._method_control.children = [self._methodselector, self._npcslider]
            else:
                self._method_control.children = [self._methodselector]

        self._methodselector.observe(toggle_npc, names="value")

        self._interpolationselector = widgets.Dropdown(
            description="Interpolation",
            options=["polynomial", "pchip"],
            value="polynomial",
        )

        self._orderslider = widgets.IntSlider(
            description="orderslider", value=1, min=1, max=6
        )

        self._interpolation_control = widgets.HBox(
            children=[self._interpolationselector, self._orderslider]
        )

        def toggle_polyorder(change):
            if self._interpolationselector.value == "polynomial":
                self._interpolation_control.children = [
                    self._interpolationselector,
                    self._orderslider,
                ]
            else:
                self._interpolation_control.children = [self._interpolationselector]

        self._interpolationselector.observe(toggle_polyorder, names="value")

        self._ranges_control = widgets.Textarea(description="Ranges:")
        self._x_limits_control = widgets.Text(description="x slice:")
        self._y_limits_control = widgets.Text(description="y slice:")
        self._limits_control = widgets.HBox(
            children=[self._x_limits_control, self._y_limits_control]
        )

        io = widgets.VBox(
            children=[
                self._uploader,
                self._processbutton,
                self._savebutton,
            ]
        )
        controls = widgets.VBox(
            children=[
                self._limits_control,
                self._method_control,
                self._interpolation_control,
                self._ranges_control,
            ]
        )

        self._input = widgets.HBox(children=[io, controls])
        display(self._input)

        if self._X is not None:
            self.process_clicked()
        else:
            self.corrected = NDDataset()

    def blcorrect_and_plot(self, clear=False):
        slice_x = _str_to_slice(self._x_limits_control.value.strip(), self._X, "x")
        slice_y = _str_to_slice(self._y_limits_control.value.strip(), self._X, "y")
        self.original = self._X[slice_y, slice_x]
        ranges = eval(self._ranges_control.value.strip())
        if self.original is not None:  # slicing was OK
            blc = BaselineCorrection(self.original)
            self.corrected = blc.compute(
                *ranges,
                interpolation=self._interpolationselector.value,
                order=self._orderslider.value,
                method=self._methodselector.value,
                npc=self._npcslider.value,
            )
            self.baseline = self.original - self.corrected

            with self._output:
                # if clear:
                #     self._output.clear_output(True)
                axes = multiplot(
                    [
                        concatenate(self.original, self.baseline, dims="y"),
                        self.corrected,
                    ],
                    labels=["Original", "Corrected"],
                    sharex=True,
                    nrow=2,
                    ncol=1,
                    fig=self._fig,
                    figsize=(7, 6),
                    dpi=96,
                    mpl_event=False,
                )
                axes["axe11"].get_xaxis().set_visible(False)
                blc.show_regions(axes["axe21"])
                self._fig = axes["axe21"].figure

            self._done = True

    def process_clicked(self, b=None):
        """(re)process dataset (slicing) and baseline correct"""
        if self._X is None:
            # no dataset loaded, read data (byte content)
            value = self._uploader.value
            dicvalue = {key: value[key]["content"] for key in value.keys()}
            ds = read(dicvalue)
            if isinstance(ds, NDDataset):
                self._X = ds
            else:
                with self._output:
                    error_("Could not read or merge uploaded files")
                return

        if not self._done:
            # first processing,
            # defines default ranges (10% of the X axis at both ends)...
            len_ = int(len(self._X.x) / 10)
            if self._initial_ranges:
                ranges = self._initial_ranges
            else:
                ranges = _round_ranges(
                    (
                        [self._X.x.data[0], self._X.x.data[len_]],
                        [self._X.x.data[-len_], self._X.x.data[-1]],
                    )
                )
            self._ranges_control.value = (
                str(ranges)
                .replace("(", "(\n")
                .replace("], ", "],\n")
                .replace(")", "\n)")
            )
            self._x_limits_control.value = _x_slice_to_str(
                slice(0, len(self._X.x), 1), self._X
            )
            self._y_limits_control.value = _y_slice_to_str(slice(0, len(self._X.y), 1))
            # ... and baseline correct with defaults
            self.blcorrect_and_plot(self._X)

        else:
            # was processed once, the user probably asks re-processing
            # with new parameters
            self.blcorrect_and_plot(clear=True)

    def save_clicked(self, b=None):
        try:
            self.corrected.write()
        except AttributeError:
            # the user has cancelled
            pass


# Utility functions


def _x_slice_to_str(s, A, decimals=2):
    return (
        f"[{round(A.x.data[s.start], decimals)} : "
        f"{round(A.x.data[s.stop-1], decimals)} : {s.step}]"
    )


def _y_slice_to_str(s):
    return f"[{s.start} : {s.stop} : {s.step}]"


def _str_to_slice(st, A, dim):
    start, stop, step = st.replace("[", " ").replace("]", " ").replace(":", " ").split()
    start = int(start) if "." not in start else float(start)
    stop = int(stop) if "." not in stop else float(stop)
    step = int(step)
    return A._get_slice(slice(start, stop, step), dim)


def _round_ranges(ranges, decimals=2):
    """round x values for better reading"""
    for i, item in enumerate(ranges):
        if isinstance(item, float):  # pragma: cover
            # can not be reached in the current implementation of baselinecorrector
            ranges[i] = round(item, decimals)
        else:
            for j, sub_item in enumerate(item):
                ranges[i][j] = round(sub_item, decimals)
    return ranges