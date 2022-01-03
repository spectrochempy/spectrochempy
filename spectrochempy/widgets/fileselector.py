# ------------------------------------------------------------------
# Modified from intake.gui
#
# Copyright (c) 2012 - 2018, Anaconda, Inc. and Intake contributors
#
# BSD 2-Clause "Simplified" License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ------------------------------------------------------------------
"""
Widgets for the Jupyter notebook and Jupyter lab.
"""
from contextlib import contextmanager

from IPython.core.interactiveshell import InteractiveShell
import IPython

from ..utils import pathclean

__all__ = ["FileSelector"]

import ipywidgets as widgets


@contextmanager
def ignore(ob):
    try:
        ob.ignore = True
        yield
    finally:
        ob.ignore = False


class Base(object):
    done_callback = None

    def stop(self, ok=False):
        done = self.done_callback is not None
        if ok and done:
            self.done_callback(ok)
        elif done:
            self.done_callback(None)

    def __repr__(self):
        return (
            "To get widget to display, you must "
            "install ipy/jupyter-widgets, run in a notebook and, in "
            "the case of jupyter-lab, install the jlab extension."
        )

    def _ipython_display_(self, **kwargs):

        # from IPython.Widget._ipython_display_
        if InteractiveShell.initialized():
            if self.widget._view_name is not None:
                plaintext = repr(self)
                data = {
                    "text/plain": plaintext,
                    "application/vnd.jupyter.widget-view+json": {
                        "version_major": 2,
                        "version_minor": 0,
                        "model_id": self.widget._model_id,
                    },
                }
                IPython.display.display(data, raw=True)
                self.widget._handle_displayed(**kwargs)


class FileSelector(Base):
    """
    IPyWidgets interface for picking files.
    """

    def __init__(self, done_callback=None, path=None, filters=None):
        """
        The current path is stored in ``.path`` and the current selection is stored in ``.value``.

        Parameters
        ----------
        done_callback : function
            Called when the tick or cross buttons are clicked. Expects signature func(path, ok=True|False).
        filters : list of str or None
            Only show files ending in one of these strings. Normally used for picking file extensions. None is an
            alias for [''], passes all files.
        """
        path = pathclean(path)
        self.startpath = path
        self.startname = path.name

        self.done_callback = done_callback
        if filters:
            if not isinstance(filters, (list, tuple)):
                filters = [filters]
            self.filters = list(filters)
        else:
            self.filters = [""]
        if not path or not path.exists():
            self.path = path.cwd()
        else:
            self.path = path
        self.main = widgets.Select(rows=7)

        self.button = widgets.Button(
            icon="chevron-left",
            tooltip="Parent",
            layout=widgets.Layout(flex="1 1 auto", width="auto"),
        )
        self.button.on_click(self.up)

        self.label = widgets.Label(
            layout=widgets.Layout(flex="100 1 auto", width="auto")
        )
        self.x = widgets.Button(
            icon="close", tooltip="Close Selector", layout=widgets.Layout(width="auto")
        )
        self.x.on_click(lambda ev: self.stop())

        self.ok = widgets.Button(
            icon="check", tooltip="OK", layout=widgets.Layout(width="auto")
        )
        self.ok.on_click(lambda ev: self._ok())

        self.make_options()
        self.main.observe(self.changed, "value")
        self.upper = widgets.Box(children=[self.button, self.label])
        self.right = widgets.VBox(children=[self.x, self.ok])
        self.lower = widgets.HBox(children=[self.main, self.right])
        self.widget = widgets.VBox(children=[self.upper, self.lower])
        self.ignore = False

    def _ok(self):
        fn = self.path / self.main.value
        if fn.is_dir():
            self.stop()
        else:
            self.stop(fn)

    def make_options(self):
        self.ignore = True
        self.label.value = (
            str(self.path).replace(str(self.startpath.parent), "..")
            if str(self.startpath) in str(self.path)
            else str(self.path)
        )
        out = []
        for f in sorted(self.path.glob("[a-zA-Z0-9]*")):  # os.listdir()):
            if f.is_dir() or any(
                ext in f.suffix
                for ext in self.filters + list(map(str.upper, self.filters))
            ):
                out.append(f.name)
        self.main.value = self.value = None
        self.fullpath = self.path
        self.main.options = out
        self.ignore = False

    def up(self, *args):
        self.path = (
            self.path.parent
        )  # os.path.dirname(self.path.rstrip('/')).rstrip('/') + '/'
        self.make_options()

    def changed(self, ev):
        if self.ignore:
            self.value = None
            self.fullpath = None
            return
        with ignore(self):
            fn = self.path / ev["new"]
            if fn.is_dir():
                self.path = fn
                self.make_options()
            self.value = self.main.value
            self.fullpath = self.path / self.value
