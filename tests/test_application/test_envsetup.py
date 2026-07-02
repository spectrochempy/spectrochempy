# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

from spectrochempy.application import envsetup


class _FakeIPython:
    def __init__(self, repr_text="ipykernel"):
        self.repr_text = repr_text
        self.magics = []

    def run_line_magic(self, name, value):
        self.magics.append((name, value))

    def __str__(self):
        return self.repr_text


def test_setup_environment_notebook_defaults_to_inline(monkeypatch):
    ip = _FakeIPython()

    monkeypatch.setattr(envsetup, "is_terminal", lambda: False)
    monkeypatch.setattr(envsetup, "is_notebook", lambda: True)
    monkeypatch.setattr(envsetup, "get_ipython", lambda: ip)
    monkeypatch.setattr(envsetup, "setup_jupyter_css", lambda: None)
    monkeypatch.setattr(envsetup.sys, "argv", ["python"])
    monkeypatch.delenv("DOC_BUILDING", raising=False)

    no_display, _, is_pytest = envsetup.setup_environment()

    assert no_display is False
    assert is_pytest is False
    assert ip.magics == [("matplotlib", "inline")]


def test_setup_environment_notebook_uses_inline_for_nbsphinx(monkeypatch):
    ip = _FakeIPython()

    monkeypatch.setattr(envsetup, "is_terminal", lambda: False)
    monkeypatch.setattr(envsetup, "is_notebook", lambda: True)
    monkeypatch.setattr(envsetup, "get_ipython", lambda: ip)
    monkeypatch.setattr(envsetup, "setup_jupyter_css", lambda: None)
    monkeypatch.setattr(
        envsetup.sys,
        "argv",
        [
            "ipykernel_launcher",
            "--InlineBackend.rc={'figure.dpi': 96}",
        ],
    )
    monkeypatch.delenv("DOC_BUILDING", raising=False)

    envsetup.setup_environment()

    assert ip.magics == [("matplotlib", "inline")]
