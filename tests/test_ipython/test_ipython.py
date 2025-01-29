# ruff: noqa: S101, T201

import spectrochempy as scp


def test_magic_addscript(ip):
    if ip is None:
        scp.warning_("ip is None - pss this test ")
        return

    ip.run_line_magic("load_ext", "spectrochempy.ipython")

    ip.run_cell("from spectrochempy import *")

    assert "preferences" in ip.user_ns

    ip.run_cell("print(preferences.available_styles)", store_history=True)
    ip.run_cell("project = Project()", store_history=True)
    x = ip.run_line_magic("addscript", "-p project -o prefs -n preferences 2")

    print("x", x)
    assert x.strip() == "Script prefs created."

    # with cell contents
    x = ip.run_cell(
        "%%addscript -p project -o essai -n preferences\n"
        "print(preferences.available_styles)"
    )

    print("result\n", x.result)
    assert x.result.strip() == "Script essai created."
