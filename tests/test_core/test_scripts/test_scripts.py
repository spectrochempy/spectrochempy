# -*- coding: utf-8 -*-
# flake8: noqa


from spectrochempy import Script, info_


def test_script():
    Script("name", "print(2)")

    try:
        Script("0name", "print(3)")
    except Exception:
        info_("name not valid")
