from __future__ import annotations

from spectrochempy.ci import install_plugins


def test_pip_install_places_global_flags_before_editable_target(tmp_path, monkeypatch):
    plugin_dir = tmp_path / "plugins" / "spectrochempy-nmr"
    plugin_dir.mkdir(parents=True)

    captured = {}

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = cmd

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(install_plugins.subprocess, "run", fake_run)

    rc = install_plugins._pip_install(
        "spectrochempy-nmr",
        tmp_path / "plugins",
        editable=True,
        pip_cmd=["python", "-m", "pip"],
        no_deps=True,
        no_build_isolation=True,
    )

    assert rc == 0
    assert captured["cmd"] == [
        "python",
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-build-isolation",
        "-e",
        str(plugin_dir),
    ]
