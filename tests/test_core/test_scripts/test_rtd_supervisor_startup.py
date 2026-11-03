"""Finite fault experiments; the production supervisor is never modified."""
# ruff: noqa: S603

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux ownership measurement"
)
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / ".github/workflows/scripts/rtd_supervisor_startup_probe.py"


@pytest.fixture
def observer_sentinel():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True
    )
    try:
        yield process
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)


def run_case(
    tmp_path,
    isolated,
    scenario="baseline",
    actor_mode="block",
    timeout=0.3,
    delay=0.6,
    extra_env=None,
    expect_final=True,
    regular_output=True,
):
    directory = tmp_path / (scenario + str(isolated))
    kwargs = {
        "isolated": isolated,
        "directory": str(directory),
        "timeout": timeout,
        "scenario": scenario,
        "delay": delay,
        "actor_mode": actor_mode,
        "extra_env": extra_env,
    }
    code = f"""
import importlib.util,json,sys
spec=importlib.util.spec_from_file_location('startup_probe',{str(SCRIPT)!r})
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
module=probe.load_supervisor()
assert not any(row[0]==__import__('os').getpid() for row in module._process_table().values())
previous=module._subreaper(True)
try:
    print(json.dumps(probe.measure(module,**json.loads(sys.argv[1]))))
finally:
    module._subreaper(previous)
"""
    sentinel = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True
    )
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", code, json.dumps(kwargs)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        row = json.loads(result.stdout)
        assert not row["observer_guard_expired"]
        assert (
            row["observer_live_pids"]
            == row["remaining_actor_pids"]
            == row["observer_cleanup_errors"]
            == []
        )
        assert not row["partial_evidence"]
        if expect_final:
            assert row["final_report"]["cleanup_complete"]
            assert row["final_report"]["live_pids"] == []
            assert not Path(f"/proc/{row['final_report']['root_pid']}").exists()
        else:
            assert row["final_report"] is None
        assert not Path(f"/proc/{row['controller_pid']}").exists()
        assert all(not Path(f"/proc/{item['pid']}").exists() for item in row["actors"])
        assert all(item["regular_output"] == regular_output for item in row["actors"])
        assert sentinel.poll() is None
        return row
    finally:
        sentinel.terminate()
        sentinel.wait(timeout=5)


def timeline(row):
    return {item["stage"]: item["time"] for item in row["controller_timeline"]}


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize(
    "actor_mode,expected",
    [("success", 0), ("detached", 124), ("orphan", 0), ("noevents", 124)],
)
def test_both_modes_preserve_exit_capture_and_cleanup(
    tmp_path, isolated, actor_mode, expected
):
    row = run_case(tmp_path, isolated, actor_mode=actor_mode, timeout=0.8)
    assert row["controller_code"] == row["final_report"]["exit_code"] == expected
    times = timeline(row)
    assert (
        times["body_enter"]
        <= times["stdlib_imports_finished"]
        < times["main.call"]
        < times["supervise.call"]
    )
    assert (
        times["supervise.call"]
        < times["deadline_clock_initialized_observed"]
        < times["popen.call"]
        < times["popen.return"]
    )
    assert (
        times["cleanup.call"]
        <= times["cleanup.return"]
        < times["report.call"]
        <= times["report.return"]
        < times["controller_exit"]
    )
    info = row["controller_timeline"][-1]["environment"]
    assert info["isolated"] == info["no_site"] == int(isolated)
    if actor_mode in ("detached", "orphan"):
        assert row["final_report"]["forced_kill"]
        assert len(row["actors"]) == 2
    if actor_mode == "detached":
        assert row["final_report"]["active_count"] == 2
    if actor_mode == "noevents":
        assert row["final_report"]["events"] == 0


@pytest.mark.parametrize(
    "scenario,stage",
    [
        ("pre_scan", "injected_pre_clock_scan"),
        ("launch", "injected_launch"),
        ("launch_return", "injected_launch_return"),
        ("loop_scan", "injected_loop_scan"),
        ("temporary_file", "injected_temporary_file"),
        ("output", "injected_output"),
    ],
)
def test_finite_delays_expose_cooperative_timeout_coverage(tmp_path, scenario, stage):
    row = run_case(tmp_path, True, scenario=scenario, timeout=0.2)
    assert row["controller_code"] == 124
    times = timeline(row)
    assert times[stage + ".return"] - times[stage + ".call"] >= 0.55
    clock = times["deadline_clock_initialized_observed"]
    if scenario == "pre_scan":
        assert times[stage + ".return"] < clock
        assert clock - times["supervise.call"] >= 0.6
    else:
        assert clock < times[stage + ".call"] < times[stage + ".return"]
        assert row["final_report"]["elapsed"] >= 0.6
    if scenario == "launch_return":
        assert times["popen.return"] < times[stage + ".call"]


def test_sitecustomize_delay_occurs_before_controller_clock(tmp_path):
    custom = tmp_path / "custom"
    custom.mkdir()
    (custom / "sitecustomize.py").write_text("import time;time.sleep(.6)\n")
    rows = [
        run_case(tmp_path, flag, timeout=0.2, extra_env={"PYTHONPATH": str(custom)})
        for flag in (False, True)
    ]
    normal, isolated = rows
    assert timeline(normal)["body_enter"] - normal["launched"] >= 0.55
    assert (timeline(normal)["body_enter"] - normal["launched"]) - (
        timeline(isolated)["body_enter"] - isolated["launched"]
    ) >= 0.4
    assert normal["controller_timeline"][-1]["environment"]["sitecustomize_loaded"]
    assert not isolated["controller_timeline"][-1]["environment"][
        "sitecustomize_loaded"
    ]
    assert all(
        row["controller_code"] == 124 and row["final_report"]["events"] == 0
        for row in rows
    )


def test_usercustomize_and_pth_only_apply_where_user_site_is_enabled(tmp_path):
    userbase = tmp_path / "userbase"
    usersite = (
        userbase
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    usersite.mkdir(parents=True)
    marker = tmp_path / "startup-markers"
    (usersite / "controlled.pth").write_text(
        f"import os;open({str(marker)!r},'a').write('pth '+str(os.getpid())+'\\n')\n"
    )
    (usersite / "usercustomize.py").write_text(
        f"import os,time\nwith open({str(marker)!r},'a') as f:f.write('user '+str(os.getpid())+'\\n')\ntime.sleep(.6)\n"
    )
    normal = run_case(
        tmp_path, False, timeout=0.2, extra_env={"PYTHONUSERBASE": str(userbase)}
    )
    if not normal["controller_timeline"][-1]["environment"]["user_site_enabled"]:
        pytest.skip("interpreter disables user site (e.g. virtualenv)")
    isolated = run_case(
        tmp_path, True, timeout=0.2, extra_env={"PYTHONUSERBASE": str(userbase)}
    )
    marks = marker.read_text().splitlines()
    assert f"pth {normal['controller_pid']}" in marks
    assert f"user {normal['controller_pid']}" in marks
    assert f"pth {isolated['controller_pid']}" not in marks
    assert f"user {isolated['controller_pid']}" not in marks
    assert any(
        mark.startswith("user ") and mark != f"user {normal['controller_pid']}"
        for mark in marks
    )


def test_startup_child_can_prevent_timer_and_hold_capture_after_parent_exit(tmp_path):
    custom = tmp_path / "custom"
    custom.mkdir()
    child_code = "import os,json,time,stat;from pathlib import Path;p=Path(os.environ['SCPY_STARTUP_EVIDENCE']).parent;row={'pid':os.getpid(),'ppid':os.getppid(),'pgid':os.getpgrp(),'sid':os.getsid(0),'regular_output':stat.S_ISREG(os.fstat(1).st_mode)};(p/('startup-'+str(os.getpid())+'.json')).write_text(json.dumps(row));time.sleep(1.5)"
    (custom / "sitecustomize.py").write_text(
        f"import subprocess,sys\nsubprocess.Popen([sys.executable,'-I','-S','-c',{child_code!r}],start_new_session=True)\n"
    )
    env = {"PYTHONPATH": str(custom)}
    normal = run_case(
        tmp_path,
        False,
        timeout=0.2,
        extra_env=env,
        expect_final=False,
        regular_output=False,
    )
    assert normal["controller_code"] == 1
    times = timeline(normal)
    assert "supervise.call" in times
    assert "deadline_clock_initialized_observed" not in times
    assert normal["controller_timeline"][-1]["exception_type"] == "RuntimeError"
    assert next(
        row
        for row in normal["controller_timeline"]
        if row["stage"] == "preflight_children"
    )["pids"]
    assert normal["capture_complete_observed"] - times["controller_exit"] >= 0.5
    isolated = run_case(tmp_path, True, timeout=0.3, extra_env=env)
    assert isolated["controller_code"] == 124
    assert isolated["final_report"]["cleanup_complete"]


def test_complete_observer_reports_two_real_controller_failures(tmp_path):
    output = tmp_path / "comparison.json"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(SCRIPT),
            "compare",
            "--timeout",
            ".8",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=12,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert [row["isolated"] for row in report["results"]] == [False, True]
    assert all(
        row["controller_code"] == 124 and row["final_report"]["exit_code"] == 124
        for row in report["results"]
    )


def test_proposed_yaml_preserves_installs_but_never_builds_docs():
    yaml = pytest.importorskip("yaml")
    original = yaml.safe_load((ROOT / ".readthedocs.yaml").read_text())
    proposed = yaml.safe_load((ROOT / ".readthedocs.startup-probe.yaml").read_text())
    commands = proposed["build"]["jobs"]["build"]["html"]
    assert commands == [
        "/bin/sh .github/workflows/scripts/rtd_startup_preflight.sh",
        "python -I -S -u .github/workflows/scripts/rtd_supervisor_startup_probe.py compare --timeout 3",
    ]
    assert original["build"]["jobs"]["build"]["html"] == commands
    original["build"]["jobs"]["build"]["html"] = commands
    assert original == proposed


@pytest.mark.parametrize("isolated", [False, True])
def test_temporary_collector_backpressure_keeps_timeout_and_final_report(
    tmp_path, isolated
):
    supervisor = SCRIPT.with_name("docs_build_diagnostics.py")
    code = f"import sys,time;sys.path.insert(0,{str(ROOT)!r});from docs.tools.lifecycle_trace import emit;emit('backpressure','START',span='wait');print('synthetic'*40000,flush=True);time.sleep(5)"
    command = [
        sys.executable,
        *(["-I", "-S"] if isolated else []),
        "-u",
        str(supervisor),
        "--timeout",
        ".6",
        "--grace",
        ".1",
        "--kill-grace",
        ".2",
        "--",
        sys.executable,
        "-u",
        "-c",
        code,
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env={**os.environ, "SCPY_DOCS_TRACE": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        time.sleep(0.15)
        output, errors = process.communicate(timeout=5)
        assert process.returncode == 124, errors
        final = json.loads(
            next(
                line.split("] ", 1)[1]
                for line in output.splitlines()
                if line.startswith("[docs recovery FINAL] ")
            )
        )
        assert final["cleanup_complete"] and final["live_pids"] == []
        assert not final["log_delivery_failed"]
        assert '"phase": "backpressure"' in output
        assert not Path(f"/proc/{final['root_pid']}").exists()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()


def test_observer_guard_keeps_partial_startup_evidence_and_cleans(
    tmp_path, observer_sentinel
):
    custom = tmp_path / "custom"
    custom.mkdir()
    (custom / "sitecustomize.py").write_text(
        "import json,os,time\n"
        "with open(os.environ['SCPY_STARTUP_EVIDENCE'],'a') as f: "
        "f.write(json.dumps({'stage':'controlled_site_enter','time':time.monotonic(),'pid':os.getpid()})+'\\n')\n"
        "time.sleep(9)\n"
    )
    output = tmp_path / "comparison.json"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(SCRIPT),
            "compare",
            "--timeout",
            ".2",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(custom)},
        capture_output=True,
        text=True,
        timeout=14,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    normal, isolated = json.loads(output.read_text())["results"]
    assert normal["observer_guard_expired"]
    assert normal["controller_code"] == -9
    assert normal["final_report"] is None
    assert [row["stage"] for row in normal["controller_timeline"]] == [
        "controlled_site_enter"
    ]
    assert not isolated["observer_guard_expired"]
    assert isolated["controller_code"] == 124
    for row in (normal, isolated):
        assert (
            row["observer_live_pids"]
            == row["observer_cleanup_errors"]
            == row["remaining_actor_pids"]
            == []
        )
        assert not Path(f"/proc/{row['controller_pid']}").exists()
        if row["final_report"]:
            assert row["final_report"]["cleanup_complete"]
            assert not Path(f"/proc/{row['final_report']['root_pid']}").exists()
