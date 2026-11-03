"""Real isolated process and Sphinx tests for bounded diagnostic recovery."""
# ruff: noqa: S603

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

if sys.platform != "linux":
    pytest.skip("Linux subreaper/pidfd supervisor", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / ".github/workflows/scripts/docs_build_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("docs_build_diagnostics", SCRIPT)
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)


@pytest.fixture
def sentinel():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(90)"], start_new_session=True
    )
    try:
        yield process
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)


def report(output):
    prefix = "[docs recovery FINAL] "
    return json.loads(
        next(
            line[len(prefix) :]
            for line in output.splitlines()
            if line.startswith(prefix)
        )
    )


def invoke(command, timeout, tmp_path):
    started = time.monotonic()
    env = {**os.environ, "SCPY_DOCS_TRACE": "1", "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(SCRIPT),
            "--timeout",
            str(timeout),
            "--grace",
            "0.15",
            "--kill-grace",
            "0.3",
            "--",
            *command,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout + 8,
        check=False,
    )
    assert time.monotonic() - started < timeout + 7
    summary = report(result.stdout)
    assert result.returncode == summary["exit_code"]
    assert summary["cleanup_complete"] and not summary["live_pids"]
    assert not Path(f"/proc/{summary['root_pid']}").exists()
    return result, summary


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("success", 0),
        ("exception", 1),
        ("block", 124),
        ("ignore", 124),
        ("detached", 124),
        ("held_fd", 0),
        ("flood", 124),
    ],
)
def test_process_recovery_after_supervisor_exit(mode, expected, tmp_path, sentinel):
    identities = tmp_path / "pids.jsonl"
    child = (
        "import os,signal,time,json,stat\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"with open({str(identities)!r}, 'a') as f: f.write(json.dumps(os.getpid())+'\\n')\n"
        "assert stat.S_ISREG(os.fstat(1).st_mode)\n"
        "time.sleep(30)\n"
    )
    code = (
        "import os,signal,time,json,stat,subprocess,sys\n"
        "from docs.tools.lifecycle_trace import emit\n"
        f"with open({str(identities)!r}, 'a') as f: f.write(json.dumps(os.getpid())+'\\n')\n"
        "assert stat.S_ISREG(os.fstat(1).st_mode)\n"
        "emit('controlled', 'START', span='controlled', document='case.py')\n"
    )
    if mode in ("detached", "held_fd"):
        code += f"subprocess.Popen([sys.executable, '-u', '-c', {child!r}], start_new_session={mode == 'detached'})\n"
        code += (
            f"while len(open({str(identities)!r}).readlines()) < 2: time.sleep(.005)\n"
        )
    if mode == "exception":
        code += "raise RuntimeError('controlled failure')\n"
    elif mode == "flood":
        code += "print('x' * 3000000, end='', flush=True)\nemit('gallery.execute', 'START', span='last-useful', document='last-example.py')\ntime.sleep(30)\n"
    elif mode not in ("success", "held_fd"):
        if mode == "ignore":
            code += "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        code += "time.sleep(30)\n"
    else:
        code += "emit('controlled', 'END', span='controlled')\n"
    result, summary = invoke([sys.executable, "-u", "-c", code], 0.7, tmp_path)
    assert result.returncode == expected
    assert summary["events"] >= 1 and summary["trace_complete"]
    assert '"document": "case.py"' in result.stdout
    if mode in ("ignore", "held_fd", "detached"):
        assert summary["forced_kill"]
    if mode == "flood":
        assert summary["stream_truncated"]
        assert '"document": "last-example.py"' in result.stdout
    assert all(
        not Path(f"/proc/{pid}").exists()
        for pid in map(json.loads, identities.read_text().splitlines())
    )
    assert sentinel.poll() is None


def test_cleanup_failure_is_reported_after_real_cleanup(tmp_path, sentinel):
    code = f"""
import importlib.util,sys
spec=importlib.util.spec_from_file_location('diagnostics', {str(SCRIPT)!r})
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
original=module._cleanup
def injected(process,budget):
    live,errors=original(process,budget)
    return live,errors+['InjectedCleanupError']
module._cleanup=injected
sys.exit(module.supervise([sys.executable,'-c','pass'],timeout=.5,grace=.1,kill_grace=.2))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    summary = report(result.stdout)
    assert result.returncode == summary["exit_code"] == 70
    assert not summary["cleanup_complete"]
    assert summary["cleanup_errors"] == ["InjectedCleanupError"]
    assert summary["live_pids"] == []
    assert not Path(f"/proc/{summary['root_pid']}").exists()


def test_external_term_preserves_final_events(tmp_path, sentinel):
    ready = tmp_path / "ready"
    code = f"from docs.tools.lifecycle_trace import emit;import time;emit('waiting','START',span='wait');open({str(ready)!r},'w').close();time.sleep(30)"
    process = subprocess.Popen(
        [sys.executable, str(SCRIPT), "--", sys.executable, "-c", code],
        cwd=ROOT,
        env={**os.environ, "SCPY_DOCS_TRACE": "1", "PYTHONPATH": str(ROOT)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        process.terminate()
        output, errors = process.communicate(timeout=5)
        assert process.returncode == 143, errors
        assert report(output)["cleanup_complete"]
        assert '"phase": "waiting"' in output
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=20)
        process.stdout.close()
        process.stderr.close()


def test_trace_report_allowlist_and_partial_record(tmp_path):
    path = tmp_path / "events"
    path.write_text(
        '[docs lifecycle] {"phase":"test","status":"START","span":"one","scientific_data":"NEVER_LOG"}\n'
    )
    data = supervisor._trace_report(path)
    assert data["complete"] and data["active"]
    assert "NEVER_LOG" not in json.dumps(data)
    path.write_text(path.read_text() + '{"partial":')
    assert not supervisor._trace_report(path)["complete"]


def test_fast_exiting_output_is_drained_before_final(tmp_path, sentinel):
    code = "import sys;sys.stdout.write('x'*400000 + '\\nEND_OF_BUILD\\n')"
    result, summary = invoke([sys.executable, "-c", code], 2, tmp_path)
    assert result.returncode == 0
    assert "x" * 400000 + "\nEND_OF_BUILD\n" in result.stdout
    assert not summary["stream_truncated"]
    assert not summary["stream_incomplete"]
    assert not summary["log_delivery_failed"]


@pytest.mark.parametrize(
    "failure", ["capability", "temporary_directory", "temporary_file"]
)
def test_setup_failure_still_has_final_report(tmp_path, sentinel, failure):
    code = f"""
import importlib.util,sys
spec=importlib.util.spec_from_file_location('diagnostics', {str(SCRIPT)!r})
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def unavailable(*args,**kwargs): raise OSError(38,'injected setup failure')
target={failure!r}
if target=='capability': module._pidfd_open=unavailable
elif target=='temporary_directory': module.tempfile.TemporaryDirectory=unavailable
else: module.tempfile.TemporaryFile=unavailable
sys.exit(module.supervise([sys.executable,'-c','raise RuntimeError("must not run")']))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={**os.environ, "TMPDIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    summary = report(result.stdout)
    assert result.returncode == summary["exit_code"] == 1
    assert summary["root_pid"] is None and summary["cleanup_complete"]
    assert summary["trace_error"] == "OSError"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("blocked", [False, True])
def test_real_sphinx_gallery_notebook_and_recovery(tmp_path, sentinel, blocked):
    if not hasattr(sys, "monitoring"):
        pytest.skip("fine-grained observation requires CPython 3.12+")
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("sphinx_gallery")
    source = tmp_path / "source"
    examples = tmp_path / "examples"
    source.mkdir()
    examples.mkdir()
    (examples / "README.rst").write_text("Examples\n========\n")
    body = "import time; time.sleep(30)" if blocked else "answer = 42"
    example = examples / "plot_controlled.py"
    example.write_text(f'"""Example\n=======\n"""\n{body}\n')
    original = example.read_bytes()
    (source / "conf.py").write_text(
        f"extensions=['sphinx_gallery.gen_gallery','nbsphinx']\nmaster_doc='index'\nhtml_theme='basic'\nsphinx_gallery_conf={{'examples_dirs':{str(examples)!r},'gallery_dirs':'gallery','image_scrapers':(),'download_all_examples':False}}\nnbsphinx_execute='always'\nexclude_patterns=['gallery/*.ipynb','gallery/**/*.ipynb']\n"
    )
    (source / "index.rst").write_text(
        "Test\n====\n\n.. toctree::\n\n   gallery/index\n   notebook\n"
    )
    notebook = source / "notebook.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("answer = 42")]),
        notebook,
    )
    before = notebook.read_bytes()
    code = f"from docs.tools.lifecycle_trace import observe;from sphinx.application import Sphinx\nwith observe():\n app=Sphinx({str(source)!r},{str(source)!r},{str(tmp_path / 'html')!r},{str(tmp_path / 'doctrees')!r},'html');app.build(force_all=True)\n"
    result, summary = invoke(
        [sys.executable, "-u", "-c", code], 6 if blocked else 20, tmp_path
    )
    assert result.returncode == (124 if blocked else 0)
    assert summary["trace_complete"]
    assert '"phase": "sphinx.initialize"' in result.stdout
    assert '"phase": "gallery.execute"' in result.stdout
    if not blocked:
        assert '"phase": "notebook.execute"' in result.stdout
        assert '"phase": "kernel.wait"' in result.stdout
        events = [
            json.loads(line.split("] ", 1)[1])
            for line in result.stdout.splitlines()
            if line.startswith("[docs lifecycle] ")
        ]
        kernels = {
            row["kernel_pid"]
            for row in events
            if row["phase"] == "kernel.launch" and row["status"] == "END"
        }
        assert len(kernels) == 1
        assert all(not Path(f"/proc/{pid}").exists() for pid in kernels)
        assert (tmp_path / "html" / "notebook.html").exists()
    else:
        assert summary["active_count"] >= 2
        assert '"document":' in result.stdout
    assert example.read_bytes() == original
    assert notebook.read_bytes() == before
