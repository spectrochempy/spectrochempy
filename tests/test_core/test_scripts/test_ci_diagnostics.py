"""CI summaries must describe every pytest invocation, including empty passes."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def diagnostics():
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "ci_diagnostics", root / ".github/workflows/scripts/ci_diagnostics.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_includes_all_invocations(tmp_path, monkeypatch, capsys, diagnostics):
    reports = tmp_path / "reports"
    reports.mkdir()
    for name, count, exitstatus in (("primary-1", 3, 0), ("serial", 0, 5)):
        (reports / f"{name}.json").write_text(
            json.dumps(
                {"collected": count, "executed": count, "exitstatus": exitstatus}
            ),
            encoding="utf-8",
        )
    log = tmp_path / "pytest.log"
    log.write_text("=== no tests ran in 0.01s ===\n", encoding="utf-8")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    diagnostics.print_job_summary(log, None, reports)
    text = summary.read_text(encoding="utf-8")
    assert "| primary-1 | 3 | 3 | 0 |" in text
    assert "| serial | 0 | 0 | 5 |" in text
    assert "Total test executions: **3**" in text
    assert "Total test executions: 3" in capsys.readouterr().out


def test_missing_report_directory_is_safe(tmp_path, diagnostics):
    assert diagnostics.read_execution_reports(tmp_path / "missing") == []


def test_corrupt_report_does_not_hide_valid_reports(tmp_path, capsys, diagnostics):
    (tmp_path / "primary-1.json").write_text("not json", encoding="utf-8")
    (tmp_path / "serial.json").write_text(
        '{"collected": 1, "executed": 1, "exitstatus": 1}', encoding="utf-8"
    )
    assert diagnostics.read_execution_reports(tmp_path) == [
        {"invocation": "serial", "collected": 1, "executed": 1, "exitstatus": 1}
    ]
    assert "primary-1.json" in capsys.readouterr().err


def test_log_fallback_is_preserved(tmp_path, monkeypatch, diagnostics):
    log = tmp_path / "pytest.log"
    log.write_text("=== 2 passed, 1 skipped in 1.23s ===\n", encoding="utf-8")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    diagnostics.print_job_summary(log, None)
    assert "passed: `2`" in summary.read_text(encoding="utf-8")
