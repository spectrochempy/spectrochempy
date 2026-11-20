"""Count test executions on the pytest controller and reject empty CI jobs."""

import json
import os
import sys
from pathlib import Path


class ExecutionCounts:
    def __init__(self):
        self.executed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call" and (
            not report.skipped or hasattr(report, "wasxfail")
        ):
            self.executed += 1

    def pytest_sessionfinish(self, session, exitstatus):
        if hasattr(session.config, "workerinput"):
            return
        path = Path(os.environ["SCPY_CI_TEST_COUNTS"])
        path.write_text(
            json.dumps(
                {
                    "collected": session.testscollected,
                    "executed": self.executed,
                    "exitstatus": int(exitstatus),
                }
            ),
            encoding="utf-8",
        )


def pytest_configure(config):
    config.pluginmanager.register(ExecutionCounts(), "ci-execution-counts")


def check_execution(directory):
    reports = sorted(Path(directory).glob("*.json"))
    executed = 0
    if not reports:
        raise ValueError("No pytest execution reports were produced")
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        count = report["executed"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"Invalid executed-test count in {path.name}")
        executed += count
    if executed == 0:
        raise ValueError("The CI job did not execute any tests")
    return executed


if __name__ == "__main__":
    try:
        count = check_execution(sys.argv[1])
    except (ValueError, KeyError, OSError) as error:
        print(f"::error::{error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(f"CI job executed {count} tests")
