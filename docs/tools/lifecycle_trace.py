"""Opt-in lifecycle observation without replacing dependency functions."""
# ruff: noqa: PLC0415

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from contextlib import suppress
from datetime import UTC
from datetime import datetime
from pathlib import Path


def emit(phase, status, **fields):
    if os.environ.get("SCPY_DOCS_TRACE") != "1":
        return
    record = {
        "utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "monotonic": time.monotonic(),
        "pid": os.getpid(),
        "phase": phase,
        "status": status,
        **fields,
    }
    line = "[docs lifecycle] " + json.dumps(record, sort_keys=True)
    with suppress(OSError, ValueError):
        print(line + "\n", end="", file=sys.__stderr__, flush=True)
    if filename := os.environ.get("SCPY_DOCS_TRACE_FILE"):
        with suppress(OSError, ValueError), Path(filename).open(
            "a", encoding="utf-8"
        ) as stream:
            print(line + "\n", end="", file=stream, flush=True)


@contextmanager
def span(phase):
    if os.environ.get("SCPY_DOCS_TRACE") != "1":
        yield
        return
    started = time.monotonic()
    token = f"{os.getpid()}:{threading.get_ident()}:{time.monotonic_ns()}"
    emit(phase, "START", span=token)
    outcome = "ok"
    try:
        yield
    except BaseException:
        outcome = "error"
        raise
    finally:
        emit(
            phase,
            "END",
            span=token,
            duration=time.monotonic() - started,
            outcome=outcome,
        )


def _targets():
    from jupyter_client.provisioning.local_provisioner import LocalProvisioner
    from nbconvert.preprocessors import ExecutePreprocessor
    from sphinx.application import Sphinx
    from sphinx.builders import Builder
    from sphinx.builders.html import StandaloneHTMLBuilder
    from sphinx.util.parallel import ParallelTasks
    from sphinx.util.parallel import SerialTasks
    from sphinx_gallery.gen_rst import execute_script
    from sphinx_gallery.gen_rst import generate_file_rst

    return {
        Sphinx.__init__.__code__: "sphinx.initialize",
        Sphinx.build.__code__: "sphinx.build",
        Builder.build.__code__: "sphinx.generate",
        Builder.read.__code__: "sphinx.read",
        Builder.read_doc.__code__: "sphinx.document",
        Builder.write.__code__: "sphinx.write",
        StandaloneHTMLBuilder.handle_page.__code__: "sphinx.html",
        StandaloneHTMLBuilder.finish.__code__: "sphinx.finish.schedule",
        SerialTasks.join.__code__: "sphinx.tasks.join",
        ParallelTasks.join.__code__: "sphinx.tasks.join",
        generate_file_rst.__code__: "gallery.generate",
        execute_script.__code__: "gallery.execute",
        ExecutePreprocessor.preprocess.__code__: "notebook.execute",
        ExecutePreprocessor.preprocess_cell.__code__: "notebook.cell",
        LocalProvisioner.launch_kernel.__code__: "kernel.launch",
        LocalProvisioner.wait.__code__: "kernel.wait",
    }


def _context(phase, frame):
    values = frame.f_locals
    context = {}
    if phase.startswith("gallery."):
        if phase == "gallery.generate":
            context["document"] = str(Path(values["src_dir"]) / values["fname"])
        else:
            variables = values["script_vars"]
            if not variables["execute_script"]:
                return None
            context["document"] = str(variables["src_file"])
    elif phase.startswith("notebook."):
        resources = values.get("resources") or getattr(values["self"], "resources", {})
        context["document"] = resources.get("nbsphinx_docname", "<notebook>")
        if phase == "notebook.cell":
            context["cell_index"] = values["index"]
    elif phase.startswith("sphinx."):
        if "docname" in values:
            context["document"] = values["docname"]
        elif "pagename" in values:
            context["document"] = values["pagename"]
    elif phase.startswith("kernel."):
        context["kernel_pid"] = getattr(values["self"], "pid", None)
    return context


@contextmanager
def observe():
    """Observe pinned dependency code with local CPython monitoring events."""
    if os.environ.get("SCPY_DOCS_TRACE") != "1":
        yield
        return
    monitoring = getattr(sys, "monitoring", None)
    if monitoring is None:
        emit("monitoring", "UNAVAILABLE", reason="requires CPython 3.12 or newer")
        yield
        return
    try:
        targets = _targets()
    except (ImportError, AttributeError):
        emit("monitoring", "UNAVAILABLE", reason="dependency code not supported")
        yield
        return
    tool = None
    for candidate in range(6):
        try:
            monitoring.use_tool_id(candidate, "scpy-docs-lifecycle")
        except ValueError:
            continue
        tool = candidate
        break
    if tool is None:
        emit("monitoring", "UNAVAILABLE", reason="all monitoring IDs in use")
        yield
        return
    active = {}
    after_cells = {}

    def start(code, offset):
        frame = sys._getframe(1)
        try:
            phase = targets[code]
            context = _context(phase, frame)
            if context is None:
                return
            key = (os.getpid(), threading.get_ident(), id(frame))
            started = time.monotonic()
            token = ":".join(map(str, (*key, time.monotonic_ns())))
            active[key] = (started, phase, context, token)
            emit(phase, "START", span=token, **context)
        except Exception:
            emit("monitoring", "ERROR", reason="start observation failed")

    def end(code, offset, result):
        finish(sys._getframe(1), result, False)

    def unwind(code, offset, exception):
        if code in targets:
            finish(sys._getframe(1), None, True, type(exception).__name__)

    def finish(frame, result, failed, exception_type=None):
        try:
            key = (os.getpid(), threading.get_ident(), id(frame))
            item = active.pop(key, None)
            if item is None:
                return
            started, phase, context, token = item
            fields = dict(context)
            outcome = "error" if failed else "ok"
            if exception_type:
                fields["exception_type"] = exception_type
                if "Timeout" in exception_type:
                    outcome = "timeout"
            if (
                phase == "gallery.execute"
                and "formatted_exception" in frame.f_locals["script_vars"]
            ):
                outcome = "error"
            if phase == "gallery.generate" and result is not None:
                variables = result[3]
                if "formatted_exception" in variables:
                    outcome = "error"
                elif "stale" in variables:
                    outcome = "cached"
            if (
                phase == "notebook.cell"
                and not failed
                and any(
                    output.get("output_type") == "error"
                    for output in result[0].get("outputs", [])
                )
            ):
                outcome = "cell_error_allowed"
            if phase == "notebook.cell" and not failed:
                values = frame.f_locals
                if values["index"] == len(values["self"].nb.cells) - 1:
                    after_cells[(os.getpid(), id(values["self"]))] = (
                        time.monotonic(),
                        token + ":after_cells",
                    )
                    emit(
                        "notebook.after_cells",
                        "START",
                        span=token + ":after_cells",
                        **context,
                    )
            if phase == "notebook.execute":
                after = after_cells.pop((os.getpid(), id(frame.f_locals["self"])), None)
                if after is not None:
                    emit(
                        "notebook.after_cells",
                        "END",
                        span=after[1],
                        duration=time.monotonic() - after[0],
                        outcome=outcome,
                        **context,
                    )
            if phase.startswith("kernel."):
                fields["kernel_pid"] = getattr(frame.f_locals["self"], "pid", None)
                if phase == "kernel.wait" and not failed:
                    fields["returncode"] = result
            emit(
                phase,
                "END",
                span=token,
                duration=time.monotonic() - started,
                outcome=outcome,
                **fields,
            )
        except Exception:
            emit("monitoring", "ERROR", reason="end observation failed")

    events = monitoring.events
    try:
        monitoring.register_callback(tool, events.PY_START, start)
        monitoring.register_callback(tool, events.PY_RETURN, end)
        monitoring.register_callback(tool, events.PY_UNWIND, unwind)
        monitoring.set_events(tool, events.PY_UNWIND)
        for code in targets:
            monitoring.set_local_events(tool, code, events.PY_START | events.PY_RETURN)
        emit("monitoring", "START", functions=len(targets))
        yield
    finally:
        monitoring.set_events(tool, 0)
        for code in targets:
            monitoring.set_local_events(tool, code, 0)
        for event in (events.PY_START, events.PY_RETURN, events.PY_UNWIND):
            monitoring.register_callback(tool, event, None)
        monitoring.free_tool_id(tool)
        for started, phase, context, token in active.values():
            if token.startswith(f"{os.getpid()}:"):
                emit(
                    phase,
                    "INCOMPLETE",
                    span=token,
                    duration=time.monotonic() - started,
                    **context,
                )
        active.clear()
        after_cells.clear()
        emit("monitoring", "END")
