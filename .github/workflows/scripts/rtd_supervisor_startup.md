# F.1 startup experiment — local investigation, not a deployed fix

Publication-branch adaptation: this branch activates only the installed-environment
startup comparison, preceded by a separate completed shell command. The shell
marker is independent of Python but cannot prove that the following Python exec
occurred. The candidate `docs_controller_entry.py` is tested locally for a future
CLI launch; it is deliberately not inserted into the two measured controllers,
which continue to call the same validated core. No full documentation build is
authorized. Do not merge this experimental configuration.

The entry candidate catches escaped import/main exceptions and emits a separate
`[docs controller FAILURE]` record with code 1. It adds no cleanup and explicitly
does not infer the supervisor's ownership/cleanup state. It preserves the
pre-existing-child refusal and cannot report an interpreter that never reaches
this entry code. The pairing intentionally does not activate this candidate:
the measured core remains the same in both modes.

Protocol provenance: `89bba37bc8119355afdeb507d7a528efb3e2c38b` (phase F).
Publication base: `b583ccb2c0f560086ee6460f0252b87db1f3e7cc` (normal master).
The production supervisor is unchanged. The original F.1 audit worktree remains
unpublished and unchanged; only this dedicated publication branch activates the
short configuration, after local validation, review and maintainer approval.

## What E.2 versus F does and does not establish

| Dimension | E.2 | F |
|---|---|---|
| Controller | Python 3.13.14, `-I -S -u` | Python 3.13.14, `-u` |
| Documentation packages | Not installed | Installed with the three pins and plugins |
| Child | Isolated controlled actors | Normal `python -u docs/make.py ...` |
| Deadline/grace/KILL | 3 / 0.3 / 0.3 seconds | 600 / 15 / 5 seconds |
| Retained result | Actual RTD 124, final report, cleanup | Global timeout, command status/output unavailable |

These experiments differ in more than isolation flags. There is no recorded
phase-F `PYTHONPATH`, startup-module inventory, first Python instruction, or
pre-existing-child snapshot. Missing output does not prove that no controller
instruction executed: RTD saves completed-command output. Several earlier normal
Python installation commands succeeded; an unconditional startup hang affecting
every interpreter is therefore not established either.

## Static chronology in the unchanged supervisor

| Point | Operations | Time protection |
|---|---|---|
| Before script body | Interpreter setup, `site`, path search, executable `.pth`, `sitecustomize`/`usercustomize` where enabled | Outside the internal timer |
| Before `main()` | stdlib imports; `ctypes.CDLL(None)` | Outside |
| Inside `main()` | argparse and command/default path construction | Outside |
| Before `started` assignment | Argument validation, `/proc` scan, refusal of pre-existing children | Outside; refusal precedes the final-report try block |
| After clock, before `Popen` | Output fd setup, subreaper, pidfd preflight, signal handlers, temporary files/environment | Elapsed time accrues, but no asynchronous interruption |
| During `Popen` | OS launch and exec-error-pipe handling | A blocking call cannot be preempted by this timer |
| Supervision loop | Poll, ancestry scan, regular-file read and output flush, then deadline check | Cooperative checks only after synchronous calls return |
| Cleanup | Signals, `/proc` reads, wait/poll, final reap budget | Cooperative budget; uninterruptible OS calls remain outside guarantees |
| Report/exit | Trace parser, nonblocking output/flushes, handler/subreaper/fd restoration, interpreter finalization | Separate bounded loops, not a hard real-time global kill |

The timer is a monotonic timestamp plus comparisons, not a preemptive alarm.
Finite injected delays demonstrate coverage boundaries; they do not demonstrate
that the corresponding OS operation blocked on RTD.

## Measurement program

`rtd_supervisor_startup_probe.py` is stdlib-only and calls the existing
supervisor once per measured controller. It does not invoke Sphinx, Gallery,
nbsphinx or a kernel. The child remains normal Python in both modes, preserving
its installed startup environment; only the controller flags change.

Run a local paired observation:

```bash
python -I -S -u .github/workflows/scripts/rtd_supervisor_startup_probe.py compare --timeout 3
```

The isolated measuring process launches a normal controller and an isolated
controller, each through the same unchanged supervisor. Its `communicate` guard
and teardown are test-harness safety, not a second documentation supervisor.
It reuses the existing subreaper/pidfd cleanup primitives only to collect its
own measurement processes on teardown. It refuses pre-existing children.

Milestones are recorded to a task-specific evidence file before calls and at
returns; the external observer harvests them even if a controller cannot finish.
The first `body_enter` timestamp follows a minimal builtin `time` import: it is
not an instrumented first opcode. Launch-to-body also includes OS scheduling
and process creation. Profiling and metadata I/O add overhead, warm stdlib
imports and can perturb timings; the measurements are not a native phase-F
performance replay. The first `time.monotonic` return inside `supervise` observes
initialization of the elapsed-time clock, within callback overhead.

`controller_exit` is a pre-kernel-exit announcement, not an independently timed
waitpid result. `capture_complete_observed` is when `communicate` completes both
capture and waiting; it must not be relabelled as an independently measured EOF.
The observer reports the actual reaped controller code separately.

Environment output is allowlisted: flags, customization-module presence,
scientific-module presence, working directory/fd destinations, and `.pth` names
and executable-line counts. `.pth` inventory alone is not proof that a file ran.
No raw environment values, `.pth` payloads, cell contents, scientific results or
secrets are exported. Evidence files are temporary, not durable RTD artifacts.

## Controlled counter-examples

- Slow `sitecustomize` or a controlled user-site hook delays the normal
  controller before its body/clock; `-I -S` bypasses that controller startup path.
  The normal child still inherits the customization and is stopped by the timer.
- Delays before `Popen`, before its return, in loop scans, temporary-file setup
  and output flush can overrun the numerical deadline before it is checked.
- A controlled startup customization can create a child retaining the capture
  pipe. The normal controller starts, refuses the pre-existing child before
  clock initialization and produces no supervisor FINAL. Capture can then wait
  for that child despite a controller exit announcement. The observer safely
  reaps the finite experiment. With an isolated controller, the same startup
  behavior occurs in the supervised child's regular-file capture instead.

The last case is a reproduced possible mechanism, not evidence that phase F
actually contained a startup child or customization. Do not remove the existing
child-free safety check or start killing arbitrary pre-existing processes.

## Proposed RTD gate — additional authorization required

In the original audit worktree, `.readthedocs.startup-probe.yaml` was an inactive
proposal. In this dedicated publication branch, `.readthedocs.yaml` mirrors that
configuration and adds the separate shell milestone before the paired observer.
Phase-F Ubuntu/Python/docs installations, all pins and checkout/standalone plugins
are retained. Publication still requires configuration review and exact GitHub
text approval; do not modify PR #1828. No data download, notebook sync or
scientific documentation execution follows installation.

The observer returns 0 only if both controllers really returned 124 with complete
reports and safe cleanup. Otherwise it returns failure while reporting both
outcomes. Thus RTD command 0 is an observation result, **not** a claim that RTD
directly recorded either inner 124 or that a documentation build succeeded.
A static infrastructure-only HTML marker satisfies RTD's success artifact
requirement; no diagnostics are automatically published into HTML.

Normal paired runtime is roughly twice the short controller budget; each
observation guard is 8 seconds plus bounded teardown. OS/FS responsiveness and
collectable output remain assumptions. A globally killed or blocked observer
cannot recover its own report; this proposal does not claim otherwise. No
startup fix or new RTD run is authorized by preparing these local files.
