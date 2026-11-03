# Bounded documentation diagnostics (phase E, local prototype)

No RTD configuration is changed by this prototype. Do not merge the phase-D
probe configuration or treat this mechanism as a fix for a scientific hang.

## Local invocation

```bash
SCPY_DOCS_TRACE=1 python -u .github/workflows/scripts/docs_build_diagnostics.py
```

The default command is the existing full `docs/make.py html --warning-is-error
-j auto`, invoked once. Do not use it for an RTD experiment without separate
authorization. For a controlled small corpus or process test:

```bash
SCPY_DOCS_TRACE=1 python -u .github/workflows/scripts/docs_build_diagnostics.py \
  --timeout 3 --grace 0.3 --kill-grace 0.3 -- python controlled_fixture.py
```

The child command must be trusted. The supervisor requires a dedicated Linux
interpreter without pre-existing children, readable `/proc`, and libc/Linux
subreaper/pidfd support. It never imports scientific libraries. Fine-grained
Sphinx/Gallery/nbconvert observation additionally requires CPython 3.12+ and the
supported documentation dependency versions; unavailability is explicitly logged.

## Recovery contract

- A new session isolates the command. Both output descriptors point to a
  regular temporary file, not the supervisor's RTD output pipe. Child processes
  cannot inherit the supervisor's private diagnostic descriptor through exec.
- Linux subreaping adopts orphaned descendants, including `setsid()` daemons.
  Ownership follows ancestry/adoption, not names or process groups. Signalling
  uses pidfds with process-start identity checks. No global process kill occurs.
- Deadline: TERM, grace, then KILL. Operator INT/TERM is propagated and followed
  by the same bounded escalation. A last KILL/reap pass runs on exceptions too.
- Exit codes: command status (negative signal status normalized), 124 for an
  internal deadline even after escalation, 128+signal for external interruption,
  70 for incomplete/erroneous cleanup, and 74 for diagnostic delivery failure
  when there is no earlier nonzero failure. Cleanup errors override other codes.
- Normal output sent to RTD is capped at 1 MiB; the report announces truncation.
  The final report is emitted after process cleanup, not by a later RTD command.
  It repeats the last lifecycle metadata and unfinished spans independently of
  scientific output volume. No cell source, cell results, or exception messages
  are included in the structured report. Existing ordinary build output remains
  ordinary build output, not a new scientific data collection mechanism.
- Lifecycle parsing is bounded at 32 MiB, 2 seconds, 16 KiB per record, 256
  tracked spans, 32 displayed active spans and 20 recent events. Budget exhaustion,
  malformed/partial records, unavailable instrumentation and omitted active
  spans are reported, not silently presented as complete diagnostics.
- Temporary files are disposable. The recovery artifact is the command's final
  stdout as captured by its caller after it exits, not an assumed durable file.

## Provenance and behavior preservation

The PID ownership, adoption and pidfd functions derive unchanged from
`ef092a238:docs/tools/supervise_build.py`. Output descriptor setup/teardown is
exception-safe; the control loop adds final structured reporting, bounded normal
streaming, explicit cleanup-failure status and TERM-first deadlines. This avoids
runtime retrieval of an unrelated Git object and introduces no dependency.

`docs/tools/lifecycle_trace.py` is the reviewed phase-C file, SHA256
`b8f0631b4e77601a7a7a6e4e1b1164d984d667c0759395b6ac5186d63137e664`.
Its original staged worktree remains untouched. `docs/make.py` uses its opt-in
span/observation contexts without changing Sphinx options, example resets,
notebook execution policy, data downloads, API generation or HTML validation.
The new upstream `--del-nb` correction is retained; phase A's staged correction
is preserved in its own worktree and is unnecessary for this integration.

## A versus B

| Criterion | A: bounded foreground command | B: sequential checkpoints |
|---|---|---|
| Termination | Dedicated owner, deadline and bounded cleanup | Each running command still needs supervision |
| Recovery | Report before the same command returns nonzero | A later report is skipped after a failing/hung command |
| RTD | Normal completed-command capture | Shared files do not imply background-process survival |
| Orphans | Subreaper plus identity-checked signalling | Requires additional cross-command ownership protocol |
| Complexity | Reuses established primitives | Must serialize state, offsets and final status |
| Runtime | Small observation/report cost to measure | More commands/startup; accidental repeated preparation possible |
| Content | Sphinx initialization/build remain together | Must not split live Sphinx state or rerun notebooks/Gallery |
| Portability | Linux supervision, optional cross-platform traces | Shell/file sharing alone is insufficient on any platform |

Choose A for the next controlled probe. B is not claimed impossible, but its
background variant was not established by phase D and is not needed here.

## Time budget and limits

The provisional local default is 600 seconds from supervisor startup, chosen
against successful documentary commands of roughly 436 seconds, not against
the 24-minute creation-to-failure observations. Add 15 seconds TERM grace,
5 seconds escalation, up to 5 seconds last cleanup, 2 seconds parsing and
bounded flushes: reserve at least 630 seconds from command startup. If an outer
guard is later approved, allow at least 660 seconds rather than killing before
the internal cleanup completes. The project's actual remaining RTD budget must
be confirmed first; its public project API does not expose that limit. These
numbers are not an authorization or a guaranteed RTD time allocation.

No recovery guarantee survives supervisor SIGKILL/OOM, worker destruction, an
unwritable/full diagnostic filesystem, an externally blocked/broken collector,
uninterruptible kernel tasks, unsupported ownership facilities, or adversarial
process behavior. Output writes are nonblocking and bounded, so sink failure
must fail rather than indefinitely delay cleanup; delivery itself cannot be
guaranteed to a broken sink. Persistent live children are a reported failure,
never a successful termination. High-volume or malformed trace files have an
explicitly incomplete summary. OS scheduling and `/proc` access are assumptions,
not hard-real-time guarantees.

## Proposed next gate — no current remote execution

One short RTD experiment, no scientific installation/build: a controlled actor
emits a named START event, creates an owned detached TERM-ignoring descendant,
then blocks. Use a 3-second internal deadline and short coherent grace/KILL
budgets. Expected outcome: final report, last named event, explicit cleanup
status, empty survivor list, and exit 124 in the persisted RTD command log.
RTD must mark this build failed; do not hide 124 or depend on `post_build`.
Obtain a new maintainer decision before publishing/triggering this experiment.
Only after it passes and the remaining service budget is confirmed may a full
documentation experiment be considered. No external log transfer is proposed.
