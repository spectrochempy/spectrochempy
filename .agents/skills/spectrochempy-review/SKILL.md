---
name: spectrochempy-review
description: >-
  Independent code review in a temporary subagent session. Use when a separate review is
  needed: the reviewer starts from the need, the diff, and sources — not the
  implementer's history. Checks assumptions, finds counter-examples and
  regressions, evaluates whether tests prove the expected behavior.
metadata:
  audience: maintainer
  workflow: review
---

## Trigger

Use this skill for a requested **separate review** and for the mandatory
independent review before every PR merge. This review runs in a **temporary
subagent session** attached to the current task, with the review prompt and
repository access but without the implementation history. Return the report
to the parent agent.

For a PR intended for merge, use this skill even for documentation,
formatting, or test-only changes.

---

## Context

The reviewer works from:

1. **The original need** — problem statement or requirement provided in the
   handoff prompt.
2. **Applicable instructions** — `AGENTS.md`, `CONTRIBUTING.md`, and only the
   relevant conventions from the `spectrochempy-dev` skill (read the specific
   sections needed, not the skill in full).
3. **The diff** — exact base and head commits or the local diff snapshot.
4. **Relevant sources** — the implementation code and tests.

The implementer's self-report is **information to verify**, not authority.
Trust the code and tests, not the summary.

---

## Inputs

A handoff prompt from the implementer containing:

* Need / requirement.
* Branch name or PR number.
* Exact base and head commits, or exact base commit and local diff snapshot.
* Sensitive points flagged by the implementer.

**Do not ask the maintainer for missing elements.** First, try to find them:
* Need: read the PR description or linked issue.
* Base and commit: use `gh pr view` or `git log` to identify the merge base
  and the commit to review.
* Sensitive points: examine the diff and identify them yourself.

Only ask the maintainer if a genuine ambiguity remains after investigation.

---

## Procedure

### 1. Establish the diff

For uncommitted work, verify the supplied base commit and review the exact
diff snapshot from the handoff. Report which snapshot the verdict covers.
For committed work, use the base and head commits as follows.

```bash
git diff <base>..<commit>
# or
gh pr diff <PR_NUMBER>
```

Verify the base and commit match what was requested. If the branch has been
modified since the handoff prompt, stop and report — the review targets a
specific commit.

### 2. Reconstruct the need

From the handoff prompt, restate the requirement in your own words. Identify:
* What problem does this solve?
* What behavior should change?
* What behavior must be preserved?

### 3. Check assumptions

For each change in the diff, identify the underlying assumptions:

* **Domain assumptions** — what does the code assume about the data (shape,
  range, monotonicity, units)?
* **Contract assumptions** — what does the code assume about callers, API
  consumers, or serialization format?
* **Environmental assumptions** — platform, dependencies, ordering.

Verify each assumption against the actual code, documentation, or tests.
Flag assumptions that are undocumented, untested, or violated by existing
usage.

### 4. Search for counter-examples

For each behavioral change, try to find inputs that would produce incorrect or
unexpected results:

* Edge cases (empty arrays, single element, negative values, NaN, infinity).
* Type variations (different dtypes, units, axis orderings).
* Interactions with existing features not explicitly tested.

If a counter-example is reproducible, report it with the exact input and
expected vs. actual behavior.

### 5. Evaluate regression risk

* Does the change modify code paths used by other features?
* Are existing tests sufficient to catch regressions?
* Could the change interact with concurrent work?
* Is backward compatibility preserved (API, serialization, file formats)?

### 6. Evaluate test adequacy

For each behavioral claim in the implementation:

* Is there a test that demonstrates the expected behavior?
* Does the test actually exercise the changed code path?
* Is the test assertion precise enough to catch the regression?
* Does the test cover both success and failure cases?

If tests are missing or insufficient, report what should be added — but do
**not** write or modify code unless explicitly asked.

### 7. Check project conventions

* Code follows existing patterns.
* No unnecessary new dependencies.
* Commit messages use correct prefix.
* Changelog entries link to the correct PR in the owning repository, or the
  `no-changelog` label is justified.

---

## Deliverable

Return the report to the parent session for the task record. A review does not
require a separate maintainer file. Preserve the findings, validation, verdict
and exact reviewed revision; promote reusable conclusions into an existing
shared document only when they warrant it.

Produce a review report with the following structure:

### Summary

One or two sentences assessing what the change does relative to the stated
need.

### Findings

For each finding, report:

* **Location** — file and line number.
* **Defect** — what is wrong or at risk.
* **Consequence** — what happens if the defect is not addressed.
* **Evidence** — the input, test, or code path that demonstrates it.

Classify findings as:

| Category | Meaning |
|---|---|
| **Established defect** | The code is wrong or will fail under documented conditions. |
| **Uncertainty** | An assumption is unverified or could not be confirmed from sources. |
| **Preference** | A style or design suggestion — not a correctness issue. |

### Verdict

State one explicit verdict and record the exact reviewed head commit or local diff snapshot.

* `APPROVE` — no unresolved actionable findings for this head.
* `REQUEST_CHANGES` — established defects require correction.
* `NEEDS_DISCUSSION` — an uncertainty or design decision requires resolution.

Only `APPROVE` for the current PR head satisfies the independent review
requirement. Any new head requires a fresh independent session.

### Validation

Propose targeted test commands that would confirm or refute the key concerns.

---

## Rules

* **Do not modify code** unless explicitly asked. This is a review, not an
  implementation session.
* **Trust code and tests**, not the implementer's summary.
* **Distinguish facts from judgments** — established defects vs.
  uncertainties vs. preferences.
* **Do not delegate the review.** Return the report to the parent agent
  and stop.

---

## Validation (Skill Exit)

The review is complete when:

* All behavioral changes have been examined.
* Assumptions are verified or flagged.
* Findings are reported with location, consequence, and evidence.
* Verdict is justified by the findings.
* Validation commands are proposed.
