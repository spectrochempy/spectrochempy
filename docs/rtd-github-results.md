# Prepare notebooks and galleries on GitHub, publish on Read the Docs

The documentation can keep Read the Docs hosting, versions, search, subprojects
and pull request previews while executing its notebooks and galleries on GitHub.
RTD installs the documentation dependencies, retrieves the prepared sources for
its exact commit, verifies them and renders the integrated Sphinx site.

The two execution jobs run in parallel in `docs-artifact-prototype.yml`. Its
render job checks that their results build together, including the released NMR
plugin. Failed execution or strict rendering prevents publication. Every build
executes fresh sources; this does not introduce a cross-commit execution cache.

## Transfer and access

`docs-rtd-transfer.yml` runs trusted code from `master` after a successful source
workflow. It verifies that the commit is still the current `master` or the head
of an open PR targeting `master`. It then obtains temporary GitHub storage links
and publishes them in a GitHub check for that RTD project. It starts the RTD
build using the existing `SCPY_RTD_API_TOKEN` Actions secret and refreshes the
links every 20 seconds until the build finishes, for at most 25 minutes.

Automatic transfers are serialized per RTD project, source repository and
branch. Two forks with the same branch name have independent queues. A queued
transfer checks the current PR head before starting RTD and skips superseded
commits.

RTD uses the public check and public GitHub metadata to verify the source run,
commit, artifact IDs and archive digests. The existing source manifests also
verify the package version and each restored file. RTD receives no GitHub token.
The transfer job never checks out or executes PR code and its RTD credential
remains on the trusted GitHub runner. PR jobs do not receive private Corn data;
they retain the existing public Corn download. Trusted master execution retains
the separate private Corn input. Only documentation outputs are transferred.

## Enable and verify

1. Merge the workflows with automatic operation disabled: leave the GitHub
   repository variable `RTD_RESULTS_PROJECT` unset.
2. On the RTD test project `spectrochempy-artifacts-test`, add the **public**
   environment variable `SCPY_RTD_GITHUB_RESULTS=1`. Public is necessary for
   external PR previews; the variable contains no credential. Leave the legacy
   `SCPY_RTD_DOCS_ARTIFACTS` option disabled.
3. Dispatch **Documentation artifact prototype** on `master`. After it succeeds,
   dispatch **Publish prepared documentation on RTD** with its run ID and the
   test project. The transfer workflow must succeed, and RTD logs must show
   downloads of both artifacts and rendering from that same run, without
   notebook or gallery reexecution. Inspect figures, notebook downloads, gallery
   pages, API mini-galleries, navigation and NMR links. Measure RTD installation,
   download and rendering time separately from the GitHub execution time.
4. Set `RTD_RESULTS_PROJECT=spectrochempy-artifacts-test` on GitHub to test
   automatic master builds. Prepared PR builds have a separate gate: enable
   external builds on the test project, add the public RTD flag
   `SCPY_RTD_GITHUB_RESULTS_PR=1` and set `RTD_RESULTS_PREVIEWS=1` on GitHub.
   Test a PR from a fork, including all Corn examples, before enabling this
   extra flag on the main project. RTD's initial webhook build exits with code
   183 before
   installation when the prepared sources are not ready. The transfer job starts
   the actual RTD build when preparation succeeds. A failed GitHub job leaves
   the previously published site intact and must be fixed or rerun.
5. After the hosted `latest` tests succeed, add `SCPY_RTD_GITHUB_RESULTS=1`
   on the main RTD project and change `RTD_RESULTS_PROJECT` to `spectrochempy`.
   Verify a real `latest` build there. PR previews retain native RTD execution
   until their separate flags are enabled after successful fork validation.

Public Corn data are fetched once per execution job, with a 15-second network
timeout. A successful download must match the pinned size and SHA-256; the
verified local file and ZIP are then reused by all examples. A malformed or
changed download fails the job. A network failure marks Corn as unavailable:
only the four Corn gallery examples and notebook cells tagged `requires-corn`
are left unexecuted. Their source remains visible, with an explicit notice.
The artifact manifests and Actions summary list the omitted results.
Each affected gallery page displays a named **Corn unavailable** notice.
The HTML build checks that these notices are present before publication.

Trusted master builds still require their separately verified private Corn
fixture. They never silently fall back to incomplete scientific documentation.
Enable the PR gates only after an actual fork preview has validated both the
successful-download and unavailable-data paths. PRs opened from older branches
should incorporate the current `master` before using the prepared results;
in particular they need the public Corn preparation and omission handling.
No raw Corn input is included in public artifacts.

The main flag selects `latest`; the additional PR flag selects external versions.
Release tags, `stable`
and older immutable checkouts keep their native RTD build. Automatic preparation
on master is gated by `RTD_RESULTS_PROJECT`. PR preparation runs
on the official repository without relying on configuration variables in the
fork context. The separate `RTD_RESULTS_PREVIEWS` gate controls the trusted
transfer after preparation; removing it leaves RTD previews native when the
RTD PR flag is also removed. Manual producer dispatch remains available.
Manual transfer dispatch accepts only the official and test projects.
A transfer for a superseded commit is skipped. The transfer fails if RTD fails,
exceeds its budget or publishes a different commit.

## Roll back

Remove `RTD_RESULTS_PROJECT` and `RTD_RESULTS_PREVIEWS` from GitHub and remove
`SCPY_RTD_GITHUB_RESULTS` and `SCPY_RTD_GITHUB_RESULTS_PR` on
RTD, then rebuild the desired version. RTD will execute notebooks and galleries
natively again. Do not activate the legacy token-based option on external PRs.
GitHub Pages workflows are independent and are not needed for this arrangement.
