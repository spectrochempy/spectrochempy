# GitHub Pages migration

Status: prepared, disabled until qualification. Read the Docs remains available
for comparison and rollback. No DNS change is part of installing these files.

This is a fallback option while native Read the Docs build optimizations are
evaluated. These workflows do not change RTD execution or decide a cutover.

## Publication model

Each repository owns its Sphinx build, Pages site and `gh-pages` snapshot branch.
The Core and NMR sites have independent release histories. The shared navigation
links both projects; it does not assume equal versions or equal PR numbers.

| Route | Content |
| --- | --- |
| `/latest/` | Current successful `master` build |
| `/stable/` | Highest published final release, never a release candidate |
| `/<tag>/` | Only the current final release; older HTML is downloadable from Releases |
| `/previews/pr-<number>/` | Current successful open-PR build, including forks |
| `/` | Redirect to the highest published final release, or `latest` initially |

Tags accepted are `spectrochempy-vX.Y.Z` (Core),
`spectrochempy-nmr-vX.Y.Z` (NMR) and `vX.Y.Z`, optionally followed by `rcN`.
Existing RTD `latest` links in the generated HTML are rewritten to the
configured Core/NMR targets; historical release URLs are preserved.

A published GitHub release also triggers the build. Its complete HTML is saved
as `documentation-html.zip` on that existing release before updating Pages.
The workflow never creates a release or replaces an existing archive asset.
Tag-only builds wait for a published release before publication; the release
publication event starts a fresh build. Release candidates are archived as ZIPs
without taking online storage or moving `stable`.

Only tags containing these workflows can trigger their automatic build.
Historical tags are not rewritten or retrospectively presented as having docs.

`docs-pages-build.yml` executes documentation with read-only repository access.
Core reuses the existing notebook/gallery artifact pipeline; NMR executes
Sphinx and gallery examples serially using its existing dependency resolution
and fixture fetch. Neither workflow pretends that a non-executing preview
validates scientific examples. Core's trusted master/tag builds retain the
existing private Corn fixture action; fork/PR builds receive no fixture key.

`docs-pages-publish.yml` runs from trusted `master` after a successful build.
It verifies the source revision against the current PR/master/tag, downloads
only `pages-html`, rejects unsafe paths and links, and never executes code
from that artifact. It replaces only the intended subtree, saves `gh-pages`,
and publishes the active site through the official Pages actions.
Publication jobs are serialized. Closing a PR removes its preview. A successful
PR publication adds a **Documentation preview** check with the direct link.
Previews are public HTML under the site's origin and marked `noindex,nofollow`;
they must contain no private information, credentials or authenticated apps.
The build and publication logs remain available in Actions.

## Configure and qualify

Before landing these workflows, apply the `docs-pages-qualify` label to their
pull request to run the read-only build and inspect the `pages-html` artifact.
The label does not enable publication or private Corn access. Removing it
prevents subsequent qualification builds while `DOCS_PAGES_ENABLED` is unset.

1. Land the reviewed workflows and helpers on `master` in both repositories.
   Do not activate an unreviewed publisher. Its files must exist on the default
   branch for `workflow_run` and closed-PR cleanup to work.
2. In Settings → Pages, select **GitHub Actions** as the publishing source.
   Configure the `github-pages` environment to allow the default branch.
   A required environment reviewer will also gate PR preview publication.
   Allow the workflow's token to write repository contents, Pages and checks;
   branch rules must permit the bot to replace the dedicated `gh-pages` snapshot
   with a guarded force push. No source branches are rewritten.
   Do not use that branch for any other content or publishing workflow.
   An existing branch without the `.pages-managed` marker is refused: inspect
   and migrate or rename it explicitly before activating the publisher.
3. Leave `DOCS_PAGES_ENABLED` unset and run **Documentation Pages build**
   manually on `master` in each repository. Download and inspect `pages-html`.
   This builds without publishing. Compare with RTD: index, galleries, API,
   downloads, figures, `objects.inv` and local search.
4. Set repository variable `DOCS_PAGES_ENABLED=true` and run the build again
   on `master`. This enables automatic master, tag and PR builds/publication.
   The default URLs are
   `https://spectrochempy.github.io/spectrochempy/latest/` and
   `https://spectrochempy.github.io/spectrochempy_nmr/latest/`.
5. Open a small test PR in each repository, then a fork PR. Check the preview
   check's URL, version menu, nested pages, static assets and local search.
   Push another commit and verify the preview updates. Close the PR and verify
   the directory disappears without changing `latest` or version archives.
   Test published final and candidate releases: both must get an archive ZIP;
   only the final release advances `stable`. The previous final release must
   disappear from Pages while its release ZIP remains downloadable. The
   `stable/` URL is a small redirect to the current tag, without copying assets.
6. Verify both sites before changing the domain or disabling RTD. Preserve old
   paths explicitly when planning DNS: two project sites do not automatically
   recreate RTD's `/projects/nmr/` routing or redirect its hosted domains.

Optional variables in **both repositories**:

- `DOCS_CORE_URL`: deployed Core documentation URL for cross-project navigation;
  also the NMR development Intersphinx target.
- `DOCS_NMR_URL`: deployed NMR documentation URL for cross-project navigation.
- `DOCS_CORE_RELEASE_URL` (**NMR only**): immutable, compatible Core documentation
  URL for NMR tag builds, for example
  `https://spectrochempy.github.io/spectrochempy/spectrochempy-v1.4.0/`.
  Deploy and verify that archive before building a NMR release. NMR release
  validation rejects a moving `latest`/`stable` inventory. This explicit mapping
  is independent of the version of Core installed by requirements-rtd.txt.

Set custom-domain URLs explicitly; the Pages deployment returns its own URL
for preview links. NMR's existing `NMR_CORE_DOCS_URL`/Intersphinx configuration
resolves Core objects; navigation links alone do not provide object resolution.
Core has no new mandatory NMR inventory dependency, so it can bootstrap first.
Coordinate Core/NMR stable updates: older Core tag URLs leave Pages when a new
stable arrives. Archived NMR HTML retains its original external links; those
links are not guaranteed to stay online. Preserve the corresponding Core ZIP
for historical documentation, and choose an available inventory when rebuilding.
Sphinx's local full-text search stays available independently on each site.
RTD's federated search, build dashboard and redirects are not reproduced by
these workflows. A unified search index would be separate work.

## Limits, recovery and validation

The publisher refuses files of 95 MB or larger and a retained tree of 950 MB
or larger, leaving room below the Git/Pages limits. This budget applies
separately to Core and NMR. There is only one full stable tree, one latest tree,
and the active previews. There are no historical full trees online. Measure
actual complete builds before cutover; the presence of small local page
previews is not evidence that the complete site fits. PR previews persist
until close, without automatic age expiration. If storage is
full, close obsolete PRs or reduce the built site deliberately, then rebuild.
The publisher uses `queue: max` so up to 100 publication/cleanup runs wait in
order without replacing one another. Rerun a rejected run if that queue fills;
obsolete PR and master revisions are checked again at publication time.

Unset `DOCS_PAGES_ENABLED` to stop automatic builds and publication. Keep RTD
active until migration acceptance; restore its domain settings if rolling back
a later domain cutover. `gh-pages` is a disposable single-commit snapshot, replaced with a lease to
prevent overwriting a concurrent update. It does not accumulate old binary
sites in branch history. Durable archives belong to Releases. A failed Pages
deployment can be retried without discarding the active site. To refresh
`latest`, rerun the build on current `master`. Builds of obsolete PR revisions
are deliberately ignored. No external deployment has been validated merely
by passing the helper tests below.

Run the focused checks from either repository:

```console
python -m pytest --noconftest -o addopts= tests/test_docs/test_pages_publication.py -q
node --test .github/pages/test_resolve.cjs
```

Keep the identical publication helpers/tests in both repositories synchronized
when changing the shared protocol. Full Sphinx builds and a real Actions/Pages
rehearsal remain required before retiring RTD.

References: [custom Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages),
[Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits),
[deploy-pages](https://github.com/actions/deploy-pages).
