# Read the Docs prototype

This directory holds the optional Read the Docs configuration of the **example
gallery** project. The main documentation project uses `.readthedocs.yaml` at
the root of the repository and publishes the integrated site.

Both projects build the same sources with the same `docs/make.py` driver, and
differ only by the documentation profile they select. The separate gallery
configuration is retained for now, but is no longer the primary published
experience.

| Project | Configuration | Profile | Content |
|---|---|---|---|
| main | `.readthedocs.yaml` | `full` | guides, notebooks, API reference, gallery, plugins guide |
| gallery | `docs/rtd/gallery/.readthedocs.yaml` | `gallery` | optional split example site |

## Dashboard setup

Both projects are Read the Docs **Community** projects pointing at
`https://github.com/spectrochempy/spectrochempy`. Read the Docs supports several
projects per repository, and the configuration path is a per-project setting.

For the **main** project: *Admin > Settings > Advanced settings > Configuration
file*, leave it as `.readthedocs.yaml`.

For the **gallery** project: its configuration remains
`docs/rtd/gallery/.readthedocs.yaml`. It can be disabled manually after the
integrated main-site preview has been accepted. All paths inside the file stay
relative to the repository root, including `docs/make.py`.

Both projects need a GitHub App integration for PR previews. Enable
*Admin > Settings > Integrations > Add your integration*, and enable
previews under *Settings > Pull request previews*.

The optional split profiles use the
`https://spectrochempy.readthedocs.io/en/latest` and
`https://spectrochempy-gallery.readthedocs.io/en/latest` companion URLs. Update
them to the slugs actually created and keep them without a trailing slash. The
integrated `full` main site does not use a companion URL.

## Versions

Three separate mechanisms are involved. They are not interchangeable.

**Versions come from branches and tags.** Every branch and every tag of the
repository is imported as an *inactive* version, and inactive versions do not
build. The plugin tags (`spectrochempy-nmr-v0.1.13` and the others) are tags too,
so they are imported as inactive versions as well. **Leave them inactive.** Do not
activate them, and do not bulk-activate all versions, or each plugin tag would
build and consume build quota.

**The default version** is a dashboard setting. The bare project URL redirects
to it. It defaults to `latest`, which points at the default branch of the
repository, and can be changed to any active version.

**`stable`** is a reserved version name with a fixed slug. It is created
automatically only when the repository has a tag or branch whose name follows
semantic versioning, with or without a `v` prefix. The release tags of this
repository are prefixed (`spectrochempy-v1.1.1`), so that automatic selection
does not fire. To get a `stable` version, [create a tag or a branch named
`stable`](https://docs.readthedocs.com/platform/stable/versions.html) in the
repository; if both exist, the tag wins. The slug of `stable` is managed by
Read the Docs and cannot be renamed. Choosing a default version is a different
action and does not create a `stable` version.

A release is therefore published by two independent steps: activate the version
for the release tag, and either point the default version at it or move the
`stable` tag/branch. This prototype does not decide between them, and it does
not create the `stable` tag or branch.

### Version slugs and cross-project links

The slug is derived from the branch or tag name, lowercased, with `/` replaced
by `-`. A release tag therefore yields the slug of the whole prefixed name:

| Git ref | Version slug | URL path |
|---|---|---|
| default branch | `latest` | `/en/latest/` |
| tag `spectrochempy-v1.1.1` | `spectrochempy-v1.1.1` | `/en/spectrochempy-v1.1.1/` |
| tag or branch `stable` | `stable` | `/en/stable/` |

This is why `SCPY_DOCS_MAIN_URL` and `SCPY_DOCS_GALLERY_URL` must be fully
versioned. Read the Docs serves a documentation tree under
`/<language>/<version-slug>/` and does **not** serve `/<language>/<page>`, so a
bare project root plus `/userguide/...` is a 404. Both configuration files
therefore use the `/en/latest` form.

For the first trial both projects link to `latest`, which is enough to check that
the cross-project links resolve. A release page should instead link to the
companion release, so a `1.1.1` page links to the companion
`spectrochempy-v1.1.1` and not to `latest`. That mapping, and the default
version or `stable` tag that goes with it, is not implemented yet.

## Profiles

`docs/make.py` and `docs/conf.py` accept a profile from `SCPY_DOCS_PROFILE` or
from the `--profile/-P` flag.

| Profile | Meaning |
|---|---|
| `full` | default and primary RTD site: guides, API reference, and example gallery |
| `main` | everything except the example gallery; retained for split-build experiments |
| `gallery` | the example gallery only; retained for split-build experiments |

The default is `full`, so an unset variable keeps the current behaviour. The
generated output of the `full` profile is unchanged by this mechanism.

Two companion URLs are read from the environment:

| Variable | Used by | Purpose |
|---|---|---|
| `SCPY_DOCS_MAIN_URL` | `gallery` | replaces references to pages that only exist in the main project |
| `SCPY_DOCS_GALLERY_URL` | `main` | links the guides to the published gallery |

Both are optional. When a URL is missing the cross-project reference is replaced
by plain text, so a build never fails on an unresolvable `:ref:`. When a URL is
given it must be fully versioned, `/<language>/<version-slug>`, for the reason
given in [Versions](#versions-slugs-and-cross-project-links).

## Read the Docs specific behaviour

`docs/make.py` already supports Read the Docs:

- with `READTHEDOCS_OUTPUT` set, the HTML is written to
  `$READTHEDOCS_OUTPUT/html` and the doctree cache to
  `~doctrees_<profile>` inside `SCPY_BUILDDIR`;
- the gh-pages post-build (version manifest, stable mirroring, pruning, upload)
  is skipped, so Read the Docs never clones or rewrites the published tree;
- `build.jobs.post_checkout` unshallow the clone, because both
  `docs/make.py` and the editable install need the release tags;
- every profile keeps `index` as the root document, so `index.html` exists at
  the root of each published site.

## Plugin installation

`pip install ".[docs,plugins]"` is **not** used. The `plugins` extra resolves the
plugins published on PyPI, so a pull request that modifies a plugin of this
monorepo would build its documentation against the previously released
implementation.

Both configurations therefore install only `.[docs]`, then install the plugins
from the checkout in `build.jobs.post_install`:

```bash
for name in $(python -m spectrochempy.ci.install_plugins --list-names); do
  python -m pip install --no-deps -e "plugins/$name"; done
python -m pip install osqp scipy numpy-quaternion tensorly
```

This is the same sequence as the "Install SpectroChemPy plugins" step of
`.github/workflows/build_docs.yml`, including the runtime dependencies that
`--no-deps` skips.

## Reproducing locally

`pandoc` must be on `PATH`. The plugins are required by both profiles, the API
reference by `main` and the plugin examples by `gallery`.

```bash
python -m pip install -e ".[docs]"
for name in $(python -m spectrochempy.ci.install_plugins --list-names); do
  python -m pip install --no-deps -e "plugins/$name"; done
python -m pip install osqp scipy numpy-quaternion tensorly

# integrated main project
python docs/make.py html --profile full --warning-is-error -j auto

# optional split gallery project
SCPY_DOCS_MAIN_URL=https://spectrochempy.readthedocs.io/en/latest \
  python docs/make.py html --profile gallery --warning-is-error -j auto
```

`--no-exec` and `--no-data` shorten a smoke build considerably.

For a clean RTD-like validation, use empty directories for both
`SCPY_BUILDDIR` and `READTHEDOCS_OUTPUT`; this prevents old doctrees and HTML
from satisfying references to excluded profile content. If the build downloads
test data, also use an isolated `HOME` and `SCP_CONFIG_HOME` so the local data
directory and preferences remain untouched.

### Integrated-site preview checklist

Before accepting the main-project PR preview, validate the `full` profile as
the historical integrated site, not merely as a successful Sphinx build:

- the gallery landing page exposes every category and example page through the
  normal site navigation;
- Sphinx-Gallery cross-references are generated, including the small example
  thumbnails at the bottom of public API pages;
- inspect several API pages for objects used by gallery examples (for example
  `NDDataset`, `MCRALS`, and `PCA`): their thumbnails must be present and each
  link must resolve locally to the corresponding generated example page;
- confirm that gallery-to-API and notebook-to-gallery links are local links,
  not companion-project URLs;
- confirm that the `main` and `gallery` profile conditions do not remove the
  gallery, its backreferences, thumbnails, navigation, or local links in the
  `full` preview.

The Read the Docs preview must retain the content, navigation, and internal
links of the previous integrated documentation before the optional split gallery
project is disabled.

## Measured on the development machine

Warm caches, `-j 4`, executed notebooks and examples, test data downloaded.

| Profile | Wall time | HTML pages | HTML size | Build workspace |
|---|---:|---:|---:|---:|
| `full` (no-exec) | 2 m 03 | 555 | 83 MB | 32 MB |
| `main` | 4 m 38 | 462 | 97 MB | 85 MB |
| `gallery` | 3 m 38 | 98 | 32 MB | 13 MB |

For comparison, the no-exec smoke builds take 2 m 03 for `full` and 1 m 46 for
`main`.

`main` executes 38 notebooks, `gallery` executes 52 example files. The two
profiles partition the single site: 462 + 98 = 560 pages, against 555 for
`full`. The split removes the example work from the main project and the API
work from the gallery project, it does not reduce the total work.

Read the Docs Community provides 2 concurrent builds and 15 minutes per build.
`-j auto` resolves to the job count of the build machine, so a 2 vCPU build
image is expected to be roughly twice as slow as the measurements above.

## Not covered by this prototype

- dependency isolation per profile: both projects install the whole `docs` extra
  and all six plugins. A split such as `docs-sphinx` / `docs-notebooks` /
  `docs-gallery` in `pyproject.toml` is still to be done, and is not required
  for the split to build.
- data isolation: both profiles download the full test data directory.
- version-matched cross-project links: both projects link to `latest`, so a
  release page will point readers at the development gallery.
- the `stable` version: the prefixed release tags do not trigger Read the Docs'
  automatic selection, and no `stable` tag or branch is created here.
- cross-project `intersphinx` and canonical URLs.
- NMR as a third profile.
- old tags, which predate these configuration files.
