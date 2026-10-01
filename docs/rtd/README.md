# Read the Docs prototype

This directory holds the Read the Docs configuration of the **example gallery**
project. The main documentation project uses `.readthedocs.yaml` at the root of
the repository.

Both projects build the same sources, with the same `docs/make.py` driver, and
differ only by the documentation profile they select.

| Project | Configuration | Profile | Content |
|---|---|---|---|
| main | `.readthedocs.yaml` | `main` | guides, notebooks, API reference, plugins guide |
| gallery | `docs/rtd/gallery.readthedocs.yaml` | `gallery` | executed examples, figures, downloads |

## Dashboard setup

Both projects are Read the Docs **Community** projects pointing at
`https://github.com/spectrochempy/spectrochempy`. Read the Docs supports several
projects per repository, and the configuration path is a per-project setting.

For the **main** project: *Admin > Settings > Advanced settings > Configuration
file*, leave it as `.readthedocs.yaml`.

For the **gallery** project: set the configuration file to
`docs/rtd/gallery.readthedocs.yaml`. All paths inside the file stay relative to
the repository root, including `docs/make.py`.

Both projects need a GitHub App integration for PR previews. Enable
*Admin > Settings > Integrations > Add your integration*, and enable
previews under *Settings > Pull request previews*.

The `https://spectrochempy.readthedocs.io` and
`https://spectrochempy-gallery.readthedocs.io` URLs in the two configuration
files must be updated to the slugs actually created, because the projects link to
each other. Keep them without a trailing slash.

## Version selection

Read the Docs derives versions from branches and tags. The release tags of this
repository are prefixed (`spectrochempy-v1.1.1`), which is not the semantic
versioning form the automatic `stable` selection expects. Until the tag scheme
is validated on Read the Docs, **select the `stable` version manually** in the
dashboard and verify that plugin tags do not become active versions.

## Profiles

`docs/make.py` and `docs/conf.py` accept a profile from `SCPY_DOCS_PROFILE` or
from the `--profile/-P` flag.

| Profile | Meaning |
|---|---|
| `full` | default, the current single site published on GitHub Pages |
| `main` | everything except the example gallery |
| `gallery` | the example gallery only |

The default is `full`, so an unset variable keeps the current behaviour. The
generated output of the `full` profile is unchanged by this mechanism.

Two companion URLs are read from the environment:

| Variable | Used by | Purpose |
|---|---|---|
| `SCPY_DOCS_MAIN_URL` | `gallery` | replaces references to pages that only exist in the main project |
| `SCPY_DOCS_GALLERY_URL` | `main` | links the guides to the published gallery |

Both are optional. When a URL is missing the cross-project reference is replaced
by plain text, so a build never fails on an unresolvable `:ref:`.

## Read the Docs specific behaviour

`docs/make.py` already supports Read the Docs:

- with `READTHEDOCS_OUTPUT` set, the HTML is written to
  `$READTHEDOCS_OUTPUT/html` and the doctree cache to
  `~doctrees_<profile>` inside `SCPY_BUILDDIR`;
- the gh-pages post-build (version manifest, stable mirroring, pruning, upload)
  is skipped, so Read the Docs never clones or rewrites the published tree;
- `build.jobs.post_checkout` unshallow the clone, because `docs/make.py` reads
  the release tags to determine the version;
- every profile keeps `index` as the root document, so `index.html` exists at
  the root of each published site.

## Reproducing locally

`pandoc` must be on `PATH`. The plugins are required by both profiles, the API
reference by `main` and the plugin examples by `gallery`.

```bash
python -m spectrochempy.ci.install_plugins --editable --no-deps all

# main project
SCPY_DOCS_GALLERY_URL=https://spectrochempy-gallery.readthedocs.io \
  python docs/make.py html --profile main --warning-is-error -j auto

# gallery project
SCPY_DOCS_MAIN_URL=https://spectrochempy.readthedocs.io \
  python docs/make.py html --profile gallery --warning-is-error -j auto
```

`--no-exec` and `--no-data` shorten a smoke build considerably.

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

- dependency isolation per profile: both projects install `.[docs,plugins]`. A
  split such as `docs-sphinx` / `docs-notebooks` / `docs-gallery` in
  `pyproject.toml` is still to be done, and is not required for the split to
  build.
- data isolation: both profiles download the full test data directory.
- cross-project `intersphinx`, canonical URLs and version-pinned companion links.
- NMR as a third profile.
- old tags, which predate these configuration files.
