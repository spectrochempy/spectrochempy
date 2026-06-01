
:orphan:

What's New in Revision {{ revision }}
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-{{ revision }}.
See :ref:`release` for a full changelog, including other versions of SpectroChemPy.

..
   Do not remove the ``revision`` marker. It will be replaced during doc building.
   Also do not delete the section titles.
   Add your list of changes between (Add here) and (section) comments
   keeping a blank line before and after this list.

.. section

New Features
~~~~~~~~~~~~
.. Add here new public features (do not delete this comment)


.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)


.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)


.. section

Breaking Changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)


.. section

Developer
~~~~~~~~~
.. Add here developer changes (do not delete this comment)

- CI: Allowed documentation builds to use canonical core release tags such as
  ``spectrochempy-v0.9.2`` directly, while publishing versioned docs under the
  plain semver directory.
- DOC: Improved maintainer release documentation with explicit Zenodo
  verification steps, versioned docs checks, docs build note, and roadmap for
  modular docs.
- DOC: Clarified that official plugin versions are independent from the core
  package version and that conda dev plugin builds are not currently published
  automatically.
- DOC: Updated contributor documentation build instructions to use the
  repository-root ``docs/make.py`` entry point and current ``build/html`` output.
- DOC: Updated optional plugin installation notes so conda development
  channels no longer imply automatic plugin dev uploads.
- DOC: Aligned workflow comments and recovery notes with the current plugin
  conda policy: stable plugin releases upload to ``main`` and dev uploads are
  disabled until distinct dev versions exist.
- DOC: Updated plugin release documentation to use canonical core tags,
  current branch pushes, plugin ``__init__.py`` version bumps, and the current
  Anaconda ``main`` upload behavior.
- CI: Added plugin status table to ``release_plugin.yml`` step summary,
  listing all official plugins and whether they have changed since their
  last release tag — shows a "no changes" message when all plugins are
  unchanged, or highlights modified plugins when some need a release.
- CI: Added ``confirm_zenodo_enabled`` checkbox to
  ``prepare_new_release.yml`` so maintainers verify Zenodo is active before
  creating a release PR.
- CI: Disabled automatic dev conda uploads for official plugins
  (``build_package.yml``).  Plugin conda packages are still built in CI for
  testing, but only uploaded to Anaconda.org during a stable plugin release
  (label ``main``).  Dev uploads will be re-enabled once plugin recipes
  generate distinct dev versions.
- DEV: Added TODO/FIXME note in ``docs/sources/devguide/plugins/packaging.rst``
  explaining that dev conda uploads for plugins are disabled until distinct
  dev versions are generated.
- CI: Added ``plugin_release_status.yml`` workflow (``workflow_dispatch``)
  to inspect official plugin changes since their last release tag before
  deciding to run a release.  Writes a Markdown table (plugin, status, last
  tag, changed files count) to the workflow summary.  Read-only — no
  packages, tags, or releases are created.
