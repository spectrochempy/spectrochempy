<!-- Remove items not relevant to your change -->

Check List for a good PR:
- [ ] Closes #xxxx (optional)
- [ ] Tests added (most of the time necessary)
- [ ] User visible changes (including notable bug fixes) are documented in `CHANGELOG.md` (Developpers only relevant change are generally not necessary included in CHANGELOG)
- [ ] New methods are listed in `docs/userguide/reference/api.rst`. If an API method (ex. `core.readers.readomnic `) is also a NDDataset method, it must also be listed as `NDDataset.readomnic`
- [ ] If dependencies have been updated, the `.ci/create_requirements.py` script has been executed.
