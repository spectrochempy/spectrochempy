<!-- Remove items not relevant to your change -->

**Checklist for a good PR**:
- [ ] Close the #xxxx (optional)
- [ ] Tests have been added (mostly required)
- [ ] If dependencies have been updated, the script `.scripts/create_requirements.py`
      has been executed.
- [ ] User-visible changes (including notable bug fixes) have been documented in
      `CHANGELOG.md` (Changes relevant to developers only are generally not needed in
      CHANGELOG as they are apparent in commit messages and PR comments)
- [ ] The new methods have been listed in `docs/userguide/reference/api.rst`.
      If an API method (e.g. `core.readers.readomnic`) is a NDDataset method, it should
      also be listed as `NDDataset.readomnic`.
- [ ] If you are a new contributor, you have added your name (affiliation and ORCID
      if you have one) in the .zenodo.json in the field contributors field. Be careful
      not to break the json format (check the content of the file with the
      [JSON Validator](https://jsonformatter.curiousconcept.com/))
