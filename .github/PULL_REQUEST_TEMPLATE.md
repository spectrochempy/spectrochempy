**Checklist for a good PR**:

- [ ] Close the #xxxx (optional)
- [ ] Tests have been added (mostly required)
- [ ] If dependencies have been updated, the script `.scripts/create_requirements.py`
      has been executed.
- [ ] The docstrings have been tested with the script `script/validate_docstrings.py`.
- [ ] User-visible changes (including notable bug fixes) have been documented in
      `CHANGELOG.md` <font size=1>*(Changes relevant to developers only are generally not needed in
      CHANGELOG as they are apparent in commit messages and PR comments)*</font>
- [ ] The new methods have been listed in `docs/userguide/reference/api.rst`.
      <font size=1>*(If an API method (e.g. `core.readers.readomnic`) is a NDDataset method, it should
      also be listed as `NDDataset.readomnic`.)*</font>
- [ ] If you are a new contributor, you have added your name (affiliation and ORCID
      if you have one) in the .zenodo.json in the field contributors field. <font size=1>*Be careful
      not to break the json format (check the content of the file with the
      [JSON Validator](https://jsonformatter.curiousconcept.com/))*</font>
