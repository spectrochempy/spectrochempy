#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

import os
import json
import requests

PROJECT = "spectrochempy"
DOCDIR = os.path.dirname(os.path.abspath(__file__))
PROJECTDIR = os.path.dirname(DOCDIR)

def make_changelogs():
    """
    Utility to update changelog (using the GITHUB API)
    """
    REPO_URI = f"spectrochempy/{PROJECT}"
    API_URL = "https://api.github.com"
    CHANGELOGRST = os.path.join(DOCDIR, 'gettingstarted', 'changelog.rst')
    CHANGELOGMD = os.path.join(PROJECTDIR, "CHANGELOG.md")

    print("getting latest release tag")
    LATEST = os.path.join(API_URL, "repos", REPO_URI, "releases","latest")
    tag_name = json.loads(requests.get(LATEST).text)['tag_name']

    def get(milestone, label):
        print("getting list of issues with label ", label)
        issues = os.path.join(API_URL, "search",f"issues?q=repo:{REPO_URI}"
                                            f"+milestone:{milestone}"
                                            f"+is:issue"
                                            f"+label:{label}")
        return json.loads(requests.get(issues).text)

    # Create a versionlog file for the current target
    bugs = get(tag_name, "bug")
    features = get(tag_name, "enhancement")
    tasks = get(tag_name, "task")

    with open(os.path.join(TEMPLATES, 'versionlog.rst'), 'r') as f:
        template = Template(f.read())
    out = template.render(target=target, bugs=bugs, features=features, tasks=tasks)

    with open(os.path.join(DOCDIR, 'versionlogs', f'versionlog.{target}.rst'), 'w') as f:
        f.write(out)

    # make the full version history

    lhist = sorted(iglob(os.path.join(DOCDIR, 'versionlogs', '*.rst')))
    lhist.reverse()
    history = ""
    for filename in lhist:
        if '.'.join(filename.split('.')[-4:-1]) > target:
            continue  # do not take into account future version for change log - obviously!
        with open(filename, 'r') as f:
            history += "\n\n"
            nh = f.read().strip()
            vc = ".".join(filename.split('.')[1:4])
            nh = nh.replace(':orphan:', f".. _version_{vc}:")
            history += nh
    history += '\n'

    with open(os.path.join(TEMPLATES, 'changelog.rst'), 'r') as f:
        template = Template(f.read())
    out = template.render(history=history)

    with open(os.path.join(DOCDIR, 'gettingstarted', 'changelog.rst'), 'w') as f:
        f.write(out)

    return


if __name__ == '__main__':
    make_changelogs()
