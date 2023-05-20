
## Make a new release

* Go to Action
* Select Prepare a new release
* Click on run workflow
* Enter the version number you want to release: e.g. 0.6.4
* If the action is a success, a new PR is created automatically
* Open it
* Unfortunately as the PR was created by a bot, the action will not start.
  But in principle at this stage (release) we have already fully checked master. So we can bypass them
  And merge the PR.
* Now we go to Code and select tags.
* Choose Release and you can see the draft of the release to publish
* Click on the draft name and edit it by clicking on the pencil.
* Optionally give a description (for example a copy of the Changelog.rst converted to markdown.)
* And then Publish release.
* Zenodo will be updated automatically  as well as conda and pipit package
