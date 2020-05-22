.. _travisCI:

How to run TravisCI locally
===========================
To set up the TravisCI yaml file, it can be usefull to avoid many commit on GitHub before the  configuration get some success.
This is why it can be usefull to have a local solution to check this configuration.

The recipe here is adapted form solution discussed in
`here (how-to-run-travis-ci-locally) <https://stackoverflow.com/questions/21053657/how-to-run-travis-ci-locally>`_
and in `travis-build <https://github.com/travis-ci/travis-build>`_ repository on GitHub.

It assume you have `Docker <https://www.docker.com>`_ set up on your computer (running)

Set up the build environment using Docker

* Make up your own temporary build ID :

  .. sourcecode:: bash

     $ export BUILDID="build-$RANDOM"

* Get an image for travisCI:

  .. sourcecode:: bash

     $ docker pull travisci/ci-sardonyx:packer-1576238197-60d50014
     $ export INSTANCE="travisci/ci-sardonyx:packer-1576238197-60d50014"

* Run the headless server (from the spectrochempy folder)

  .. sourcecode:: bash

     $ docker run --name $BUILDID -v $(pwd):/home/travis/builds/AUTHOR/spectrochempy -dit $INSTANCE /sbin/init

* Run the attached client

  .. sourcecode:: bash

     $ docker exec -it $BUILDID bash -l

* Run the job

  Now you are now inside your Travis environment.

  Switch to the travis user

  .. sourcecode:: bash

     $ su - travis

  Install travis-build to generate a .sh out of .travis.yml

  .. sourcecode:: bash

     $ cd builds
     $ git clone https://github.com/travis-ci/travis-build.git
     $ cd travis-build
     $ gem install travis
     $ mkdir -p ~/.travis
     $ ln -s $PWD ~/.travis/travis-build
     $ gem install bundler
     $ bundle update --bundler
     $ bundle install --gemfile ~/.travis/travis-build/Gemfile
     $ bundler binstubs travis

Create project dir, assuming your project is AUTHOR/PROJECT on GitHub
cd ~/builds
mkdir AUTHOR
cd AUTHOR
git clone https://github.com/AUTHOR/PROJECT.git
cd PROJECT
change to the branch or commit you want to investigate

~/.travis/travis-build/bin/travis compile > ci.sh

You most likely will need to edit ci.sh as it ignores matrix and env
bash ci.sh