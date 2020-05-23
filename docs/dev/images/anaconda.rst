Push |scpy| to anaconda.org after successful build on travis
---------------------------------------------------------------------------

.. note::
   This 'push' is for those who have permission to log in to spectrocat
   account on anaconda.org

- website: `anaconda.org/spectrocat <https://anaconda.org/ambermd>`_

- install ``ruby``
rvm install 2.3.0

- install ``travis``::

  $ gem install travis

- install anaconda-client::

  $ conda install anaconda-client

- In your terminal, log in to anaconda account::

  $ anaconda login
  $ # just enter your username and password

- generate anaconda token to give travis permision to push data in ambermd channel in anaconda.org::

  $ git clone https://github.com/Amber-MD/pytraj
  $ cd pytraj
  $ # generate token
  $ TOKEN=$(anaconda auth --create --name MyToken)
  $ echo $TOKEN

- need to use ``travis`` to encrypt our token::

  $ travis encrypt TRAVIS_TO_ANACONDA=secretvalue

- make code change, commit, push to github so travis can build pytraj and libcpptraj::

  $ # after successful build, travis will push to anaconda.org by below command
  $ anaconda -t $TRAVIS_TO_ANACONDA upload --force -u ambermd -p pytraj-dev $HOME/miniconda/conda-bld/linux-64/pytraj-dev-*
  $ # check devtools/travis-ci/upload.sh and .travis.yml files for implementation.
