# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================


""" Tests folder operations. """


# Standard library imports.
import os, shutil, stat, unittest
from os.path import join

# Enthought library imports.
from spectrochempy.utils import File


class FolderTestCase(unittest.TestCase):
    """ Tests folder operations on a local file system. """

    ###########################################################################
    # 'TestCase' interface.
    ###########################################################################

    def setUp(self):
        """ Prepares the test fixture before each test method is called. """

        try:
            shutil.rmtree('data')

        except:
            pass

        os.mkdir('data')

        return

    def tearDown(self):
        """ Called immediately after each test method has been called. """

        shutil.rmtree('data')

        return

    ###########################################################################
    # Tests.
    ###########################################################################

    def test_properties(self):
        """ folder properties """

        # Properties of a non-existent folder.
        f = File('data/bogus')

        self.assert_(os.path.abspath(os.path.curdir) in f.absolute_path)
        self.assertEqual(f.children, None)
        self.assertEqual(f.ext, '')
        self.assertEqual(f.exists, False)
        self.assertEqual(f.is_file, False)
        self.assertEqual(f.is_folder, False)
        self.assertEqual(f.is_package, False)
        self.assertEqual(f.is_readonly, False)
        self.assertEqual(f.mime_type, 'content/unknown')
        self.assertEqual(f.name, 'bogus')
        self.assertEqual(f.parent.path, 'data')
        self.assertEqual(f.path, 'data/bogus')
        self.assert_(os.path.abspath(os.path.curdir) in f.url)
        self.assertEqual(str(f), 'File(%s)' % f.path)

        # Properties of an existing folder.
        f = File('data/sub')
        f.create_folder()

        self.assert_(os.path.abspath(os.path.curdir) in f.absolute_path)
        self.assertEqual(len(f.children), 0)
        self.assertEqual(f.ext, '')
        self.assertEqual(f.exists, True)
        self.assertEqual(f.is_file, False)
        self.assertEqual(f.is_folder, True)
        self.assertEqual(f.is_package, False)
        self.assertEqual(f.is_readonly, False)
        self.assertEqual(f.mime_type, 'content/unknown')
        self.assertEqual(f.name, 'sub')
        self.assertEqual(f.parent.path, 'data')
        self.assertEqual(f.path, 'data/sub')
        self.assert_(os.path.abspath(os.path.curdir) in f.url)

        # Make it readonly.
        os.chmod(f.path, stat.S_IRUSR)
        self.assertEqual(f.is_readonly, True)

        # And then make it NOT readonly so that we can delete it at the end of
        # the test!
        os.chmod(f.path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        self.assertEqual(f.is_readonly, False)

        # Properties of a Python package folder.
        f = File('data/package')
        f.create_folder()

        init = File('data/package/__init__.py')
        init.create_file()

        self.assert_(os.path.abspath(os.path.curdir) in f.absolute_path)
        self.assertEqual(len(f.children), 1)
        self.assertEqual(f.ext, '')
        self.assertEqual(f.exists, True)
        self.assertEqual(f.is_file, False)
        self.assertEqual(f.is_folder, True)
        self.assertEqual(f.is_package, True)
        self.assertEqual(f.is_readonly, False)
        self.assertEqual(f.mime_type, 'content/unknown')
        self.assertEqual(f.name, 'package')
        self.assertEqual(f.parent.path, 'data')
        self.assertEqual(f.path, 'data/package')
        self.assert_(os.path.abspath(os.path.curdir) in f.url)

        return

    def test_copy(self):
        """ folder copy """

        f = File('data/sub')
        self.assertEqual(f.exists, False)

        # Create the folder.
        f.create_folder()
        self.assertEqual(f.exists, True)
        self.assertRaises(ValueError, f.create_folder)

        self.assertEqual(len(f.children), 0)
        self.assertEqual(f.ext, '')
        self.assertEqual(f.is_file, False)
        self.assertEqual(f.is_folder, True)
        self.assertEqual(f.mime_type, 'content/unknown')
        self.assertEqual(f.name, 'sub')
        self.assertEqual(f.path, 'data/sub')

        # Copy the folder.
        g = File('data/copy')
        self.assertEqual(g.exists, False)

        f.copy(g)
        self.assertEqual(g.exists, True)

        self.assertEqual(len(g.children), 0)
        self.assertEqual(g.ext, '')
        self.assertEqual(g.is_file, False)
        self.assertEqual(g.is_folder, True)
        self.assertEqual(g.mime_type, 'content/unknown')
        self.assertEqual(g.name, 'copy')
        self.assertEqual(g.path, 'data/copy')

        # Attempt to copy a non-existent folder (should do nothing).
        f = File('data/bogus')
        self.assertEqual(f.exists, False)

        g = File('data/bogus_copy')
        self.assertEqual(g.exists, False)

        f.copy(g)
        self.assertEqual(g.exists, False)

        return

    def test_create_folder(self):
        """ folder creation """

        f = File('data/sub')
        self.assertEqual(f.exists, False)

        # Create the folder.
        f.create_folder()
        self.assertEqual(f.exists, True)

        parent = File('data')
        self.assertEqual(len(parent.children), 1)
        self.assertEqual(parent.children[0].path, join('data', 'sub'))

        # Try to create it again.
        self.assertRaises(ValueError, f.create_folder)

        return

    def test_create_folders(self):
        """ nested folder creation """

        f = File('data/sub/foo')
        self.assertEqual(f.exists, False)

        # Attempt to create the folder with 'create_folder' which requires
        # that all intermediate folders exist.
        self.assertRaises(OSError, f.create_folder)

        # Create the folder.
        f.create_folders()
        self.assertEqual(f.exists, True)
        self.assertEqual(File('data/sub').exists, True)

        # Try to create it again.
        self.assertRaises(ValueError, f.create_folders)

        return

    def test_delete(self):
        """ folder deletion """

        f = File('data/sub')
        self.assertEqual(f.exists, False)

        # Create the folder.
        f.create_folder()
        self.assertEqual(f.exists, True)
        self.assertRaises(ValueError, f.create_folder)

        self.assertEqual(len(f.children), 0)
        self.assertEqual(f.ext, '')
        self.assertEqual(f.is_file, False)
        self.assertEqual(f.is_folder, True)
        self.assertEqual(f.mime_type, 'content/unknown')
        self.assertEqual(f.name, 'sub')
        self.assertEqual(f.path, 'data/sub')

        # Delete it.
        f.delete()
        self.assertEqual(f.exists, False)

        # Attempt to delete a non-existet folder (should do nothing).
        f = File('data/bogus')
        self.assertEqual(f.exists, False)

        f.delete()
        self.assertEqual(f.exists, False)

        return

#### EOF ######################################################################
