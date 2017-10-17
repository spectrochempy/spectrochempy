# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
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

""" Tests file operations. """


# Standard library imports.
import os, shutil, stat, unittest

# Enthought library imports.
from spectrochempy.utils import File


class FileTestCase(unittest.TestCase):
    """ Tests file operations on a local file system. """

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
        """ file properties """

        # Properties of a non-existent file.
        f = File('data/bogus.xx')

        self.assert_(os.path.abspath(os.path.curdir) in f.absolute_path)
        self.assertEqual(f.children, None)
        self.assertEqual(f.ext, '.xx')
        self.assertEqual(f.exists, False)
        self.assertEqual(f.is_file, False)
        self.assertEqual(f.is_folder, False)
        self.assertEqual(f.is_package, False)
        self.assertEqual(f.is_readonly, False)
        self.assertEqual(f.mime_type, 'content/unknown')
        self.assertEqual(f.name, 'bogus')
        self.assertEqual(f.parent.path, 'data')
        self.assertEqual(f.path, 'data/bogus.xx')
        self.assert_(os.path.abspath(os.path.curdir) in f.url)
        self.assertEqual(str(f), 'File(%s)' % f.path)

        # Properties of an existing file.
        f = File('data/foo.txt')
        f.create_file()

        self.assert_(os.path.abspath(os.path.curdir) in f.absolute_path)
        self.assertEqual(f.children, None)
        self.assertEqual(f.ext, '.txt')
        self.assertEqual(f.exists, True)
        self.assertEqual(f.is_file, True)
        self.assertEqual(f.is_folder, False)
        self.assertEqual(f.is_package, False)
        self.assertEqual(f.is_readonly, False)
        self.assertEqual(f.mime_type, 'text/plain')
        self.assertEqual(f.name, 'foo')
        self.assertEqual(f.parent.path, 'data')
        self.assertEqual(f.path, 'data/foo.txt')
        self.assert_(os.path.abspath(os.path.curdir) in f.url)

        # Make it readonly.
        os.chmod(f.path, stat.S_IRUSR)
        self.assertEqual(f.is_readonly, True)

        # And then make it NOT readonly so that we can delete it at the end of
        # the test!
        os.chmod(f.path, stat.S_IRUSR | stat.S_IWUSR)
        self.assertEqual(f.is_readonly, False)

        return

    def test_copy(self):
        """ file copy """

        content = 'print "Hello World!"\n'

        f = File('data/foo.txt')
        self.assertEqual(f.exists, False)

        # Create the file.
        f.create_file(content)
        self.assertEqual(f.exists, True)
        self.assertRaises(ValueError, f.create_file, content)

        self.assertEqual(f.children, None)
        self.assertEqual(f.ext, '.txt')
        self.assertEqual(f.is_file, True)
        self.assertEqual(f.is_folder, False)
        self.assertEqual(f.mime_type, 'text/plain')
        self.assertEqual(f.name, 'foo')
        self.assertEqual(f.path, 'data/foo.txt')

        # Copy the file.
        g = File('data/bar.txt')
        self.assertEqual(g.exists, False)

        f.copy(g)
        self.assertEqual(g.exists, True)

        self.assertEqual(g.children, None)
        self.assertEqual(g.ext, '.txt')
        self.assertEqual(g.is_file, True)
        self.assertEqual(g.is_folder, False)
        self.assertEqual(g.mime_type, 'text/plain')
        self.assertEqual(g.name, 'bar')
        self.assertEqual(g.path, 'data/bar.txt')

        # Attempt to copy a non-existent file (should do nothing).
        f = File('data/bogus.xx')
        self.assertEqual(f.exists, False)

        g = File('data/bogus_copy.txt')
        self.assertEqual(g.exists, False)

        f.copy(g)
        self.assertEqual(g.exists, False)

        return

    def test_create_file(self):
        """ file creation """

        content = 'print "Hello World!"\n'

        f = File('data/foo.txt')
        self.assertEqual(f.exists, False)

        # Create the file.
        f.create_file(content)
        self.assertEqual(f.exists, True)
        self.assertEqual(open(f.path).read(), content)

        # Try to create it again.
        self.assertRaises(ValueError, f.create_file, content)

        return

    def test_delete(self):
        """ file deletion """

        content = 'print "Hello World!"\n'

        f = File('data/foo.txt')
        self.assertEqual(f.exists, False)

        # Create the file.
        f.create_file(content)
        self.assertEqual(f.exists, True)
        self.assertRaises(ValueError, f.create_file, content)

        self.assertEqual(f.children, None)
        self.assertEqual(f.ext, '.txt')
        self.assertEqual(f.is_file, True)
        self.assertEqual(f.is_folder, False)
        self.assertEqual(f.mime_type, 'text/plain')
        self.assertEqual(f.name, 'foo')
        self.assertEqual(f.path, 'data/foo.txt')

        # Delete it.
        f.delete()
        self.assertEqual(f.exists, False)

        # Attempt to delete a non-existet file (should do nothing).
        f = File('data/bogus.txt')
        self.assertEqual(f.exists, False)

        f.delete()
        self.assertEqual(f.exists, False)

        return

if __name__ == "__main__":
    unittest.main()

#### EOF ######################################################################
