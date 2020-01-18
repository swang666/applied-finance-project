"""Tests for the functions in the data.coroutines.__init__ module
"""
import os
from unittest import TestCase

from ucla_topic_analysis.data.coroutines import create_file
from ucla_topic_analysis.data.coroutines import insert

class CreateFileTestCase(TestCase):
    """Tests the create_file function.
    """
    def setUp(self):
        """sets up the tests
        """
        data_dir = os.path.dirname(os.path.realpath(__file__)) + "/data"
        old_file_name = "old-file.txt"
        new_file_name = "new-file.txt"
        self.old_file_path = os.path.join(data_dir, old_file_name)
        self.new_file_path = os.path.join(data_dir, new_file_name)
        self.old_data = "this is line1\nThis is line2\nThis is line3"
        with open(self.old_file_path, 'w') as old_file:
            old_file.write(self.old_data)

    def tearDown(self):
        """Cleans up after any tests
        """
        try:
            os.remove(self.old_file_path)
        except OSError:
            pass

        try:
            os.remove(self.new_file_path)
        except OSError:
            pass

    def test_create_file(self):
        """Tests that the function actually creates a file
        """
        self.assertTrue(create_file(self.new_file_path))
        self.assertTrue(os.path.isfile(self.new_file_path))

    def test_file_exists(self):
        """Tests that the function when the file already exists
        """
        self.assertFalse(create_file(self.old_file_path))
        self.assertTrue(os.path.isfile(self.old_file_path))
        with open(self.old_file_path, "r") as old_file:
            self.assertEqual(old_file.read(), self.old_data)

    def test_no_initial_data(self):
        """Tests that the function when the file already exists
        """
        self.assertTrue(create_file(self.new_file_path))
        self.assertTrue(os.path.isfile(self.new_file_path))
        with open(self.new_file_path, "r") as new_file:
            self.assertEqual(new_file.read(), "")

    def test_initial_data(self):
        """Tests that the function when the file already exists
        """
        initialdata = "some random data"
        self.assertTrue(create_file(self.new_file_path, initialdata))
        self.assertTrue(os.path.isfile(self.new_file_path))
        with open(self.new_file_path, "r") as new_file:
            self.assertEqual(new_file.read(), initialdata)


class InsertTestCase(TestCase):
    """Tests the insert function
    """

    def setUp(self):
        """sets up the tests
        """
        data_dir = os.path.dirname(os.path.realpath(__file__)) + "/data"
        test_file_name = "test-file.txt"
        self.test_file_path = os.path.join(data_dir, test_file_name)
        self.file_data = "This is line1\nThis line is line2\nAnd this is line3"
        with open(self.test_file_path, 'w') as test_file:
            test_file.write(self.file_data)

    def tearDown(self):
        """Cleans up after any tests
        """
        try:
            os.remove(self.test_file_path)
        except OSError:
            pass

    def data_insert(self, sub_string, index):
        """Used to get the result of inserting a sub-string into the string
        contained in the self.file_data

        Args:
            sub_string (str): The substring to insert into the data
            index (int): The position at which to make the insertion

        Returns:
            str: A new string which is the result of inserting the sub_string
            into the file_data string
        """
        return self.file_data[:index] + sub_string + self.file_data[index:]

    @property
    def file_contents(self):
        """str: The contents of the test file
        """
        with open(self.test_file_path) as test_file:
            return test_file.read()

    def test_empty_insertion(self):
        """Tests the function when the string inserted is empty
        """
        index = 11
        insertion = ""
        expected = self.data_insert(insertion, index)
        with open(self.test_file_path, "r+") as test_file:
            insert(insertion, test_file, index)
        actual = self.file_contents
        self.assertEqual(actual, expected)

    def test_empty_start(self):
        """Tests inserting an empty string at the start of the file
        """
        index = 0
        insertion = ""
        expected = self.data_insert(insertion, index)
        with open(self.test_file_path, "r+") as test_file:
            insert(insertion, test_file, index)
        actual = self.file_contents
        self.assertEqual(actual, expected)

    def test_empty_end(self):
        """Tests inserting an empty string at the end of the file
        """
        index = len(self.file_data)
        insertion = ""
        expected = self.data_insert(insertion, index)
        with open(self.test_file_path, "r+") as test_file:
            insert(insertion, test_file, index)
        actual = self.file_contents
        self.assertEqual(actual, expected)

    def test_single_char_middle(self):
        """Tests inserting a single character somewhere in the middle of the
        file
        """
        index = 11
        insertion = "g"
        expected = self.data_insert(insertion, index)
        with open(self.test_file_path, "r+") as test_file:
            insert(insertion, test_file, index)
        actual = self.file_contents
        self.assertEqual(actual, expected)

    def test_single_char_start(self):
        """Tests inserting a single character at the start of the file
        """
        index = 0
        insertion = "g"
        expected = self.data_insert(insertion, index)
        with open(self.test_file_path, "r+") as test_file:
            insert(insertion, test_file, index)
        actual = self.file_contents
        self.assertEqual(actual, expected)

    def test_single_char_end(self):
        """Tests inserting a single character at the end of the file
        """
        index = len(self.file_data)
        insertion = "g"
        expected = self.data_insert(insertion, index)
        with open(self.test_file_path, "r+") as test_file:
            insert(insertion, test_file, index)
        actual = self.file_contents
        self.assertEqual(actual, expected)

    def test_multiple_char_middle(self):
        """Tests inserting multiple characters somewhere in the middle of the
        file
        """
        index = 11
        insertion = "gasdf asdfwwr\n asdf"
        expected = self.data_insert(insertion, index)
        with open(self.test_file_path, "r+") as test_file:
            insert(insertion, test_file, index)
        actual = self.file_contents
        self.assertEqual(actual, expected)

    def test_multiple_char_start(self):
        """Tests inserting multiple characters at the start of the file
        """
        index = 0
        insertion = "gasdf asdfwwr\n asdf"
        expected = self.data_insert(insertion, index)
        with open(self.test_file_path, "r+") as test_file:
            insert(insertion, test_file, index)
        actual = self.file_contents
        self.assertEqual(actual, expected)

    def test_multiple_char_end(self):
        """Tests inserting multiple characters at the end of the file
        """
        index = len(self.file_data)
        insertion = "gasdf asdfwwr\n asdf"
        expected = self.data_insert(insertion, index)
        with open(self.test_file_path, "r+") as test_file:
            insert(insertion, test_file, index)
        actual = self.file_contents
        self.assertEqual(actual, expected)

