"""Tests the ReadFilePipeline
"""
import os
from unittest import TestCase

from tests.utils import async_test
from ucla_topic_analysis.data.coroutines.read import ReadFilePipeline


class CoroutineTestCase(TestCase):
    """Tests the coroutine function in the ReadFilePipeline class
    """

    def setUp(self):
        """sets up the tests
        """
        self.file_reader = ReadFilePipeline()
        self.data_dir = os.path.dirname(os.path.realpath(__file__)) + "/data"

    def tearDown(self):
        """cleans up after the tests
        """
        del self.file_reader

    @async_test
    async def test_empty_file(self):
        """Tests getting data from an empty file
        """
        expected = ""
        actual = await self.file_reader.coroutine(self.data_dir + "/test-file1.txt")
        self.assertEqual(expected, actual["text"])

    @async_test
    async def test_nonempty_file(self):
        """Tests getting data from a non-empty file
        """
        expected = ("Some random data\n" +
                    "Some more random data\n" +
                    "And a third line for luck")
        actual = await self.file_reader.coroutine(self.data_dir + "/test-file2.txt")
        self.assertEqual(expected, actual["text"])
