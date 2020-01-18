"""tests the SentencePipeline class
"""
from unittest import TestCase
from unittest import main

from tests.utils import async_test
from ucla_topic_analysis.data.coroutines.sentence_tokeniser import SentencePipeline


class CoroutineTestCase(TestCase):
    """Tests the coroutine function in the SentencePipeline class
    """

    def setUp(self):
        """sets up the tests
        """
        self.sentence_pipeline = SentencePipeline()

    def tearDown(self):
        """cleans up after the tests
        """
        del self.sentence_pipeline

    @async_test
    async def test_one_sentence(self):
        """Tests the function when there is only one sentence.
        """
        expected = ["This is a sentence."]
        actual = await self.sentence_pipeline.coroutine({"text": expected[0]})
        self.assertEqual(expected, actual["text"])

    @async_test
    async def test_without_full_stop(self):
        """Tests the function when there is only one sentence with no full-stop
        """
        expected = ["This is a sentence"]
        actual = await self.sentence_pipeline.coroutine({"text": expected[0]})
        self.assertEqual(expected, actual["text"])

    @async_test
    async def test_multiple_sentences(self):
        """Tests the function there are mulpiple sentences"""
        text = "This is sentence 1. This is sentence 2. This is sentence 3"
        expected = [
            "This is sentence 1.",
            "This is sentence 2.",
            "This is sentence 3"
        ]
        actual = await self.sentence_pipeline.coroutine({"text": text})
        self.assertEqual(expected, actual["text"])

    @async_test
    async def test_multiline(self):
        """Tests the function when there are new line characters thrown in
        """
        data = {
            "text": "This is sentence 1. This is \nsentence 2. This\n is sentence 3"
        }
        expected = [
            "This is sentence 1.",
            "This is \nsentence 2.",
            "This\n is sentence 3"
        ]
        actual = (await self.sentence_pipeline.coroutine(data))["text"]
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    main()
