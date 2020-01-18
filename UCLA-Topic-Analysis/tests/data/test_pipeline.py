"""Tests the Pipeline class
"""
from unittest import TestCase
from unittest import main
import asyncio

from tests.utils import async_test
from ucla_topic_analysis.data.pipeline import Pipeline

class TestPipeline(Pipeline):
    """ A test pipeline
    """
    async def coroutine(self, data):
        await asyncio.sleep(0.01)
        return data + "some test data"


class RunTestCase(TestCase):
    """Tests the run function in the Pipeline class
    """

    def setUp(self):
        """sets up the tests
        """
        self.pipeline = TestPipeline()

    def tearDown(self):
        """cleans up after the test
        """
        del self.pipeline

    @async_test
    async def test_result_is_set(self):
        """Tests that the self._result variable is properly set
        """
        expected = "some test data"
        await self.pipeline.run("")
        self.assertEqual(expected, self.pipeline._result)

    @async_test
    async def test_downstream_is_run(self):
        """Tests that the downstream pipeline is run
        """
        expected = "some test datasome test data"
        test_pipeline = TestPipeline([self.pipeline])
        await test_pipeline.run("")
        self.assertEqual(expected, self.pipeline._result)

    @async_test
    async def test_data_is_passed(self):
        """Tests that the data is passed to the coroutine
        """
        expected = "some other test data some test data"
        await self.pipeline.run("some other test data ")
        self.assertEqual(expected, self.pipeline._result)

if __name__ == "__main__":
    main()
