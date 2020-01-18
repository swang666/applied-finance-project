""" Contains a pipeline for generating a data set for use with the LightTag
platform

see: https://www.lighttag.io/
"""
import json

from ucla_topic_analysis import get_file_list
from ucla_topic_analysis.data import get_training_file_path
from ucla_topic_analysis.data.coroutines import create_file
from ucla_topic_analysis.data.coroutines import insert
from ucla_topic_analysis.data.coroutines import print_progress
from ucla_topic_analysis.data.pipeline import Pipeline
from ucla_topic_analysis.data.coroutines.read import ReadFilePipeline


class LightTagDataSetPipeline(Pipeline):
    """Pipeline for generating a dataset for the LightTag platform.

     NOTE: This pipeline is a data sink. It does not return any new data.
    """

    # The split schema for the corpus files
    SCHEMA = {
        "training": 0.8,
        "validation": 0.1,
        "testing": 0.1
    }

    @staticmethod
    def get_input_stream(schema=None):
        """This function is used to get an input stream for the
        LightTagDataSetPipeline

        Args:
            schema(:obj:`dict`): The schema for the file pipeline

        Returns:
            An iterable containing the dataset.
        """
        # Build the input stream pipeline
        files = ReadFilePipeline.get_input_stream()
        return ReadFilePipeline(
            input_stream=files,
            schema=schema
        ).output_stream()

    @classmethod
    async def generate_dataset(cls):
        """This function is used to create a dataset for the LightTag platform
        """
        #build the pipeline
        data_stream = cls.get_input_stream(cls.SCHEMA)
        pipeline = cls()

        # create the dataset
        count = 1
        total = len(get_file_list())
        async for data in data_stream:
            await pipeline.run(data)
            print_progress(count, total)
            count += 1
        print("")

    async def coroutine(self, data):
        """This function dictionaries to a json file for using in LightTag
        data sets.

        Args:
            data (:obj:`dict`): A dictionary containing data that needs to be
                tagged
        """
        file_path = get_training_file_path("LightTag-dataset.json")
        is_new_file = create_file(file_path, "[\n]")
        data_string = json.dumps(data)
        prefix = "\n" if is_new_file else ",\n"
        insertion_string = "{0}{1}".format(prefix, data_string)
        with open(file_path, "r+") as json_file:
            json_file.seek(0, 2)
            position = json_file.tell() - 2
            insert(insertion_string, json_file, position)
