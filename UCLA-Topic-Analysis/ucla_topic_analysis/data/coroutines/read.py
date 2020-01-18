"""Contains a pipeline for reading text files
"""
import os
import random
from ucla_topic_analysis import get_file_list, get_data_folder
from ucla_topic_analysis.data.pipeline import Pipeline


class ReadFilePipeline(Pipeline):
    """Pipeline that generates a list of
    """

    def __init__(self, *args, schema=None, **kwargs):
        """Initialises the pipeline

        Args:
            schema (:obj:`dict`): A dict where the keys are the categories and
                the values are the proportion of documents in that category.
                The values must be in the range [0, 1] and must add up to 1. If
                this is `None` (default) then every document will be labelled
                `None`.
        """

        # Sanity tests
        if schema and sum([schema[key] for key in schema]) != 1:
            raise ValueError("Values in schema must add up to 1")
        if schema and any([not (0 <= schema[key] <= 1) for key in schema]):
            raise ValueError("Values in schema must be in the range [0, 1]")

        # Set the instance variables
        self._schema = schema
        self._seed = 0

        # Initialise parent class
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_input_stream():
        """An input stream for the pipeline
        """
        file_paths = sorted(get_file_list())
        for file_path in file_paths:
            yield file_path

    async def coroutine(self, data):
        """Extracts text from the given data file and attaches

        Args:
            data (str): Path to the file that is to be read

        Returns:
            :obj:`dict`: A dictionary of the form::

                {
                    'text': The text in the file pointed to by the data,
                    'label': The label associated with this document,
                    'path': The relative path to the file
                }
        """
        rel_path = os.path.relpath(data, get_data_folder())
        result = {"label": self._sort_document(), "path": rel_path}
        with open(data, encoding='utf-8', mode='r') as data_file:
            result["text"] = data_file.read()
        self._seed += 1
        return result

    def _sort_document(self, seed=None):
        """Used to pick a document for the current mode

        Args:
            seed: The seed to use for the random number generator. If this is
                not set then self._seed will be used.

        Returns:
            str: the label to attach to the document
        """
        seed = seed or self._seed

        # Generate a random number
        random.seed(self._seed)
        random_number = random.random()

        # If no mode or split mechanic is set then pick every document
        if not self._schema:
            return None

        # Check if we should pick the document
        start = 0
        for key in sorted(self._schema.keys()):
            end = start + self._schema[key]
            if start <= random_number < end:
                return key
            start += self._schema[key]
        raise IndexError("Error with `split` proportions")
