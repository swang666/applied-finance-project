"""This module holds a pipeline for generating a file containing training data
for the LDA model
"""
import os
import json

from ucla_topic_analysis.data import get_training_file_path
from ucla_topic_analysis import get_file_list
from ucla_topic_analysis.data.pipeline import Pipeline
from ucla_topic_analysis.data.coroutines import print_progress
from ucla_topic_analysis.data.coroutines.dictionary import DictionaryPipeline


class LdaCorpusPipeline(Pipeline):
    """Pipeline for creating and updating a corpus for training, validating and
    testing an LDA model. This is used to synchronise the pipeline
    since LDA implementations don't generally accept async functions for
    training.

    NOTE: This pipeline is a data sink. It does not return any new data.
    """

    # The split schema for the corpus files
    SCHEMA = {
        "training": 0.8,
        "validation": 0.1,
        "testing": 0.1
    }

    def __init__(self, *args, mode="training", **kwargs):
        """Sets up the pipeline
        """
        super().__init__(*args, **kwargs)

        if mode not in self.SCHEMA:
            raise ValueError("Argument mode must be part of the schema")
        self._mode = mode

        # Used for caching the number of documents for the LDA to train on. This
        # is lazy loaded. To gurantee that you get a value call `len` with this
        # instance as the argument.
        # Note: This is probably not equal to the actual number of files the
        #       corpus is made up of
        self._num_documents = None

        # Used for caching the number of rows in the corpus file. This is lazy
        # loaded. use the `number_of_rows` property instead.
        self._num_rows = None

    @staticmethod
    def get_file_path():
        """
        Returns:
            str: the path to the file containing the corpus' data. The file name
            is 'lda-corpus.dat'.
        """
        file_name = "lda-corpus.dat"
        return get_training_file_path(file_name)

    @classmethod
    async def prepare_data(cls):
        """Runs a pipeline to generate data for training an LDA model and
        saves it to a file.
        """
        # Build the pipeline
        dictionary_input = DictionaryPipeline.get_input_stream(cls.SCHEMA)
        dictionary = DictionaryPipeline(input_stream=dictionary_input)
        bow_stream = dictionary.output_stream()
        lda_corpus_pipeline = cls()

        print("Did not find any corpus data. preparing now")
        # create the data
        count = 1
        total = len(get_file_list())
        async for data in bow_stream:
            await lda_corpus_pipeline.run(data)
            print_progress(count, total)
            count += 1
        print("")

    @property
    def number_of_rows(self):
        """int: The number of rows in the prepared corpus file
        """
        if self._num_rows is None:
            len(self)
        return self._num_rows

    def __len__(self):
        """This function is used to return the number of documents in the corpus

        Returns:
            int: The number of documents in the corpus
        """
        if self._num_documents is None:
            self._num_documents = 0
            self._num_rows = 0  # count rows while we are at it
            with open(self.get_file_path(), "r") as data_file:
                for line in data_file:
                    self._num_rows += 1
                    data = json.loads(line)
                    if data["label"] == self._mode:
                        self._num_documents += len(data["text"])
        return self._num_documents

    def __iter__(self):
        """Generates data from the corups file.

        Yields:
            :obj:`list` of :obj:`(int, int)`)
        """
        completed = 1
        with open(self.get_file_path(), "r") as data_file:
            for line in data_file:
                data = json.loads(line)
                if data["label"] == self._mode:
                    for document in data.get("text"):
                        yield document
                        print_progress(completed, len(self))
                        completed += 1
        print("")

    async def coroutine(self, data):
        """Updates the file with the documents in the data. This is a data sink
        it does not return any new data

        Args:
            data (:obj:`dict`): A dictionary containing the data for the LDA
                model.
        """
        # If the file exists. Append to it.
        if os.path.isfile(self.get_file_path()):
            with open(self.get_file_path(), "a") as data_file:
                data_file.write(json.dumps(data))
                data_file.write("\n")

        # Otherwise create a new file
        else:
            with open(self.get_file_path(), "x") as data_file:
                data_file.write(json.dumps(data))
                data_file.write("\n")

        # Add to the total number of documents if they have been loaded already
        self._num_documents = (None if self._num_documents is None
                               else self._num_documents + len(data["text"]))

        # Add to the total number of rows if they have been loaded already
        self._num_rows = (None if self._num_rows is None
                          else self._num_rows + 1)
