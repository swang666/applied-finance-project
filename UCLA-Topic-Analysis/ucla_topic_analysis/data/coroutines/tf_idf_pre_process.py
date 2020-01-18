"""This module contains a pipeline to proprocess data for training a TF-IDF
model
"""
import os
import json

from ucla_topic_analysis.data import get_training_file_path
from ucla_topic_analysis import get_file_list
from ucla_topic_analysis.data.coroutines import print_progress
from ucla_topic_analysis.data.pipeline import Pipeline
from ucla_topic_analysis.data.coroutines.read import ReadFilePipeline
from ucla_topic_analysis.data.coroutines.sentence_tokeniser import SentencePipeline
from ucla_topic_analysis.data.coroutines.words_tokeniser import WordPipeline
from ucla_topic_analysis.data.coroutines.word_lemmatise import LemmaPipeline


class TFIDFDataPreprocessor(Pipeline):
    """Pipeline for creating and updating a file with preprocessed data for
    training a tf-idf model. This is used to synchronise the pipeline since
    tf-idf implementations don't generally accept async functions for training.

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
            raise ValueError("'mode' must be one of %s" %set(self.SCHEMA.keys()))
        self._mode = mode

        # Used for caching the number of documents for TF-IDF to train on. This
        # is lazy loaded. To gurantee that you get a value call `len` with this
        # instance as the argument.
        # Note: This is probably not equal to the actual number of files the
        #       corpus is made up of
        self._num_documents = None

        # Used for caching the number of rows in the corpus file. This is lazy
        # loaded. Use the `number_of_rows` property instead.
        self._num_rows = None

    @staticmethod
    def get_file_path():
        """
        Returns:
            str: the path to the file containing the preprocessed data. The file
            name is 'tf-idf-corpus.dat'.
        """
        file_name = "tf-idf-corpus.dat"
        return get_training_file_path(file_name)

    @staticmethod
    def get_input_stream(schema=None):
        """This function builds a pipeline to for preprocessing the data for the
        model.

        Args:
            schema(:obj:`dict`): The schema for the file pipeline

        Returns:
            An iterable containing lists of words to train a dictionary with.
        """
        # Build the pipeline
        files = ReadFilePipeline.get_input_stream()
        file_stream = ReadFilePipeline(
            input_stream=files, schema=schema).output_stream()
        sent_stream = SentencePipeline(
            input_stream=file_stream).output_stream()
        word_stream = WordPipeline(input_stream=sent_stream).output_stream()
        return LemmaPipeline(input_stream=word_stream).output_stream()

    @classmethod
    async def prepare_data(cls):
        """Runs a pipeline to generate data for training a TF-IDF model and
        saves it to a file.
        """
        # Build the pipeline
        data_stream = cls.get_input_stream(cls.SCHEMA)
        pipeline = cls()

        # Process the data
        count = 1
        total = len(get_file_list())
        async for data in data_stream:
            await pipeline.run(data)
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
            str: A preprocessed document in the corpus.
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
        # Join words into a list of documents.
        data["text"] = [" ".join(document) for document in data["text"]]

        # If the data file exists just append to it.
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

