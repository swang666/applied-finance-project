"""A pipeline for generating a dictionary from a corpus
"""
import os
from gensim.corpora import Dictionary

from ucla_topic_analysis import get_file_list
from ucla_topic_analysis.data import get_training_file_path
from ucla_topic_analysis.data.pipeline import Pipeline
from ucla_topic_analysis.data.coroutines import print_progress
from ucla_topic_analysis.data.coroutines.read import ReadFilePipeline
from ucla_topic_analysis.data.coroutines.sentence_tokeniser import SentencePipeline
from ucla_topic_analysis.data.coroutines.words_tokeniser import WordPipeline
from ucla_topic_analysis.data.coroutines.word_lemmatise import LemmaPipeline


class DictionaryPipeline(Pipeline):
    """Pipeline for creating and updating a gensim dictionary and converting
    documents to a bag of words representation.
    """

    def __init__(self, *args, **kwargs):
        """Loads a dictionary for updating
        """
        super().__init__(*args, **kwargs)

        # This is only for lazy loading. Use get_dict() unless you are sure you
        # need this.
        self._dictionary = None

    @staticmethod
    def load_dictionary():
        """This function is used to load a gensim dictionary from the models
        folder.

        Returns:
            :obj:`gensim.corpora.dictionary.Dictionary`: The dictionary found
            in ucla_topic_analysis/model/dictionary.gensim or None if there was
            no dictionary.
        """
        file_name = "dictionary.gensim"
        file_path = get_training_file_path(file_name)
        if os.path.isfile(file_path):
            return Dictionary.load(file_path)
        return None

    @staticmethod
    def get_input_stream(schema=None):
        """This function is used to get a pipeline to feed into a dictionary for
        training an LDA model.

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

    async def train_dictionary(self):
        """This function trains a new gensim dictionary from the corpus.
        """
        input_stream = self.get_input_stream()
        # Train the dictionary
        count = 1
        total = len(get_file_list())
        async for data in input_stream:
            await self.run(data)
            print_progress(count, total)
            count += 1
        print("")
        self.save_dict()

    async def get_dictionary(self):
        """This function is used to get an instance of a gensim dictionary. It
        will load a dictionary from file if one has not already been loaded. If
        no previous dictionary has been loaded and no dictionary has been saved
        to file it will train a new one.

        Returns:
            :obj:`gensim.corpora.dictionary.Dictionary`: The dictionary found
            in ucla_topic_analysis/model/dictionary.gensim or None if there was
            no dictionary.
        """
        if self._dictionary is None:
            self._dictionary = self.load_dictionary()
        if self._dictionary is None:
            print("Did not find a saved dictionary. Training one now.")
            self._dictionary = Dictionary()
            await self.train_dictionary()
        return self._dictionary

    def save_dict(self):
        """Saves the updated dictionary to file
        """
        file_name = "dictionary.gensim"
        file_path = get_training_file_path(file_name)

        self._dictionary.save(file_path)

    async def coroutine(self, data):
        """Converts the documents in the data to bags of words

        Args:
            data (:obj:`dict`): A dict with the key "text" containing a list of
                lists with tokenised words that need to be changed to a bag of
                words format.
        Returns:
            :obj:`dict`: The data dict with the value associated with "text"
            replaced with a list containing a bag of words representation for
            each document.
        """
        dictionary = await self.get_dictionary()
        data["text"] = [dictionary.doc2bow(document, allow_update=True)
                        for document in data["text"]]
        self.save_dict()
        return data
