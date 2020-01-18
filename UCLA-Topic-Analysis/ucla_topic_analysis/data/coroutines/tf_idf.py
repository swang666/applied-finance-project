"""Contasins a TF-IDF data pipeline
"""
import os
import pickle

import nltk

from sklearn.feature_extraction.text import TfidfVectorizer

from ucla_topic_analysis.data.pipeline import Pipeline
from ucla_topic_analysis import log_async_time
from ucla_topic_analysis.data import get_training_file_path
from ucla_topic_analysis.data.coroutines.word_lemmatise import LemmaPipeline
from ucla_topic_analysis.data.coroutines.tf_idf_pre_process import TFIDFDataPreprocessor


class TFIDFPipeline(Pipeline):
    """Pipeline for creating a TF-IDF model. This is a data sink, it does not
    return any new data
    """

    EN_STOP = LemmaPipeline.EN_STOP

    def __init__(self, *args, **kwargs):
        """Initialises the TF-IDF pipeline
        """
        super().__init__(*args, **kwargs)

        # This is only used for lazy loading. Use self.get_model() to ensure it
        # is not None. And to create a new model if one does not exist.
        self._model = None

    @staticmethod
    def get_file_path():
        """str: the name of the model's file. It is of the form
        tf-idf.model
        """
        file_name = "tf-idf.model"
        return get_training_file_path(file_name)

    def get_model(self):
        """This function is used to get an instance of a TF-IDF model. It will
        load the model from file if it finds one, otherwise it will create a new
        one.

        Returns:
            :obj:`sklearn.feature_extraction.text.TfidfVectorizer`: A tf-idf
            model
        """
        self._model = self._model or self._load_model()
        if not self._model:
            print("No previous model found. Creating a new one for training.")
            self._model = TfidfVectorizer(
                tokenizer=nltk.word_tokenize,
                stop_words=self.EN_STOP)
        return self._model

    def _load_model(self):
        """This function is used to load a TF-IDF model from the models
        folder. Or `None` if one does not exist.

        Returns:
            :obj:`sklearn.feature_extraction.text.TfidfVectorizer`: The model
            found in ucla_topic_analysis/model/tf-idf.model or None if there was
            no tf-idf model saved.
        """
        if os.path.isfile(self.get_file_path()):
            with open(self.get_file_path(), "rb") as model_file:
                return pickle.load(model_file)
        return None

    def save_model(self, file_path=None):
        """Saves the updated model to file overwriting any existing model.
        """
        if self._model is not None:
            path = file_path or self.get_file_path()
            with open(path, "wb") as modle_file:
                pickle.dump(self._model, modle_file)
        else:
            raise Exception("Can not save. No model has been loaded.")

    @log_async_time
    async def train(self):
        """Trains a TF-IDF model from the data in the corpus file.
        """
        # Initialise vectorizer and corpus
        vectorizer = TfidfVectorizer(
            tokenizer=nltk.word_tokenize,
            stop_words=self.EN_STOP
        )
        corpus = TFIDFDataPreprocessor()

        # Make sure corpus data has been prepared
        if not os.path.isfile(corpus.get_file_path()):
            print("Did not find any corpus data. preparing now")
            await corpus.prepare_data()

        # Train the model
        vectorizer.fit(corpus)

        # Set self._model and save to file
        self._model = vectorizer
        self.save_model()

    async def coroutine(self, data):
        """empty
        """
