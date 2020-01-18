"""A pipeline for generating a dictionary from a corpus
"""
import os

from gensim.models.ldamulticore import LdaMulticore

from ucla_topic_analysis import get_workers
from ucla_topic_analysis import log_async_time, log_time
from ucla_topic_analysis.data import get_training_file_path
from ucla_topic_analysis.data.pipeline import Pipeline
from ucla_topic_analysis.data.coroutines.dictionary import DictionaryPipeline
from ucla_topic_analysis.data.coroutines.lda_corpus import LdaCorpusPipeline


class LdaPipeline(Pipeline):
    """Pipeline for creating and updating a gensim LDA model. This is a data
    sink. it does not return any new data
    """

    def __init__(self, num_topics, *args, **kwargs):
        """Initialises the LDA pipeline

        Args:
            num_topics (int): The number of topics in the LDA model
            workers (int): The number of workers to use to training. Defaults to
            num_cores - 1
        """
        super().__init__(*args, **kwargs)
        self._num_topics = num_topics
        self._workers = get_workers()

        # This is only used for lazy loading. Use self.get_model() to ensure it
        # is not None. And to create a new model if one does not exist.
        self._model = None

    @property
    def file_path(self):
        """str: the name of the model's file. It is of the form
        lda-num-topics.model
        """
        file_name = "lda-{num_topics}.model".format(num_topics=self._num_topics)
        return get_training_file_path(file_name)

    async def get_model(self):
        """This function is used to get an instance of an LdaModel. It will load
        the model from file if it finds one, otherwise it will create a new one
        using a saved dictionary if one exists. Lastly it will create a new
        dictionary from the corpus if it finds no pre-saved dictionary to use
        and no pre-saved LDA model to load.

        Returns:
            :obj:`gensim.models.ldamodel.LdaModel`: A gensim LdaModel
        """
        self._model = self._model or self._load_model()
        if not self._model:
            print("No previous model found. Creating a new one for training")
            dictionary = await DictionaryPipeline().get_dictionary()
            self._model = LdaMulticore(id2word=dictionary, workers=self._workers)
        return self._model

    def _load_model(self):
        """This function is used to load a gensim LdaModel from the models
        folder. Or `None` if one does not exist.

        Returns:
            :obj:`gensim.models.ldamodel.LdaModel`: The model found
            in ucla_topic_analysis/model/lda.model or None if there was
            no lda model saved or the number of topics does not match.
        """

        if os.path.isfile(self.file_path):
            return LdaMulticore.load(self.file_path)
        return None

    def save_model(self, file_path=None):
        """Saves the updated model to file
        """
        if self._model is not None:
            path = file_path or self.file_path
            self._model.save(path)
        else:
            raise Exception("Can not save. No model has been loaded.")

    @log_time
    def get_log_perplexity(self, mode):
        """Used to get the log perplexity for the LDA model.

        Args:
            mode (str): The label associated with the files for which to
                calculate the log perplexity.

        Returns:
            int: The log perplexity for the LDA model
        """
        # Get corpus
        corpus = LdaCorpusPipeline(mode=mode)

        # Make sure corpus data has been prepared
        if not os.path.isfile(corpus.get_file_path()):
            raise Exception("No corpus has been prepared.")

        model = self._model or self._load_model()
        if not model:
            raise Exception("No model saved model found. Please tain one first.")
        return model.log_perplexity(corpus)


    @log_async_time
    async def train(self):
        """This function trains an LDA model from the data in the corpus file.
        It will overwrite any existing model and creating a new one if one does
        not exist.
        """
        # Get the dictionary
        dictionary = await DictionaryPipeline().get_dictionary()

        # Get corpus
        corpus = LdaCorpusPipeline()

        # Make sure corpus data has been prepared
        if not os.path.isfile(corpus.get_file_path()):
            await corpus.prepare_data()

        print("Training model. This might take some time")
        model = LdaMulticore(
            corpus=corpus,
            num_topics=self._num_topics,
            id2word=dictionary,
            workers=self._workers
            )
        self._model = model
        self.save_model()

    async def coroutine(self, data):
        """Updates the model with the documents in the data. This is a data sink
        it does not return any new data

        Args:
            data (:obj:`list` of :obj:`list` of :obj:`(int, int)`): A list of
                documents, in bag of words representation
        """
        model = await self.get_model()
        model.update(data)
