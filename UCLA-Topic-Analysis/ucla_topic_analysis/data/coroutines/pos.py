"""A pipeline for tagging parts of speech.
"""
import nltk
from ucla_topic_analysis.data.pipeline import Pipeline

class POSPipeline(Pipeline):
    """Pipeline that generates parts of speech tags given a list of words.
    """

    async def coroutine(self, data):
        """Generates parts of speech tags for the given data

        Args:
            data (:obj:`list` of :obj:`str`): list of words to be tagged

        Returns:
            A list of tuples of the form::
                [
                    ('word1', 'POS tag'),
                    ('word2', 'POS tag'),
                    ('word3', 'POS tag'),
                    ...
                ]
        """
        return nltk.pos_tag(data)
