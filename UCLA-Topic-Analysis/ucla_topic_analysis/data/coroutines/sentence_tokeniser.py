"""A pipeline for breaking text into sentences.
"""

import nltk
from ucla_topic_analysis.data.pipeline import Pipeline

class SentencePipeline(Pipeline):
    """Pipeline that generates a list of sentences from string
    """

    async def coroutine(self, data):
        """Tokenises the given data into a list of sentences

        Args:
            data (:obj:`dict`): A dictionary containing the key "text" which is
                to be tokenised into sentences.

        Returns:
            :obj:`dict`: The data dict with the value associated with the key
            "text" replaced with a list strings containing the tokenised
            sentences. All other data in the dict is left untouched.
        """
        data["text"] = nltk.sent_tokenize(data["text"])
        return data
