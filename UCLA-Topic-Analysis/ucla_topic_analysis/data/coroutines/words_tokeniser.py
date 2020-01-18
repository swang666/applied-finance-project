"""A pipeline for breaking text into words.
"""
import nltk
from ucla_topic_analysis.data.pipeline import Pipeline

class WordPipeline(Pipeline):
    """Pipeline that generates a list of words from string
    """

    async def coroutine(self, data):
        """Tokenises the text in the given list of sentences into a lists of words

        Args:
            data (:obj:`dict`): A dictionary containng the key "text" which is
                an :obj:`list` of :obj:`str` containing the list of strings to
                be tokenised.

        Returns:
            :obj:`dict`: The data dict with the value associated with the key
            `text` replaced with a list of lists containing tokenised
            words. Any other data in the data dict is left untouched.
        """
        data["text"] = [nltk.word_tokenize(sentence) for sentence in data["text"]]
        return data
