"""A pipeline for combining lemmatised words into sent
"""
from ucla_topic_analysis.data.pipeline import Pipeline

class SentLemmaPipeline(Pipeline):
    """Pipeline that combine words into sent
    """

    async def coroutine(self, data):
        """join tokens with space to form sentence

        Args:
            data (:obj:`dict`): A dictionary containng the key "text" which is
                an :obj:`list` of :obj:`str` containing the list of lists of strings
                be join

        Returns:
            :obj:`dict`: The data dict with the value associated with the key
            `text` replaced with a list of sentences
        """
        data["text"] = [' '.join(tokens) for tokens in data["text"]]
        return data
