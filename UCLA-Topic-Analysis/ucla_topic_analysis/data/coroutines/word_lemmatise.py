"""A pipeline for getting the root of the word.
"""
import re
import nltk
from nltk.corpus import wordnet as wn
from ucla_topic_analysis.data.pipeline import Pipeline

class LemmaPipeline(Pipeline):
    """Pipeline that obtain the root of the word
    """

    EN_STOP = set(nltk.corpus.stopwords.words('english'))
    word_contain_num = re.compile('.*[0-9].*')
    puntuation = ['(', ')', '[', ']', '{', '}', ',', '.', '$', '#',
                    '%', '/', '!', '&']

    @staticmethod
    def get_lemma(word):
        """Get the root of the word

        Args:
            word (str): The word we want to lemmatise

        Returns:
            str: The lemmatised version of the given word
        """
        lemma = wn.morphy(word)
        if lemma is None:
            return word
        return lemma

    @classmethod
    def prepare_token_for_lda(cls, words):
        """Filter words with lemma and stopwords

        Args:
            words (:obj:`list` of :obj:`str`): The list of words that need to
                be prepared

        Returns:
            :obj:`list` of :obj:`str`: The cleaned up list of words
        """
        tokens = [word.lower() for word in words]
        tokens = [cls.get_lemma(word) for word in tokens
                  if len(word) > 3 and word not in cls.EN_STOP
                  and bool(re.match(cls.word_contain_num, word)) is False
                  and word not in cls.puntuation]
        return tokens

    async def coroutine(self, data):
        """Tokenises the given data into a list of words

        Args:
            data (:obj:`dict`): A dict with the key "text" containing a list of
            lists with tokenised words that need to be processed

        Returns:
            :obj:`dict`: The data dict with the value associated with the key
            "text" replaced with the lemmatised and filtered list of word lists.
            All other data in the dict is left untouched.
        """
        data["text"] = [self.prepare_token_for_lda(sentence)
                        for sentence in data["text"]]
        return data
