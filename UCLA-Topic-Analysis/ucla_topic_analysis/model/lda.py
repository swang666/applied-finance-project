""" This module holds code for training an LDA model.
"""

import logging
import os
import pickle
import random
import json

from gensim import corpora
import gensim

from ucla_topic_analysis.data.preprocess import tokenise_words

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Lda:
    '''this class is an object contains the text data,
    dictionary, corpus, model of the lda model
    '''
    def __init__(self, num_topics, num_passes):
        self.num_topics = num_topics
        self.num_passes = num_passes
        self.text_data = None
        self.dictionary = None
        self.corpus = None
        self.model = None

    def build_dict(self):
        '''this function builds the dictionary and corpus from
        list of words

        Args: list of words(strings)

        Returns: gensim dictionary and corpus
        '''
        self.text_data = list(tokenise_words())
        dictionary = corpora.Dictionary(self.text_data)
        corpus = [dictionary.doc2bow(text) for text in self.text_data]
        self.dictionary = dictionary
        self.corpus = corpus
        # save dictionary
        dictionary.save('ucla_topic_analysis/model/dictionary.gensim')
        pickle.dump(corpus, open('ucla_topic_analysis/model/corpus.pkl', 'wb'))
        pickle.dump(self.text_data, open('ucla_topic_analysis/model/text_data.pkl', 'wb'))
        # return dictionary, corpus

    def build_lda_model(self):
        '''this function builds the lda model from the documents
        it checks if the directory contains the dictionary and corpus,
        and if user wants to update dictionary

        Args: num_topics: number of topics the model will classify, default 10
            num_passes: number of iterations the model will run, default 1

        Returns: a gensim model
        '''
        dict_exists = os.path.isfile('ucla_topic_analysis/model/dictionary.gensim')
        corpus_exists = os.path.isfile('ucla_topic_analysis/model/corpus.pkl')
        text_exists = os.path.isfile('ucla_topic_analysis/model/text_data.pkl')
        # update_dict = (input("Update dictionary? [y/n] ") == 'y')
        if  dict_exists and corpus_exists and text_exists:
            self.dictionary = corpora.Dictionary.load(
                'ucla_topic_analysis/model/dictionary.gensim')
            with open('ucla_topic_analysis/model/corpus.pkl', 'rb') as corpus_file:
                self.corpus = pickle.load(corpus_file)
            with open('ucla_topic_analysis/model/text_data.pkl', 'rb') as text_file:
                self.text_data = pickle.load(text_file)
        else:
            self.build_dict()
        # create a tfidf corpus
        # tfidf_corpus = tfidf.create_corpus(corpus)
        self.model = gensim.models.ldamulticore.LdaMulticore(
            self.corpus, num_topics=self.num_topics, id2word=self.dictionary,
            passes=self.num_passes, workers=3)
        # ldamodel.save('ucla_topic_analysis/model/lda.gensim')
        '''
        tfidf_ldamodel = gensim.models.ldamodel.LdaModel(
            tfidf_corpus, num_topics=num_topics, id2word=dictionary, passes=num_passes)
        tfidf_ldamodel.save('ucla_topic_analysis/model/tfidf_lda.gensim')
        '''
        # topics = ldamodel.print_topics(num_words=10)
        # return ldamodel
