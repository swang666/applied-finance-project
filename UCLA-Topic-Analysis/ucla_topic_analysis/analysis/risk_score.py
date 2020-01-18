import time
import os
import re
import pandas as pd
import numpy as np
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from ucla_topic_analysis.analysis import get_score_file_path
from ucla_topic_analysis.data.pipeline import Pipeline
from ucla_topic_analysis import get_file_list
from ucla_topic_analysis.data.coroutines import print_progress
from ucla_topic_analysis.data.coroutines.read import ReadFilePipeline
from ucla_topic_analysis.data.coroutines.sentence_tokeniser import SentencePipeline
from ucla_topic_analysis.data.coroutines.words_tokeniser import WordPipeline
from ucla_topic_analysis.data.coroutines.word_lemmatise import LemmaPipeline

class RiskScorePipeline(Pipeline):
    """Pipeline for calculating a risk score
    """
    RISK_WORD = re.compile('.*risk.*')
    UNCERTAIN_WORD = re.compile('.*uncertain.*')

    def __init__(self, *args, **kwargs):
        """Loads a tfidf csv file for updating
        """
        super().__init__(*args, **kwargs)

        # This is only for lazy loading. Use get_dict() unless you are sure you
        # need this.
        self._tfidf_df = None

    @staticmethod
    def get_input_stream(schema=None):
        """This function is used to get a pipeline to get the sentences to calculate
        risk score

        Args:
            schema(:obj:`dict`): The schema for the file pipeline

        Returns:
            An iterable containing lists of sentences
        """
        # Build the pipeline
        files = ReadFilePipeline.get_input_stream()
        file_stream = ReadFilePipeline(
            input_stream=files, schema=schema).output_stream()
        sent_stream = SentencePipeline(
            input_stream=file_stream).output_stream()
        word_stream = WordPipeline(input_stream=sent_stream).output_stream()
        return LemmaPipeline(input_stream=word_stream).output_stream()

    @staticmethod
    def load_model():
        """This function loads the LDA dictionary and model 

        Returns:
            a gensim dictionary and LDA model
        """
        dictionary = Dictionary.load('ucla_topic_analysis/data/training/dictionary.gensim')
        model = LdaModel.load('ucla_topic_analysis/data/training/lda-50.model')
        return dictionary, model

    async def calc_risk(self):
        """This function calculates a risk score
        """
        dictionary, model = self.load_model()
        # for this model, risk topic id is 15
        input_stream = self.get_input_stream()
        # Train the dictionary
        count = 1
        total = len(get_file_list())
        tickers = []
        filing_dates = []
        total_num_sent = []
        total_risk_sent = []
        total_risk_top1 = []
        total_risk_top2 = []
        total_risk_top3 = []
        total_risk_top4 = []
        total_risk_words = []
        total_uncertain_words = []
        weighted_risk_score = []
        weighted_rank_score = []
        async for data in input_stream:
            list_of_tokenized_words = data['text']
            path = data['path'].split('\\')
            ticker = path[1]
            filing_date = path[-1][:10]
            total_sent = len(list_of_tokenized_words)
            scores = []
            ranks = []
            risk_num = 0
            risk_top1 = 0
            risk_top2 = 0
            risk_top3 = 0
            risk_top4 = 0
            risk_word = 0
            uncertain_word = 0
            for tokens in list_of_tokenized_words:
                bow_vector = dictionary.doc2bow(tokens)
                for idx, (topic_id, score) in enumerate(sorted(model[bow_vector], key=lambda tup: -1*tup[1])):
                    if topic_id == 15:
                        scores.append(score)
                        ranks.append(idx + 1)
                        risk_num = risk_num + 1
                        if idx < 1:
                            risk_top1 += 1
                        elif idx < 2:
                            risk_top2 += 1
                        elif idx < 3:
                            risk_top3 += 1
                        elif idx < 4:
                            risk_top4 += 1
                for token in tokens:
                    if re.match(self.RISK_WORD, token):
                        risk_word += 1
                    elif re.match(self.UNCERTAIN_WORD, token):
                        uncertain_word += 1
            tickers.append(ticker)
            filing_dates.append(filing_date)
            total_num_sent.append(total_sent)
            total_risk_sent.append(risk_num)
            total_risk_top1.append(risk_top1)
            total_risk_top2.append(risk_top2)
            total_risk_top3.append(risk_top3)
            total_risk_top4.append(risk_top4)
            total_risk_words.append(risk_word)
            total_uncertain_words.append(uncertain_word)
            if len(scores) > 0:
                weighted_risk_score.append(np.mean(scores))
                weighted_rank_score.append(np.mean(ranks))
            else:
                weighted_risk_score.append(0)
                weighted_rank_score.append(0)
            print_progress(count, total)
            count += 1
        print('')
        
        df = pd.DataFrame({
            'ticker': tickers,
            'filing dates': filing_dates,
            'total number of sentences': total_num_sent,
            'total number of risk sentences': total_risk_sent,
            'score rank 1': total_risk_top1,
            'score rank 2': total_risk_top2,
            'score rank 3': total_risk_top3,
            'score rank 4': total_risk_top4,
            'average of risk score': weighted_risk_score,
            'average of ranks': weighted_rank_score,
            'total number of risk word': total_risk_words,
            'total number of uncertain word': total_uncertain_words
        })

        file_name = "risk_score.csv"
        file_path = get_score_file_path(file_name)
        df.to_csv(file_path)


    async def coroutine(self, data):
        '''
        empty
        '''
