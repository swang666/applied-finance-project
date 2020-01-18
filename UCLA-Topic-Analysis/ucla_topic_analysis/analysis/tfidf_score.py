import time
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from ucla_topic_analysis.data.pipeline import Pipeline
from ucla_topic_analysis.data import get_training_file_path
from ucla_topic_analysis.data.coroutines import print_progress
from ucla_topic_analysis import get_file_list
from ucla_topic_analysis.analysis import get_score_file_path
from ucla_topic_analysis.data.coroutines.read import ReadFilePipeline
from ucla_topic_analysis.data.coroutines.sentence_tokeniser import SentencePipeline
from ucla_topic_analysis.data.coroutines.words_tokeniser import WordPipeline
from ucla_topic_analysis.data.coroutines.word_lemmatise import LemmaPipeline
from ucla_topic_analysis.data.coroutines.sent_lemmatise import SentLemmaPipeline

class TFIDFScorePipeline(Pipeline):
    """Pipeline for calculating a tfidf score
    """
    TOPICS = [
        'investment property distribution interest agreement',
        'regulation change law financial operation tax accounting',
        'gas price oil natural operation production input prices risks',
        'stock price share market futuredividend security stakeholder',
        'cost regulation environmental law operation liability',
        'control financial internal loss reporting history Financial condition risks',
        'financial litigation operation condition action legal liability regulatory claim lawsuit',
        'competitive industry competition highly market',
        'cost operation labor operating employee increase acquisition',
        'product candidate development approval clinical regulatory',
        'tax income asset net goodwill loss distribution impairment',
        'interest director officer trust combination share conflict',
        'product liability claim market insurance sale revenue',
        'loan real estate investment property market loss portfolio Investment',
        'personnel key retain attract management employee',
        'stock price operating stockholder fluctuate interest volatile',
        'acquisition growth future operation additional capital strategy',
        'condition economic financial market industry change affected downturn demand',
        'system service information failure product operation software network breach interruption',
        'cost contract operation plan increase pension delay',
        'customer product revenue sale supplier relationship key portion contract manufacturing rely',
        'property intellectual protect proprietary technology patent protection harm license',
        'product market service change sale demand successfully technology competition',
        'provision law control change stock prevent stockholder Delaware charter delay bylaw',
        'regulation government change revenue contract law service',
        'capital credit financial market cost operation rating access liquidity downgrade',
        'debt indebtedness cash obligation financial credit covenant',
        'operation international foreign currency rate fluctuation',
        'loss insurance financial loan reserve operation cover',
        'operation natural facility disaster event terrorist weather',
    ]

    def __init__(self, *args, **kwargs):
        """Loads a tfidf csv file for updating
        """
        super().__init__(*args, **kwargs)

        # This is only for lazy loading. Use get_dict() unless you are sure you
        # need this.
        self._model = self.load_model()
        self._topic_sparse_mat = {}
        for i in range(30):
            self._topic_sparse_mat['topic' + str(i)] = self._model.transform([self.TOPICS[i]])

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
        token_stream = LemmaPipeline(input_stream=word_stream).output_stream()
        return SentLemmaPipeline(input_stream=token_stream).output_stream()

    @staticmethod
    def load_model():
        """This function loads a tf-idf model

        Returns:
            a tf-idf model (TFIDFVectorizer)
        """
        file_name = get_training_file_path('tf-idf.model')
        with open(file_name, "rb") as model_file:
            model = pickle.load(model_file)
        return model

    async def calc_cos(self):
        """This function calculates a cos similarity score
        """
        count = 1
        total = len(get_file_list())
        file_name = "cos_score.csv"
        file_path = get_score_file_path(file_name)
        input_stream = self.get_input_stream()
        async for data in input_stream:
            sentences = data['text']
            sent_mat = self._model.transform(sentences)
            cosine_similarities = []
            for i in range(30):
                cosine_similarities.append(linear_kernel(self._topic_sparse_mat['topic' + str(i)], sent_mat).flatten())
            n = len(sentences)
            for i in range(n):
                if len(sentences[i]) > 20:
                    score_dict = {'10k_path': [data['path']],
                            'sentence_index' : [i],
                            'joined tokens' : [sentences[i]]}
                    for j in range(30):
                        score_dict['topic'+str(j)+' score'] = cosine_similarities[j][i] if cosine_similarities[j][i] > 0.1 else None
                    if os.path.isfile(file_path):
                        df = pd.DataFrame(score_dict)
                        if df.iloc[0, 3:32].sum() > 0:
                            with open(file_path, 'a', encoding = 'utf-8') as f:
                                df.to_csv(f, encoding = 'utf-8', index = False, header=False)
                    else:
                        df = pd.DataFrame(score_dict)
                        if df.iloc[0, 3:32].sum() > 0:
                            df.to_csv(file_path, encoding = 'utf-8', index = False)
            print_progress(count, total)
            count += 1
        print('')
            

    async def coroutine(self, data):
        '''
        empty
        '''