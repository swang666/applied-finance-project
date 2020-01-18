"""train a TF-IDF model
"""
import asyncio
import pickle
import os
import time
from ucla_topic_analysis.analysis.tfidf_score import TFIDFScorePipeline
from ucla_topic_analysis.data.coroutines.tf_idf import TFIDFPipeline
from ucla_topic_analysis.data.coroutines.tf_idf_pre_process import TFIDFDataPreprocessor

if __name__ == "__main__":
    file_path = TFIDFPipeline.get_file_path()
    if os.path.isfile(file_path):
        print('model already trained, start computing tfidf score')
        tfidf_score = TFIDFScorePipeline()
        asyncio.run(tfidf_score.calc_cos())
    else:
    # train model
        tfidf = TFIDFPipeline()
        asyncio.run(tfidf.train())

    
    



