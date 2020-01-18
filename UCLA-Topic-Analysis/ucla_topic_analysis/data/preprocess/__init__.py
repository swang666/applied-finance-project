""" Defines pipelines for preprocessing data
"""
import asyncio
import os
from pathlib import Path
from ucla_topic_analysis.data.coroutines.read import ReadFilePipeline
from ucla_topic_analysis.data.coroutines.sentence_tokeniser import SentencePipeline
from ucla_topic_analysis.data.coroutines.words_tokeniser import WordPipeline
from ucla_topic_analysis.data.coroutines.word_lemmatise import LemmaPipeline
from ucla_topic_analysis import get_file_list

async def tokenise_sentences():
    """
    This function tokenises sentences contained in the data files and yeilds
    them one at a time

    Yields:
        str: A single sentence
    """
    files = get_file_list()
    file_pipeline = ReadFilePipeline(input_stream=files)
    sentence_pipeline = SentencePipeline(input_stream=file_pipeline.output_stream())
    async for sentence in sentence_pipeline.output_stream():
        yield sentence

def tokenise_words():
    """
    This function tokenises words contained in sentences

    Yields:
        str: A single word
    """
    lemma_pipeline = LemmaPipeline()
    word_pipeline = WordPipeline([lemma_pipeline])
    sentence_pipeline = SentencePipeline([word_pipeline])
    file_pipeline = ReadFilePipeline([sentence_pipeline])
    loop = asyncio.get_event_loop()
    files = get_file_list()
    print(len(files))
    for file_path in files:
        loop.run_until_complete(file_pipeline.run(file_path))
        for root in lemma_pipeline.result:
            yield root
    loop.close()
