# UCLA-Topic-Analysis

[![CircleCI](https://circleci.com/gh/Ark-Paradigm/UCLA-Topic-Analysis/tree/development.svg?style=svg&circle-token=3e2461970aea2b63b07c5cdff1a11adb4440f4d6)](https://circleci.com/gh/Ark-Paradigm/UCLA-Topic-Analysis/tree/development)

This package is used for finding text belonging to a particular topic from within a financial statement.

## Table of Contents
1. [Getting Started](#getting-started)
    1. [Installation](#installation)
    1. [Prepare Data](#prepare-data)

## Getting Started
### Installation

This repository uses `pipenv` to manage its dependencies. Instructions for how to install `pipenv` can be found [here](https://pipenv.readthedocs.io/en/latest/install/#installing-pipenv). Once `pipenv` is installed we can install the package for development by running:

```
$ pipenv install --dev
```


### Prepare Data

A data pipeline is constructed by extending the Pipeline abstract base class and chaining different Pipeline objects together by passing down-stream pipelines as arguments during pipeline initialisation. For example to create a pipeline for tagging words with their parts of speech me construct build something like this:

```python
pos_tag_pipeline = FilePipeline([SentencePipeline([WordPipeline([POSPipeline()])])])
```

Where the `FilePipeline` would read a file into a string. The `SentencePipeline` would tokenise strings into sentences. The `WordPipeline` would tokenise strings into words. And the POSPipeline would tag a list of words with their part of speech. The key here is that each Pipeline passes its result on to its down-stream pipelines for further processing. So that in our example, the result of reading a file is passed to the pipeline responsible for tokenising sentences which in turn passes its result on to the word tokenisation pipeline before it finally arrives at the part of speech tagging pipeline.

For an example implementation of a Pipeline see [pos_tagging.py](data/pos_tagging.py)