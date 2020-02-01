# Sentiment Analysis of the potential Replies that a given Tweet may get

### By S. Tayebi Arasteh, M. Monajem, and S. Sitaula

This is the final project of the `Seminar Deep Learning` course (WS1920) jointly offered by the [Pattern Recognition Lab (LME)](https://lme.tf.fau.de/) of the *Computer Science Department*, and 
the [Chair of Computational Corpus Linguistics](https://www.linguistik.phil.fau.de/) of the *Department of German Language and Literature* at University of Erlangen-Nuremberg (FAU).


The main function running the training, testing and validation process is `main.py`.

Introduction
------
*What is Sentiment Analysis?* It is a machine learning technique that automatically analyses and detects the sentiment of text.
Here we do *Message-level* sentiment analysis, i.e. we classify the sentiment of the whole given message (tweet or reply).
### Goal of this project 
The aim of this project is to build a sentiment classification tool which, given a tweet predicts how likely it gets positive, negative or neutral replies. In other words, it is a tool to predict the nature of replies to a tweet is most likely to get.
## Getting Started

### Prerequisites

The software is developed in **Python 3.7**. For the deep learning, the **PyTorch 1.3.1** (including *TorchText 0.5.0*) framework is used.


Main Python modules required for the software can be installed from ./requirements in three stages:

1. Create a Python3 environment by installing the conda `environment.yml` file:

```
$ conda env create -f environment.yml
$ activate SentimentAnalysis
```



2. Install the remaining dependencies from `requirements.txt`.

***Note:*** These might take a few minutes.

3. We have also made use of **spaCy** to tokenize our data. To install spaCy, follow the instructions [here](https://spacy.io/usage) making sure to install the English models with:

```
$ python -m spacy download en
```
