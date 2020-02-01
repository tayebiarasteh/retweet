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


Approach to the challange
------
This project can be divided into three consecutive parts to reach our goal.

1. **Sentiment Analysis of the labelled tweets from [SemEval 2015 Task 10, Subtask B](http://alt.qcri.org/semeval2015/task10/) dataset.**

2. **Dataset Creation for Post Reply Datasets:**
This classifier is employed to the replies of a huge number of tweets to find the sentiment of each the reply.
Each tweet will then have a list of sentiments as many as the number of replies.
The respective tweet is then given a label based on a maximum number of occurence of a particular sentiment i.e the argmax value of the classified sentiments.
This label shows the sentiment of the replies that a tweet  is most likely to get. 
3. **Sentiment Predictor for Tweet Replies:**
Use the labelled dataset created from the second part to train another model which will predict the sentiment for Tweet-Replies.


Proposed architecture
------
The architecture used in this software is a variant of the Bi-Directional Long-Short Term Memory units (BiLSTM).
