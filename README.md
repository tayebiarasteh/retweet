# How Will Your Tweet Be Received? Predicting the Sentiment Polarity of Tweet Replies

### By [Soroosh Tayebi Arasteh](https://github.com/starasteh), et al.

[![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
[![](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/starasteh/retweet/pulls)

Overview
------

* This is the official repository of the paper **How Will Your Tweet Be Received? Predicting the Sentiment Polarity of Tweet Replies**.
* The paper comes with a public dataset, [RETWEET](https://kaggle.com/soroosharasteh/retweet/), which is accessible under this link:
https://kaggle.com/soroosharasteh/retweet/





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

Code structure
---
1. Everything can be ran from *./main.py*. 
* Set the hyper-parameters and model parameters here.
* *main_train()*, *main_test()*, and *main_manual_predict()* run codes related to Standard Tweet Sentiment Polarity Classification.
* The data preprocessing parameters and directories can be modified from *./configs/config.json*. Note that, you have to stick to the names specified in the *./configs/config.json* and name your data accordingly.
* Also, you should first choose an `experiment` name (if you are starting a new experiment) for training, in which all the evaluation and loss value statistics, tensorboard events, and model & checkpoints will be stored. Furthermore, a `config.json` file will be created for each experiment storing all the information needed.
* For testing, just load the experiment which its model you need.

2. The rest of the files:
* *./models/* directory contains all the model architectures and losses.
* *./Train_Test_Valid.py* contains the training, validation, and the inference processes.
