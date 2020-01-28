# Sentiment Analysis of the potential future reply threads of tweets

### By S. Tayebi Arasteh, M. Monajem, and S. Sitaula

This is the final project of the `Seminar Deep Learning` course (WS1920) jointly offered by the [Pattern Recognition Lab (LME)](https://lme.tf.fau.de/) of the *Computer Science Department*, and 
the [Chair of Computational Corpus Linguistics](https://www.linguistik.phil.fau.de/) of the *Department of German Language and Literature* at University of Erlangen-Nuremberg (FAU).


The main function running the training, testing and validation process is `main.py`.

## Getting Started

### Prerequisites

The software is developed in **Python 3.7**. For the deep learning, the **PyTorch 1.3** framework is used.


All Python modules required for the software can be installed from ./requirements in three stages:

1. Create a Python3 environment by installing the conda `environment.yml` file:

```
$ conda env create -f environment.yml
$ activate DeepLearningExercises
```



2. Install the remaining dependencies from `requirements.txt`.

***Note:*** These might take a few minutes.

3. We'll also make use of spaCy to tokenize our data. To install spaCy, follow the instructions [here](https://spacy.io/usage) making sure to install the English models with:

```
$ python -m spacy download en
```
