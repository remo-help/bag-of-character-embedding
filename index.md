# Character Level Embeddings with GloVe

This tutorial will show you how you how to use [GloVe](https://nlp.stanford.edu/projects/glove/) (Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014) to create your own custom character-level embeddings for machine learning purposes. We will begin by discussing embeddings in general and GloVe in particular. I will then briefly discuss in what kind of a scenario we would want to do a character-level embedding, as opposed to a word-level embedding. Finally, I will walk you through an example application for the embeddings we just created.

For your convenience, this tutorial is seperated into parts. If you are just interested in a certain part, feel free to directly go to one of the links below.
## Table of Contents
1. [Embeddings](Embeddings.md)
2. [GloVe](about_glove.md)
3. [Creating Custom Embeddings with GloVe](custom_embeddings.md)
4. [Using Your Embeddings in a Classifier (Evaluation)](evaluation.md)
5. Synopsis



## What You Will Need

### Embeddings
In order to create your GloVe embeddings you will need a c-compiler (such as gcc). If you do not have a c-compiler on your os, you may want to go ahead and install one of your choosing. If you are on Ubuntu or Debian, you will already have a c-compiler installed on your system.

### Evaluation
In order to run the evaluation/classifier, you will need python and the following packages for python:
- numpy
- sklearn
