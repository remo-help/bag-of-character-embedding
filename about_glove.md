# GloVe

In this section of the tutorial we will discuss the intuition behind the GloVe embeddings and how they are created.

## Word2Vec

Before we discuss GloVe vectors, we should _briefly_ discuss word2vec, as both combine the principle of distributional semantics (you can tell a word's meaning by the words surrounding it) and unsupervised machine learning. The intuition behind word2vec is the following: Instead of implementend a count-based methods directly, let us train a classifier instead that will predict a word from its k-window context (or the other way around). We are not actually interested in the classifier itself, instead we use its internal layer as our embeddings.

Essentially, how this works in practice that we randomly assign each word a vector and randomly assign vectors to each words' contexts. The size of the context is defined by the k-window we use. We then train a classifier to either predict the word from the context (CBOW) or the context from the word (skip-gram). These are the two architectures of word2vec. This is done via a three layer neural network. The middle hidden layer contains weights which are continuously updated during training. After the model is done training, we throw out the classifier and keep those weights as our word vectors. This is the very abridged version of word2vec.

## GloVe

GloVe was introduced by Pennigton et. al 2014 and it combines the insights of word2vec, that local context is a powerful predictor, with _global co-occurence statistics_. Word2vec exclusively relies on local context and its model iterates over the data multiple times until it converges, which can be quite costly with large corpora. So even though word2vec iterates over the _entire_ corpus, it only ever consideres local information. This means, for example, that if there are words which are quite frequent globally, word2vec may overestimate their importance locally. Here global means the entire corpus, whereas local refers to the k-window.


### References
Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation.
Mikolov, Tomas (2013). "Distributed representations of words and phrases and their compositionality". Advances in Neural Information Processing Systems
