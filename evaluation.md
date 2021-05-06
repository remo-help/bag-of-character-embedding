# Using Your Embeddings in a Classifier (Evaluation)

## Cosine Similarity

Now that we have created our custom embeddings, we need to evaluate how good they are. A common technique for evaluating word embeddings is cosine similarity. For cosine similarity we calculate the angle between two vectors and take the cosine of that angle. The smaller the angle, the more similar the two vectors are, hence the more similar they are in meaning or distribution. 
Cosine similarity is defined as the dot-product of two vectors divided by the product of their respective magnitudes:

![img](https://latex.codecogs.com/svg.latex?%5Ccos%28%5Ctheta%29%3D%7B%5Cmathbf%7BA%7D%5Ccdot%5Cmathbf%7BB%7D%5Cover%5C%7C%5Cmathbf%7BA%7D%5C%7C%5C%7C%5Cmathbf%7BB%7D%5C%7C%7D)

If you want to calculate the cosine similarities of your own vectors in python, you can do it with this simple formula:
```python
import numpy as np
vec1 = [2, 3, 4, 5] #our vectors
vec2 = [4, 8, 2, 2]
cosine_sim = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
print(cosine_sim)
## 
0.7253235664820743
```
## Why cosine similarity is no good for character level embeddings

With word embeddings, we can test our vectors by checking how similar two word vectors are through cosine similarities. This can help us to evaluate our vectors. For example, our vector for "giraffe" should be closer to the vector for "lion" than it should be to the vector for "Antarctica". If semantically distant words have high cosine similarity in our vectors, then we have a problem.

Since we assume that word-embeddings carry some information about the distribution and semantics of words, we can leverage the semantics to test how "accurate" our vector representations are. Semantically close words should be close in the vector space. 

Characters, however, are semantically void. The character "e" in the English language has no semantic content. Only when combined into a word, do characters have meaning. So the classic cosine similarity technique will not help us to evaluate our character-level vectors. Cosine similarity here will only show us which characters pattern together. 

Now, we might consider leveraging our knowledge of English phonology to evaluate these vectors. The English language has rules about what sounds can pattern together, perhaps we can use that to evaluate our vectors? This might be a good approach for a language where the orthography (i.e. the letters) is largely phonetic (i.e. translates directly to sound). English is unfortunately notorious for having an orthography that is particularly intransparent about the actual sounds of the word. For example, can you tell me the difference in _sound_ between "knight" and "night"?

So, as you can see, cosine similarity will only be marginally helpful at best in evaluating our character level embeddings, we need to find a different way.

## Evaluation through a task

Instead, we will evaluate our embeddings with a task. Simply put, we will use our embeddings against another style of embedding in an artificially constructed task and see which performs better.

For this task we will construct a logistic regression classifier in [scikit-learn](https://sklearn.org/) using our own glove embeddings. We will train this classifier to read in a word from English, French, or German and make a decision as to which language the word belongs to. Then we will test our classifier with a series of unknown words, asking it to make a prediction on them. Note how this task would be impossible for most word-level embeddings, as they usually cannot deal with unknown words (exceptions to this include [fasttext](https://fasttext.cc/)).

We will train this classifier once with our pretrained embeddings from the tutorial, and then test it. After that, we will train the same classifier again, using embeddings from the [CountVectorizer](https://sklearn.org/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer) library of sci-kit learn. We will then compare the accuracy and f1 score of each implementation to evaluate our vectors. (spoiler: CountVectorizer wins!)

## [Practical Implementation](glove_classifier.md)

For the practical implementation of this, complete with Jupyter notebook to follow along, please go to the [next page of the tutorial](glove_classifier.md)


[return to main page](index.md)
