# Creating Our Own Classifier for Evaluation

Welcome to the evaluation part of this tutorial. In this part we will construct a logistic regression classifier using sklearn. We will then train it to recognize which words are English, French, or German. Then, we will present it with a sequence of _unknown words_ and ask it to make predictions. We will do this once with our own custom GloVe embeddings, and then with the sklearn CountVectorizer embeddings. We will compare the performance of the two embedding styles to see which one comes out on top.

**Necessary packages:** In order to follow along, you will need python3 with jupyter notebook and the following python libraries: sklearn, numpy
If you do not have these libraries, you can install them using:

```bash
pip install scikit-learn
pip install numpy
```

## Downloading the notebook

If you haven't already downloaded this repository, you will need to do so now, if you wish to follow along. I provide a [jupyter notebook](Glove_Char_Classifier.ipynb) with which you can follow along every step of the way. So go ahead and clone the repository and open your jupyter notebook

```bash
git clone https://github.com/remo-help/character-embedding-with-glove
jupyter notebook
```

Once you are in your jupyter notebook browser, open the "Glove_Char_Classifier.ipynb" notebook. You can now follow along. You can also copy all the code I paste here and do this live in your python interpreter.

## Imports
We will start off by importing the necessary packages:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

import numpy as np
```

This will be enough to implement our classifier.

## Reading in the embeddings
Unfortunately, sklearn by itself does not support GloVe vector files, so we need to write a little function that allows us to read in the embeddings we need. This function will take a path to a file with glove-style embeddings and return a dictionary where the keys are the word or character that is embedded and the value is the respective vector. This way every character has a unique GloVe vector that we can easily access:
```python
def glove(path):
        
    embeddings_dict={}
        
    f = open(path,'r',encoding='utf8') #reading in the input data
    vector_file = f.read()
    f.close()
    vector_file=vector_file.split("\n")
        
    for line in vector_file:
        if line[0]:
            line=line.split()
            token = line[0]
            vector=line[1:]
            vector= [float(x) for x in vector]
            vector = np.array(line[1:])
            vector = vector.astype('float64')
            embeddings_dict[token]=vector
            
    return embeddings_dict
```

