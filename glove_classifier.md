# Creating Our Own Classifier for Evaluation

Welcome to the evaluation part of this tutorial. In this part we will construct a logistic regression classifier using sklearn. We will then train it to recognize which words are English, French, or German. Then, we will present it with a sequence of _unknown words_ and ask it to make predictions. We will do this once with our own custom GloVe embeddings, and then with the sklearn CountVectorizer embeddings. We will compare the performance of the two embedding styles to see which one comes out on top.

**Necessary packages:** In order to follow along, you will need python3 with jupyter notebook and the following python libraries: sklearn, numpy
If you do not have these libraries, you can install them using:

```bash
pip install scikit-learn
pip install numpy
```

## Downloading the notebook

If you haven't already downloaded this repository, you will need to do so now, if you wish to follow along. I provide a [jupyter notebook](Glove_Char_Classifier.ipynb) with which you can follow along every step of the way. So go ahead and clone the repository and open your jupyter notebook. It is imperative that you clone or download the entire repository, as it contains the data the notebook is working with.

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
## Defining the Classifier
We will define our classifier as a class. We will give this class a few attributes and functions:

### train()
This function will train our classifier on the GloVe vectors we provide. It will take the vector-dictionary created by the glove() function as 1st argument and the path to the training data as a second argument.
### train_count()
This function will train our classifier on the embeddings provided by the CountVectorizer of the sklearn library. This is a count-based embedding technique. It takes the path to the training data as an argument.
### predict_labels()
This function will take in test data and make predictions based on the test data. The output of the function will be encoded labels (integers). The output will consist of a tuple of lists. One list will contain all the predicted labels. The other list will contain the gold labels. We can inverse_transform those labels with the LabelEncoder if we want to see the strings.
### predict_labels_count()
This is the same as above, except for the CountVectorizer embeddings.

```python
class Classifier:
    def __init__(self):
        """
        Initializes the classifier.
        """
        self.label_encoder = LabelEncoder()
        
           
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 10)) #we are using the "word" parameter because our characters are already seperated

        
        self.model = LogisticRegression(solver="lbfgs", multi_class="multinomial", max_iter=5000,verbose=1)
    
    def train(self, vectors, train_data_path):
        """
        trains on the GloVe embeddings
        """

        f = open(train_data_path,'r',encoding="utf8") #reading in the input data
        input_string = f.read()
        f.close()
        
        input_list=input_string.split("\n") #storing each datum as a string in a list
        input_list=input_list[0:-2] #getting rid of the last empty newline
        feature_strings= []
        label_strings= []
        
        for datum in input_list:
            temp = datum.split("\t") #separating the word from its label
            feature_strings.append(temp[0])
            label_strings.append(temp[1])
        del input_list #deleting the initial list to be economic
        
        vector_list=[] #here we will store our feature vectors
        
        for feature in feature_strings: #selecting a feature
            feature = feature[0:-1].split(" ") #getting rid of trailing space and splitting on space
            temp_list = [] #here we will compile the feature vector
            for char in feature: #iterating over characters of the word
                if char in vectors.keys(): #making sure we are not running into unknown chars
                    vector = vectors[char] #getting the vector associated with the character
                    temp_list.append(vector)
            
            if len(temp_list)==0:
                print("cannot find:",feature)
                
            base = temp_list[0] #selecting the first character-vector

            for i in range(1,len(temp_list)): 
                base=np.add(base,temp_list[i]) #adding all other character vectors item-wise
            array=base/len(temp_list) #dividing by the total amount of characters
        
            vector_list.append(array) #putting the new and averaged array into our list
                
                
                
        
        x_train = vector_list #transforming the feature strings into glove vectors based on our pretrained embedding

        y_train = self.label_encoder.fit_transform(label_strings) #fitting and transforming the labels to integers
        
        if len(x_train)!=len(y_train): #making sure we have as many feature vectors as we have labels
            print("features:",len(x_train),"\n","labels:",len(y_train))
        
        self.model.fit(x_train, y_train)
        
        print("The classifier has finished training")
        
    def train_count(self, train_data_path):
        """
        This trains the classifier on vectors created by the CountVectorizer, as opposed to pretrained embeddings
        """

        f = open(train_data_path,'r',encoding="utf8") #reading in the input data
        input_string = f.read()
        f.close()
        
        input_list=input_string.split("\n") #storing each datum as a string in a list
        input_list=input_list[0:-2] #getting rid of the last empty newline
        feature_strings= []
        label_strings= []
        
        for datum in input_list:
            temp = datum.split("\t") #separating the word from its label
            feature_strings.append(temp[0])
            label_strings.append(temp[1])
        del input_list #deleting the initial list
        

        
        x_train = self.vectorizer.fit_transform(feature_strings) #transforming the feature strings into vectors with the count vectorizer

        y_train = self.label_encoder.fit_transform(label_strings) #fitting and transforming the labels to integers
        
        self.model.fit(x_train, y_train) #training the model
        
        print("The classifier has finished training (CountVectorizer)")
        
        
    
    def predict_labels(self, vectors, test_data_path):
        """
        Takes a testfile where tokens and labels are seperated with \t
        returns the sequence of gold-lables and the sequence of predicted labels in INTEGER FORM
        """
        f = open(test_data_path,'r',encoding="utf8") #reading in the input data
        input_string = f.read()
        f.close()
        
        input_list=input_string.split("\n") #storing each datum as a string in a list
        input_list=input_list[0:-2] #getting rid of the last empty newline
        feature_strings= []
        label_strings= []
        
        for datum in input_list:
            temp = datum.split("\t") #separating the word from its label
            feature_strings.append(temp[0])
            label_strings.append(temp[1])
        del input_list #deleting the initial list to be economic
        
        vector_list=[] #here we will store our feature vectors
        
        for feature in feature_strings: #selecting a feature
            feature = feature[0:-1].split(" ") #getting rid of trailing space and splitting on space
            temp_list = [] #here we will compile the feature vector
            for char in feature: #iterating over characters of the word
                if char in vectors.keys(): #making sure we are not running into unknown chars
                    vector = vectors[char] #getting the vector associated with the character
                    temp_list.append(vector)
            
            if len(temp_list)==0:
                print("cannot find:",feature)
                
            base = temp_list[0] #selecting the first character-vector

            for i in range(1,len(temp_list)): 
                base=np.add(base,temp_list[i]) #adding all other character vectors item-wise
            array=base/len(temp_list) #dividing by the total amount of characters
        
            vector_list.append(array) #putting the new and averaged array into our list
        
        x_test=vector_list

        predictions = self.model.predict(x_test) #makes the predictions
        
        gold_labels = self.label_encoder.transform(label_strings)
        
        return predictions,gold_labels
    
    def predict_labels_count(self, test_data_path):
        """
        Takes a testfile where tokens and labels are seperated with \t
        returns the sequence of gold-lables and the sequence of predicted labels in INTEGER FORM
        This is the CountVectorizer based implementation
        """
        f = open(test_data_path,'r') #reading in the test data
        input_string = f.read()
        f.close()
        
        input_list=input_string.split("\n") #storing each datum as a string in a list
        input_list=input_list[0:-2] #getting rid of the last empty newline
        feature_strings= []
        label_strings= []
        
        for datum in input_list:
            temp = datum.split("\t") #separating the word from its label
            feature_strings.append(temp[0])
            label_strings.append(temp[1])
        del input_list #deleting the initial list to be economic
        
        x_test = self.vectorizer.transform(feature_strings) #transforming the feature strings into vectors with the count vectorizer


        predictions = self.model.predict(x_test) #makes the predictions
        
        gold_labels = self.label_encoder.transform(label_strings)
        
        return predictions,gold_labels
    
```

## Training and testing the Classifier
Now that we have constructed our classifier, it is time to train and test it once with each embedding type. First we will train and test it with our own custom embeddings. We will calculate an accuracy score and a f1 score and will save that for later:

```python
classifier = Classifier()
vec= glove('data/dickens_vectors.txt')
classifier.train(vec,'data/train_tokens.txt')

predictions = classifier.predict_labels(vec,"data/test_tokens.txt")
glove_f1= f1_score(predictions[1], predictions[0], average='weighted')
glove_accuracy=accuracy_score(predictions[1],predictions[0])
```
After that it is time to train and test the CountVectorizer embeddings. For this we will simply refit the Classifier using the CountVectorizer embeddings:
```python
classifier = Classifier()
classifier.train_count('data/count_train_tokens.txt')

predictions = classifier.predict_labels_count("data/count_test_tokens.txt")
count_accuracy=accuracy_score(predictions[1],predictions[0])
count_f1=f1_score(predictions[1], predictions[0], average='weighted')
```
If you are particularly observant, you probably noticed that I am using a different data file for the CountVectorizer. This is because this vectorizer is already character based, so there is no need to seperate the characters with spaces. The files contain the exact same as their GloVe equivalents. If you are unconvinced, go open the repository and take a look.
## Compare
Now we can finally compare the two:
```python
print("Here are the results of the GloVe embeddings:\n","F1_score: ",glove_f1,"\n", "accuracy: ", glove_accuracy,"\n")
print("Here are the results of the CountVectorizer embeddings:\n","F1_score: ",count_f1,"\n", "accuracy: ", count_accuracy)
```
