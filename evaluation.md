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

With word embeddings, we can test our vectors by checking how similar two word vectors are through cosine similarities. This can help 

[return to main page](index.md)
