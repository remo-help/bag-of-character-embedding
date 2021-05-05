# Embeddings
In this part of the tutorial, I will briefly discuss the notion of word embeddings and character level embeddings. This section is aimed at newcomers. If you are already familiar with this subject and are just here for the practical parts of the tutorial, feel free to switch to the section discussing [GloVe specifically](about_glove.md).

## Why do we need embeddings?

To a human who speaks English, the meaning of "dog" is fairly obvious. Not only do we understand that a dog is a mid-sized furry mammal, we also have some idea in our head of what a dog looks like, what it behaves like, what it sounds like, etc. Not only that, we also understand that "dog" is a noun and that means we understand facts about the _syntactic distribution_ of "dog", meaning we know that it takes a determiner, that it can be modified by adjectives, that "dogs" is the plural form, and that it can be the argument of a verb. Some of these facts we may know only subconsciously, but that does not matter, what is important is that we do know these facts.

To a computer "dog" is just a string of characters. All the information that you have in your head about "dog", is stored in some abstract sense in your brain, but how can we teach it to a computer? You might think you could store it like a dictionary (in the real world sense), where we just tell the program: "dog":"mid-sized furry mammal". But, this type of representation is entirely useless to a computer or a neural net. These are just series of strings that do not carry any information apart from the characters itself. So how can we teach a computer an approximation of all the knowledge you have about the word "dog"? The answer is numbers. With numbers we can encode statistics and other information about words in a way that can be computed. This is essentially what embeddings are.

## What are embeddings?

Word embeddings are numerical representations of words, usually represented in a multi-dimensional vector space. You are probably familiar with the notion of a two-dimensional vector, that is a vector that can be represented on a 2-D plane. Embedding vectors are usually multi-dimensional, meaning they have more than two dimensions. For example, the [GloVe project](https://nlp.stanford.edu/projects/glove/) offers pretrained vectors in dimensions from 25D-300D. A 300D vector will have 300 numbers inside it, each represention one dimension.

Word embeddings are primarily used in machine learning to compute an approximation of the meaning or some other information about the word. The idea is that we store that information in the vector as a numerical representation. How can we do that? Well we have to either do this manually (which nobody in their right mind would do with a 50D vector) or we can let a program or neural net do it for us. 

## Why and how do we train embeddings?



[Return to main page](index.md)
