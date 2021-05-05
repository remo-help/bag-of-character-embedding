# Custom Embeddings

Welcome to the first applied part of this tutorial. Here I will show you how you can use GloVe to create your own custom embeddings. First we have to start by finding and prepping our data.

## Prepping your Data

For demonstration purposes, I have created a dataset for you from scratch. You can find the prepped dataset for vectorization in the [repository](https://github.com/remo-help/character-embedding-with-glove) for this page in the [vector_tokens](/data/vector_tokens.txt) file. But if you would like to know how to prep your data yourself, here is a quick rundown on what I did:

### Find a source for your data
First, you will need to find the data you want to work on. It is important to use the right dataset for your usecase. For example, if you want to create some application that deals with Gen Z slang, then the Wall Street Journal Corpus is not going to be a good source for your data.

I simply downloaded a series of books from [Project Gutenberg](https://www.gutenberg.org/), which is a library for books in the public domain. Since all the books used are in the public domain, you can find my dataset in the repository. With our later [evaluation method](glove_classifier.md) in mind, I downloaded a series of English, French, and German language books. I used some of them for training data and set some aside for later testing. Here are the books used:
- **English**: A Tale of Two Cities - Dickens, The Adventures of Sherlock Holmes - Arthur Conan Doyle
- **French**: Les misérables Tome V: Jean Valjean - Victor Hugo, Eureka - Edgar Allan Poe (French), Les Trois Mousquetaire - Alexandre Dumas
- **German**: Sämmtliche Werke 3: Abende auf dem Gutshof bei Dikanka - Gogol (Translated by Frieda Ichak), Celsissimus - Arthur Achleitner, Wunderbare Reise des kleinen Nils Holgersson mit den Wildgänsen - Selma Lagerlöf (translated by Pauline Klaiber)
You can find these books in .txt format in the [test](/test/) and [training](/training/) folders of this repository.
### Preprocess your data
