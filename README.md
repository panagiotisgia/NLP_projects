# NLP Assignments Repository

This repository contains a series of assignments focusing on Natural Language Processing (NLP).

## Assignment 1

In this assignment, we have built a bigram and trigram language model. The core functionalities implemented in this assignment include:

1. **Building n-gram language models**: Developed bigram and trigram language models using either Laplace smoothing or Kneser-Ney smoothing. All the language models were trained on a subset of a corpus, only considering words that occur at least 10 times in the training data. We've also introduced a special token `*UNK*` to handle Out-Of-Vocabulary (OOV) words.

2. **Calculating Language Cross-Entropy and Perplexity**: Computed cross-entropy and perplexity of the language models on a test subset of the corpus.

3. **Creating a Context-Aware Spelling Corrector**: Built a spelling corrector using the bigram language model, a beam search decoder, and several equations related to NLP and sequence prediction.

Below are some detailed descriptions of each task:

### 1. Building n-gram language models
Implemented bigram and trigram language models using Laplace or Kneser-Ney smoothing. The models were trained on a subset of a corpus, considering only words occurring at least 10 times. A special token `*UNK*` was introduced for OOV words.

### 2. Calculating Language Cross-Entropy and Perplexity
Computed the cross-entropy and perplexity of the two models on a test subset of the corpus, treating the entire test subset as a single sequence of sentences.

### 3. Context-Aware Spelling Corrector
A context-aware spelling corrector was developed, which handles both types of errors. A beam search decoder was utilized for this task, following the formulas from the course material.

This repository includes a detailed report that consists of the following:

- A concise description of the algorithms/methods used, including any data preprocessing steps.
- Cross-entropy and perplexity scores for each model (bigram, trigram).
- Input/output examples demonstrating how the spelling corrector works, including interesting cases that were handled correctly or incorrectly.

The solution leverages NLTK for sentence splitting, tokenization, counting n-grams, and computing Levenshtein distances, but the remaining functionalities were implemented from scratch.


## Assignment 2

In this assignment, we developed a sentiment classifier for text data, such as tweets or product reviews. Here is a high-level overview of the tasks carried out:

1. **Sentiment Classification**: Developed a sentiment classifier for text data using a pre-existing sentiment analysis dataset with at least two classes. We experimented with different feature representations, including Boolean, TF, or TF-IDF features corresponding to words or n-grams.

2. **Feature Selection and Dimensionality Reduction**: Applied feature selection or dimensionality reduction methods as appropriate. We also explored the use of centroids of pre-trained word embeddings.

3. **Model Building**: Implemented Logistic Regression (or Multinomial Logistic Regression for more than two classes) and optionally additional learning algorithms such as Naive Bayes, k-NN.

Below are more details on each of the tasks:

### 1. Sentiment Classification
A sentiment classifier was built for selected text data using a sentiment analysis dataset. The dataset consists of at least two mutually exclusive classes.

### 2. Feature Selection and Dimensionality Reduction
Feature selection or dimensionality reduction methods were applied as needed. This stage also involved using centroids of pre-trained word embeddings.

### 3. Model Building
Used Logistic Regression or Multinomial Logistic Regression for classification. We also experimented with additional learning algorithms such as Naive Bayes, k-NN.

This assignment includes a detailed report that comprises:

- Precision, recall, F1, and precision-recall AUC scores for each class and classifier, separately for the training, development, and test subsets.
- Macro-averaged precision, recall, F1, precision-recall AUC scores for each classifier, separately for the training, development, and test subsets.
- Learning curves showing macro-averaged F1 computed on the training data, the entire development subset, and the entire test subset for each classifier.
- A concise description of the methods and datasets used, including statistics about the datasets and a description of the preprocessing steps performed.
