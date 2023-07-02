# NLP Assignments Repository

Welcome to this Natural Language Processing (NLP) repository! This collection of assignments explores various aspects of NLP, from language modeling and sentiment analysis to advanced techniques like multi-layer perceptrons (MLP) and recurrent neural networks (RNN).

## Table of Contents

1. [Assignment 1: Language Models and Spelling Correction](#assignment-1)
2. [Assignment 2: Sentiment Analysis](#assignment-2)
3. [Assignment 3: Sentiment Analysis with MLP](#assignment-3)
4. [Assignment 4: Text Classification with RNN and Self-Attention MLP](#assignment-4)

<a name="assignment-1"></a>
## Assignment 1: Language Models and Spelling Correction

This assignment focuses on the creation of bigram and trigram language models, computing language cross-entropy and perplexity, and creating a context-aware spelling corrector. The key aspects are:

- Building n-gram language models: Bigram and trigram language models are developed with Laplace or Kneser-Ney smoothing.
- Calculating language cross-entropy and perplexity: The models are evaluated on a test subset of the corpus.
- Creating a context-aware spelling corrector: The bigram model is leveraged to build a spelling corrector using a beam search decoder. 

[Detailed Description & Report](#assignment-1)

<a name="assignment-2"></a>
## Assignment 2: Sentiment Analysis

In this assignment, a sentiment classifier is developed for text data. The core tasks include:

- Sentiment Classification: Developing a sentiment classifier for text data using a pre-existing sentiment analysis dataset with at least two classes.
- Feature Selection and Dimensionality Reduction: Application of feature selection or dimensionality reduction methods, and exploration of pre-trained word embeddings' centroids.
- Model Building: Implementing Logistic Regression (or Multinomial Logistic Regression for more than two classes) and optionally additional learning algorithms such as Naive Bayes, k-NN.

[Detailed Description & Report](#assignment-2)

<a name="assignment-3"></a>
## Assignment 3: Sentiment Analysis with MLP

This assignment extends the sentiment analysis by using a Multi-Layer Perceptron (MLP) for the classification task. The main tasks in this assignment are:

- MLP Classifier: Building an MLP classifier with different feature representations.
- Hyperparameter Tuning: Tuning hyperparameters on the development subset.
- Training and Evaluation: Monitoring the performance of the MLP during training and evaluation of the model on training, development, and test subsets.

[Detailed Description & Report](#assignment-3)

<a name="assignment-4"></a>
## Assignment 4: Text Classification with RNN and Self-Attention MLP

In this assignment, text classification is performed using advanced neural network architectures. The key tasks include:

- RNN and Self-Attention MLP: Implementing a bi-directional stacked RNN and a self-attention MLP.
- Hyperparameter Tuning: Tuning hyperparameters such as the number of stacked RNNs, number of hidden layers in the self-attention MLP, and dropout probability on the development subset.
- Training and Evaluation: Monitoring the performance of the RNN during training and evaluating the model on the training, development, and test subsets.

[Detailed Description & Report](#assignment-4)