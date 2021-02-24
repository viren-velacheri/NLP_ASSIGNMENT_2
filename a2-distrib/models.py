# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, V, g, W, log_softmax, V_weight, W_weight, ffnn, optimizer, word_embeddings):
        self.V = V
        self.g = g
        self.W = W
        self.log_softmax = log_softmax
        self.V_weight = V_weight
        self.W_weight = W_weight
        self.ffnn = ffnn
        self.optimizer = optimizer
        self.word_embeddings = word_embeddings
    
    def predict(self, ex_words: List[str]) -> int:
        initArray = np.zeros(300)
        for word in ex_words:
            wordEmbed = self.word_embeddings.get_embedding(word)
            initArray = initArray + wordEmbed
        x = torch.from_numpy(initArray).float() / len(ex_words)
        # x = torch.tensor([word_embeddings.get_embedding(w) for w in sentimentExampleWords], dtype=torch.float)
        log_probs = self.ffnn.forward(x)
        prediction = torch.argmax(log_probs)
        return prediction
    
    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]
    


class FFNN(nn.Module):
    """
    Defines the core neural network for doing multiclass classification over a single datapoint at a time. This consists
    of matrix multiplication, tanh nonlinearity, another matrix multiplication, and then
    a log softmax layer to give the ouputs. Log softmax is numerically more stable. If you take a softmax over
    [-100, 100], you will end up with [0, 1], which if you then take the log of (to compute log likelihood) will
    break.

    The forward() function does the important computation. The backward() method is inherited from nn.Module and
    handles backpropagation.
    """
    def __init__(self, inp, hid, out):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param inp: size of input (integer)
        :param hid: size of hidden layer(integer)
        :param out: size of output (integer), which should be the number of classes
        """
        super(FFNN, self).__init__()
        self.V = nn.Linear(inp, hid)
        self.g = nn.Tanh()
        # self.g = nn.ReLU()
        self.W = nn.Linear(hid, out)
        self.log_softmax = nn.LogSoftmax(dim=0)
        # Initialize weights according to a formula due to Xavier Glorot.
        # nn.init.xavier_uniform_(self.V.weight)
        # nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)
        # Initialize with zeros instead
        # nn.init.zeros_(self.V.weight)
        # nn.init.zeros_(self.W.weight)

    def forward(self, x):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """
        return self.log_softmax(self.W(self.g(self.V(x))))


def form_input(x) -> torch.Tensor:
    """
    Form the input to the neural network. In general this may be a complex function that synthesizes multiple pieces
    of data, does some computation, handles batching, etc.

    :param x: a [num_samples x inp] numpy array containing input data
    :return: a [num_samples x inp] Tensor
    """
    return torch.from_numpy(x).float()



def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """

    batchTest = True
    if batchTest:
        feat_vec_size = word_embeddings.get_embedding_length()
        embedding_size = 300
        batch_size = 865
        # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
        # slightly more compact code for the binary case is possible.
        num_classes = 2
        # RUN TRAINING AND TEST
        num_epochs = 5
        ffnn = FFNN(feat_vec_size, 202, num_classes)
        initial_learning_rate = 0.01
        optimizer = optim.Adam(ffnn.parameters(), lr=initial_learning_rate)
        for epoch in range(0, num_epochs):
            ex_indices = [i for i in range(0, len(train_exs))]
            random.shuffle(ex_indices)
            total_loss = 0.0
            idx = 0
            while idx < len(ex_indices):
                limit = idx + batch_size
                vectors = []
                actualVals = []
                while idx < limit and idx < len(ex_indices):
                    sentimentExampleWords = train_exs[ex_indices[idx]].words
                    initArray = np.zeros(feat_vec_size)
                    for word in sentimentExampleWords:
                        wordEmbed = word_embeddings.get_embedding(word)
                        initArray = initArray + wordEmbed
                    x = initArray / len(sentimentExampleWords)
                    y = train_exs[ex_indices[idx]].label
                    idx += 1
                # Build one-hot representation of y. Instead of the label 0 or 1, y_onehot is either [0, 1] or [1, 0]. This
                # way we can take the dot product directly with a probability vector to get class probabilities.
                    y_onehot = [0, 0]
                # scatter will write the value of 1 into the position of y_onehot given by y
                    y_onehot[y] = 1
                    actualVals.append(y_onehot)
                    vectors.append(x)
                vectors = torch.tensor(vectors, dtype=torch.float)
                actualVals = torch.tensor(actualVals, dtype=torch.float)

            # for idx in ex_indices:
            #     sentimentExampleWords = train_exs[idx].words
            #     initArray = np.zeros(feat_vec_size)
            #     for word in sentimentExampleWords:
            #         wordEmbed = word_embeddings.get_embedding(word)
            #         initArray = initArray + wordEmbed
            #     x = form_input(initArray) / len(sentimentExampleWords)
                # x = torch.tensor([word_embeddings.get_embedding(w) for w in sentimentExampleWords], dtype=torch.float)
                # y = train_exs[idx].label
                # # Build one-hot representation of y. Instead of the label 0 or 1, y_onehot is either [0, 1] or [1, 0]. This
                # # way we can take the dot product directly with a probability vector to get class probabilities.
                # y_onehot = torch.zeros(num_classes)
                # # scatter will write the value of 1 into the position of y_onehot given by y
                # y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.int64)), 1)
                # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
                ffnn.zero_grad()
                log_probs = ffnn.forward(vectors)
                # Can also use built-in NLLLoss as a shortcut here but we're being explicit here
                loss = 0.0
                for index in range(0, batch_size):
                    loss += torch.neg(log_probs[index]).dot(actualVals[index])
                    
                total_loss += loss
                    
                loss.backward()
                optimizer.step()
                # loss = torch.sum(torch.neg(log_probs).dot(actualVals))
                # total_loss += loss
                # Computes the gradient and takes the optimizer step
                # loss.backward()
                # optimizer.step()
            print("Total loss on epoch %i: %f" % (epoch, total_loss))
        return NeuralSentimentClassifier(ffnn.V, ffnn.g, ffnn.W, ffnn.log_softmax, ffnn.V.weight, ffnn.W.weight, ffnn, optimizer, word_embeddings)
    else:
        feat_vec_size = word_embeddings.get_embedding_length()
        embedding_size = 300
        # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
        # slightly more compact code for the binary case is possible.
        num_classes = 2
        # RUN TRAINING AND TEST
        num_epochs = 5
        ffnn = FFNN(feat_vec_size, 32, num_classes)
        initial_learning_rate = 0.01
        optimizer = optim.Adam(ffnn.parameters(), lr=initial_learning_rate)
        for epoch in range(0, num_epochs):
            ex_indices = [i for i in range(0, len(train_exs))]
            random.shuffle(ex_indices)
            total_loss = 0.0
            for idx in ex_indices:
                sentimentExampleWords = train_exs[idx].words
                initArray = np.zeros(feat_vec_size)
                for word in sentimentExampleWords:
                    wordEmbed = word_embeddings.get_embedding(word)
                    initArray = initArray + wordEmbed
                x = form_input(initArray) / len(sentimentExampleWords)
                y = train_exs[idx].label
                # Build one-hot representation of y. Instead of the label 0 or 1, y_onehot is either [0, 1] or [1, 0]. This
                # way we can take the dot product directly with a probability vector to get class probabilities.
                y_onehot = torch.zeros(num_classes)
                # scatter will write the value of 1 into the position of y_onehot given by y
                y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.int64)), 1)
                # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
                ffnn.zero_grad()
                log_probs = ffnn.forward(x)
                # Can also use built-in NLLLoss as a shortcut here but we're being explicit here
                
                loss = torch.neg(log_probs).dot(y_onehot)
                    
                total_loss += loss
                    
                loss.backward()
                optimizer.step()
            print("Total loss on epoch %i: %f" % (epoch, total_loss))
        return NeuralSentimentClassifier(ffnn.V, ffnn.g, ffnn.W, ffnn.log_softmax, ffnn.V.weight, ffnn.W.weight, ffnn, optimizer, word_embeddings)

