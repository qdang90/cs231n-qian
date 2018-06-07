import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  n_obs = np.shape(X)[0]
  n_classes = max(y)+1
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  L = np.zeros(n_obs)
  for i in range(n_obs):
        Probability = np.zeros(n_classes)
        for c in range(n_classes):
            Probability[c] = np.dot(X[i], W[:,c])
        Probability = np.exp(Probability)
        total = np.sum(Probability)
        Probability /= total
        L[i] = -np.log(Probability[y[i]])
        for c in range(n_classes):
            if c==y[i]:
                dW[:,c] += (Probability[c]-1)*X[i]
            else:
                dW[:,c] += Probability[c]*X[i]

  loss = 1/n_obs* np.sum(L) + reg*np.sum(W*W)
  dW /= n_obs
  dW += reg*2*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  n_obs = np.shape(X)[0]
  n_classes = max(y)+1  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # loss
  e = np.exp(np.dot(X,W))
  total = np.sum(e, axis=1)
  Prob = e/total[:,np.newaxis]
  L = -np.log(Prob[np.arange(n_obs),y])
  loss = 1/n_obs*np.sum(L)+reg*np.sum(W*W)
  # gradient
  Prob[np.arange(n_obs),y] -= 1
  multiple = Prob[:,np.newaxis,:]*X[:,:,np.newaxis]
  dW = np.sum(multiple, axis = 0)/n_obs
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

