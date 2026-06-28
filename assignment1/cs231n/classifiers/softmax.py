from builtins import range
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

    # compute the loss and the gradient
    num_classes = W.shape[1]  
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W) # (D,) * (D, C) = (C, )
        scores -= np.max(scores)
        
        # forward pass
        exp_scores = np.exp(scores)
        softmax_num = exp_scores[y[i]]
        softmax_den = np.sum(exp_scores)
        softmax = softmax_num / softmax_den
        loss += -np.log(softmax)
        
        # backprop
        dexp_scores = np.zeros_like(exp_scores) # construct the same shape array
        dexp_scores[y[i]] += -1 / softmax_num
        dexp_scores += 1 / softmax_den
        dscores = dexp_scores * exp_scores # upstream gradient * local gradient
        
        dW += np.outer(X[i], dscores) # (D,) *outer* (C,) = (D, C)
        
    # normalized hinge loss plus regularization
    loss /= num_train
    dW /= num_train
    
    loss += reg * np.sum(W ** 2)
    dW += 2 * reg * W
    
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
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

    N = X.shape[0]
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    
    # forward pass
    scores = X.dot(W) # (N, C)
    scores -= np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
     
    loss1 = -scores[np.arange(N), y] + np.log(np.sum(exp_scores, axis=1))
    loss = np.sum(loss1) / N + reg * np.sum(W ** 2)
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    dloss1 = np.ones_like(loss1) / N  # (N, )
    dscores_local = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    dscores_local[np.arange(N), y] -= 1
    dscores = dloss1.reshape(N, 1) * dscores_local
    dW = X.T.dot(dscores) + 2 * reg * W
    

    return loss, dW
