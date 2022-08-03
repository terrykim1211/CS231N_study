from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] = dW[:, j] + X[i]    # My code added
                dW[:, y[i]] = dW[:, y[i]] - X[i]   # My code added
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train   # My code added

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dw = dW + 2 * W * reg   # My code added (derivative of W squared is 2W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    
    # Step 1: Compute the entire score labels
    scores = X.dot(W)

    # Step 2: Extract the correct label values from score
    num_train = X.shape[0]
    correct_label_val = scores[np.arange(num_train), y]


    
    # Step 3: Compute (sj -syi + 1) for each image using broadcasting
    correct_label_val = correct_label_val.reshape(num_train, -1)
    margin = scores - correct_label_val + 1

    # Step 4: Replace the labels with maximum comparison with 0, 
    # don't count correct labels for computation
    margin = np.maximum(margin, 0)
    margin[np.arange(num_train), y] = 0


    
    loss = np.sum(margin)/num_train
    loss += reg * np.sum(W * W)
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Step 1: Obtain computation information for loss labels
    # Since dW = dL/dW, the margin tells how many loss label computation 
    # is needed for each image  
    computation = np.maximum(0, margin)
    computation[computation > 0] = 1

    # Step 2: Since Sj computation # = Syi computation #, account for those correct labels

    correct_margin_number = np.sum(computation, axis = 1)
    computation[np.arange(num_train), y] -= correct_margin_number
   
   
 
    # Step 3: Use the dot product property to calculate the dL/dW, 
    # which is corresponding image vector sums
    # Transpose X, to make each image as column vector, allowing matrix 
    # multiplication to be combination of image columns
    dW = np.dot(X.T,computation)

    dW = dW / num_train

    dW = dW + 2 * W * reg   # (derivative of W squared is 2W)
    
    


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
