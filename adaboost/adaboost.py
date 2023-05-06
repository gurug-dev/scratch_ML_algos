import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

def accuracy(y, pred):
    return np.sum(y == pred) / float(len(y))

def parse_spambase_data(filename):
    """ Given a filename return X and Y numpy arrays

    X is of size number of rows x num_features
    Y is an array of size the number of rows
    Y is the last element of each row. (Convert 0 to -1)
    """
    data = np.loadtxt(Path(filename), delimiter=",")
    X = data[:, :-1]
    Y = np.where(data[:, -1] == 0, -1, 1)
    return X, Y

def adaboost(X, y, num_iter, max_depth=1):
    """Given an numpy matrix X, a array y and num_iter return trees and weights 
   
    Input: X, y, num_iter
    Outputs: array of trees from DecisionTreeClassifier
             trees_weights array of floats
    Assumes y is {-1, 1}
    """
    trees = []
    trees_weights = [] 
    N, _ = X.shape
    d = np.ones(N)/N

    for i in range(num_iter):
        # fit a decision tree on weighted data
        tree = DecisionTreeClassifier(max_depth=max_depth)
        tree.fit(X, y, sample_weight=d)
        trees.append(tree)

        # compute error and alpha
        y_pred = tree.predict(X)
        err = np.sum(d * (y != y_pred))
        if err==0: #what if err is 0?? Fix that tree as the final tree.
            trees = [tree]
            trees_weights = [1]
            break
        alpha = np.log((1 - err) / err)

        # update weights
        # d *= np.exp(-alpha * y * y_pred)
        d*= np.exp(alpha * (y != y_pred))
        # d /= np.sum(d)

        trees_weights.append(alpha)

    return trees, trees_weights

def adaboost_predict(X, trees, trees_weights):
    """Given X, trees and weights predict Y
    """
    # X input, y output
    N, _ =  X.shape
    y = np.zeros(N)

    for i, tree in enumerate(trees):
        y += trees_weights[i] * tree.predict(X)

    return np.sign(y)
