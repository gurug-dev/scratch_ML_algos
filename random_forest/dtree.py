import numpy as np
from scipy import stats
from scipy.stats import mode
from sklearn.metrics import r2_score, accuracy_score

class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild
        
    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        if x_test[self.col] <= self.split:
            return self.lchild.predict(x_test)
        return self.rchild.predict(x_test)

    def leaf(self, x_test):
        if isinstance(self, LeafNode): return self
        elif x_test[self.col] <= self.split:
            return self.lchild.leaf(x_test)
        return self.rchild.leaf(x_test)
        
        
class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction

    def predict(self, x_test):
        # return prediction
        return self.prediction

    def leaf(self, x_test):
        return self
        
def gini(y):
    """
    Return the gini impurity score for values in y
    Assume y = {0,1}
    Gini = 1 - sum_i p_i^2 where p_i is the proportion of class i in y
    """
    prob_sum = 0
    for val in [0,1]:
        prob_y = np.sum(y==val) / len(y)
        prob_sum += prob_y**2
    return 1 - prob_sum

def rf_find_best_split(X, y, max_features, loss, min_samples_leaf):
    best = {
            'col':-1,
            'split':-1,
            'loss':loss(y)
           }
    cols_subset = np.random.choice(range(0,X.shape[1]), max(round(max_features * X.shape[1]),1), replace=False)
    for var_col in cols_subset:
        
        if len(X[:,0]) > 11:
            split_values = np.random.choice(X[:, var_col], 11, replace=False) ## take 11 if observations > 11
        else:   
            split_values = list(X[:, var_col]) ## else take all
            
        for split_value in split_values:
            y_left = y[X[:,var_col] <= split_value]
            y_right = y[X[:,var_col] > split_value]
            
            if len(y_left) < min_samples_leaf or len(y_right) < min_samples_leaf: 
                continue
                
            weighted_loss = (len(y_left) * loss(y_left) + len(y_right) * loss(y_right)) / len(y)
            
            if weighted_loss==0: 
                return var_col, split_value
            
            if weighted_loss < best['loss']: 
                best = {
                    'col':var_col, 
                    'split':split_value, 
                    'loss':weighted_loss
                    } 
    
    return best['col'], best['split']
    
def find_best_split(X, y, loss, min_samples_leaf):
    best = {
            'col':-1,
            'split':-1,
            'loss':loss(y)
           }
    
    for var_col in range(0,X.shape[1]):
        
        if len(X[:,0]) > 11:
            split_values = np.random.choice(X[:, var_col], 11, replace=False) ## take 11 if observations > 11
        else:   
            split_values = list(X[:, var_col]) ## else take all
            
        for split_value in split_values:
            y_left = y[X[:,var_col] <= split_value]
            y_right = y[X[:,var_col] > split_value]
            
            if len(y_left) < min_samples_leaf or len(y_right) < min_samples_leaf: 
                continue
                
            weighted_loss = (len(y_left) * loss(y_left) + len(y_right) * loss(y_right)) / len(y)
            
            if weighted_loss==0: 
                return var_col, split_value
            
            if weighted_loss < best['loss']: 
                best = {
                    'col':var_col, 
                    'split':split_value, 
                    'loss':weighted_loss
                    } 
    
    return best['col'], best['split']
    
    
class DecisionTree621:
    def __init__(self, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss # loss function; either np.var for regression or gini for classification
        
    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for  either a classifier or regression.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressions predict the average y
        for observations in that leaf.

        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)


    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classification or regression.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621.create_leaf() depending
        on the type of self.

        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """
        if len(X) <= self.min_samples_leaf:  
            return self.create_leaf(y)
            # return DecisionNode(-1, -1, None, None, self.create_leaf(y))
        
        col, split = rf_find_best_split(X, y, self.max_features, self.loss, self.min_samples_leaf)
        if col == -1:   
            return self.create_leaf(y)
            # return DecisionNode(-1, -1, None, None, self.create_leaf(y))
        
        left_mask = X[:,col] <= split
        right_mask = X[:,col] > split
        
        left_child = self.fit_(X[left_mask], y[left_mask])
        right_child = self.fit_(X[right_mask], y[right_mask])
        
        return DecisionNode(col, split, left_child, right_child)

    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        return np.apply_along_axis(self.root.predict, axis=1, arr=X_test)


class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1, max_features=0.3):
        super().__init__(min_samples_leaf, loss=np.var)
        self.max_features = max_features
    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        return r2_score(y_test, self.predict(X_test))

    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))
        # return np.mean(y)


class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1,max_features=0.3):
        super().__init__(min_samples_leaf, loss=gini)
        self.max_features = max_features
    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        return accuracy_score(y_test, self.predict(X_test))

    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor. Feel free to use scipy.stats to use the mode function.
        """
        return LeafNode(y, int(mode(y)[0]))
        # return int(mode(y)[0])
