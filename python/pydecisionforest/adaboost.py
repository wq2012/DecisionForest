import numpy as np
from .core import Data
from .decision_tree import train_decision_tree, run_decision_tree

class AdaBoost:
    def __init__(self, forest_size=10, depth=5, noc=10):
        self.forest_size = forest_size
        self.depth = depth
        self.noc = noc
        self.trees = []
        self.weights = [] # Alpha values
        self.feature_importance = None
    
    def fit(self, X, Y):
        n = X.shape[0]
        d = X.shape[1]
        W = np.ones(n) / n
        K = int(np.max(Y)) # Number of classes
        
        self.trees = []
        self.weights = []
        self.feature_importance = np.zeros(d)
        
        for i in range(self.forest_size):
            tree = train_decision_tree(X, Y, self.depth, self.noc, W)
            self.trees.append(tree)
            
            # Aggregate importance
            self.feature_importance += tree.get_importance()
            
            # Predict
            Y_pred, _ = run_decision_tree(X, tree)
            
            # Weighted error
            incorrect = (Y_pred != Y)
            err = np.sum(W[incorrect])
            
            # Avoid division by zero
            if err == 0:
                err = 1e-10
            elif err >= (1 - 1/K):
                err = 1 - 1/K - 1e-10
                
            # Alpha
            alpha = np.log((1 - err) / err) + np.log(K - 1)
            self.weights.append(alpha)
            
            # Update weights
            # W <- W * exp(alpha * I(y != h(x)))
            W = W * np.exp(alpha * incorrect)
            W = W / np.sum(W)
            
        self.feature_importance /= np.sum(self.feature_importance)

    def predict(self, X):
        Y, _ = self.run(X)
        return Y
        
    def run(self, X):
        if not self.trees:
            raise ValueError("AdaBoost not trained yet")
            
        n = X.shape[0]
        # Get nol
        _, P0 = run_decision_tree(X, self.trees[0])
        nol = P0.shape[1]
        
        P_total = np.zeros((n, nol))
        
        for i in range(self.forest_size):
            Y_weak, _ = run_decision_tree(X, self.trees[i])
            alpha = self.weights[i]
            
            # Add alpha to predicted class
            # P_total[row, label-1] += alpha
            # Vectorized add:
            rows = np.arange(n)
            cols = Y_weak - 1
            P_total[rows, cols] += alpha
            
        # Normalize
        sum_p = np.sum(P_total, axis=1, keepdims=True)
        # Avoid zero division if sum_p is 0 (shouldn't happen with valid alpha)
        sum_p[sum_p == 0] = 1.0
        P = P_total / sum_p
        
        Y = np.argmax(P, axis=1) + 1
        return Y, P

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(path):
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

def train_adaboost(X, Y, forest_size=10, depth=5, noc=10):
    model = AdaBoost(forest_size, depth, noc)
    model.fit(X, Y)
    return model.weights, model.feature_importance, model

def run_adaboost(X, model):
    return model.run(X)
