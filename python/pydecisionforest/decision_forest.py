import numpy as np
from .decision_tree import train_decision_tree, run_decision_tree

class DecisionForest:
    def __init__(self, forest_size=10, depth=5, noc=10):
        self.forest_size = forest_size
        self.depth = depth
        self.noc = noc
        self.trees = []
    
    def fit(self, X, Y):
        """
        Train the Decision Forest.
        """
        self.trees = []
        for i in range(self.forest_size):
            tree = train_decision_tree(X, Y, self.depth, self.noc)
            self.trees.append(tree)
            
    def predict(self, X):
        """
        Predict labels for X.
        """
        Y, _ = self.run(X)
        return Y
        
    def predict_proba(self, X):
        """
        Predict probabilities for X.
        """
        _, P = self.run(X)
        return P
        
    def run(self, X):
        """
        Run the forest on X.
        Returns (Y, P).
        """
        if not self.trees:
            raise ValueError("Forest not trained yet")
            
        n = X.shape[0]
        # Get nol from first tree's prediction to initialize P
        _, P0 = run_decision_tree(X, self.trees[0])
        nol = P0.shape[1]
        
        P_total = P0
        for i in range(1, self.forest_size):
            _, P_tree = run_decision_tree(X, self.trees[i])
            P_total += P_tree
            
        P_avg = P_total / self.forest_size
        Y = np.argmax(P_avg, axis=1) + 1
        
        return Y, P_avg

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(path):
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

def train_decision_forest(X, Y, forest_size=10, depth=5, noc=10):
    """
    Functional API for training a forest.
    """
    forest = DecisionForest(forest_size, depth, noc)
    forest.fit(X, Y)
    return forest

def run_decision_forest(X, forest):
    """
    Functional API for running a forest.
    """
    return forest.run(X)
