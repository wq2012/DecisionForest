from .core import Data, Tree

def train_decision_tree(X, Y, depth=5, noc=10, W=None):
    """
    Train a single Decision Tree.
    
    Args:
        X (array-like): n x d training data.
        Y (array-like): n x 1 labels.
        depth (int): Maximum depth of the tree.
        noc (int): Number of candidates at each node.
        W (array-like, optional): n x 1 weights.
        
    Returns:
        Tree: Trained decision tree object.
    """
    data = Data(X, Y, W)
    tree = Tree(depth, noc)
    tree.train(data)
    return tree

def run_decision_tree(X, tree):
    """
    Run a Decision Tree.
    
    Args:
        X (array-like): n x d testing data.
        tree (Tree): Trained decision tree object.
        
    Returns:
        tuple: (Y_pred, P) where Y_pred is predicted labels and P is probabilities.
    """
    return tree.run(X)
