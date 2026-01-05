import pytest
import numpy as np
from pydecisionforest import train_decision_tree, run_decision_tree, train_decision_forest, run_decision_forest, train_adaboost, run_adaboost

def create_synthetic_data(n=1000, d=10, k=3):
    X = np.random.randn(n, d)
    # Axis-aligned boundary:
    # If x[0] > 0 and x[1] > 0 -> Class 1
    # Else if x[0] < 0 and x[1] < 0 -> Class 2
    # Else -> Class 3
    # This is slightly complex but axis aligned.
    
    Y = np.zeros(n, dtype=int)
    mask1 = (X[:, 0] > 0) & (X[:, 1] > 0)
    mask2 = (X[:, 0] <= 0) & (X[:, 1] <= 0)
    
    Y[mask1] = 1
    Y[mask2] = 2
    Y[(~mask1) & (~mask2)] = 3
    
    # Add a little noise to features to make it not perfect? 
    # Actually let's keep it clean to ensure high accuracy for unit test.
    return X, Y

def test_decision_tree():
    X_train, Y_train = create_synthetic_data(n=1000)
    X_test, Y_test = create_synthetic_data(n=500)
    
    tree = train_decision_tree(X_train, Y_train, depth=5, noc=1000)
    Y_pred, _ = run_decision_tree(X_test, tree)
    
    accuracy = np.mean(Y_pred == Y_test)
    print(f"Decision Tree Accuracy: {accuracy}")
    assert accuracy > 0.8

def test_decision_forest():
    X_train, Y_train = create_synthetic_data(n=1000)
    X_test, Y_test = create_synthetic_data(n=500)
    
    forest = train_decision_forest(X_train, Y_train, forest_size=10, depth=5, noc=1000)
    Y_pred, _ = run_decision_forest(X_test, forest)
    
    accuracy = np.mean(Y_pred == Y_test)
    print(f"Decision Forest Accuracy: {accuracy}")
    assert accuracy > 0.85

def test_adaboost():
    X_train, Y_train = create_synthetic_data(n=1000)
    X_test, Y_test = create_synthetic_data(n=500)
    
    weights, importance, model = train_adaboost(X_train, Y_train, forest_size=10, depth=5, noc=1000)
    Y_pred, _ = run_adaboost(X_test, model)
    
    accuracy = np.mean(Y_pred == Y_test)
    print(f"AdaBoost Accuracy: {accuracy}")
    assert accuracy > 0.85
    assert len(weights) == 10
    assert len(importance) == 10
