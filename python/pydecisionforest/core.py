import numpy as np
import math
import random

class Data:
    def __init__(self, X, Y, W=None):
        self.X = np.array(X, dtype=float)
        self.Y = np.array(Y, dtype=int)
        self.n, self.d = self.X.shape
        if W is None:
            self.W = np.ones(self.n) / self.n
        else:
            self.W = np.array(W, dtype=float)
        
        # Calculate stats
        self.mean = np.mean(self.X, axis=0)
        self.std = np.std(self.X, axis=0)
        self.nol = int(np.max(self.Y)) # Assuming Y is 1-based labels 1..M

class TreeNode:
    def __init__(self, feature=-1, threshold=0.0, nol=0):
        self.feature = feature
        self.threshold = threshold
        self.param = np.zeros(nol)
        self.left = None
        self.right = None

class Tree:
    def __init__(self, depth, noc):
        self.depth = depth
        self.noc = noc
        self.root = None
        self.d = 0
        self.nol = 0
        self.importance = None
        
        # Constants
        self.eps = 1e-10
        self.inf = 1e12
        self.min_list = 10
        self.search_range = 3

    def train(self, data):
        self.d = data.d
        self.nol = data.nol
        self.importance = np.zeros(self.d)
        
        if self.min_list < data.n / 1000:
            self.min_list = int(data.n / 1000)
            
        indices = np.arange(data.n)
        self.root = self._train_node(indices, data, level=1)

    def _train_node(self, indices, data, level):
        node = TreeNode(nol=data.nol)
        
        # Case 1: Leaf node conditions
        if (level == self.depth or 
            len(indices) < self.min_list or 
            self._pure_list(indices, data)):
            node.feature = -1
            for idx in indices:
                node.param[data.Y[idx] - 1] += data.W[idx]
            return node

        # Case 2: Split node
        best_feature = -1
        best_threshold = 0.0
        largest_entropy_decrease = -self.inf
        
        candidates = self._get_candidates(data)
        
        for cand_feature, cand_threshold in candidates:
            entropy_decrease = self._get_entropy_decrease(data, indices, cand_feature, cand_threshold)
            if entropy_decrease > largest_entropy_decrease:
                largest_entropy_decrease = entropy_decrease
                best_feature = cand_feature
                best_threshold = cand_threshold
                
        # Update importance
        if best_feature != -1:
            self.importance[best_feature] += largest_entropy_decrease * len(indices)
            
        node.feature = best_feature
        node.threshold = best_threshold
        
        # Split data
        left_indices = []
        right_indices = []
        
        for idx in indices:
            if data.X[idx, best_feature] <= best_threshold:
                left_indices.append(idx)
            else:
                right_indices.append(idx)
                
        if not left_indices or not right_indices:
            # Failed to split (can happen if all features same but labels diff?)
            # Fallback to leaf
            node.feature = -1
            for idx in indices:
                node.param[data.Y[idx] - 1] += data.W[idx]
            return node
            
        node.left = self._train_node(np.array(left_indices), data, level + 1)
        node.right = self._train_node(np.array(right_indices), data, level + 1)
        
        return node

    def _get_candidates(self, data):
        candidates = []
        for _ in range(self.noc):
            feature = random.randint(0, self.d - 1)
            r = random.uniform(-1, 1)
            threshold = data.mean[feature] + data.std[feature] * self.search_range * r
            candidates.append((feature, threshold))
        return candidates

    def _get_entropy_decrease(self, data, indices, feature, threshold):
        # Vectorized implementation
        subset_X = data.X[indices, feature]
        subset_Y = data.Y[indices] - 1
        subset_W = data.W[indices]

        left_mask = subset_X <= threshold
        right_mask = ~left_mask
        
        left_W = subset_W[left_mask]
        right_W = subset_W[right_mask]
        
        left_weight = np.sum(left_W)
        right_weight = np.sum(right_W)
        total_weight = left_weight + right_weight
        
        if total_weight == 0: return 0.0
        
        entropy_decrease = 0.0
        
        # Left entropy
        if left_weight > self.eps:
            left_counts = np.bincount(subset_Y[left_mask], weights=left_W, minlength=self.nol)
            left_probs = left_counts / left_weight
            valid = left_probs > self.eps
            left_entropy = -np.sum(left_probs[valid] * np.log(left_probs[valid]))
            entropy_decrease -= (left_weight / total_weight) * left_entropy

        # Right entropy
        if right_weight > self.eps:
            right_counts = np.bincount(subset_Y[right_mask], weights=right_W, minlength=self.nol)
            right_probs = right_counts / right_weight
            valid = right_probs > self.eps
            right_entropy = -np.sum(right_probs[valid] * np.log(right_probs[valid]))
            entropy_decrease -= (right_weight / total_weight) * right_entropy
            
        return entropy_decrease

    def _pure_list(self, indices, data):
        if len(indices) == 0: return True
        first_label = data.Y[indices[0]]
        for idx in indices[1:]:
            if data.Y[idx] != first_label:
                return False
        return True

    def run(self, X):
        n = X.shape[0]
        Y = np.zeros(n, dtype=int)
        P = np.zeros((n, self.nol))
        
        for i in range(n):
            node = self._decide_recursive(self.root, X[i,:])
            sum_p = np.sum(node.param)
            if sum_p > self.eps:
                P[i,:] = node.param / sum_p
            else:
                P[i,:] = 1.0 / self.nol # Uniform fallback
            Y[i] = np.argmax(P[i,:]) + 1
            
        return Y, P

    def _decide_recursive(self, node, feature_vector):
        if node.feature == -1 or node.left is None:
            return node
        
        if feature_vector[node.feature] <= node.threshold:
            return self._decide_recursive(node.left, feature_vector)
        else:
            return self._decide_recursive(node.right, feature_vector)
    
    def get_importance(self):
        return self.importance

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(path):
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
