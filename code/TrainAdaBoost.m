function [weights, importance] = TrainAdaBoost(X, Y, forestPath, forestSize, depth, noc)
%TrainAdaBoost Trains an AdaBoost ensemble of decision trees.
%
%   Usage:
%       [weights, importance] = TrainAdaBoost(X, Y, forestPath, forestSize, depth, noc)
%
%   Inputs:
%       X           - n*d matrix, training data.
%       Y           - n*1 vector, labels (1...M).
%       forestPath  - String, the folder path to save the forest.
%       forestSize  - Integer, the number of decision trees in the ensemble.
%       depth       - Integer, the maximum depth of each decision tree.
%       noc         - Integer, number of candidates at each tree node.
%
%   Outputs:
%       weights     - forestSize*1 vector of voting weights for each tree (alpha).
%       importance  - d*1 vector of feature importance.

%   Copyright (C) 2013 Quan Wang <wangq10@rpi.edu>
%   Refactored for AdaBoost.
%   Signal Analysis and Machine Perception Laboratory
%   Department of Electrical, Computer, and Systems Engineering
%   Rensselaer Polytechnic Institute, Troy, NY 12180, USA
%
%   Related publications:
%   [1] Quan Wang, Yan Ou, A. Agung Julius, Kim L. Boyer and Min Jun Kim, 
%       "Tracking Tetrahymena Pyriformis Cells using Decision Trees", 
%       2012 21st International Conference on Pattern Recognition (ICPR), 
%       Pages 1843-1847, 11-15 Nov. 2012.
%   [2] Quan Wang, Dijia Wu, Le Lu, Meizhu Liu, Kim L. Boyer, and Shaohua 
%       Kevin Zhou, "Semantic Context Forests for Learning-Based Knee 
%       Cartilage Segmentation in 3D MR Images", 
%       MICCAI 2013: Workshop on Medical Computer Vision.
%   [3] Quan Wang. "Exploiting Geometric and Spatial Constraints for Vision
%       and Lighting Applications".
%       Ph.D. dissertation, Rensselaer Polytechnic Institute, 2014.

if exist(forestPath, 'dir') ~= 7
    mkdir(forestPath);
end

n = size(X, 1);
d = size(X, 2);
W = ones(n, 1) / n; % Initialize weights
weights = zeros(forestSize, 1);
importance = zeros(d, 1);

% Multi-class AdaBoost (SAMME algorithm)
K = max(Y); % number of classes

for i = 1:forestSize
    treeFile = fullfile(forestPath, [num2str(i) '.tree']);
    
    % Train weak learner with weights
    imp = TrainDecisionTree(X, Y, treeFile, depth, noc, W);
    importance = importance + imp;
    
    % Predict on training data
    [Y_pred, ~] = RunDecisionTree(X, treeFile);
    
    % Calculate weighted error
    incorrect = (Y_pred ~= Y);
    err = sum(W(incorrect));
    
    % Avoid division by zero
    if err == 0
        err = 1e-10;
    elseif err >= (1 - 1/K)
        err = 1 - 1/K - 1e-10;
    end
    
    % Calculate alpha
    alpha = log((1 - err) / err) + log(K - 1);
    weights(i) = alpha;
    
    % Update weights
    % incorrect: W * exp(alpha)
    % correct: W * 1
    % Since we normalize later, we can multiply all by exp(alpha) only where incorrect?
    % SAMME update: w <-- w * exp(alpha * I(y != h(x)))
    W = W .* exp(alpha * incorrect);
    
    % Normalize weights
    W = W / sum(W);
end

% Normalize importance
importance = importance / sum(importance);

% Save weights to a file in the forest directory
save(fullfile(forestPath, 'weights.mat'), 'weights');

end
