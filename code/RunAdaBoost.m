function [Y, P] = RunAdaBoost(X, forestPath)
%RunAdaBoost Runs an AdaBoost ensemble on testing data.
%
%   Usage:
%       [Y, P] = RunAdaBoost(X, forestPath)
%
%   Inputs:
%       X           - n*d matrix, testing data.
%       forestPath  - String, the folder path where the ensemble is saved.
%
%   Outputs:
%       Y           - n*1 vector, decision labels.
%       P           - n*nol matrix, weighted voting scores (probabilities).

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

% Load weights
weightsFile = fullfile(forestPath, 'weights.mat');
if exist(weightsFile, 'file')
    load(weightsFile, 'weights');
else
    error('Weights file not found in forest path.');
end

treeFiles = dir(fullfile(forestPath, '*.tree'));
forestSize = length(treeFiles);

if forestSize == 0
    error('Error: no decision trees found.');
end

if length(weights) ~= forestSize
    % Fallback if weights file doesn't match file count (unlikely if trained correctly)
    warning('Mismatch between weights count and tree files. Using uniform weights.');
    weights = ones(forestSize, 1);
end

% Initialize P
% We need to know nol (number of labels). 
% RunDecisionTree returns P with size n*nol. We can infer nol from the first run.
[~, P0] = RunDecisionTree(X, fullfile(forestPath, treeFiles(1).name));
nol = size(P0, 2);
n = size(X, 1);
P = zeros(n, nol);

% Weighted voting
for i = 1:forestSize
    treeFile = fullfile(forestPath, treeFiles(i).name);
    [Y_weak, ~] = RunDecisionTree(X, treeFile);
    
    % The output Y_weak are labels 1..nol.
    % We add alpha to the column corresponding to the predicted label.
    % To vectorize:
    
    % Create indices for accumulation
    % P(j, Y_weak(j)) += weights(i)
    
    linear_ind = sub2ind([n, nol], (1:n)', Y_weak);
    P(linear_ind) = P(linear_ind) + weights(i); 
end

% Normalize P (softmax-like or just normalize sum to 1)
% Here we treat them as scores.
P = P ./ repmat(sum(P, 2), 1, nol);

[~, Y] = max(P, [], 2);

end
