function TrainDecisionForest(X, Y, forestPath, forestSize, depth, noc)
%TrainDecisionForest Trains a decision forest.
%
%   A decision forest is saved as a folder, and each decision tree is a file
%   in this folder, named as 1.tree, 2.tree, 3.tree, ...
%
%   Usage:
%       TrainDecisionForest(X, Y, forestPath, forestSize, depth, noc)
%
%   Inputs:
%       X           - n*d matrix, training data, each row is one instance.
%       Y           - n*1 vector, labels, each row is one instance.
%                     Important: for M classes, labels should be 1, 2, ..., M.
%       forestPath  - String, the folder path to save the forest.
%       forestSize  - Integer, the number of decision trees in the forest.
%       depth       - Integer, the maximum depth of each decision tree.
%       noc         - Integer, number of candidates at each tree node.
%
%   See also RunDecisionForest, TrainDecisionTree, RunDecisionTree.

%   Copyright (C) 2013 Quan Wang <wangq10@rpi.edu>
%   Signal Analysis and Machine Perception Laboratory
%   Department of Electrical, Computer, and Systems Engineering
%   Rensselaer Polytechnic Institute, Troy, NY 12180, USA

if exist(forestPath, 'dir') ~= 7
    mkdir(forestPath);
end

for i=1:forestSize
    treeFile=[forestPath '/' num2str(i) '.tree'];
    TrainDecisionTree(X,Y,treeFile,depth,noc);
end
