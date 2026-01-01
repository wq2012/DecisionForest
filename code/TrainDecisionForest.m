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

for i=1:forestSize
    treeFile=[forestPath '/' num2str(i) '.tree'];
    TrainDecisionTree(X,Y,treeFile,depth,noc);
end
