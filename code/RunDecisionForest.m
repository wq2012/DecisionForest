function [Y, P] = RunDecisionForest(X, forestPath)
%RunDecisionForest Runs a decision forest on testing data.
%
%   Usage:
%       [Y, P] = RunDecisionForest(X, forestPath)
%
%   Inputs:
%       X           - n*d matrix, testing data, each row is one instance.
%       forestPath  - String, the folder path where the forest is saved.
%
%   Outputs:
%       Y           - n*1 vector, decision labels for each instance.
%       P           - n*nol matrix, probabilities. 
%                     P(i,j) is the probability that instance i belongs to class j.
%                     nol is the number of unique labels.
%
%   See also TrainDecisionForest, RunDecisionTree, TrainDecisionTree.

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

treeFiles=dir([forestPath '/*.tree']);

forestSize=length(treeFiles);

if forestSize==0
    error('Error: no decision trees found. ');
end

for i=1:forestSize
    treeFile=[forestPath '/' treeFiles(i).name];
    [~,P0]=RunDecisionTree(X,treeFile);
    if i==1
        P=P0;
    else
        P=P+P0;
    end
end

P=P/forestSize;

[~,Y]=max(P,[],2);


