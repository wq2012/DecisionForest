% This package implements the decision tree technique used in [1], and it
% is also a simplified version of the random forest method used in [2].
% However, the original code of [2] is a property of Siemens Corporate
% Research. Thus this re-implementation by the author only contains part of
% the functionalities. 
% We have tested this package on Windows 7 and Mac OS, both 64-bit, and
% both MATLAB 2012b. 

% This file is a demo showing the use of our decision tree/forest package.  
% To use this package you need to compile C++ code with MEX. 
% The four functions are TrainDecisionTree(), RunDecisionTree(), 
% TrainDecisionForest(), and RunDecisionForest(). 
% Note: if there are M classes, then the labels should be 1, 2, ..., M. 

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

clear; clc; close all;

%% compile C++ code

mex TrainDecisionTree.cpp;
mex RunDecisionTree.cpp;

%% training a decision tree

load TrainingData.mat;

depth = 5;       % tree depth
noc = 1000;      % number of candidates at each tree node
treeFile = 'tree.txt'; % tree file name

tic;
TrainDecisionTree(X, Y+1, treeFile, depth, noc);
t1 = toc;

fprintf('Training decision tree of depth %d completed, using time %f seconds \n',depth,t1);

%% testing a decision tree

load TestingData.mat;

tic;
[Y1,P]=RunDecisionTree(X,treeFile);
t2=toc;

Y1 = Y1 - 1;

error = sum(Y1 ~= Y);
accuracy = 1 - error / length(Y);

fprintf('Testing decision tree of depth %d completed, using time %f seconds \n',depth,t2);
fprintf('Decision tree classification accuracy: %f \n\n',accuracy);

clear;

%% training a decision forest

load TrainingData.mat;

depth = 5;       % tree depth
noc = 1000;      % number of candidates at each tree node
forestSize = 10; % how many trees does the forest contain
forestPath = 'forest'; % forest directory name

tic;
TrainDecisionForest(X, Y+1, forestPath, forestSize, depth, noc);
t3 = toc;

fprintf('Training decision forest of size %d completed, using time %f seconds \n',forestSize,t3);

%% testing a decision forest

load TestingData.mat;

tic;
[Y1, P] = RunDecisionForest(X, forestPath);
t4 = toc;

Y1 = Y1 - 1;

error = sum(Y1 ~= Y);
accuracy = 1 - error / length(Y);

fprintf('Testing decision forest of size %d completed, using time %f seconds \n',forestSize,t4);
fprintf('Decision forest classification accuracy: %f \n\n',accuracy);

