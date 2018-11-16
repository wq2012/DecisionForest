% This is the function to train a decision forest. 
% A decision forest is saved as a folder, and each decision tree is a file
% in this folder, named as 1.tree, 2.tree, 3.tree, ...

function TrainDecisionForest(X,Y,forestPath,forestSize,depth,noc)
% X: n*d training data, each row is one instance
% Y: n*1 labels, each row is one instance
%     Important: for M classes, labels should be 1, 2, ..., M
% forestPath: the folder path to save the forest
% forestSize: the number of decision trees in the forest
% depth: the maximum depth of each decision tree
% noc: number of candidates at each tree node

if exist(forestPath,'dir')~=7
    mkdir(forestPath);
end

for i=1:forestSize
    treeFile=[forestPath '/' num2str(i) '.tree'];
    TrainDecisionTree(X,Y,treeFile,depth,noc);
end
