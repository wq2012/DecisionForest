% This is the function to run a decision forest on testing data. 

function [Y,P]=RunDecisionForest(X,forestPath)
% X: n*d testing data, each row is one instance
% forestPath: the folder path where the forest is saved
% Y: n*1 decision labels, each row is one instance
% P: n*nol probabilities
%     nol: the number of unique labels, or number of classes
%     P(i,j): the probability that instance i belongs to class j

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


