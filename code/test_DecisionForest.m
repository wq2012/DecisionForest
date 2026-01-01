function test_DecisionForest()
%TEST_DECISIONFOREST Test suite for DecisionForest library.
%   This function runs a series of tests to verify the correctness of the
%   library. It checks if the accuracy is within an acceptable range for
%   both decision tree and decision forest.

    fprintf('Running tests for DecisionForest...\n');
    
    % Compile C++ code
    try
        fprintf('Compiling C++ code...\n');
        mex TrainDecisionTree.cpp;
        mex RunDecisionTree.cpp;
        fprintf('Compilation successful.\n');
    catch e
        error('Compilation failed: %s', e.message);
    end

    % ------------------------
    % Test 1: Single Decision Tree
    % ------------------------
    fprintf('Test 1: Single Decision Tree...\n');
    load('TrainingData.mat'); % loads X and Y
    
    depth = 5;
    noc = 1000;
    treeFile = 'test_tree.txt';
    
    % Test w/o importance
    TrainDecisionTree(X, Y+1, treeFile, depth, noc);
    
    if ~exist(treeFile, 'file')
        error('Tree file was not created.');
    end
    
    load('TestingData.mat'); % loads X and Y
    [Y1, ~] = RunDecisionTree(X, treeFile);
    Y1 = Y1 - 1;
    
    error_count = sum(Y1 ~= Y);
    accuracy = 1 - error_count / length(Y);
    
    fprintf('Decision Tree Accuracy: %.4f\n', accuracy);
    assert(accuracy > 0.8, 'Decision Tree accuracy is too low.');
    
    % Test w/ importance (optional output)
    imp = TrainDecisionTree(X, Y+1, treeFile, depth, noc);
    assert(length(imp) == size(X, 2), 'Importance vector size mismatch');
    fprintf('Feature importance calculated.\n');
    
    delete(treeFile);
    
    % ------------------------
    % Test 2: Decision Forest
    % ------------------------
    fprintf('\nTest 2: Decision Forest...\n');
    load('TrainingData.mat');
    
    forestSize = 5;
    forestPath = 'test_forest';
    
    if exist(forestPath, 'dir')
        rmdir(forestPath, 's');
    end
    
    TrainDecisionForest(X, Y+1, forestPath, forestSize, depth, noc);
    
    load('TestingData.mat');
    [Y1, ~] = RunDecisionForest(X, forestPath);
    Y1 = Y1 - 1;
    
    error_count = sum(Y1 ~= Y);
    accuracy = 1 - error_count / length(Y);
    
    fprintf('Decision Forest Accuracy: %.4f\n', accuracy);
    assert(accuracy > 0.85, 'Decision Forest accuracy is too low.');
    
    rmdir(forestPath, 's');

    % ------------------------
    % Test 3: AdaBoost
    % ------------------------
    fprintf('\nTest 3: AdaBoost...\n');
    load('TrainingData.mat');
    
    forestPath = 'test_adaboost';
    if exist(forestPath, 'dir')
        rmdir(forestPath, 's');
    end
    
    [weights, importance] = TrainAdaBoost(X, Y+1, forestPath, forestSize, depth, noc);
    
    assert(length(weights) == forestSize, 'Weights vector size mismatch');
    assert(length(importance) == size(X, 2), 'Importance vector size mismatch');
    
    load('TestingData.mat');
    [Y1, ~] = RunAdaBoost(X, forestPath);
    Y1 = Y1 - 1;
    
    error_count = sum(Y1 ~= Y);
    accuracy = 1 - error_count / length(Y);
    
    fprintf('AdaBoost Accuracy: %.4f\n', accuracy);
    assert(accuracy > 0.85, 'AdaBoost accuracy is too low.');
    
    rmdir(forestPath, 's');
    
    fprintf('\nAll tests passed!\n');
end
