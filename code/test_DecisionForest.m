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
    
    % Assert accuracy is reasonable (e.g., > 80%)
    % Note: Accuracy might vary slightly due to randomness in training,
    % but for this dataset it should be high.
    assert(accuracy > 0.8, 'Decision Tree accuracy is too low.');
    
    % Clean up
    delete(treeFile);
    
    % ------------------------
    % Test 2: Decision Forest
    % ------------------------
    fprintf('Test 2: Decision Forest...\n');
    load('TrainingData.mat');
    
    forestSize = 5;
    forestPath = 'test_forest';
    
    if exist(forestPath, 'dir')
        rmdir(forestPath, 's');
    end
    
    TrainDecisionForest(X, Y+1, forestPath, forestSize, depth, noc);
    
    if ~exist(forestPath, 'dir')
        error('Forest directory was not created.');
    end
    
    load('TestingData.mat');
    [Y1, ~] = RunDecisionForest(X, forestPath);
    Y1 = Y1 - 1;
    
    error_count = sum(Y1 ~= Y);
    accuracy = 1 - error_count / length(Y);
    
    fprintf('Decision Forest Accuracy: %.4f\n', accuracy);
    
    assert(accuracy > 0.85, 'Decision Forest accuracy is too low.');
    
    % Clean up
    rmdir(forestPath, 's');
    
    fprintf('All tests passed!\n');
end
