/**
 * This is the C/MEX code for training a decision tree
 *
 * Copyright (C) 2013 Quan Wang <wangq10@rpi.edu>,
 * Signal Analysis and Machine Perception Laboratory,
 * Department of Electrical, Computer, and Systems Engineering,
 * Rensselaer Polytechnic Institute, Troy, NY 12180, USA
 *
 * Related publications:
 * [1] Quan Wang, Yan Ou, A. Agung Julius, Kim L. Boyer and Min Jun Kim,
 *     "Tracking Tetrahymena Pyriformis Cells using Decision Trees",
 *     2012 21st International Conference on Pattern Recognition (ICPR),
 *     Pages 1843-1847, 11-15 Nov. 2012.
 * [2] Quan Wang, Dijia Wu, Le Lu, Meizhu Liu, Kim L. Boyer, and Shaohua
 *     Kevin Zhou, "Semantic Context Forests for Learning-Based Knee
 *     Cartilage Segmentation in 3D MR Images",
 *     MICCAI 2013: Workshop on Medical Computer Vision.
 * [3] Quan Wang. "Exploiting Geometric and Spatial Constraints for Vision
 *     and Lighting Applications".
 *     Ph.D. dissertation, Rensselaer Polytechnic Institute, 2014.
 *
 * compile:
 *     mex TrainDecisionTree.cpp
 *
 * usage:
 *     importance = TrainDecisionTree(X,Y,path,depth,noc,W)
 *       X: n*d training data, each row is one instance, double
 *       Y: n*1 labels, each row is one instance, each number is an integer between 1 and nol
 *       path: the file path of the resulting tree
 *       depth: the maximum depth of the tree
 *       noc: number of candidates at each node
 *       W (optional): n*1 weights, each row is one instance, double
 *       importance (optional): d*1 vector of feature importance
 */

#include "mex.h"
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include "DecisionTree.h"

/* the gateway function */
void mexFunction(
    int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[])
{
    double *X;
    int *Y;
    double *Y1;
    double *W = NULL;
    double *importance;
    long n;    // number of instances
    long d;    // dimension of features
    int depth; // the maximum depth of the tree
    long noc;  // number of candidates at each node
    char *path;

    /*  check for proper number of arguments */
    if (nrhs < 5 || nrhs > 6)
    {
        mexErrMsgIdAndTxt(
            "MATLAB:TrainDecisionTree:invalidNumInputs",
            "Five or six inputs required.");
    }
    if (nlhs > 1)
    {
        mexErrMsgIdAndTxt(
            "MATLAB:TrainDecisionTree:invalidNumOutputs",
            "At most one output.");
    }

    /*  get X */
    X = mxGetPr(prhs[0]);
    n = mxGetM(prhs[0]);
    d = mxGetN(prhs[0]);

    /*  get Y */
    Y1 = mxGetPr(prhs[1]);
    if (mxGetM(prhs[1]) != n || mxGetN(prhs[1]) != 1)
    {
        mexErrMsgIdAndTxt(
            "MATLAB:TrainDecisionTree:dimNotMatch",
            "Dimension of input Y is incorrect");
    }
    Y = new int[n];
    for (long i = 0; i < n; i++)
    {
        Y[i] = (int)Y1[i];
    }

    /*  get path */
    path = mxArrayToString(prhs[2]);

    /*  get depth */
    if (!mxIsDouble(prhs[3]) || mxIsComplex(prhs[3]) ||
        mxGetN(prhs[3]) * mxGetM(prhs[3]) != 1)
    {
        mexErrMsgIdAndTxt(
            "MATLAB:TrainDecisionTree:depthNotScalar",
            "Input depth must be a scalar.");
    }

    depth = (int)mxGetScalar(prhs[3]);

    if (depth < 1)
    {
        mexErrMsgIdAndTxt(
            "MATLAB:TrainDecisionTree:depthWrongRange",
            "Input depth must be larger than 0.");
    }

    /*  get noc */
    if (!mxIsDouble(prhs[4]) || mxIsComplex(prhs[4]) ||
        mxGetN(prhs[4]) * mxGetM(prhs[4]) != 1)
    {
        mexErrMsgIdAndTxt(
            "MATLAB:TrainDecisionTree:nocNotScalar",
            "Input noc must be a scalar.");
    }

    noc = (long)mxGetScalar(prhs[4]);

    if (noc < 1)
    {
        mexErrMsgIdAndTxt(
            "MATLAB:TrainDecisionTree:nocWrongRange",
            "Input noc must be larger than 0.");
    }

    /*  get W */
    if (nrhs == 6)
    {
        W = mxGetPr(prhs[5]);
        if (mxGetM(prhs[5]) != n || mxGetN(prhs[5]) != 1)
        {
            mexErrMsgIdAndTxt(
                "MATLAB:TrainDecisionTree:dimNotMatch",
                "Dimension of input W is incorrect");
        }
    }

    /*  call the C++ subroutine */
    Data *data = new Data(X, Y, n, d, W);
    Tree *tree = new Tree(depth, noc);
    tree->trainTree(data);
    tree->saveTree(path);

    /*  return importance */
    if (nlhs >= 1) {
        plhs[0] = mxCreateDoubleMatrix(d, 1, mxREAL);
        importance = mxGetPr(plhs[0]);
        double *treeImportance = tree->getImportance();
        for (long i = 0; i < d; i++) {
            importance[i] = treeImportance[i];
        }
    }

    delete data;
    delete tree;
    delete[] Y;

    return;
}
