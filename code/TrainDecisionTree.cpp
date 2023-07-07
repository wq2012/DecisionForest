/**
 * This is the C/MEX code for training a decision tree
 *
 * compile:
 *     mex TrainDecisionTree.cpp
 *
 * usage:
 *     TrainDecisionTree(X,Y,path,depth,noc)
 *       X: n*d training data, each row is one instance, double
 *       Y: n*1 labels, each row is one instance, each number is an integer between 1 and nol
 *       path: the file path of the resulting tree
 *       depth: the maximum depth of the tree
 *       noc: number of candidates at each node
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
    long n;    // number of instances
    long d;    // dimension of features
    int depth; // the maximum depth of the tree
    long noc;  // number of candidates at each node
    char *path;

    /*  check for proper number of arguments */
    if (nrhs != 5)
    {
        mexErrMsgIdAndTxt(
            "MATLAB:TrainDecisionTree:invalidNumInputs",
            "Five inputs required.");
    }
    if (nlhs > 0)
    {
        mexErrMsgIdAndTxt(
            "MATLAB:TrainDecisionTree:invalidNumOutputs",
            "Zero output required.");
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

    /*  call the C++ subroutine */
    Data *data = new Data(X, Y, n, d);
    Tree *tree = new Tree(depth, noc);
    tree->trainTree(data);
    tree->saveTree(path);

    delete data;
    delete tree;
    delete[] Y;

    return;
}
