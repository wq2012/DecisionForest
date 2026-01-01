/**
 * This is the C/MEX code for running a decision tree
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
 *     mex RunDecisionTree.cpp
 *
 * usage:
 *     [Y,P]=RunDecisionTree(X,path)
 *       X: n*d testing data, each row is one instance, double
 *       path: the file path of the resulting tree
 *       Y: n*1 decision labels, each row is one instance, each number is an integer between 1 and nol
 *       P: n*nol probabilities
 */

#include "mex.h"
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include "DecisionTree.h"

/* the gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    double *X;
    double *Y;
    double *P;
    long n; // number of instances
    long d; // dimension of features
    char *path;

    /*  check for proper number of arguments */
    if (nrhs != 2)
    {
        mexErrMsgIdAndTxt(
            "MATLAB:RunDecisionTree:invalidNumInputs",
            "Two inputs required.");
    }
    if (nlhs > 2)
    {
        mexErrMsgIdAndTxt(
            "MATLAB:RunDecisionTree:invalidNumOutputs",
            "At most two outputs.");
    }

    /*  get X */
    X = mxGetPr(prhs[0]);
    n = mxGetM(prhs[0]);
    d = mxGetN(prhs[0]);

    /*  get path */
    path = mxArrayToString(prhs[1]);

    /*  call the C++ subroutine */
    Tree *tree = new Tree(path);

    /*  set the output pointers to the output matrix */
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(n, tree->nol, mxREAL);

    /*  create C++ pointers to a copies of the output matrix */
    Y = mxGetPr(plhs[0]);
    P = mxGetPr(plhs[1]);

    /*  call the C++ subroutine */
    tree->runDecision(X, Y, P, n, d);

    delete tree;

    return;
}
