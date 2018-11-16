/** 
 * This is the C/MEX code for running a decision tree
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
void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    double *X;
    double *Y;
    double *P;
    long n; // number of instances
    long d; // dimension of features
    char *path;
    
    /*  check for proper number of arguments */
    if(nrhs!=2)
    {
        mexErrMsgIdAndTxt( "MATLAB:TrainDecisionTree:invalidNumInputs",
                "Two inputs required.");
    }
    if(nlhs>2)
    {
        mexErrMsgIdAndTxt( "MATLAB:TrainDecisionTree:invalidNumOutputs",
                "At most two outputs.");
    }
    
    /*  get X */
    X=mxGetPr(prhs[0]);
    n=mxGetM(prhs[0]);
    d=mxGetN(prhs[0]);
    
    /*  get path */
    path=mxArrayToString(prhs[1]);

    /*  call the C++ subroutine */
    Tree *tree=new Tree(path);
    
    /*  set the output pointers to the output matrix */
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(n, tree->nol, mxREAL);

    /*  create C++ pointers to a copies of the output matrix */
    Y=mxGetPr(plhs[0]);
    P=mxGetPr(plhs[1]);

    /*  call the C++ subroutine */
    tree->runDecision(X,Y,P,n,d);
    
    delete tree;
    
    return;   
}

