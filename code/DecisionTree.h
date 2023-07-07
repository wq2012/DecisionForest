/**
 * Copyright (C) 2013 Quan Wang <wangq10@rpi.edu>,
 * Signal Analysis and Machine Perception Laboratory,
 * Department of Electrical, Computer, and Systems Engineering,
 * Rensselaer Polytechnic Institute, Troy, NY 12180, USA
 */

/**
 * Related publications:
 * [1] Quan Wang, Yan Ou, A. Agung Julius, Kim L. Boyer and Min Jun Kim, 
 *     "Tracking Tetrahymena Pyriformis Cells using Decision Trees", 
 *     2012 21st International Conference on Pattern Recognition (ICPR), 
 *     Pages 1843-1847, 11-15 Nov. 2012.
 * [2] Quan Wang, Dijia Wu, Le Lu, Meizhu Liu, Kim L. Boyer, and Shaohua 
 *     Kevin Zhou, "Semantic Context Forests for Learning-Based Knee 
 *     Cartilage Segmentation in 3D MR Images", 
 *     MICCAI 2013: Workshop on Medical Computer Vision. 
 */

/** 
 * This is the C++ implementation of the decision tree data structure. 
 * Implemented classes:
 *     1. Data: to hold the training data.
 *     2. List: to hold the list of indices of data instances. 
 *     3. TreeNode: to represent a node of the tree. 
 *     4. Tree: to represent the decision tree. 
 *
 * The connection between Tree and TreeNode is that there is a hash table 
 * in Tree, which maps a node index to a TreeNode: 
 *     HashTable<TreeNode*> *map;
 *
 * A decision tree can be saved into a text file using the saveTree() 
 * function. The first line of the file is tree information, and each of 
 * the following lines is one node.  
 */

#ifndef DecisionTree_H
#define DecisionTree_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <iostream>
#include "HashTable.h"

using namespace std;
        
/**********************************************
* Declaration part
**********************************************/

class Data
{
public:
    double *X;
    int *Y;
    long n; // number of instances
    long d; // dimension of each instance
    int nol; // number of unique labels
    double * mean; // mean value of each dimension
    double * std; // standard deviation of each dimension

    Data(double *X_, int *Y_, long n_, long d_);
    ~Data();
    double getFeature(long i, long feature);
};

class List
{
public:
	long *list;
	long num; // number of instances
    List(long num_);
    ~List();
};

class TreeNode
{
public:
	long feature;
	double threshold;
	double *param; // parameters or probabilities
    TreeNode();
    TreeNode(long feature_, double threshold_, int nol);
    ~TreeNode();
};

class Tree
{
private:
    int depth; // maximum depth of tree
    long noc; // number of candidates
    long d; // dimension of each instance
    double eps; // constant
    double inf; // constant
    double searchRange; // range of threshold K: mu +/- K * sigma
    int minList; // minimum size of a splittable list
    HashTable<TreeNode*> *map; // the data structure to hold tree nodes
    
public:
    int nol; // number of unique labels
    void initialize(); // called by constructors to set constants
    Tree(int depth_, long noc_);
    Tree(char *path); // load a tree from a file
    ~Tree();
    void saveTree(char *path); // save tree to file
    
    long leftChild(long n);
    long rightChild(long n);
    long parent(long n);
    int treeLevel(long n);
    
    TreeNode * getCandidates(Data *data); // get candidates for one node
    void trainTree(Data *data); // train decision tree using data
    void trainTreeNode(long n, List *list, Data *data); // train one node (recursive)
    double getEntropyDecrease(Data *data, TreeNode node, List *list);
    bool pureList(List *list, Data *data); // check if a list contains only one kind of label
    
    TreeNode *decideTree(long n, double *feature); // make decisions given one instance (recursive)
    void runDecision(double *X, double *Y, double *P, long n, long d); // make decisions given tesing data
};



/**********************************************
* Implementation part
**********************************************/

Data::Data(double *X_, int *Y_, long n_, long d_)
{
    X=X_;
    Y=Y_;
    n=n_;
    d=d_;
    
    mean=new double[d];
    std=new double[d];
    
    for(long i=0;i<d;i++)
    {
        // mean
        double sum=0;
        for(long j=0;j<n;j++)
        {
            sum+=X[j+i*n];
        }
        mean[i]=sum/n;
        
        // std
        sum=0;
        for(long j=0;j<n;j++)
        {
            sum+=((X[j+i*n]-mean[i])*(X[j+i*n]-mean[i]));
        }
        std[i]=sqrt(sum/n);
    }
    
    nol=0;
    for(long i=0;i<n;i++)
    {
        if(Y[i]<1)
        {
            cout<<"Error: entries of Y should be between 1 and nol. \n";
            exit(1);
        }
        
        if(Y[i]>nol)
        {
            nol=Y[i];
        }
        
        if(nol==0)
        {
            cout<<"Error: entries of Y should be between 1 and nol. \n";
            exit(1);
        }
    }
}

Data::~Data()
{
    delete[] mean;
    delete[] std;
}

double Data::getFeature(long i, long feature)
{
	return X[i+feature*n];
}

List::List(long num_)
{
    num=num_;
    list=new long[num];
}

List::~List()
{
    delete[] list;
}

TreeNode::TreeNode()
{
    param=NULL;
}

TreeNode::TreeNode(long feature_, double threshold_, int nol)
{
    feature=feature_;
    threshold=threshold_;
    param=new double[nol];
    for(int i=0;i<nol;i++)
    {
        param[i]=0;
    }
}

TreeNode::~TreeNode()
{
    if(param!=NULL)
    {
        delete[] param;
    }
}

void Tree::initialize()
{
    eps=0.00000000001;
    inf=1000000000000;
    minList=10;
    searchRange=3;
    map=new HashTable<TreeNode*>(10000);
}

Tree::Tree(int depth_, long noc_)
{
    initialize();
    
    depth=depth_;
    noc=noc_;
}
    
Tree::Tree(char *path)
{
    initialize();

    FILE * pFile;
	pFile = fopen (path , "r");
	if (pFile == NULL)
    {
        cout<<"Error opening "<<path<<endl; 
        exit (1);
    }
    
	char * line=new char[200];
	char * word;

    // read information
    if (fgets(line , 200 , pFile) == NULL)
    {
        cerr<<"Error reading "<<path<<endl; 
    }
	word=strtok(line,"\t\r\n");
	depth=atoi(word); // depth of tree
    word=strtok(NULL,"\t\r\n");
    d=atoi(word); // dimension of instances
    word=strtok(NULL,"\t\r\n");
    nol=atoi(word); // number of unique labels
    word=strtok(NULL,"\t\r\n");
    int numLines=atoi(word); // number of nodes
 
	for(long i=0;i<numLines;i++)
	{
		TreeNode *node=new TreeNode(0,0,nol);
        
        if (fgets(line, 200, pFile) == NULL)
        {
            cerr<<"Error reading "<<path<<endl; 
        }

		word=strtok(line,"\t\r\n");
        long n=atol(word); // node index
        
        word=strtok(NULL,"\t\r\n");
		node->feature=atol(word); // feature

		word=strtok(NULL,"\t\r\n");
		node->threshold=atof(word); // threshold
        
        // parameters
        if(node->feature==-1)
        {
            for(int j=0;j<nol;j++)
            {
                word=strtok(NULL,"\t\r\n");
                node->param[j]=atof(word);
            }
        }
        map->add(n,node);
	}

    delete[] line;
	fclose (pFile);
}

Tree::~Tree()
{
    delete map;
}

long Tree::leftChild(long n)
{
	return n*2+1;
}

long Tree::rightChild(long n)
{
	return n*2+2;
}

long Tree::parent(long n)
{
	return (long)((n-1)/2);
}

int Tree::treeLevel(long n)
{
	return (int)floor(log((double)n+1)/log(2.0)+eps)+1;
}

TreeNode * Tree::getCandidates(Data *data)
{
	int feature;
	double threshold;
	TreeNode * candidates=new TreeNode[noc];
    
    srand(time(NULL));
    
	for(long i=0;i<noc;i++)
	{
		// feature
        feature=i%d;
        candidates[i].feature=feature;
        
        // threshold
        double r=((double)rand()/(RAND_MAX))*2-1;
        threshold=data->mean[feature]+data->std[feature]*searchRange*r;
        candidates[i].threshold=threshold;
	}
	return candidates;
}


void Tree::trainTree(Data *data)
{
	d=data->d;
    nol=data->nol;
    List *list=new List(data->n);
    for(long i=0;i<data->n;i++)
    {
        list->list[i]=i;
    }
    
    if(minList<data->n/1000)
    {
        minList=data->n/1000;
    }
    
    // recursive call
    trainTreeNode(0,list,data);
    
    delete list;
}

void Tree::trainTreeNode(long n, List *list, Data *data)
{
	// Case 1: leaf node, stop splitting
    int level=treeLevel(n);
	if(level==depth || list->num<minList || pureList(list, data))
	{
		TreeNode *node=new TreeNode(-1,0,data->nol);

		for(long i=0;i<list->num;i++)
		{
			node->param[data->Y[list->list[i]]-1]++;
		}
        map->add(n,node);
		return;
	}

    // Case 2: non-leaf node
    TreeNode *bestNode;
	double *entropyDecrease=new double[noc];
	double largestEntropyDecrease=-inf;

	TreeNode * candidates=getCandidates(data);
    
    // get best node
	for(long i=0;i<noc;i++)
	{
		entropyDecrease[i]=getEntropyDecrease(data, candidates[i], list);
		if(entropyDecrease[i]>largestEntropyDecrease)
		{
			bestNode=&candidates[i];
			largestEntropyDecrease=entropyDecrease[i];
		}
	}

	map->add(n,new TreeNode(bestNode->feature,bestNode->threshold,nol));

    delete[] entropyDecrease;
    
    // generate lists for children
	List *leftList=new List(list->num);
    List *rightList=new List(list->num);
    leftList->num=0;
    rightList->num=0;
   
	for(long i=0;i<list->num;i++)
	{
		double feature=data->getFeature(list->list[i], bestNode->feature);
		if(feature<=bestNode->threshold)
		{
			leftList->list[leftList->num]=list->list[i];
			leftList->num++;
		}
		else
		{
			rightList->list[rightList->num]=list->list[i];
			rightList->num++;
		}
	}
    
    // only delete candidates after bestNode is not used
    delete[] candidates;

    // recursive call
    trainTreeNode(leftChild(n),leftList,data);
    delete leftList;
    
	trainTreeNode(rightChild(n),rightList,data);
    delete rightList;		  
}

double Tree::getEntropyDecrease(Data *data, TreeNode node, List *list)
{
	double feature;

	double entropyDecrease=0;
	double leftEntropy=0;
	double rightEntropy=0;
	long leftSize=0;
    long rightSize=0;
  
    double *leftLabel=new double[nol];
    double *rightLabel=new double[nol];
    for(int i=0;i<nol;i++)
    {
        leftLabel[i]=0;
        rightLabel[i]=0;
    }

    // splitting
	for(long i=0;i<list->num;i++)
	{
		feature=data->getFeature(list->list[i], node.feature); 
		if(feature<=node.threshold)
		{
			leftSize++;
            leftLabel[data->Y[list->list[i]]-1]++;
		}
		else
		{
			rightSize++;
            rightLabel[data->Y[list->list[i]]-1]++;
		}
	}
 
    // get left entropy
	if(leftSize>eps)
	{
		for(int i=0;i<nol;i++)
        {
            leftLabel[i]/=(double)leftSize;
            if(leftLabel[i]>eps)
            {
                leftEntropy-=leftLabel[i]*log(leftLabel[i]);
            }
        }
	}

    // get right entropy
	if(rightSize>eps)
	{
		for(int i=0;i<nol;i++)
        {
            rightLabel[i]/=(double)rightSize;
            if(rightLabel[i]>eps)
            {
                rightEntropy-=rightLabel[i]*log(rightLabel[i]);
            }
        }
	}

    // get entropy decrease
	entropyDecrease= -(double)leftSize/list->num*( leftEntropy )
		-(double)rightSize/list->num*( rightEntropy );

	delete[] leftLabel;
    delete[] rightLabel;
    
	return entropyDecrease;		  
}

void Tree::saveTree(char *path)
{
	FILE * pFile;
	pFile = fopen (path, "w");
	if (pFile == NULL) 
    {
        cout<<"Error opening "<<path<<endl; 
        exit(1);
    }

    // save information
	fprintf(pFile,"%d\t%ld\t%d\t%ld\n",depth,d,nol,map->size());
    
    // save nodes
    for(map->begin(); map->hasNext(); )
    {
        HashNode<TreeNode*> *hnode=map->next();
        long n=hnode->key;
        TreeNode *node=hnode->data;
        fprintf(pFile,"%ld\t%ld\t%f\t",n,node->feature,node->threshold);
        if(node->feature==-1)
        {
            for(int i=0;i<nol;i++)
            {
                fprintf(pFile,"%f\t",node->param[i]);
            }
        }
        fprintf(pFile,"\n");
    }
	
	fclose(pFile);
}

bool Tree::pureList(List *list, Data *data)
{
	int *Y=data->Y;
	for(long i=1;i<list->num;i++)
	{
		if( Y[list->list[i]] != Y[list->list[0]] )
		{
			return false;
		}
	}
	return true;
}

TreeNode *Tree::decideTree(long n, double *feature)
{
	TreeNode *node=map->get(n);
	if(node->feature==-1)
	{
		return node;
	}
    
    // recursive call
	if(feature[node->feature]<=node->threshold)
	{
		return decideTree(leftChild(n), feature);
	}
	else
	{
		return decideTree(rightChild(n), feature);
	}
}

void Tree::runDecision(double *X, double *Y, double *P, long n_, long d_)
{
    if(d!=d_)
    {
        cout<<"Error: testing data dimension does not match. \n";
        exit(1);
    }
    
    double *feature=new double[d];
    for(long i=0;i<n_;i++)
    {
        // constructing features
        for(long j=0;j<d;j++)
        {
            feature[j]=X[i+j*n_];
        }
        
        // recursive call
        TreeNode *node=decideTree(0,feature);
        
        // computing probabilities
        double sum=0;
        for(long j=0;j<nol;j++)
        {
            P[i+j*n_]=node->param[j];
            sum+=P[i+j*n_];
        }
        for(long j=0;j<nol;j++)
        {
            P[i+j*n_]/=(sum+eps);
        }

        // decide labels
        Y[i]=1;
        double maxP=P[i];
        for(long j=1;j<nol;j++)
        {
            if(P[i+j*n_]>maxP)
            {
                maxP=P[i+j*n_];
                Y[i]=j+1;
            }
        }
    }
    delete[] feature;    
}

#endif
