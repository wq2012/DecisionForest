# Decision Tree and Decision Forest

[![View Decision Tree and Decision Forest on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/matlabcentral/fileexchange/39110-decision-tree-and-decision-forest)
[![Octave application](https://github.com/wq2012/DecisionForest/actions/workflows/octave.yml/badge.svg)](https://github.com/wq2012/DecisionForest/actions/workflows/octave.yml)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
  - [Prerequisites](#prerequisites)
  - [Compiling](#compiling)
  - [Training a Decision Tree](#training-a-decision-tree)
  - [Testing a Decision Tree](#testing-a-decision-tree)
  - [Training a Decision Forest](#training-a-decision-forest)
  - [Testing a Decision Forest](#testing-a-decision-forest)
- [Copyright and Citation](#copyright-and-citation)

## Overview

This package implements **Decision Tree** and **Decision Forest** (Random Forest) techniques in C++, optimized for efficiency. It provides a MEX interface to be easily called from **MATLAB** or **Octave**.

The algorithm is based on standard information gain principles and has been utilized in multiple research publications.

## Features

- **Efficient Implementation**: Core logic written in C++ for performance.
- **Easy Integration**: MEX wrappers allow direct usage in MATLAB/Octave.
- **Decision Tree**: Train and test single decision trees.
- **Decision Forest**: Train and test ensembles of trees (Random Forest) for improved robustness and accuracy.
- **Cross-Platform**: Tested on Windows, Mac OS, and Linux.

## Usage

### Prerequisites
- MATLAB or Octave installed.
- C++ compiler compatible with MEX (e.g., `gcc`, `clang`, or MSVC).

### Compiling
Before using the package, compile the C++ source files. Run the following commands in MATLAB/Octave:
```matlab
mex TrainDecisionTree.cpp
mex RunDecisionTree.cpp
```

### Training a Decision Tree
To train a single decision tree:
```matlab
% X: n x d training data (n instances, d dimensions)
% Y: n x 1 labels (1, 2, ..., M)
% treeFile: path to save the tree
% depth: maximum depth of the tree
% noc: number of candidates to split at each node

TrainDecisionTree(X, Y, treeFile, depth, noc);
```

### Testing a Decision Tree
To test a single decision tree:
```matlab
% X: n x d testing data
% treeFile: path to the saved tree
% Y_pred: predicted labels
% P: probabilities

[Y_pred, P] = RunDecisionTree(X, treeFile);
```

### Training a Decision Forest
To train a decision forest (ensemble of trees):
```matlab
% forestPath: directory to save the forest
% forestSize: number of trees in the forest

TrainDecisionForest(X, Y, forestPath, forestSize, depth, noc);
```

### Testing a Decision Forest
To test a decision forest:
```matlab
% forestPath: directory where the forest is saved

[Y_pred, P] = RunDecisionForest(X, forestPath);
```

## Copyright and Citation

```
Copyright (C) 2012 Quan Wang <wangq10@rpi.edu>,
Signal Analysis and Machine Perception Laboratory,
Department of Electrical, Computer, and Systems Engineering,
Rensselaer Polytechnic Institute, Troy, NY 12180, USA
```

If you use this software for academic purposes, please cite our papers:

**Plain Text:**

> Quan Wang, Yan Ou, A. Agung Julius, Kim L. Boyer and Min Jun Kim.
> "Tracking Tetrahymena Pyriformis Cells using Decision Trees."
> 2012 21st International Conference on Pattern Recognition (ICPR),
> Pages 1843-1847, 11-15 Nov. 2012.

> Quan Wang, Dijia Wu, Le Lu, Meizhu Liu, Kim L. Boyer, and Shaohua Kevin Zhou.
> "Semantic Context Forests for Learning-Based Knee Cartilage Segmentation in 3D MR Images."
> MICCAI 2013: Workshop on Medical Computer Vision.

> Quan Wang.
> Exploiting Geometric and Spatial Constraints for Vision and Lighting Applications.
> Ph.D. dissertation, Rensselaer Polytechnic Institute, 2014.

**BibTeX:**

```bibtex
@inproceedings{wang2012tracking,
  title={Tracking tetrahymena pyriformis cells using decision trees},
  author={Wang, Quan and Ou, Yan and Julius, A Agung and Boyer, Kim L and Kim, Min Jun},
  booktitle={Proceedings of the 21st International Conference on Pattern Recognition (ICPR2012)},
  pages={1843--1847},
  year={2012},
  organization={IEEE}
}

@inproceedings{wang2013semantic,
  title={Semantic context forests for learning-based knee cartilage segmentation in 3D MR images},
  author={Wang, Quan and Wu, Dijia and Lu, Le and Liu, Meizhu and Boyer, Kim L and Zhou, Shaohua Kevin},
  booktitle={International MICCAI Workshop on Medical Computer Vision},
  pages={105--115},
  year={2013},
  organization={Springer}
}

@phdthesis{wang2014exploiting,
  title={Exploiting Geometric and Spatial Constraints for Vision and Lighting Applications},
  author={Quan Wang},
  year={2014},
  school={Rensselaer Polytechnic Institute},
}
```
