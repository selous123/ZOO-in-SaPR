# ZOO-in-SaPR
ZerOing Operator in Sign-aware Perturbations Regression

## Introduction
This paper presents the first study on Sign-aware Perturbations Regression (SaPR), where the ob- served response variables contain the sign-aware (negative or positive) perturbations. In order to predict the non-perturbation response variables, we propose a novel parameter estimator ZOO (i.e.,ZerOing Operator), which aims at taking full advantage of the aware perturbations information to correct the mistake values in the estimation process with computationally efficiency. In this paper, the two aspects of theoretical analysis are proposed to deeply understand our method. Firstly, we establish the perturbation parameter error upper bound and prove consistency guarantee in the linear regression scenario. Secondly, we introduce the generalization error bound for the proposed ZOO, which indicates that the error bound is related to the value and the number of negative and positive perturbations. The effectiveness of the proposed approach is well validated by the experimental results on both synthetic and real datasets.Detailed description of the method can be found in our paper.
## Prerequisites
- python 2.7
- scikit-learn 0.19.1
## Installation
- Clone the repo
```
git clone https://github.com/selous123/ZOO-in-SaPR.git yourDirName
cd yourDirName
```
- Install scikit-learn

## Datasets
### 1)synthetic dataset
```
python data_generator.py
```
### 2)real dataset
```
python data_music.py
```
## Getting Started
```
python main.py
#python main_music.py
```

## Result

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

Except where otherwise noted, this content is published under a [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license, which means that you can copy, remix, transform and build upon the content as long as you do not use the material for commercial purposes and give appropriate credit and provide a link to the license.
