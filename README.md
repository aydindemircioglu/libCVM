# LibCVM Toolkit Version: 2.2 (beta) (Backup)

This is a backup of LibCVM, by Ivor W. Tsang, Andras Kocsor and James T. Kwok,
as the original webpage http://c2inet.sce.ntu.edu.sg/ivor/cvm.html
does not work for me.

You can find the Linux version with a propoer Makefile in the main directory
as well as the unzipped original  software package in the bvm_src_Jan05_2009
folder and the last known webpage, archived from the Wayback machine.


## Description from the original webpage

Last Update: Aug 29, 2011


### Introduction

The LibCVM Toolkit is a C++ implementation of the improved Core Vector Machine (CVM) and recently developed Ball Vector Machine (BVM), which are fast Support Vector Machine (SVM) training algorithms using core-set approximation on very large scale data sets. It is adapted from the LIBSVM implementation (version 2.85). The code has been used on a large range of problems, including network intrusion detection, face detection and implicit surface modeling.

The main features of the toolkit are the following:

more stable core set selection
adaptive epsilon approximation solution
sparse caching of kernel evaluations
can handle millions of training examples
supports standard and several other nonlinear kernel functions
supports dense and sparse vector representations
supports multiple training data files and label renaming
supports BVM/CVM/CVM-LS for large-scale classification
supports Core Vector Data Description (CVDD) for large-scale novelty detection
supports Core Vector Regression (CVR) for large-scale sparse least-squares regression
Pending features:

multiclass CVM/BVM
probabilistic output estimation for CVM
If you have any suggestions or bug findings, please email to ivor.tsang@gmail.com. Thank you!

### Download

#### LibCVM

The toolkit is free for research purpose. The software is not tested under Linux/Unix platform, and the performance under Linux/Unix platform may not be reliable.	The software must not be further distributed without prior permission of the authors. The author is not responsible for implications from the use of this software. If you use LibCVM Toolkit in your scientific work, please cite as

Ivor W. Tsang, James T. Kwok, Pak-Ming Cheung. Core vector machines: Fast SVM training on very large data sets. Journal of Machine Learning Research, 6:363-392, 2005.
Ivor W. Tsang, Andras Kocsor, James T. Kwok. Simpler core vector machines with enclosing balls. Proceedings of the Twenty-Fourth International Conference on Machine Learning (ICML), Corvallis, Oregon, USA, June 2007.


#### How to Use

The toolkit consists of two modules, bvm_train for SVM training and bvm_predict for SVM prediction. Since the LibCVM Toolbox is adapted from the LIBSVM, some options are used for LIBSVM only. The options of bvm_train are:

```
Usage: bvm-train [options] training_set_file [model_file]

options:
-s svm_type : set type of SVM (default 9)
	 0 -- C-SVC
	 1 -- nu-SVC
	 2 -- one-class SVM
	 3 -- epsilon-SVR
	 4 -- nu-SVR
	 5 -- CVDD (Core Vector Data Description for novelty detection)
	      [Tsang, Kwok, Cheung, JMLR 2005]
	 6 -- CVM (sqr. hinge-loss for classification) [Tsang, Kwok, Cheung, JMLR 2005]
	 7 -- CVM-LS (sqr. eps.-insensitive loss for sparse least-squares classification)
	      [Tsang, Kwok, Lai, ICML 2005]
	 8 -- CVR [Tsang, Kwok, Lai, ICML 2005], [Tsang, Kwok, Zurada, TNN 2006]
	 9 -- BVM [Tsang, Kocsor, Kwok, ICML 2007]
-t kernel_type : set type of kernel function (default 2)
	0 -- linear: u'*v
	1 -- polynomial: (gamma*u'*v + coef0)^degree
	2 -- radial basis function: exp(-gamma*|u-v|^2)
	3 -- sigmoid: tanh(gamma*u'*v + coef0)
	4 -- precomputed kernel (kernel values in training_set_file)
	5 -- laplacian: exp(-sqrt(gamma)*|u-v|)
	6 -- normalized poly: ((gamma*u'*v+coef0)/sqrt((gamma*u'*u+coef0)*(gamma*v'*v+coef0)))^degree
	7 -- inverse distance: 1/(sqrt(gamma)*|u-v|+1)
	8 -- inverse square distance: 1/(gamma*|u-v|^2+1)
-d degree : set degree in kernel function (default 3)
-g gamma : set gamma in kernel function (default -1, which sets 1/averaged distance between patterns)
-r coef0 : set coef0 in kernel function (default 0)
-c cost : set the regularization parameter C(= cost) of C-SVC, eps.-SVR, nu-SVR, BVM and CVDD/CVM,
          and the regularization parameter C/(mu*m) in CVM-LS s.t. (mu = s_ratio/(cost*m))
          (default 100 for BVM/CVDD/CVM/CVM-LS)
-C s_ratio : set the scale parameter C = s_ratio*max|Y_i| in CVR,
             and the scale parameter C = s_ratio in CVM-LS (same as the scale parameter in LASSO)
             (default 10000 for CVR/CVM-LS)
-u mu_ratio : set the regularization parameter mu = mu_ratio*max|Y_i| in CVR (default = 0.02)
-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
-m cachesize : set cache memory size in MB (default 200)
-e epsilon : set tolerance of termination criterion
 (In CVM/BVM, default eps=-1 which sets eps according to the bound |f(x)-f(x)^*|; default 1e-3 for others)
-f max #CVs : MAX number of Core Vectors in binary CVM and BVM (default 50000)
-h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
-b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
-wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)
-v n: n-fold cross validation mode
-a size: sample size for probabilistic sampling (default 60)
```

Example: Zero/one digit classification using the Gaussian kernel:

    bvm_train -s 9 -t 2 -c 100 zero_one.txt zero_one.model.txt

Example: Forest Cover Type binary class data using the Gaussian kernel:

    bvm_train -s 9 -t 2 -c 10000 -g 1e-4 forest.txt forest.model.txt

Example: KDDCUP-99 Intrusion detection binary class data using the Gaussian kernel:

    bvm_train -s 9 -t 2 -c 1000000 intrusion.txt intrusion.model.txt

Example: Regression using the Gaussian kernel:

    bvm_train -s 8 -t 2 -u 0.01 -C 20000 census.txt census.model.txt
    bvm_train -s 8 -t 2 -u 0.03 -C 100000 cpu.txt cpu.model.txt

Note that -g -1 option can be used to set the width (gamma) of the Gaussian kernel exp(-gamma*|u-v|^2), where

    1/gamma = sum ||x_i-x_j||^2/m^2

is the average distance between patterns from a subsampled training set (with 5000 patterns).

The usage of bvm_predict is:

```
Usage: bvm_predict [options] test_file model_file output_file

options:

   -b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0);

      one-class SVM not supported yet

   -r Ranking: whether to output the ranking score or not, 0 or 1 (default 0)
```

Example: Zero/one digit classification:

    bvm_predict zero_one.test.txt zero_one.model.txt zero_one.output.txt

Example: Forest cover type binary class classification:

    bvm_predict forest.test.txt forest.model.txt forest.output.txt

Example: KDDCUP-99 intrusion detection binary class classification:

    bvm_predict intrusion.test.txt intrusion.model.txt intrusion.output.txt

Example: Regression:

    bvm_predict census.test.txt census.model.txt census.output.txt
    bvm_predict cpu.test.txt cpu.model.txt cpu.output.txt


#### File Format

The LibCVM Toolkit supports both sparse and dense data representations. The sparse format is the same as in SVMlight and LIBSVM. Each line represents one training example and is of the form:
```
    <line> .=. <target> <feature>:<value> <feature>:<value> ... <feature>:<value>

    <target> .=. <label> | <float>
    <feature> .=. <unsigned integer>

    <value> .=. <float>
	```


For the dense format, each line represents one training example and is of the form:

```
    <line> .=. <value>,<value>, ... <value>,<target>

    <target> .=. <label> | <float>
    <value> .=. <float>
		```

For the multiple training file mode, the input file is started with a header:

```
#m
N M
<file name 1>
...
<file name N>
<label renaming rule 1>
...
<label renaming rule M>
```

The label renaming rule is defined as:

```
<label renaming rule> .=. <original label>><new label>
E.g.:
#m
3 6
usps1.txt
usps2.txt
usps3.txt
0>1
1>-1
2>1
3>-1
4>1
5>-1
```
