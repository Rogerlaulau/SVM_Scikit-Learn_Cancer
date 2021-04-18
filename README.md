# About Support Vector Machines (SVM)

Comparing with other classifiers such as decision tresss and logistic regression, Support Vector Machines (SVM) provides with high accuracy. SVM separates data points using a hyperplane with the largest amount of margin. It could classify new data points based on this hyperplane.


## How does SVM works?
The objective is to select a hyperplane with the maximum possible margin between support vectors in the given dataset.


Generally, SVM is considered to be a classification approach, but it can be employed in both types of classification and regression problems. Multiple continuous and categorical variables could be handled. SVM constructs a hyperplane in multidimensional space to separate different classes.

For some problems, linear hyperplanes could be used to do the classification. However, some problems could not be simply addressed with linear hyperplanes, and thus a kernel trick is introduced to transform low-dimensional input space to a higher dimensional space and thus linear separation could be applied. In other words, the kernel converts non-separable problems to separable problems by adding more dimensions to it. It is most useful in non-linear separation problems. Kernel trick helps to build a more accurate classifier.

### Note that Kernel could be
* Linear Kernel
* Polynomial Kernel
* Radial Basis Function Kernel
* Sigmoid Kernel
* Gaussian Kernel
* etc.


## Advantages:
* offer good accuracy
* perform faster prediction compared to Naïve Bayes
* less memory used since a subset of training data points are used in the decision phase.
* clear margin of separation
* high dimensional space

## Disadvantages:
* not suitable for larger datasets due to huge training time taken compared to Naïve Bayes
* works poorly with overlapping classes
* sensitive to kernel applied


<br>

# About this exercise

Implementing SVM in Python using scikit-learn

A dataset consists of 569 records and 30 features is used to build a model that could determine new cases into either malignant (harmful) or  benign (not harmful). The dataset is available in the scikit-learn library or could be downloaded from the UCI Machine Learning Library.


* Step 1: Load Data
    - Loading the data from scikit-learn dataset library

* Step 2: Understand Data
    - understand the features and target names (e.g. ['malignant' 'benign'])

* Step 3: Split Data
    - In order to evaluate the performace of trained model, it is suggested to divided the dataset into a training set and a test set.
    - Split the dataset by using the function train_test_split(). 
    - 3 parameters features, target, and test_set size are needed to provided.

* Step 4: Build Model
    - First, import the SVM module and create support vector classifier object by passing argument kernel as the linear kernel in SVC() function.
    - Then, fit the model on train set using fit() and perform prediction on the test set using predict().


* Step 5: Evaluate Model
    - Estimate how accurately the classifier or model can predict the breast cancer of patients.
    - Accuracy can be computed by comparing actual test set values and predicted values using metrics.accuracy_score() by importing the metrics module from scikit-learn
    - In addition, precision and recall of the model could be further evaluated with metrics.precision_score() and metrics.recall_score()


![alt text](image/model%20accuracy.png "Title Text")

More details:
https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python

