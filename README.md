# Credit Risk Classification

Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. In this Challenge, you’ll use various techniques to train and evaluate models with imbalanced classes. You’ll use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.



# Credit Risk Analysis Report

## Overview of the Analysis

Differentiating creditworthy borrowers from others is important in the financial industry. Supervised learning is applied to predict such borrowers.

Borrowers' data columns are inclusive of loan_size, interest_rate, borrower_income, debt_to_income, num_of_accounts, derogatory_marks, total_debt, and loan_status. Loan_status, a binary value indicating whether a borrower's loan healthy or high-risky, is what we'd like to predict; therefore, it's defined as `target`. Other data columns are defined as `features`; they are used to train supervised learning model.

Basic stages of supervised learning model is as followed: model, fit, and predict. In stage model,  we need to determine which model would mathematically best represent the real world.  In this example, `LogisticRegression(random_state=1)` is called to make such model. Now, the model is untrained;  in stage fit, we fit ( or train ) the model with data, which is `features` and `target`. Once the model is fit, it makes predictions for new data.

To make convincing predictions, the resampling method `RandomOverSampler(random_state=1)` is applied to solve a common classification problem: imbalanced classes, which is when the size of one class greatly exceeding the other. 



## Results

* model using original data
  * balanced accuracy score: 0.952
  * precision for  the minority class: 0.85
  * recall for  the minority class:  0.91




* model using resampled data
  * balanced accuracy score:  0.993
  * precision for  the minority class:  0.84
  * recall for  the minority class:  0.99



## Summary

Regarding to balanced accuracy score, model using resampled data ( abbreviated as MURD ) outperforms model using original data  ( abbreviated as MUOD ) , 0.99 versus 0.95. In other words, MURD has better performance of recognizing true positive and true negatives. 

In terms of recall  for the minority class, MURD exceeds MUOD: 0.99 versus 0.91, meaning that MURD is better at correctly classifying data.

Concerning to precision for the minority class, however, MUOD has slightly higher score than MURD: 0.85 versus 0.84, which means MUOD is, to a small extent, better at detecting targets.

In sum, I would suggest the model using resampled data; it has much higher recall than MROD at the slight expense of precision.



## Technologies

This project leverages python 3.7 with the following packages:

* [imbalanced-learn](https://imbalanced-learn.org/stable/) -Imbalanced-learn (imported as `imblearn`) is an open source, MIT-licensed library relying on scikit-learn (imported as `sklearn`) and provides tools when dealing with classification with imbalanced classes.
* [PyDotPlus](https://pydotplus.readthedocs.io/) -PyDotPlus is an improved version of the old pydot project that provides a Python Interface to Graphviz’s Dot language.



## Installation Guide

Before running the application first install the following dependencies

```python
#Install imbalance-learn
conda activate dev
conda install -c conda-forge imbalanced-learn
onda list imbalanced-learn

#Install PyDotPlus
conda activate dev
conda install -c conda-forge pydotplus
conda list pydotplus
```



