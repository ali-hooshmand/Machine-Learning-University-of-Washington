# coding=utf-8

# TASKS:================================================================================================================
# Exploring Ensemble Methods
# In this homework we will explore the use of boosting. For this assignment, we will use the pre-implemented gradient
# boosted trees. You will:
# - Use Pandas to do some feature engineering.
# - Train a boosted ensemble of decision-trees (gradient boosted trees) on the lending club dataset.
# - Predict whether a loan will default along with prediction probabilities (on a validation set).
# - Evaluate the trained model and compare it with a baseline.
# - Find the most positive and negative loans using the learned model.
# - Explore how the number of trees influences classification performance.

# ======================================================================================================================

import pandas as pd
import numpy as np
import string
import math
from sklearn.ensemble import GradientBoostingClassifier
import time

start = time.time()


# ======================================================================================================================
# 1. load data
loans = pd.read_csv('./lending-club-data.csv')

# ======================================================================================================================
# 2. we re-assign the target to have +1 as a safe (good) loan, and -1 as a risky (bad) loan.
# Next, we select four categorical features:
# - grade of the loan
# - the length of the loan term
# - the home ownership status: own, mortgage, rent
# - number of years of employment.

features = ['grade',                     # grade of the loan (categorical)
            'sub_grade_num',             # sub-grade of the loan as a number from 0 to 1
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'payment_inc_ratio',         # ratio of the monthly payment to income
            'delinq_2yrs',               # number of delinquincies
             'delinq_2yrs_zero',          # no delinquincies in last 2 years
            'inq_last_6mths',            # number of creditor inquiries in last 6 months
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'open_acc',                  # number of open credit accounts
            'pub_rec',                   # number of derogatory public records
            'pub_rec_zero',              # no derogatory public records
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
            'int_rate',                  # interest rate of the loan
            'total_rec_int',             # interest received to date
            'annual_inc',                # annual income of borrower
            'funded_amnt',               # amount committed to the loan
            'funded_amnt_inv',           # amount committed by investors for the loan
            'installment',               # monthly payment owed by the borrower
           ]
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans.drop('bad_loans', axis=1, inplace=True)
target = 'safe_loans'
loans = loans[features + [target]]

# ======================================================================================================================
# Skipping observations with missing values
# Recall that one common approach to coping with missing values is to skip observations that contain missing values.
print len(loans.index)
# loans.dropna(axis=0, how='any', inplace=True)
# loans.dropna(inplace=True)
# print len(loans.index)

# Then follow the following steps:
# - Apply one-hot encoding to loans.
# - Load the JSON files into the lists train_idx and test_idx.
# - Perform train/validation split using train_idx and test_idx.
loans_one_hot_coded = pd.get_dummies(loans)
features = list(loans_one_hot_coded)
features.remove('safe_loans')

train_idx = pd.read_json('./module-8-assignment-2-train-idx.json')
valid_idx = pd.read_json('./module-8-assignment-2-test-idx.json')

train_data = loans_one_hot_coded.iloc[train_idx[0]]
# train_data = loans.iloc[train_idx[0]]
train_data.dropna(axis=0, how='any', inplace=True)
validation_data = loans_one_hot_coded.iloc[valid_idx[0]]
# validation_data = loans.iloc[valid_idx[0]]
validation_data.dropna(axis=0, how='any', inplace=True)

# ======================================================================================================================
# Gradient boosted tree classifier
# Gradient boosted trees are a powerful variant of boosting methods; they have been used to win many Kaggle competitions
# , and have been widely used in industry. We will explore the predictive power of multiple decision trees as opposed to
#  a single decision tree.
# We will now train models to predict safe_loans using the features above. In this section, we will experiment with
# training an ensemble of 5 trees.
# 9. Now, let's use the built-in scikit learn gradient boosting classifier (sklearn.ensemble.GradientBoostingClassifier)
# to create a gradient boosted classifier on raining data. You will need to import sklearn, sklearn.ensemble, and numpy.

# You will have to first convert dataFrame into a numpy data matrix. You will also have to extract the label column.
# Make sure to set max_depth=6 and n_estimators=5.
dt = GradientBoostingClassifier(n_estimators=5, max_depth=6)

train_matrix = train_data[features].values # np.asarray(train_data)
train_target = train_data[target].values

dt.fit(train_matrix, train_target)

# ======================================================================================================================
# Making predictions
# Just like we did in previous sections, let us consider a few positive and negative examples from the validation set.
# We will do the following:
# Predict whether or not a loan is likely to default.
# Predict the probability with which the loan is likely to default.
# 10. First, let's grab 2 positive examples and 2 negative examples.
validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)

# ======================================================================================================================
# 11. For each row in the sample_validation_data, write code to make model_5 predict whether or not the loan is
# classified as a safe loan.

# Quiz question:
# What percentage of the predictions on sample_validation_data did model_5 get correct?

predictions = dt.predict(sample_validation_data[features].values)
print predictions
# [ 1 -1 -1  1]
print dt.score(validation_data[features].values, validation_data[target].values) # accuracy
# ======================================================================================================================




end = time.time()

print 'Time of Process was: ' + str(round((end - start), 2)) + '[sec]'
