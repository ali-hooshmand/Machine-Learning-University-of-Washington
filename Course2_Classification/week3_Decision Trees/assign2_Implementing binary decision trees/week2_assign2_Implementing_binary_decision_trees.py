# coding=utf-8

# TASKS:================================================================================================================
# Implementing binary decision trees:
# The goal of this notebook is to implement your own binary decision tree classifier. You will:
# Use SFrames to do some feature engineering.
# Transform categorical variables into binary variables.
# Write a function to compute the number of misclassified examples in an intermediate node.
# Write a function to find the best feature to split on.
# Build a binary decision tree from scratch.
# Make predictions using the decision tree.
# Evaluate the accuracy of the decision tree.
# Visualize the decision at the root node.
# Important Note: In this assignment, we will focus on building decision trees where the data contain only binary
# (0 or 1) features. This allows us to avoid dealing with:
# - Multiple intermediate nodes in a split
# - The thresholding issues of real-valued features.
# ======================================================================================================================

import pandas as pd
import numpy as np
import string
import math
import time

start = time.time()


# ======================================================================================================================
# 1. load data
loans = pd.read_csv('./lending-club-data.csv')

# ======================================================================================================================
# 2. reassign the labels to have +1 for a safe loan, and -1 for a risky (bad) loan.
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x == 0 else -1)
loans = loans.drop('bad_loans', 1)

# ======================================================================================================================
# 3. Unlike the previous assignment, we will only be considering these four features:
features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
            ]
target = 'safe_loans'

# Extract these feature columns from the dataset, and discard the rest of the feature columns.
loans = loans[features + [target]]
loans_one_hot_coded = pd.get_dummies(loans)
features = list(loans_one_hot_coded)
features.remove('safe_loans')
print features
# ======================================================================================================================
# Apply one-hot encoding to loans. Your tool may have a function for one-hot encoding.
# Load the JSON files into the lists train_idx and test_idx.
# Perform train/validation split using train_idx and test_idx. In Pandas, for instance:

'''
# function for one-hot encoding. Alternatively, see #7 for implementation hints.
categorical_features = []
for feat in features:
    if loans[feat].dtype == object:
        categorical_features.append(feat)

print categorical_features

def encode_target(df, target_column_list):
    """
    Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    for target_column in target_column_list:
        targets = df_mod[target_column].unique()
        map_to_int = {name: n for n, name in enumerate(targets)}
        df[target_column+'_code'] = df[target_column].replace(map_to_int)
        df_mod[target_column] = df_mod[target_column].replace(map_to_int)

    return (df, df_mod, targets)

# for feat in categorical_features:
df, df_mod, targets = encode_target(loans, categorical_features)
loans = df
loans_coded = df_mod
'''

train_idx = pd.read_json('./module-5-assignment-2-train-idx.json')
test_idx = pd.read_json('./module-5-assignment-2-test-idx.json')

train_data = loans_one_hot_coded.iloc[train_idx[0]]
test_data = loans_one_hot_coded.iloc[test_idx[0]]

'''
# ======================================================================================================================
# 4. Just as we did in the previous assignment, we will undersample the larger class (safe loans) in order to balance
# out our dataset. This means we are throwing away many data points. You should have code analogous to
safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

# Since there are less risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
safe_loans = safe_loans_raw.sample(frac=percentage)
risky_loans = risky_loans_raw
loans_data = risky_loans.append(safe_loans)

print "Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data))
print "Percentage of risky loans                :", len(risky_loans) / float(len(loans_data))
print "Total number of loans in our new dataset :", len(loans_data)
'''

# ======================================================================================================================
# Decision tree implementation
# In this section, we will implement binary decision trees from scratch. There are several steps involved in building a
# decision tree. For that reason, we have split the entire assignment into several sections.

# ======================================================================================================================
# Function to count number of mistakes while predicting majority class


def intermediate_node_num_mistakes(labels_in_node):
    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0
    # Count the number of 1's (safe loans)
    number_of_safe_loans = sum(labels_in_node == 1)

    # Count the number of -1's (risky loans)
    number_of_risky_loans = sum(labels_in_node == -1)

    # Return the number of mistakes that the majority classifier makes.
    if number_of_safe_loans >= number_of_risky_loans:
        majority_class = 1
        num_of_mistakes = number_of_risky_loans
    else:
        majority_class = -1
        num_of_mistakes = number_of_safe_loans
    return num_of_mistakes

# ======================================================================================================================
# Quiz question:
# What was the feature that my_decision_tree first split on while making the prediction for test_data[0]?


# Function to find the best feature for splitting:


def best_splitting_feature(data, features, target):
    best_feature = None
    best_error = 10
    for feat in features:
        left_split = data[data[feat] == 0]
        left_split_error = intermediate_node_num_mistakes(left_split[target])

        right_split = data[data[feat] == 1]
        right_split_error = intermediate_node_num_mistakes(right_split[target])

        feat_majority_class_error = float(left_split_error+right_split_error)/len(data[target])
        if feat_majority_class_error < best_error:
            best_error = feat_majority_class_error
            best_feature = feat

    return best_feature

print train_data.head()
# Test:
best_feature = best_splitting_feature(train_data, features, target)
print best_feature
if best_feature == 'term_ 36 months':
    print 'Test passed!'
else:
    print 'Test failed... try again!'

# ======================================================================================================================
# Building the tree

end = time.time()

print 'Time of Process was: ' + str(round((end - start), 2)) + '[sec]'
