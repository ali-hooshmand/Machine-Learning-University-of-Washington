# coding=utf-8

# TASKS:================================================================================================================
# Identifying safe loans with decision trees
# The LendingClub is a peer-to-peer leading company that directly connects borrowers and potential lenders/investors.
# In this notebook, you will build a classification model to predict whether or not a loan provided by LendingClub is
# likely to default.
# In this notebook you will use data from the LendingClub to predict whether a loan will be paid off in full or the loan
#  will be charged off and possibly go into default. In this assignment you will:
# - Use SFrames to do some feature engineering.
# - Train a decision-tree on the LendingClub dataset.
# - Visualize the tree.
# - Predict whether a loan will default along with prediction probabilities (on a validation set).
# - Train a complex tree model and compare it to simple tree model.
# ======================================================================================================================

import pandas as pd
import numpy as np
import string
import math
import time

start = time.time()


# ======================================================================================================================
# 1. Load the data set consisting of baby product reviews on Amazon.com.
loans = pd.read_csv('./lending-club-data.csv')

# ======================================================================================================================
# Exploring some features
# 2. Let's quickly explore what dataset looks like. First, print out column names to see what features we've in dataset.
# print loans.keys()

# ======================================================================================================================
# Exploring the target column
# The target column (label column) of the dataset that we are interested in is called bad_loans.
# In this column 1 means a risky (bad) loan 0 means a safe loan.
# In order to make this more intuitive and consistent with the lectures, we reassign the target to be:
# +1 as a safe loan
# -1 as a risky (bad) loan
# 3. We put this in a new column called safe_loans.
loans['safe_loans'] = loans['bad_loans'].apply(lambda l: +1 if l == 0 else -1)
loans = loans.drop('bad_loans', 1)

# ======================================================================================================================
# 4. Now, let us explore the distribution of the column safe_loans. This gives us a sense of how many safe and risky
# loans are present in the dataset. Print out the percentage of safe loans and risky loans in the data frame.
# You should have:
# - Around 81% safe loans
# - Around 19% risky loans
# It looks like most of loans are safe (thankfully). But this makes our problem of identifying risky loans challenging.
safe_loans = sum(loans['safe_loans'] == +1)
risky_loans = sum(loans['safe_loans'] == -1)

safe_loans_percentage = 100.0 * safe_loans / len(loans.index)
risky_loans_percentage = 100.0 * risky_loans / len(loans.index)

print np.round(safe_loans_percentage, decimals=1)
print np.round(risky_loans_percentage, decimals=1)
# 81.1
# 18.9

# ======================================================================================================================
# Features for the classification algorithm
# 5. In this assignment, we will be using a subset of features (categorical and numeric). The features we will be using
# are described in the code comments below. If you are a finance geek, the LendingClub website has a lot more details
# about these features. Extract feature columns and target column from the dataset. We will only use these features.

features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'                    # prediction target (y) (+1 means safe, -1 is risky)
# Extract the feature columns and target column
loans = loans[features + [target]]
#print loans['term'].dtype
# ======================================================================================================================
# - For converting categorical features values into binary values, Apply one-hot encoding to loans. Your tool may have a
# function for one-hot encoding. Alternatively, see #7 for implementation hints.
# - Load the JSON files into the lists train_idx and validation_idx.
# - Perform train/validation split using train_idx and validation_idx. In Pandas, for instance:
train_idx = pd.read_json('./module-5-assignment-1-train-idx.json')
validation_idx = pd.read_json('./module-5-assignment-1-validation-idx.json')

train_data = loans.iloc[train_idx[0]]
validation_data = loans.iloc[validation_idx[0]]

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
df, df_mod, targets = encode_target(train_data, categorical_features)
train_data = df
train_data_coded = df_mod

#print train_data.head(100)
# ======================================================================================================================
# Build a decision tree classifier

#import string
#def remove_punctuations(text):
#     return text.translate(None, string.punctuation)

#for feat in features:
#    if train_data[feat].dtype == object:
#        train_data[feat] = train_data[feat].apply(remove_punctuations)

# 1- Quiz Question: What percentage of the predictions on sample_validation_data did decision_tree_model get correct?
feature_matrix = np.array(train_data_coded[features])
target_array = np.array(train_data_coded['safe_loans'])

from sklearn import tree
dt = tree.DecisionTreeClassifier(max_depth=6)
dt.fit(feature_matrix, target_array)
# dt.fit(train_data[features], train_data['safe_loans'])

target = 'safe_loans'
validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]

sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
# Validation data should be one hot coded with the same code numbers of training data
for feat in categorical_features:
    unique_column = sample_validation_data[feat].unique()
    for i in range(len(unique_column.tolist())):
        related_code_index = next((j for j, v in enumerate(train_data[feat].tolist()) if v == unique_column.tolist()[i]), -123456)
        code = train_data[feat+'_code'].tolist()[related_code_index]
        obj = unique_column.tolist()[i]
        sample_validation_data[feat] = sample_validation_data[feat].replace(obj, code)

predictions = dt.predict(np.array(sample_validation_data[features]))
print 'predicted classes:'
print predictions
print 'real classes:'
print sample_validation_data['safe_loans']
correctPredictions = sum(predictions == np.array(sample_validation_data['safe_loans']))
total = len(predictions)
print correctPredictions
print total

# 2- Quiz Question: Which loan has the highest probability of being classified as a safe loan?
predict_proba = dt.predict_proba(sample_validation_data[features])
print predict_proba[:,1]
# 3- Checkpoint: Can you verify that for all the predictions with probability >= 0.5, the model predicted the label +1?
for i in range(len(predict_proba[:, 1])):
    if predict_proba[i, 1] >= 0.5:
        print predictions[i] - 1
    else:
        print predictions[i] - (-1)
# 4- Quiz Question: Notice that probability predictions are exact same for the 2nd and 3rd loans. Why would this happen?
small_dt = tree.DecisionTreeClassifier(max_depth=2)
small_dt.fit(feature_matrix, target_array)
small_dt_predictions = small_dt.predict_proba(sample_validation_data[features])
print small_dt_predictions

# (start) converting into the pdf file =================================================================================
with open("small_dt.dot", "w") as f:
    f = tree.export_graphviz(small_dt, out_file=f)
# The above code will convert the trained decision tree classifier into graphviz object and then store the contents into
#  the fsmall_dt.dot file. Next to convert the dot file into pdf file you can use the below command in terminal.
# $dot -Tpdf small_dt.dot -o small_dt.pdf
# To preview the created pdf file you can use the below command.
# $ open -a preview small_dt.pdf
# ( end ) converting into the pdf file =================================================================================

# 5- Quiz Question: Based on the visualized tree, what prediction would you make for this data point
# (according to small_model)? (If you don't have Graphviz, you can answer this quiz question by executing next part.)
# 6- Checkpoint: You should see that the small_model performs worse than the decision_tree_model on the training data.
# 7- Quiz Question: What is the accuracy of decision_tree_model on the validation set, rounded to the nearest .01?
# 8- Checkpoint: We should see that big_model has even better performance on training set than decision_tree_model did
# on the training set.
# 9- Quiz Question: How does the performance of big_model on the validation set compare to decision_tree_model on
# validation set? Is this a sign of overfitting?
big_dt = tree.DecisionTreeClassifier(max_depth=10)
big_dt.fit(feature_matrix, target_array)

for feat in categorical_features:
    unique_column = validation_data[feat].unique()
    for i in range(len(unique_column.tolist())):
        related_code_index = next((j for j, v in enumerate(train_data[feat].tolist()) if v == unique_column.tolist()[i]), -123456)
        code = train_data[feat+'_code'].tolist()[related_code_index]
        obj = unique_column.tolist()[i]
        validation_data[feat] = validation_data[feat].replace(obj, code)

big_dt_predictions = big_dt.predict(validation_data[features])
big_dt_correct_predictions = sum(big_dt_predictions == validation_data['safe_loans'])
total = len(big_dt_predictions)
big_dt_accuracy =  (1.0*big_dt_correct_predictions) / total

dt_predictions = dt.predict(validation_data[features])
dt_correct_predictions = sum(dt_predictions == validation_data['safe_loans'])
dt_accuracy = (1.0*dt_correct_predictions) / total

print 'big DT accuracy:'
print big_dt_accuracy
print 'DT accuracy:'
print dt_accuracy
# big DT accuracy:
# 0.628931495045
# DT accuracy:
# 0.630547177941
# Answer: Overfitting

# 10- Quiz Question: Let's assume each mistake costs us money: a false negative costs $10,000, while a false positive
# positive costs $20,000. What is the total cost of mistakes made by decision_tree_model on validation_data?




end = time.time()

print 'Time of Process was: ' + str(round((end - start), 2)) + '[sec]'
