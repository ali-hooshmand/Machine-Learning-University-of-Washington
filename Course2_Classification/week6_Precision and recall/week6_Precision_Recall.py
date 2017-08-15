# coding=utf-8

# TASKS:================================================================================================================
# Exploring precision and recall
# The goal of this assignment is to understand precision-recall in the context of classifiers.
# - Use Amazon review data in its entirety.
# - Train a logistic regression model.
# - Explore various evaluation metrics: accuracy, confusion matrix, precision, recall.
# - Explore how various metrics can be combined to produce a cost of making an error.
# - Explore precision and recall curves.
# ======================================================================================================================

import pandas as pd
import numpy as np
import string
import math
import time
import matplotlib.pyplot as plt

start = time.time()


# ======================================================================================================================
# 1. load data
products = pd.read_csv('./amazon_baby.csv')

# ======================================================================================================================
# 2. Perform text cleaning
# We start by removing punctuation, so that words "cake." and "cake!" are counted as the same word.
# - Write a function remove_punctuation that strips punctuation from a line of text
# - Apply this function to every element in the review column of products, and save result to a new column review_clean.

products = products.fillna({'review': ''})  # fill in N/A's in the review column


def remove_punctuation(text):
    return text.translate(None, string.punctuation)

products['review_clean'] = products['review'].apply(remove_punctuation)

# ======================================================================================================================
# Extract Sentiments
# 3. We will ignore all reviews with rating = 3, since they tend to have a neutral sentiment.
products = products[products['rating'] != 3]

# ======================================================================================================================
# 4. Now, we will assign reviews with a rating of 4 or higher to be positive reviews, while the ones with rating of 2 or
#  lower are negative. For the sentiment column, we use +1 for the positive class label and -1 for the negative class
# label. A good way is to create an anonymous function that converts a rating into a class label and then apply that
# function to every element in the rating column.
products['sentiment'] = products['rating'].apply(lambda rating: +1 if rating > 3 else -1)

# ======================================================================================================================
train_idx = pd.read_json('./module-9-assignment-train-idx.json')
test_idx = pd.read_json('./module-9-assignment-test-idx.json')

train_data = products.iloc[train_idx[0]]
test_data = products.iloc[test_idx[0]]

# ======================================================================================================================
# Build the word count vector for each review
# 6. We will now compute the word count for each word that appears in the reviews. A vector consisting of word counts is
#  often referred to as bag-of-word features. Since most words occur in only a few reviews, word count vectors are
# sparse. For this reason, scikit-learn and many other tools use sparse matrices to store a collection of word count
# vectors. Refer to appropriate manuals to produce sparse word count vectors. General steps for extracting word count
# vectors are as follows:
# - Learn a vocabulary (set of all words) from the training data. Only the words that show up in the training data will
# be considered for feature extraction.
# - Compute the occurrences of the words in each review and collect them into a row vector.
# - Build a sparse matrix where each row is word count vector for the corresponding review. Call this it train_matrix.
# - Using the same mapping between words and columns, convert the test data into a sparse matrix test_matrix.

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
# Use this token pattern to keep single-letter words
# First, learn vocabulary from the training data and assign columns to words
# Then convert the training data into a sparse matrix
train_matrix = vectorizer.fit_transform(train_data['review_clean'])
# Second, convert the test data into a sparse matrix, using the same word-column mapping
test_matrix = vectorizer.transform(test_data['review_clean'])


# ======================================================================================================================
# Train a sentiment classifier with logistic regression
# 7. Learn a logistic regression classifier using the training data. If you are using scikit-learn, you should create an
#  instance of the LogisticRegression class and then call the method fit() to train the classifier. This model should
# use the sparse word count matrix (train_matrix) as features and the column sentiment of train_data as the target. Use
# the default values for other parameters. Call this model model.
from sklearn import linear_model
model = linear_model.LogisticRegression()
model.fit(train_matrix, train_data['sentiment'])

# ======================================================================================================================
# Model Evaluation
# We will explore the advanced model evaluation concepts that were discussed in the lectures.

# ======================================================================================================================
# Accuracy
# 8. One performance metric we will use for our more advanced exploration is accuracy, which we have seen many times in
# past assignments. Recall that the accuracy is given by

# accuracy = # correctly classified data points / # total data points

# Compute the accuracy on the test set using your tool of choice.

accuracy = model.score(test_matrix, test_data['sentiment'])
print accuracy
# 0.93

# ======================================================================================================================
# Baseline: Majority class prediction
# 9. Recall from an earlier assignment that we used the majority class classifier as a baseline (i.e reference) model
# for a point of comparison with a more sophisticated classifier. The majority classifier model predicts the majority
# class for all data points.
# Typically, a good model should beat the majority class classifier. Since the majority class in this dataset is the
# positive class (i.e., there are more positive than negative reviews), the accuracy of the majority class classifier
# is simply the fraction of positive reviews in the test set:
baseline = float(len(test_data[test_data['sentiment'] == 1])) / len(test_data)
print "Baseline accuracy (majority class classifier): %s" % baseline
# Baseline accuracy (majority class classifier): 0.84

# ======================================================================================================================
# Quiz question:
# Using accuracy as evaluation metric, was logistic regression model better than baseline (majority class classifier)?
# Answer: yes

# ======================================================================================================================
# Confusion Matrix
# 10. The accuracy, while convenient, does not tell the whole story. For a fuller picture, we turn to the confusion
# matrix. In the case of binary classification, the confusion matrix is a 2-by-2 matrix laying out correct and incorrect
#  predictions made in each label as follows:
# Using your tool, print out the confusion matrix for a classifier. For instance, scikit-learn provides the method
# confusion_matrix for this purpose:

from sklearn.metrics import confusion_matrix
cmat = confusion_matrix(y_true=test_data['sentiment'].values,
                        y_pred=model.predict(test_matrix),
                        labels=model.classes_)    # use the same order of class as the LR model.
print ' predicted_label | target_label | count '
print '--------------+-----------------+-------'
# Print out the confusion matrix.
# NOTE: Your tool may arrange entries in a different order. Consult appropriate manuals.
for i, target_label in enumerate(model.classes_):
    for j, predicted_label in enumerate(model.classes_):
        print '{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, cmat[j,i])

# predicted_label |   target_label  | count
# ----------------+-----------------+-------
#        -1       |       -1        |  3788
#        -1       |        1        |   804
#         1       |       -1        |  1453
#         1       |        1        | 27291

# ======================================================================================================================
# Quiz Question:
# How many predicted values in the test set are false positives?
# Answer: 1453

# ======================================================================================================================
# Computing the cost of mistakes
# 11. Put yourself in the shoes of a manufacturer that sells a baby product on Amazon.com and you want to monitor your
# product's reviews in order to respond to complaints. Even a few negative reviews may generate a lot of bad publicity
# about the product. So you don't want to miss any reviews with negative sentiments --- you'd rather put up with false
# alarms about potentially negative reviews instead of missing negative reviews entirely. In other words, false
# positives cost more than false negatives. (It may be the other way around for other scenarios, but let's stick with
# the manufacturer's scenario for now.)
# Suppose you know the costs involved in each kind of mistake:
# $100 for each false positive.
# $1 for each false negative.
# Correctly classified reviews incur no cost.

# Quiz Question:
# Given the stipulation, what is the cost associated with the logistic regression classifier's
# performance on the test set?
cost_of_mistakes = 100 * 1453 + 1 * 804
print cost_of_mistakes
# 146104

# ======================================================================================================================
# Precision and Recall
# 12. You may not have exact dollar amounts for each kind of mistake. Instead, you may simply prefer to reduce the
# percentage of false positives be less than, say, 3.5% of all positive predictions. This is where precision comes in:

# [precision]=[# positive data points with positive predictions]/[# all data points with positive predictions]
#            =[# true positives]/[# true positives]+[# false positives]

# So to keep the percentage of false positives below 3.5% of positive predictions, we must raise precision >= 96.5% .
# First, let us compute the precision of the logistic regression classifier on the test_data.
# Scikit-learn provides a predefined method for computing precision.

from sklearn.metrics import precision_score
precision = precision_score(y_true=test_data['sentiment'].values,
                            y_pred=model.predict(test_matrix))
print "Precision on test data: %s" % precision
# Precision on test data: 0.95

# ======================================================================================================================
# Quiz Question:
# Out of all reviews in the test set that are predicted to be positive, what fraction of them are false positives?
print 1 - precision
# Answer: 0.05

# ======================================================================================================================
# Quiz Question:
# Based on what we learned in lecture, if we wanted to reduce fraction of false positives to be below 3.5%, we would:
# Answer: increasing the threshold (it is used to decide if a probability score is +1 sentiment or -1)

# ======================================================================================================================
# 13. A complementary metric is recall, which measures the ratio between the number of true positives and that of
# (ground-truth) positive reviews:

# [recall]=[# positive data points with positive predicitions]/[# all positive data points]
#         =[# true positives]/ ([# true positives]+[# false negatives])

# Let us compute the recall on the test_data. Scikit-learn provides a predefined method for computing recall as well.
from sklearn.metrics import recall_score
recall = recall_score(y_true=test_data['sentiment'].values,
                      y_pred=model.predict(test_matrix))
print "Recall on test data: %s" % round(recall,2)
# Recall on test data: 0.97

# ======================================================================================================================
# Quiz Question:
# What fraction of the positive reviews in the test_set were correctly predicted as positive by the classifier?
# 0.97

# Quiz Question:
# What is the recall value for a classifier that predicts +1 for all data points in the test_data?
# answer: 1

# ======================================================================================================================
# Precision-recall trade-off
# In this part, we will explore the trade-off between precision and recall discussed in the lecture.
# We first examine what happens when we use a different threshold value for making class predictions.
# We then explore a range of threshold values and plot the associated precision-recall curve.

# ======================================================================================================================
# Varying the threshold
# 14. False positives are costly in our example, so we want to be more conservative about making positive predictions.
# To achieve this, instead of thresholding class probabilities at 0.5, we can choose a higher threshold.
# Write a function called apply_threshold that accepts two things
# - probabilities: an SArray of probability values
# - threshold: a float between 0 and 1
# The function should return an array, where each element is set to +1 or -1 depending whether the corresponding
# probability exceeds threshold.


# def apply_threshold(probabilities, threshold):
#    sentiments = np.zeros(len(probabilities))
#    for i in range(len(probabilities)):
#        if probabilities[i] >= threshold:
#            sentiments[i] = 1
#        else:
#            sentiments[i] = -1
#    return sentiments


def apply_threshold(probabilities, threshold):
    sentiments = np.array([+1 if p >= threshold else -1 for p in probabilities])
    return sentiments

# ======================================================================================================================
# 15. Using the model you trained, compute the class probability values P(y=+1|x,w) for the data points in test_data.
# Then use thresholds set at 0.5 (default) and 0.9 to make predictions from these probability values.

# Note. If you are using scikit-learn, make sure to use predict_proba() function, not decision_function().
# Also, note that the predict_proba() function returns the probability values for both classes +1 and -1.
# So make sure to extract the second column, which correspond to the class +1.
probabilities = model.predict_proba(test_matrix)[:,1]

sentiments_5 = apply_threshold(probabilities, 0.5)
print sum(sentiments_5)
# 24152
sentiments_9 = apply_threshold(probabilities, 0.9)
print sum(sentiments_9)
# 16806
# ======================================================================================================================
# Quiz question:
# What happens to the number of positive predicted reviews as the threshold increased from 0.5 to 0.9?
# Answer: number of positive predicted reviews reduced from 24152 to 16806

# ======================================================================================================================
# Precision-recall curve
# 17. Now, we will explore various different values of tresholds, compute the precision and recall scores, and
# then plot the precision-recall curve. Use 100 equally spaced values between 0.5 and 1. In Python, we run
threshold_values = np.linspace(0.5, 1, num=100)
# print threshold_values

# For each of the values of threshold, we first obtain class predictions using that threshold and then compute the
# precision & recall scores. Save precision scores & recall scores to lists precision_all and recall_all, respectively.
precision_all = []
recall_all = []
for threshold in threshold_values:
    sentiments = apply_threshold(probabilities, threshold)

    precision = precision_score(y_true=test_data['sentiment'].values,
                                y_pred=sentiments)
    precision_all.append(precision)

    recall = recall_score(y_true=test_data['sentiment'].values,
                          y_pred=sentiments)
    recall_all.append(recall)

# ======================================================================================================================
# 18. Let's plot the precision-recall curve to visualize the precision-recall tradeoff as we vary the threshold.
# Implement the function plot_pr_curve that generates a connected scatter plot from the lists of precision and recall
# scores. The function would be implemented in matplotlib as follows; for other tools, consult appropriate manuals.
plt.interactive(False)

def plot_pr_curve(precision, recall, title):
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis = 'x', nbins = 5)
    plt.plot(precision, recall, 'b-', linewidth=4.0, color = '#B0017F')
    plt.title(title)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.rcParams.update({'font.size': 16})


plot_pr_curve(precision_all, recall_all, 'Precision recall curve (all)')

plt.show(block=True)

# ======================================================================================================================
# Quiz Question:
# Among all the threshold values tried, what is the smallest threshold value that achieves a precision
# of 96.5% or better? Round your answer to 3 decimal places.
for i in range(len(threshold_values)):
    print str(threshold_values[i]) + ', ' + str(precision_all[i]) + ', ' + str(recall_all[i])

# Answer: 0.7070707

# ======================================================================================================================
# Quiz Question:
# Using threshold = 0.98, how many false negatives do we get on the test_data? This is the number of false negatives
# (i.e the number of reviews to look at when not needed) that we have to deal with using this classifier.
sentiments_98 = apply_threshold(probabilities, 0.98)
test_sentiments = np.asarray(test_data['sentiment'])
print len(sentiments_98)
print len(test_data)
false_negatives = 0

for i in range(len(sentiments_98)):
    if sentiments_98[i] == -1 and test_sentiments[i]==1:
        false_negatives += 1

print 'false_negatives is ' + str(false_negatives)
# false_negatives is 8208

# ======================================================================================================================
# Evaluating specific search terms
# So far, we looked at the number of false positives for the entire test set. In this section, let's select reviews
# using a specific search term and optimize the precision on these reviews only. After all, a manufacturer would be
# interested in tuning the false positive rate just for their products (the reviews they want to read) rather than that
# of the entire set of products on Amazon.
# ======================================================================================================================
# Precision-Recall on all baby related items
# 20. From the test set, select all the reviews for all products with the word 'baby' in them. If you are using SFrame,
# generate a binary mask with apply() and index test_data with the mask.

baby_reviews = test_data[test_data['name'].apply(lambda x: 'baby' in str(x).lower())]

# ======================================================================================================================
# 21. Now, let's predict the probability of classifying these reviews as positive. Make sure to convert the column
# review_clean of baby_reviews into a 2D array before computing class probability values. In scikit-learn, this task
# would be implemented as follows:

baby_matrix = vectorizer.transform(baby_reviews['review_clean'])
probabilities = model.predict_proba(baby_matrix)[:,1]

# ======================================================================================================================
# 22. Let's plot the precision-recall curve for the baby_reviews dataset. We again use 100 equally spaced values between
#  0.5 and 1 for the threshold.

threshold_values = np.linspace(0.5, 1, num=100)

# ======================================================================================================================
# For each of the values of threshold, we first obtain class predictions for baby_reviews using that threshold.
# Then we compute the precision and recall scores for baby_reviews. Save the precision scores and recall scores to
# lists precision_all and recall_all, respectively.
precision_all = []
recall_all = []
for threshold in threshold_values:
    sentiments = apply_threshold(probabilities, threshold)

    precision = precision_score(y_true=baby_reviews['sentiment'].values,
                                y_pred=sentiments)
    precision_all.append(precision)

    recall = recall_score(y_true=baby_reviews['sentiment'].values,
                          y_pred=sentiments)
    recall_all.append(recall)

# ======================================================================================================================
# Quiz Question:
# Among all the threshold values tried, what is the smallest threshold value that achieves a precision of 96.5% or
# better for the reviews of data in baby_reviews? Round your answer to 3 decimal places.
for i in range(len(threshold_values)):
    print str(threshold_values[i]) + ', ' + str(precision_all[i]) + ', ' + str(recall_all[i])

# Answer: 737373737374

# Quiz Question:
# Is this threshold value smaller or larger than the threshold used for the entire dataset to achieve the same specified
#  precision of 96.5%?
# Answer: larger

# ======================================================================================================================
# 23. Plot the precision-recall curve for baby_reviews only by running
plot_pr_curve(precision_all, recall_all, "Precision-Recall (Baby)")















end = time.time()

print 'Time of Process was: ' + str(round((end - start), 2)) + '[sec]'
