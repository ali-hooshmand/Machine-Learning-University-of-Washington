# coding=utf-8

# TASKS:================================================================================================================
# Predicting sentiment from product reviews
# The goal of this assignment is to explore logistic regression and feature engineering
# In this assignment, you will use product review data from Amazon.com to predict whether the sentiments about a product
#  (from its reviews) are positive or negative. You will:
# - Use SFrames to do some feature engineering
# - Train a logistic regression model to predict the sentiment of product reviews.
# - Inspect the weights (coefficients) of a trained logistic regression model.
# - Make a prediction (both class and probability) of sentiment for a new product review.
# - Given logistic regression weights, predictors & ground truth labels, write a function to compute accuracy of model.
# - Inspect the coefficients of the logistic regression model and interpret their meanings.
# - Compare multiple logistic regression models.
# ======================================================================================================================

import pandas as pd
import numpy as np
import string
import math
import time

start = time.time()


# ======================================================================================================================
# 1. Load the data set consisting of baby product reviews on Amazon.com.
products = pd.read_csv('./amazon_baby.csv')

# ======================================================================================================================
# Perform text cleaning
# 2. We start by removing punctuation, so that words "cake." and "cake!" are counted as the same word.
# Write a function remove_punctuation that strips punctuation from a line of text
# Apply this function to every element in review column of products, and save the result to a new column review_clean.
# Refer to your tool's manual for string processing capabilities. Python lets us express the operation in a succinct way
# as follows:

# IMPORTANT. Make sure to fill n/a values in the review column with empty strings (if applicable). The n/a values
# indicate empty reviews. For instance, Pandas's the fillna() method lets you replace all N/A's in the review columns
# as follows:

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
# lower are negative. For the sentiment column, we use +1 for the positive class label and -1 for the negative class
# label. A good way is to create an anonymous function that converts a rating into a class label and then apply that
# function to every element in the rating column.

products['sentiment'] = products['rating'].apply(lambda rating: -1 if rating <= 2 else 1)

# ======================================================================================================================
train_data_indices = pd.read_json('./module-2-assignment-train-idx.json')
test_data_indices = pd.read_json('./module-2-assignment-test-idx.json')

train_data = products.iloc[train_data_indices[0]]
test_data = products.iloc[test_data_indices[0]]

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
# - Build a sparse matrix where each row is word count vector for corresponding review. Call this matrix train_matrix.
# - Using the same mapping between words and columns, convert the test data into a sparse matrix test_matrix.

# The following cell uses CountVectorizer in scikit-learn. Notice the token_pattern argument in the constructor.

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
# We will now use logistic regression to create a sentiment classifier on the training data.
# 7. Learn a logistic regression classifier using the training data. If you are using scikit-learn, you should create an
#  instance of the LogisticRegression class and then call the method fit() to train the classifier. This model should
# use the sparse word count matrix (train_matrix) as features and the column sentiment of train_data as the target.
# Use the default values for other parameters. Call this model sentiment_model.

from sklearn import linear_model

sentiment_model = linear_model.LogisticRegression()

sentiment_model.fit(train_matrix, train_data['sentiment'])

# ======================================================================================================================
# 8. There should be over 100,000 coefficients in this sentiment_model. Recall from the lecture that positive weights
# w_j correspond to weights that cause positive sentiment, while negative weights correspond to negative sentiment.
# Calculate the number of positive (>= 0, which is actually nonnegative) coefficients.
# Quiz question:
# How many weights are >= 0?
number_of_positive_weights = 0
number_of_negative_weights = 0
for i in range(len(sentiment_model.coef_[0])):
    if sentiment_model.coef_[0][i] >= 0:
        number_of_positive_weights += 1
    else:
        number_of_negative_weights += 1

print 'number_of_positive_weights is ' + str(number_of_positive_weights)
print 'number_of_negative_weights is ' + str(number_of_negative_weights)
# Answer:
# number_of_positive_weights is 87241
# number_of_negative_weights is 34471

# ======================================================================================================================
# Making predictions with logistic regression
# 9. Now that a model is trained, we can make predictions on the test data. In this section, we will explore this in the
#  context of 3 data points in the test data. Take the 11th, 12th, and 13th data points in the test data and save them
# to sample_test_data. The following cell extracts three data points from the SFrame test_data and print their content:

sample_test_data = test_data[10:13]

# Let's dig deeper into the first row of the sample_test_data. Here's the full review:
print sample_test_data.iloc[0]['review']

# That review seems pretty positive.

# Now, let's see what the next row of the sample_test_data looks like. As we could guess from the rating (-1), the
# review is quite negative.
print sample_test_data.iloc[1]['review']

# ======================================================================================================================
# 10. We will now make a class prediction for the sample_test_data. The sentiment_model should predict +1 if the
# sentiment is positive and -1 if the sentiment is negative. Recall from the lecture that the score (sometimes called
# margin) for the logistic regression model is defined as:
# scorei=w⊺h(xi)
# where h(xi) represents the features for data point i. We will write some code to obtain the scores. For each row, the
# score (or margin) is a number in the range (-inf, inf). Use a pre-built function in your tool to calculate the score
# of each data point in sample_test_data. In scikit-learn, you can call the decision_function() function.
# Hint: You'd probably need to convert sample_test_data into the sparse matrix format first.
sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
scores = sentiment_model.decision_function(sample_test_matrix)
print scores
# Answer: [  5.60200766  -3.17012859 -10.42429792]

# ======================================================================================================================
# Prediciting Sentiment
# 11. These scores can be used to make class predictions as follows:
# y^i = {+1  if w⊺h(xi)>0
#        −1  if w⊺h(xi)≤0
# Using scores, write code to calculate predicted labels for sample_test_data.
# Checkpoint: Make sure your class predictions match with the ones obtained from sentiment_model. The logistic
# regression classifier in scikit-learn comes with the predict function for this purpose.
for i in range(len(scores)):
    if scores[i] > 0:
        print 1
    else:
        print -1
predictions = sentiment_model.predict(sample_test_matrix)
print predictions
# [ 1 -1 -1]

# ======================================================================================================================
# 12. Recall from the lectures that we can also calculate the probability predictions from the scores using:
# P(yi=+1|xi,w)=1/ 1+exp(−w⊺h(xi))
# Using the scores calculated previously, write code to calculate the probability that a sentiment is positive using
# the above formula. For each row, the probabilities should be a number in the range [0, 1].
# Checkpoint: Make sure your probability predictions match the ones obtained from sentiment_model.
print "The probability calculated by us: "
for i in range(len(scores)):
    print 1.0 / (1.0 + math.exp(-scores[i]))

print "The probability calculated by the model: "
print sentiment_model.predict_proba(sample_test_matrix)[:, 1]
# The probability calculated by us:
# 0.996323122214
# 0.0403054409916
# 2.97010659338e-05
# The probability calculated by the model:
# [  9.96323122e-01   4.03054410e-02   2.97010659e-05]

# ======================================================================================================================
# Quiz question:
# Of the three data points in sample_test_data, which one (first, second, or third) has the lowest probability of being
# classified as a positive review?
# Answer:
# Third review

# ======================================================================================================================
# Find the most positive (and negative) review
# 13. We now turn to examining the full test dataset, test_data, and use sklearn.linear_model.LogisticRegression to form
#  predictions on all of the test data points.
# Using the sentiment_model, find the 20 reviews in the entire test_data with the highest probability of being
# classified as a positive review. We refer to these as the "most positive reviews."
# To calculate these top-20 reviews, use the following steps:
# - Make probability predictions on test_data using the sentiment_model.
# - Sort the data according to those predictions and pick the top 20.
test_data_proba_predictions = sentiment_model.predict_proba(test_matrix)
test_data['probability'] = test_data_proba_predictions[:, 1]

test_data.sort_values('probability')

print test_data.sort_values('probability').tail(20)


# ======================================================================================================================
# Quiz Question:
# Which of the following products are represented in the 20 most positive reviews?

# ======================================================================================================================
# 14. Now, let us repeat this exercise to find the "most negative reviews." Use the prediction probabilities to find the
#  20 reviews in the test_data with the lowest probability of being classified as a positive review. Repeat the same
# steps above but make sure you sort in the opposite order.
print test_data.sort_values('probability').head(20)


# ======================================================================================================================
# Quiz Question:
# Which of the following products are represented in the 20 most negative reviews?

# ======================================================================================================================
# Compute accuracy of the classifier
# 15. We will now evaluate the accuracy of the trained classifier. Recall that the accuracy is given by
# accuracy= # correctly classified examples / # total examples
# This can be computed as follows:
# Step 1: Use the sentiment_model to compute class predictions.
# Step 2: Count the number of data points when the predicted class labels match the ground truth labels.
# Step 3: Divide the total number of correct predictions by the total number of data points in the dataset.
predicted_sentiment = sentiment_model.predict(test_matrix)
test_data['predicted_sentiment'] = predicted_sentiment
test_data['diff_sentiment'] = predicted_sentiment - test_data['sentiment']
no_of_correctly_classified = np.sum(sum(test_data['diff_sentiment'] == 0))
total = len(predicted_sentiment)
accuracy = no_of_correctly_classified*1.0/total
# for i in range(len(predictions)):
#    if predictions[i] == test_data['sentiment'][i]:
#        no_of_correctly_classified += 1

print "no_of_correctly_classified is " + str(no_of_correctly_classified)
# no_of_correctly_classified is 31079

print "total number of data points is " + str(total)
# total number of data points is 33336

print " accuracy is " + str(accuracy)

# Quiz Question:
# What is the accuracy of the sentiment_model on the test_data? Round your answer to 2 decimal places.
# Answer:
#  accuracy is 0.932295416367

# Quiz Question:
# Does a higher accuracy value on the training_data always imply that the classifier is better?
# Answer: No

# ======================================================================================================================
# Learn another classifier with fewer words
# 16. There were a lot of words in the model we trained above. We will now train a simpler logistic regression model
# using only a subet of words that occur in the reviews. For this assignment, we selected 20 words to work with.
# These are:
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves',
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed',
      'work', 'product', 'money', 'would', 'return']

# Compute a new set of word count vectors using only these words. The CountVectorizer class has a parameter that lets
# you limit the choice of words when building word count vectors:
vectorizer_word_subset = CountVectorizer(vocabulary=significant_words) # limit to 20 words
train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_data['review_clean'])
test_matrix_word_subset = vectorizer_word_subset.transform(test_data['review_clean'])

# ======================================================================================================================
# Train a logistic regression model on a subset of data
# 17. Now build a logistic regression classifier with train_matrix_word_subset as features and sentiment as the target.
# Call this model simple_model.
simple_model = linear_model.LogisticRegression()
simple_model.fit(train_matrix_word_subset, train_data['sentiment'])

# ======================================================================================================================
# 18. Let us inspect weights (coefficients) of simple_model. First, build a table to store (word, coefficient) pairs.
simple_model_coef_table = pd.DataFrame({'word':significant_words,
                                         'coefficient':simple_model.coef_.flatten()})

# Sort the data frame by the coefficient value in descending order.
sorted_simple_model_coef_table = simple_model_coef_table.sort_values('coefficient', ascending=False)

# ======================================================================================================================
# Quiz Question:
# Consider the coefficients of simple_model. How many of the 20 coefficients (corresponding to the 20 significant_words)
#  are positive for the simple_model?
print sorted_simple_model_coef_table

print 'number of positive coeffs is ' + str(sum(simple_model_coef_table['coefficient']>=0))
# number of positive coeffs is 10

# ======================================================================================================================
# Quiz Question:
# Are the positive words in the simple_model also positive words in the sentiment_model?

# ======================================================================================================================
# Comparing models
# 19. We will now compare the accuracy of the sentiment_model and the simple_model.
# First, compute the classification accuracy of the sentiment_model on the train_data.
predicted_sentiment_train = sentiment_model.predict(train_matrix)
train_data['predicted_sentiment'] = predicted_sentiment_train
train_data['diff_sentiment'] = predicted_sentiment_train - train_data['sentiment']
no_of_correctly_classified = np.sum(sum(train_data['diff_sentiment'] == 0))
total = len(predicted_sentiment_train)
accuracy_sentiment = no_of_correctly_classified*1.0/total
print 'the classification accuracy of the sentiment_model on the train_data is ' + str(accuracy_sentiment)
# the classification accuracy of the sentiment_model on the train_data is 0.96849703184

# Now, compute the classification accuracy of the simple_model on the train_data.
predicted_simple_train = simple_model.predict(train_matrix_word_subset)
train_data['predicted_simple'] = predicted_simple_train
train_data['diff_simple'] = predicted_simple_train - train_data['sentiment']
no_of_correctly_classified = np.sum(sum(train_data['diff_simple'] == 0))
total = len(predicted_simple_train)
accuracy_simple = no_of_correctly_classified*1.0/total
print 'the classification accuracy of the simple_model on the train_data is ' + str(accuracy_simple)
# the classification accuracy of the simple_model on the train_data is 0.866822570007

# ======================================================================================================================
# Quiz Question:
# Which model (sentiment_model or simple_model) has higher accuracy on the TRAINING set?
# Answer:
# sentiment_model

# ======================================================================================================================
# 20. we repeat this exercise on test_data.
# Start by computing classification accuracy of sentiment_model on test_data.
# From above it is 0.93
# Next, compute the classification accuracy of the simple_model on the test_data.
predicted_simple_test = simple_model.predict(test_matrix_word_subset)
test_data['predicted_simple'] = predicted_simple_test
test_data['diff_simple'] = predicted_simple_test - test_data['sentiment']
no_of_correctly_classified = np.sum(sum(test_data['diff_simple'] == 0))
total = len(predicted_simple_test)
accuracy_simple_test = no_of_correctly_classified*1.0/total
print 'the classification accuracy of the simple_model on the test_data is ' + str(accuracy_simple_test)
# the classification accuracy of the simple_model on the test_data is 0.869360451164

# ======================================================================================================================
# Quiz Question:
# Which model (sentiment_model or simple_model) has higher accuracy on the TEST set?
# sentiment_model

# ======================================================================================================================
# Baseline: Majority class prediction
# 21. It is quite common to use the majority class classifier as the a baseline (or reference) model for comparison with
#  your classifier model. The majority classifier model predicts the majority class for all data points.
# At the very least, you should healthily beat majority class classifier, otherwise, the model is (usually) pointless.

# Quiz Question:
# Enter the accuracy of majority class classifier model on the test_data. Round your answer to two decimal places
accuracy_of_majorityClassifier_on_test_data = round(float(sum(test_data['sentiment'] ==1))/len(test_data.index),2)
print accuracy_of_majorityClassifier_on_test_data
# 0.84

# Quiz Question:
# Is the sentiment_model definitely better than the majority class classifier (the baseline)?
# Yes

end = time.time()

print 'Time of Process was: ' + str(round((end - start), 2)) + '[sec]'
