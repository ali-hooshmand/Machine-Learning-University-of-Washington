# coding=utf-8

# TASKS:================================================================================================================
# Implementing logistic regression from scratch
# The goal of this assignment is to implement your own logistic regression classifier. You will:
# - Extract features from Amazon product reviews.
# - Convert an SFrame into a NumPy array.
# - Implement the link function for logistic regression.
# - Write a function to compute the derivative of the log likelihood function with respect to a single coefficient.
# - Implement gradient ascent.
# - Given a set of coefficients, predict sentiments.
# - Compute classification accuracy for the logistic regression model.
# ======================================================================================================================

import pandas as pd
import numpy as np
import string
import math
import time

start = time.time()


# ======================================================================================================================
# 1. Load the data set consisting of baby product reviews on Amazon.com.
products = pd.read_csv('./amazon_baby_subset.csv')
positives = sum(products['sentiment']==1)
print positives
negatives = sum(products['sentiment']==-1)
print negatives

# ======================================================================================================================
# Apply text cleaning on the review data
# 3. In this section, we will perform some simple feature cleaning using data frames. The last assignment used all words
#  in building bag-of-words features, but here we limit ourselves to 193 words (for simplicity). We compiled a list of
# 193 most frequent words into the JSON file named important_words.json. Load the words into a list important_words.

important_words = pd.read_json('./important_words.json')[0].tolist()

print important_words

# ======================================================================================================================
# 4. Let us perform 2 simple data transformations:
# - Remove punctuation
# - Compute word counts (only for important_words)
# We start with the first item as follows:
# If your tool supports it, fill n/a values in the review column with empty strings. The n/a values indicate empty
# reviews. For instance, Pandas's the fillna() method lets you replace all N/A's in the review columns as follows:

products = products.fillna({'review': ''})  # fill in N/A's in the review column

# Write a function remove_punctuation that takes a line of text and removes all punctuation from that text.
# The function should be analogous to the following Python code:


def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation)

# Apply the remove_punctuation function on every element of review column and assign result to new column review_clean.
products['review_clean'] = products['review'].apply(remove_punctuation)

# ======================================================================================================================
# 5. Now we proceed with the second item. For each word in important_words, we compute a count for the number of times
# the word occurs in the review. We will store this count in a separate column (one for each word). The result of this
# feature processing is a single column for each word in important_words which keeps a count of the number of times the
# respective word occurs in the review text.

# Note: There are several ways of doing this. One way is to create an anonymous function that counts the occurrence of
# a particular word and apply it to every element in the review_clean column. Repeat this step for every word in
# important_words. Your code should be analogous to the following:

for word in important_words:
    products[word] = products['review_clean'].apply(lambda s : s.split().count(word))

# ======================================================================================================================
# 6. After #4 and #5, the data frame products should contain one column for each of the 193 important_words.
# As an example, the column perfect contains a count of the number of times the word perfect occurs in each of reviews.
# 7. Now, write some code to compute the number of product reviews that contain the word perfect.
total_items = len(products.index)
print total_items
# 53072
products_without_word_perfect = sum(products['perfect'] == 0)
print products_without_word_perfect
# 50117
products_with_word_perfect = total_items - products_without_word_perfect
print products_with_word_perfect
# 2955

# Quiz Question.
# How many reviews contain the word perfect?
# 2955

# ======================================================================================================================
# Convert data frame to multi-dimensional array
# 8. It is now time to convert our data frame to a multi-dimensional array.
# Write a function that extracts columns from a data frame and converts them into a multi-dimensional array.
# The function should accept three parameters:
# - dataframe: a data frame to be converted
# - features: a list of string, containing the names of the columns that are used as features.
# - label: a string, containing the name of the single column that is used as class labels.
# The function should return two values:
# - one 2D array for features
# - one 1D array for class labels

# The function should do the following:
# - Prepend a new column constant to dataframe and fill it with 1's. This column takes account of the intercept term.
# Make sure that the constant column appears first in the data frame.
# - Prepend a string 'constant' to the list features. Make sure the string 'constant' appears first in the list.
# - Extract columns in dataframe whose names appear in the list features.
# - Convert the extracted columns into a 2D array using a function in the data frame library.
# If you are using Pandas, you would use as_matrix() function.
# - Extract the single column in dataframe whose name corresponds to the string label.
# - Convert the column into a 1D array.
# - Return the 2D array and the 1D array.


def get_numpy_data(dataframe, features, label):
    dataframe['constant'] = 1
    features = ['constant'] + features
    features_frame = dataframe[features]
    feature_matrix = features_frame.as_matrix()
    label_df = dataframe[label]
    label_array = label_df.as_matrix()
    return(feature_matrix, label_array)


# ======================================================================================================================
# 9. Using the function written in #8, extract two arrays feature_matrix and sentiment. The 2D array feature_matrix
# would contain the content of the columns given by the list important_words. The 1D array sentiment would contain the
# content of the column sentiment.
(feature_matrix, sentiment_array) = get_numpy_data(products, important_words, 'sentiment')

# Quiz Question: How many features are there in the feature_matrix?
print feature_matrix.shape[1]
# 194 (including intercept)

# Quiz Question: Assuming that the intercept is present, how does the number of features in feature_matrix relate to the
# number of features in the logistic regression model?
# it should be 193

# ======================================================================================================================
# Estimating conditional probability with link function
# 10. Recall from lecture that the link function is given by
# P(yi=+1|xi,w) = 1 / 1+exp(−w⊺h(xi)),
# where the feature vector h(xi) represents the word counts of important_words in the review xi. Write a function named
# predict_probability that implements the link function.
# - Take two parameters: feature_matrix and coefficients.
# - First compute the dot product of feature_matrix and coefficients.
# - Then compute the link function P(y=+1|x,w).
# - Return the predictions given by the link function.


def predict_probability(feature_matrix, coefficients):
    # Take dot product of feature_matrix and coefficients
    scores = np.dot(feature_matrix, coefficients)
    # Compute P(y_i = +1 | x_i, w) using the link function
    predictions = 1.0 / (1.0 + np.exp(scores))
    return predictions

# ======================================================================================================================
# Compute derivative of log likelihood with respect to a single coefficient
# 11. Recall from lecture:
# ∂ℓ/∂wj = ∑i=1toN hj(xi)( 1[yi=+1] − P(yi=+1|xi,w) )
# We will now write a function feature_derivative that computes the derivative of log likelihood with respect to a
# single coefficient w_j. The function accepts two arguments:
# - errors: vector whose i-th value contains
# 1[yi=+1] − P(yi=+1|xi,w)
# - feature: vector whose i-th value contains
#  hj(xi)
# This corresponds to the j-th column of feature_matrix.
# The function should do the following:
# Take two parameters errors and feature.
# Compute the dot product of errors and feature.
# Return the dot product. This is the derivative with respect to a single coefficient w_j.


def feature_derivative(errors, feature):
    derivative = np.dot(feature, errors)
    return derivative

# ======================================================================================================================
# 12. Due to its numerical stability, we will use the log-likelihood instead of the likelihood to assess the algorithm.
# The log-likelihood is computed using the following formula:
# ℓℓ(w)= ∑i=1toN ((1[yi=+1]−1)w⊺h(wi)−ln(1+exp(−w⊺h(xi))))
# Write a function compute_log_likelihood that implements the equation.


def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment == 1)
    scores = np.dot(feature_matrix, coefficients)
    lp = np.sum((indicator-1)*scores - np.log(1. + np.exp(-scores)))
    return lp

# ======================================================================================================================
# Taking gradient steps
# 13. Now we are ready to implement our own logistic regression. All we have to do is to write a gradient ascent
# function that takes gradient steps towards the optimum.
# Write a function logistic_regression to fit a logistic regression model using gradient ascent.
# The function accepts the following parameters:
# - feature_matrix: 2D array of features
# - sentiment: 1D array of class labels
# - initial_coefficients: 1D array containing initial values of coefficients
# - step_size: a parameter controlling the size of the gradient steps
# - max_iter: number of iterations to run gradient ascent
# The function returns the last set of coefficients after performing gradient ascent.

# The function carries out the following steps:
# 1- Initialize vector coefficients to initial_coefficients.
# 2- Predict class probability P(yi=+1|xi,w) using your predict_probability function & save it to variable predictions.
# 3- Compute indicator value for (yi=+1) by comparing sentiment against +1. Save it to variable indicator.
# 4- Compute the errors as difference between indicator and predictions. Save the errors to variable errors.
# 5- For each j-th coefficient, compute per-coefficient derivative by calling feature_derivative with the j-th column
#  of feature_matrix. Then increment the j-th coefficient by (step_size*derivative).
# 6- Once in a while, insert code to print out the log likelihood.
# 7- Repeat steps 2-6 for max_iter times.

from math import sqrt


def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    for itr in xrange(max_iter):
        # Predict P(y_i = +1|x_1,w) using your predict_probability() function
        predictions = predict_probability(feature_matrix, coefficients)

        # Compute indicator value for (y_i = +1)
        indicator = (sentiment == 1)

        # Compute the errors as indicator - predictions
        errors = indicator - predictions

        for j in xrange(len(coefficients)): # loop over each coefficient
            # Recall that feature_matrix[:,j] is the feature column associated with coefficients[j]
            # compute the derivative for coefficients[j]
            derivative = feature_derivative(errors, feature_matrix[:,j])

            # add the step size times the derivative to the current coefficient
            coefficients[j] = coefficients[j] + step_size * derivative

        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp)
    return coefficients

# ======================================================================================================================
# 14. Now, let us run the logistic regression solver with the parameters below:
# - feature_matrix = feature_matrix extracted in #9
feature_matrix
# - sentiment = sentiment extracted in #9
sentiment_array
# - initial_coefficients = a 194-dimensional vector filled with zeros
initial_coefficients = np.zeros(194)

step_size = 1e-7
max_iter = 301
# Save the returned coefficients to variable coefficients.

coefficients = logistic_regression(feature_matrix, sentiment_array, initial_coefficients, step_size, max_iter)
# iteration   0: log likelihood of observed labels = -36780.91768478
# iteration   1: log likelihood of observed labels = -36775.12821198
# ...
# iteration 300: log likelihood of observed labels = -35178.09158697

# ======================================================================================================================
# Quiz question:
# As each iteration of gradient ascent passes, does the log likelihood increase or decrease?
# Answer: Increase

# ======================================================================================================================
# Predicting sentiments
# 15. Recall from lecture that class predictions for a data point x can be computed from the coefficients w using the
# following formula:
# y^i={+1 if x⊺iw>0
#      −1 if x⊺iw≤0

# Now, we write some code to compute class predictions. We do this in two steps:
# First compute the scores using feature_matrix and coefficients using a dot product.
score_predictions = np.dot(feature_matrix, coefficients)

# Then apply threshold 0 on the scores to compute the class predictions. Refer to the formula above.
predictions = np.zeros(len(score_predictions))
predicted_sentiment = np.array([+1 if s > 0 else -1 for s in score_predictions])

# ======================================================================================================================
# Quiz question: How many reviews were predicted to have positive sentiment?
# positive_sentiment = np.sum(predicted_sentiment + np.ones(len(predicted_sentiment)))/2
positive_sentiment = sum(predicted_sentiment == +1)
print positive_sentiment
# 11492

# ======================================================================================================================
# Measuring accuracy
# 16. We will now measure the classification accuracy of the model. Recall from the lecture that the classification
# accuracy can be computed as follows:
# accuracy = # correctly classified data points / # total data points
errors = sentiment_array - predicted_sentiment
accuracy = float(sum(errors == 0))/len(predicted_sentiment)
print np.round(accuracy, decimals=2)
# ======================================================================================================================
# Quiz question:
# What is the accuracy of the model on predictions made above? (round to 2 digits of accuracy)
# Answer:
# 0.67

# ======================================================================================================================
# Which words contribute most to positive & negative sentiments
# 17. Recall that in the earlier assignment, we were able to compute the "most positive words".
# These are words that correspond most strongly with positive reviews. In order to do this, we will first do following:
# - Treat each coefficient as a tuple (word, coefficient_value). TIntercept has no corresponding word, so throw it out.
# - Sort all the (word, coefficient_value) tuples by coefficient_value in descending order.
# - Save the sorted list of tuples to word_coefficient_tuples.

coefficients = list(coefficients[1:]) # exclude intercept
word_coefficient_tuples = [(word, coefficient) for word, coefficient in zip(important_words, coefficients)]
word_coefficient_tuples = sorted(word_coefficient_tuples, key=lambda x:x[1], reverse=True)

# ======================================================================================================================
# Ten "most positive" words
# 18. Compute 10 words that have most positive coefficient values. These words are associated with positive sentiment.
# Quiz question: Which word is not present in the top 10 "most positive" words?
print word_coefficient_tuples[0:10]

# ======================================================================================================================
# Ten "most negative" words
# 19. Next, we repeat this exerciese on the 10 most negative words. That is, we compute the 10 words that have the most
# negative coefficient values. These words are associated with negative sentiment.
# Quiz question: Which word is not present in the top 10 "most negative" words?
word_coefficient_tuples = sorted(word_coefficient_tuples, key=lambda x:x[1], reverse=False)
print word_coefficient_tuples[0:10]




end = time.time()

print 'Time of Process was: ' + str(round((end - start), 2)) + '[sec]'
