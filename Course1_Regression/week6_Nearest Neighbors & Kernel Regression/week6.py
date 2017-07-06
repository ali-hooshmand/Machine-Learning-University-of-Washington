# coding=utf-8

# TASKS:================================================================================================================
# Predicting house prices using k-nearest neighbors regression
# In this notebook, you will implement k-nearest neighbors regression. You will:
# - Find the k-nearest neighbors of a given query input
# - Predict the output for the query input using the k-nearest neighbors
# - Choose the best value of k using a validation set
# ======================================================================================================================

import pandas as pd
import numpy as np
import time

start = time.time()


dtype_dict = {'bathrooms':float,
              'waterfront':int,
              'sqft_above':int,
              'sqft_living15':float,
              'grade':int, 'yr_renovated':int,
              'price':float,
              'bedrooms':float,
              'zipcode':str,
              'long':float,
              'sqft_lot15':float,
              'sqft_living':float,
              'floors':float,
              'condition':int,
              'lat':float,
              'date':str,
              'sqft_basement':int,
              'yr_built':int,
              'id':str,
              'sqft_lot':int,
              'view':int}

sales = pd.read_csv('./kc_house_data_small.csv', dtype = dtype_dict)
sales_train = pd.read_csv('./kc_house_data_small_train.csv', dtype = dtype_dict)
sales_test = pd.read_csv('./kc_house_data_small_test.csv', dtype = dtype_dict)
sales_valid = pd.read_csv('./kc_house_data_validation.csv', dtype = dtype_dict)

# ======================================================================================================================
# 3. To efficiently compute pairwise distances among data points, we will convert the SFrame (or dataframe) into a 2D
# Numpy array. First import the numpy library and then copy and paste get_numpy_data() (or equivalent). The function
# takes a dataset, a list of features (e.g. [‘sqft_living’, ‘bedrooms’]) to be used as inputs, and a name of the output
# (e.g. ‘price’). It returns a ‘features_matrix’ (2D array) consisting of a column of ones followed by columns
# containing the values of the input features in the data set in the same order as the input list. It also returns an
# ‘output_array’, which is an array of the values of the output in the dataset (e.g. ‘price’).


def get_numpy_data(data, features, output):

    data['constant'] = 1.  # add a constant column to an SFrame
    # prepend variable 'constant' to the features list
    features = ['constant'] + features
    # select the columns of data_SFrame given by the ‘features’ list into the SFrame ‘features_sframe’
    features_data = data[features]
    # this will convert the features_data into a numpy matrix
    features_matrix = np.asarray(features_data, dtype=float)
    # assign the column of data_sframe associated with the target to the variable ‘output_sarray’
    output_data = data[output]
    # this will convert the SArray into a numpy array:
    output_array = np.asarray(output_data)

    return(features_matrix, output_array)

# ======================================================================================================================
# 4. Similarly, copy and paste the normalize_features function (or equivalent) from Module 5 (Ridge Regression).
# Given a feature matrix, each column is divided (element-wise) by its 2-norm. The function returns two items:
# (i) a feature matrix with normalized columns and (ii) the norms of the original columns.


def normalize_features(feature_matrix):
    norms = np.sqrt(np.sum(feature_matrix ** 2, axis=0))
    # norms = np.linalg.norm(X, axis=0)
    normalized_features = feature_matrix / norms
    return (normalized_features, norms)

# ======================================================================================================================
# 5. Using get_numpy_data (or equivalent), extract numpy arrays of the training, test, and validation sets.
features = list(sales.keys())
features.remove('id')
features.remove('date')
features.remove('price')

(features_matrix_train, output_array_train) = get_numpy_data(sales_train, features, 'price')
(features_matrix_test, output_array_test) = get_numpy_data(sales_test, features, 'price')
(features_matrix_valid, output_array_valid) = get_numpy_data(sales_valid, features, 'price')

# ======================================================================================================================
# 6. In computing distances, it is crucial to normalize features. Otherwise, for example, the ‘sqft_living’ feature
# (typically on the order of thousands) would exert a much larger influence on distance than the ‘bedrooms’ feature
# (typically on the order of ones). We divide each column of the training feature matrix by its 2-norm, so that the
# transformed column has unit norm.

# IMPORTANT: Make sure to store the norms of the features in the training set. The features in the test and validation
# sets must be divided by these same norms, so that the training, test, & validation sets are normalized consistently.

(features_train_normalized, norms) = normalize_features(features_matrix_train)
features_test_normalized = features_matrix_test / norms
features_valid_normalized = features_matrix_valid / norms

# ======================================================================================================================
# Compute a single distance
# 7. To start, let's just explore computing the “distance” between two given houses. We will take our query house to be
# the first house of the test set and look at the distance between this house and the 10th house of the training set.
# To see the features associated with the query house, print the first row (index 0) of the test feature matrix.
# You should get an 18-dimensional vector whose components are between 0 and 1. Similarly, print the 10th row (index 9)
# of the training feature matrix.

print features_test_normalized[0]
print features_train_normalized[9]

# ======================================================================================================================
# 8. Quiz Question:
# What is the Euclidean distance between the query house and the 10th house of the training set?
distance = features_test_normalized[0] - features_train_normalized[9]
euclidean_distance = np.sqrt(np.dot(distance, distance))
print euclidean_distance
# Answer: 0.0597235937448

# ======================================================================================================================
# 9. Of course, to do nearest neighbor regression, we need to compute the distance between our query house and all
# houses in the training set.
# To visualize this nearest-neighbor search, let's first compute the distance from our query house (features_test[0])
# to the first 10 houses of the training set (features_train[0:10]) and then search for the nearest neighbor within
# this small set of houses. Through restricting ourselves to a small set of houses to begin with, we can visually scan
# the list of 10 distances to verify that our code for finding the nearest neighbor is working.
# Write a loop to compute the Euclidean distance from the query house to each of first 10 houses in the training set.
euclidean_distance_10 = np.zeros(10)
for i in range(0,10):
    distance = features_test_normalized[0] - features_train_normalized[i]
    euclidean_distance_10[i] = np.sqrt(np.dot(distance, distance))

# ======================================================================================================================
# 10. Quiz Question:
# Among the first 10 training houses, which house is the closest to the query house?
print str(np.argmin(euclidean_distance_10)) + 'th house is the closest ( Euclidean distance is ' \
      + str(round(np.min(euclidean_distance_10), 4)) + ')'
# Answer:
# 8th house is the closest ( Euclidean distance is 0.0524)

# ======================================================================================================================
# 11. It is computationally inefficient to loop over computing distances to all houses in our training dataset.
# Fortunately, many of the numpy functions can be vectorized, applying the same operation over multiple values or
# vectors. We now walk through this process.

# ======================================================================================================================
# Perform 1-nearest neighbor regression
# 12. Now that we have the element-wise differences, it is not too hard to compute the Euclidean distances between our
# query house and all of the training houses. First, write a single-line expression to define a variable ‘diff’ such
# that ‘diff[i]’ gives the element-wise difference between the features of the query house and the i-th training house.
diff = features_train_normalized - features_test_normalized[0]

# print diff[-1].sum()
# -0.0934371531122

# ======================================================================================================================
# 13. The next step in computing the Euclidean distances is to take these feature-by-feature differences in ‘diff’,
# square each, and take the sum over feature indices. That is, compute the sum of squared feature differences for each
# training house (row in ‘diff’).
# By default, ‘np.sum’ sums up everything in the matrix and returns a single number. To instead sum only over a row or
# column, we need to specifiy the ‘axis’ parameter described in the np.sum documentation. In particular, ‘axis=1’
# computes the sum across each row.
# 14. With this result in mind, write a single-line expression to compute the Euclidean distances from the query to all
# the instances. Assign the result to variable distances.
# Hint: don't forget to take the square root of the sum of squares.
euclidean_distance_trainData = np.sqrt(np.sum(diff**2, axis=1) )

# Hint: distances[100] should contain 0.0237082324496.
print euclidean_distance_trainData[100]
# 0.0237082381289

# ======================================================================================================================
# 15. Now you are ready to write a function that computes the distances from a query house to all training houses.
# The function should take two parameters: (i) the matrix of training features and (ii) the single feature vector
# associated with the query.


def compute_distances(features_instances, features_query):
    difference = features_instances - features_query
    distances = np.sqrt(np.sum(difference**2, axis=1))
    return distances

# ======================================================================================================================
# 16. Quiz Question:
# Take the query house to be third house of the test set (features_test[2]). What is the index of the house in the
# training set that is closest to this query house?
distances = compute_distances(features_train_normalized, features_test_normalized[2])
print np.argmin(distances)
# Answer: 382

# ======================================================================================================================
# 17. Quiz Question:
# What is the predicted value of the query house based on 1-nearest neighbor regression?
print 'The predicted value is $' + str(output_array_train[np.argmin(distances)])
# The predicted value is $249000.0

# ======================================================================================================================
# Perform k-nearest neighbor regression
# 18. Using the functions above, implement a function that takes in
# - the value of k;
# - the feature matrix for the instances; and
# - the feature of the query
# and returns the indices of the k closest training houses. For instance, with 2-nearest neighbor, a return value of
# [5, 10] would indicate that the 6th and 11th training houses are closest to the query house.


def k_nearest_neighbors(k, feature_train, features_query):
    distances = compute_distances(feature_train, features_query)
    sorted_indices_of_distances = np.argsort(distances)
    neighbors = sorted_indices_of_distances[0:k]
    return neighbors

# ======================================================================================================================
# 19. Quiz Question:
# Take the query house to be third house of the test set (features_test[2]). What are the indices of
# the 4 training houses closest to the query house?

neighbors = k_nearest_neighbors(4, features_train_normalized, features_test_normalized[2])
print neighbors
# [382 1149 4087 3142]

# ======================================================================================================================
# 20. Now that we know how to find the k-nearest neighbors, write a function that predicts the value of a given query
# house. For simplicity, take the average of the prices of the k nearest neighbors in the training set. The function
# should have the following parameters:
# - the value of k;
# - the feature matrix for the instances;
# - the output values (prices) of the instances; and
# - the feature of the query, whose price we’re predicting.


def predict_output_of_query(k, features_train, output_train, features_query):
    neighbors = k_nearest_neighbors(k, features_train, features_query)
    prediction = np.average(output_train[neighbors])
    return prediction

# ======================================================================================================================
# 21. Quiz Question:
# Again taking the query house to be third house of the test set (features_test[2]), predict the
# value of query house using k-nearest neighbors with k=4 and simple averaging method described and implemented above.
prediction = predict_output_of_query(4, features_train_normalized, output_array_train, features_test_normalized[2])
print '$' + str(prediction)
# $413987.5

# ======================================================================================================================
# 22. Finally, write a function to predict the value of each and every house in a query set. (The query set can be any
# subset of the dataset, be it the test set or validation set.) The idea is to have a loop where we take each house in
# the query set as the query house and make a prediction for that specific house. The new function should take the
# following parameters:
# - the value of k;
# - the feature matrix for the training set;
# - the output values (prices) of the training houses; and
# - the feature matrix for the query set.
# - The function should return a set of predicted values, one for each house in the query set.


def predict_output(k, features_train, output_train, features_query):
    predictions = np.zeros(len(features_query[:, 1]))
    for i in range(0, len(features_query[:, 1])):
        predictions[i] = predict_output_of_query(k, features_train, output_train, features_query[i])

    return predictions

# ======================================================================================================================
# 23. Quiz Question:
# Make predictions for the first 10 houses in the test set, using k=10. What is the index of the house in this query
# set that has the lowest predicted value? What is the predicted value of this house?

predictions = predict_output(10, features_train_normalized, output_array_train, features_test_normalized[range(0, 10)])

print 'house number ' + str(np.argmin(predictions)) + ' with predicted price equals to $'+str(np.min(predictions))
# Answer:
# house number 6 with predicted price equals to $350032.0

# ======================================================================================================================
# Choosing the best value of k using a validation set
# 24. There remains a question of choosing the value of k to use in making predictions. Here, we use a validation set
# to choose this value. Write a loop that does the following:
# For k in [1, 2, … 15]:
# - Make predictions for the VALIDATION data using the k-nearest neighbors from the TRAINING data.
# - Compute the RSS on VALIDATION data
RSS = np.zeros(15)
for k in range(1, 16):
    predictions = predict_output(k, features_train_normalized, output_array_train, features_valid_normalized)
    errors = output_array_valid - predictions
    RSS[k-1] = np.dot(errors, errors)

# Report which k produced the lowest RSS on validation data.
print 'k = ' + str(np.argmin(RSS)+1) + ' produced the lowest RSS on validation data.'
# k = 8 produced the lowest RSS on validation data.

# ======================================================================================================================
# 25. Quiz Question:
# What is the RSS on the TEST data using the value of k found above? To be clear, sum over all houses in the TEST set.
test_predictions = predict_output(8, features_train_normalized, output_array_train, features_test_normalized)
test_errors = output_array_test - test_predictions
RSS_testData = np.dot(test_errors, test_errors)

print 'RSS on the TEST data is ' + str(RSS_testData)
# RSS on the TEST data is 1.33118823552e+14



end = time.time()

print 'Time of Process was: ' + str(round((end - start), 2)) + '[sec]'
