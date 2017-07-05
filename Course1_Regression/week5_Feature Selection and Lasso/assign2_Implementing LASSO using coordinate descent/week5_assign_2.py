# coding=utf-8

# TASKS:================================================================================================================
# Regression Week 5: LASSO Assignment 2
# In this notebook, you will implement your very own LASSO solver via coordinate descent. You will:
# - Write a function to normalize features
# - Implement coordinate descent for LASSO
# - Explore effects of L1 penalty
# ======================================================================================================================

import pandas as pd
import numpy as np
import time
import math

start = time.time()

# ======================================================================================================================
# 1. Load the sales dataset using Pandas:

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int,
              'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float,
              'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int,
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('./kc_house_data.csv', dtype=dtype_dict)

# ======================================================================================================================
# 3. Next, from Module 2 (Multiple Regression), copy and paste the ‘get_numpy_data’ function (or equivalent) that takes
# a data set, a list of features (e.g. [‘sqft_living’, ‘bedrooms’]), to be used as inputs, and a name of the output
# (e.g. ‘price’). This function returns a ‘feature_matrix’ (2D array) consisting of first a column of ones followed by
# columns containing the values of the input features in the data set in the same order as the input list. It also
# returns an ‘output_array’ which is an array of the values of the output in the data set (e.g. ‘price’).


def get_numpy_data(data, features, output):

    data['constant'] = 1  # add a constant column to an SFrame
    # prepend variable 'constant' to the features list
    features = ['constant'] + features
    # select the columns of data_SFrame given by the ‘features’ list into the SFrame ‘features_sframe’
    features_data = data[features]
    # this will convert the features_data into a numpy matrix
    features_matrix = np.asarray(features_data)
    # assign the column of data_sframe associated with the target to the variable ‘output_sarray’
    output_data = data[output]
    # this will convert the SArray into a numpy array:
    output_array = np.asarray(output_data)

    return(features_matrix, output_array)

# ======================================================================================================================
# 4. Similarly, copy and paste the ‘predict_output’ function (or equivalent) from Module 2. This function accepts a
# 2D array ‘feature_matrix’ and a 1D array ‘weights’ and return a 1D array ‘predictions’.


def predict_outcome(feature_matrix, weights):
    predictions = feature_matrix.dot(weights)
    return(predictions)

# ======================================================================================================================
# 5. In the house dataset, features vary wildly in their relative magnitude: ‘sqft_living’ is very large overall
# compared to ‘bedrooms’, for instance. As a result, weight for ‘sqft_living’ would be much smaller than weight for
# ‘bedrooms’. This is problematic because “small” weights are dropped first as l1_penalty goes up.

# To give equal considerations for all features, we need to normalize features as discussed in the lectures: we divide
# each feature by its 2-norm so that the transformed feature has norm 1.

# 6. Write a short function called ‘normalize_features(feature_matrix)’, which normalizes columns of a given feature
# matrix. The function should return a pair ‘(normalized_features, norms)’, where the second item contains the norms
# of original features. As discussed in the lectures, we will use these norms to normalize the test data in the same way
#  as we normalized the training data.


def normalize_features(feature_matrix):
    norms = np.sqrt(np.sum(feature_matrix ** 2, axis=0))
    # norms = np.linalg.norm(X, axis=0)
    normalized_features = feature_matrix / norms
    return (normalized_features, norms)

# ======================================================================================================================
# 7 & 8. Review of Coordinate Descent

# ======================================================================================================================
# Effect of L1 penalty
# 9. Consider a simple model with 2 features: ‘sqft_living’ and ‘bedrooms’. The output is ‘price’.
# - First, run get_numpy_data() (or equivalent) to obtain a feature matrix with 3 columns (constant column added).
# Use the entire ‘sales’ dataset for now.
features = ['sqft_living', 'bedrooms']
(features_matrix, output_array) = get_numpy_data(sales, features, 'price')
# - Normalize columns of the feature matrix. Save the norms of original features as ‘norms’.
(normalized_features, norms) = normalize_features(features_matrix)
# - Set initial weights to [1,4,1].
weights = np.array([1, 4, 1])
# - Make predictions with feature matrix and initial weights.
predictions = predict_outcome(normalized_features, weights)
# - Compute values of ro[i], where
# ro[i] = SUM[ [feature_i]*(output - prediction + w[i]*[feature_i]) ]
ro = np.zeros(3)
for i in range(0, 3):
    ro[i] = np.sum(normalized_features[:, i] * (output_array - predictions + weights[i] * normalized_features[:, i]))

print np.round(ro, 1)
# [ 79400300.   87939470.8  80966698.7]

# ======================================================================================================================
# 10. Quiz Question:
# Recall that, whenever ro[i] falls between -l1_penalty/2 and l1_penalty/2, the corresponding weight w[i] is sent to
# zero. Now suppose we were to take one step of coordinate descent on either feature 1 or feature 2. What range of
# values of l1_penalty would not set w[1] 0, but would set w[2] to zero, if we were to take a step in that coordinate?
# Answer:
l1_penalty_range = [2 * ro[2], 2 * ro[1]]
print np.round(l1_penalty_range, 1)
# [  1.61933397e+08   1.75878942e+08]

# ======================================================================================================================
# 11. Quiz Question:
# What range of values of l1_penalty would set both w[1] and w[2] to zero, if we were to take a step in that coordinate?
# Answer:
# [1.75878942e+08, inf]

# ======================================================================================================================
# Single Coordinate Descent Step:
# 12. Using the formula above, implement coordinate descent that minimizes the cost function over a single feature i.
# Note that the intercept (weight 0) is not regularized. The function should accept feature matrix, output, current
# weights, l1 penalty, and index of feature to optimize over. The function should return new weight for feature i.


def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    # compute prediction
    prediction = predict_outcome(feature_matrix, weights)
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    ro_i = np.sum(feature_matrix[:, i] * (output - prediction + weights[i] * feature_matrix[:, i]))

    if i == 0:  # intercept -- do not regularize
        new_weight_i = ro_i
    elif ro_i < -l1_penalty / 2.:
        new_weight_i = ro_i + l1_penalty / 2.
    elif ro_i > l1_penalty / 2.:
        new_weight_i = ro_i - l1_penalty / 2.
    else:
        new_weight_i = 0.

    return new_weight_i

# test:
# print lasso_coordinate_descent_step(1, np.array([[3./math.sqrt(13),1./math.sqrt(10)],
#                   [2./math.sqrt(13),3./math.sqrt(10)]]), np.array([1., 1.]), np.array([1., 4.]), 0.1)
# 0.425558846691

# ======================================================================================================================
# Cyclical coordinate descent
# 13. Now that we have a function that optimizes the cost function over a single coordinate, let us implement cyclical
# coordinate descent where we optimize coordinates 0, 1, ..., (d-1) in order and repeat.

# When do we know to stop? Each time we scan all the coordinates (features) once, we measure the change in weight for
# each coordinate. If no coordinate changes by more than a specified threshold, we stop.

# For each iteration:

# - As you loop over features in order and perform coordinate descent, measure how much each coordinate changes.
# - After the loop, if the maximum change across all coordinates is falls below the tolerance, stop. Otherwise,
# go back to the previous step.

# Return weights

# The function should accept the following parameters:
# - Feature matrix
# - Output array
# - Initial weights
# - L1 penalty
# - Tolerance


def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):

    max_change = 2 * tolerance
    weights = initial_weights
    while max_change > tolerance:
        max_change = 0.
        for i in range(len(weights)):
            old_weight_i = weights[i]
            weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
            if np.abs(weights[i] - old_weight_i) > max_change:
               max_change = np.abs(weights[i] - old_weight_i)
    return weights

# ======================================================================================================================
# 14. Let us now go back to the simple model with 2 features: ‘sqft_living’ and ‘bedrooms’.
# Using ‘get_numpy_data’, extract the feature matrix and the output array from from the house dataframe.
features = ['sqft_living', 'bedrooms']
(features_matrix, output_array) = get_numpy_data(sales, features, 'price')
# Then normalize the feature matrix using ‘normalized_features()’ function.
(normalized_features, norms) = normalize_features(features_matrix)
# Using the following parameters, learn the weights on the sales dataset.
# - Initial weights = all zeros
initial_weights = np.zeros(3)
# - L1 penalty = 1e7
l1_penalty = 1e7
# - Tolerance = 1.0
tolerance = 1.0
weights = lasso_cyclical_coordinate_descent(normalized_features, output_array, initial_weights, l1_penalty, tolerance)
print weights
# ======================================================================================================================
# 15. Quiz Question:
# What is the RSS of the learned model on the normalized dataset?

predictions = predict_outcome(normalized_features, weights)
errors = output_array - predictions
RSS = np.dot(errors, errors)

print 'RSS of the learned model on the normalized dataset is ' + str(np.round(RSS, 1))
# RSS of the learned model on the normalized dataset is 1.63049247672e+15

# ======================================================================================================================
# 16. Quiz Question:
# Which features had weight zero at convergence?
print np.round(weights, 1)
# Answer: bedrooms

# ======================================================================================================================
# Evaluating LASSO fit with more features:
# 17. Please down the corresponding csv files for train and test data.
sales_train = pd.read_csv('./wk3_kc_house_train_data.csv', dtype=dtype_dict)
sales_test = pd.read_csv('./wk3_kc_house_test_data.csv', dtype=dtype_dict)

# ======================================================================================================================
# 18. Create a normalized feature matrix from the TRAINING data with the following set of features.
# bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade, sqft_above, sqft_basement,
# yr_built, yr_renovated

# Make sure you store the norms for the normalization, since we’ll use them later.
all_features = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront',
                'view',
                'condition',
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built',
                'yr_renovated']
(features_matrix, output_array) = get_numpy_data(sales_train, all_features, 'price')
(normalized_features, norms) = normalize_features(features_matrix)


# ======================================================================================================================
# 19. First, learn the weights with l1_penalty=1e7, on the training data. Initialize weights to all zeros, and set the
# tolerance=1. Call resulting weights’ weights1e7’, you will need them later.
l1_penalty = 1e7
tolerance = 1
initial_weights = np.zeros(14)
weights1e7 = lasso_cyclical_coordinate_descent(normalized_features, output_array, initial_weights, l1_penalty, tolerance)

# ======================================================================================================================
# 20. Quiz Question:
# What features had non-zero weight in this case?
print pd.Series(weights1e7,index=['intercept'] + all_features)
# intercept        2.386469e+07
# bedrooms         0.000000e+00
# bathrooms        0.000000e+00
# sqft_living      3.049555e+07
# sqft_lot         0.000000e+00
# floors           0.000000e+00
# waterfront       1.901634e+06
# view             5.705765e+06
# condition        0.000000e+00
# grade            0.000000e+00
# sqft_above       0.000000e+00
# sqft_basement    0.000000e+00
# yr_built         0.000000e+00
# yr_renovated     0.000000e+00

# ======================================================================================================================
# 21. Next, learn the weights with l1_penalty=1e8, on the training data. Initialize weights to all zeros, and set the
# tolerance=1. Call resulting weights ‘weights1e8’, you will need them later.
l1_penalty = 1e8
initial_weights = np.zeros(14)
weights1e8 = lasso_cyclical_coordinate_descent(normalized_features, output_array, initial_weights, l1_penalty, tolerance)

# ======================================================================================================================
# 22. Quiz Question:
# What features had non-zero weight in this case?
print pd.Series(weights1e8, index=['intercept'] + all_features)

# ======================================================================================================================
# 23. Finally, learn the weights with l1_penalty=1e4, on the training data. Initialize weights to all zeros, and set
# the tolerance=5e5. Call resulting weights ‘weights1e4’, you will need them later. (This case will take quite a bit
# longer to converge than the others above.)
l1_penalty = 1e4
tolerance = 5e5
initial_weights = np.zeros(14)
weights1e4 = lasso_cyclical_coordinate_descent(normalized_features, output_array, initial_weights, l1_penalty, tolerance)

# ======================================================================================================================
# 24. Quiz Question:
# What features had non-zero weight in this case?
print pd.Series(weights1e4, index=['intercept'] + all_features)

# ======================================================================================================================
# Rescaling learned weights
# 25. Recall that we normalized our feature matrix, before learning the weights. To use these weights on a test set, we
# must normalize the test data in the same way. Alternatively, we can rescale the learned weights to include the
# normalization, so we never have to worry about normalizing the test data:

# ======================================================================================================================
# 26. Create a normalized version of each of the weights learned above. (‘weights1e4’, ‘weights1e7’, ‘weights1e8’).
# To check your results, if you call ‘normalized_weights1e7’ the normalized version of ‘weights1e7’, then
weights1e7_normalized = weights1e7 / norms
weights1e8_normalized = weights1e8 / norms
weights1e4_normalized = weights1e4 / norms

print norms[3]
print weights1e7_normalized[3]

# ======================================================================================================================
# Evaluating each of the learned models on the test data

# ======================================================================================================================
# 27. Let's now evaluate the three models on the test data. Extract the feature matrix and output array from the TEST
# set. But this time, do NOT normalize the feature matrix. Instead, use the normalized version of weights to make
# predictions.
(features_matrix_test, output_array_test) = get_numpy_data(sales_test, all_features, 'price')
# Compute the RSS of each of the three normalized weights on the (unnormalized) feature matrix.
predictions_weights1e7 = predict_outcome(features_matrix_test, weights1e7_normalized)
predictions_weights1e8 = predict_outcome(features_matrix_test, weights1e8_normalized)
predictions_weights1e4 = predict_outcome(features_matrix_test, weights1e4_normalized)

errors__weights1e7 = output_array_test - predictions_weights1e7
errors__weights1e8 = output_array_test - predictions_weights1e8
errors__weights1e4 = output_array_test - predictions_weights1e4

RSS_weights1e7 = np.dot(errors__weights1e7, errors__weights1e7)
RSS_weights1e8 = np.dot(errors__weights1e8, errors__weights1e8)
RSS_weights1e4 = np.dot(errors__weights1e4, errors__weights1e4)
# ======================================================================================================================
# 28. Quiz Question:
# Which model performed best on the test data?
print RSS_weights1e7
print RSS_weights1e8
print RSS_weights1e4
# 1.63103564165e+14
# 2.8471892521e+14
# 1.29085259902e+14
# Answer: 3rd model with 1e4 as l1_penalty






end = time.time()

print 'Time of Process was: ' + str(round((end - start), 2)) + '[sec]'
