# coding: utf-8

# <h2>Homework 1: Linear Regression on Boston Housing Data</h2>
#
# You're asked to do the following tasks to predict boston house price:
# <lu>
# <li>Use scikit-learn</li>
# <li>Implement analytical solution (normal equation) to obtain weights</li>
# <li>Implement numerical solution (gradient descent) to obtain weights</li>
# </lu>
#     Note: the accuracy of your implementations should be close to that of a linear model from scikit-learn
#
# In addition, you need to show the resulting intercept and coefficents, calculate errors on training dataset and testing dataset, and plot a figure to show your predictions and real prices on the testing dataset.

# In[1]:

# The modules we're going to use
from __future__ import print_function
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


def calcError(actual, predicted):
    return np.mean(((actual - predicted) ** 2)) / 2.0


if __name__ == '__main__':
    # When you execute a code to plot with a simple SHIFT-ENTER, the plot will be shown directly under the code cell
    # get_ipython().magic('matplotlib inline')

    # In[2]:

    # Load data from scikit-learn, which returns (data, target)
    # note: if you call "boston = load_boston()", it returns a dictionary-like object

    data, target = datasets.load_boston(True)

    # Split the data into two parts: training data and testing data
    train_data, test_data, train_target, test_target = train_test_split(data, target[:, np.newaxis], test_size=0.2,
                                                                        random_state=42)

    # <h4>Use scikit-learn library in the following cell</h4>

    # In[3]:

    # Task 1-1: use linear regression in sklearn
    regr = linear_model.LinearRegression()
    regr.fit(train_data, train_target)

    # Task 1-2: show intercept and coefficents
    print('Intercept: ', regr.intercept_, '\nWeights: ', regr.coef_, '\n')

    # Task 1-3: show errors on training dataset and testing dataset
    # Training Dataset
    train_predict = regr.predict(train_data)
    test_predict = regr.predict(test_data)
    print('Mean Squared Training Error: ', calcError(train_target, train_predict))
    print('Mean Squared Testing Error: ', calcError(test_target, test_predict))

    # Task 1-4: show plot a figure to show your predictions and real prices on the testing dataset
    plt.scatter(test_target.flatten(), test_predict.flatten(), color='black')
    plt.xticks(())
    plt.yticks(())
    plt.show()


    #
    # # <h4>Use analytical solution (normal equation) to perform linear regression in the following cell</h4>
    #
    # # In[4]:
    #
    # # Task 2-1: Implement a function solving normal equation
    # # Inputs: Training data and  training label
    # # Output: Weights
    # def myNormalEqualFun(X, y):
    #     return
    #
    #
    # # Task 2-2: Implement a function performing prediction
    # # Inputs: Testing data and weights
    # # Output: Predictions
    # def myPredictFun(X, w):
    #     return
    #
    #
    # # Here we insert a column of 1s into training_data and test_data (to be consistent with our lecture slides)
    # train_data_intercept = np.insert(train_data, 0, 1, axis=1)
    # test_data_intercept = np.insert(test_data, 0, 1, axis=1)
    #
    # # Here we call myNormalEqual to train the model and get weights
    # w = myNormalEqualFun(train_data_intercept, train_target)
    #
    # # Task 2-3: show intercept and coefficents
    #
    #
    # # Task 2-4: show errors on training dataset and testing dataset
    #
    #
    # # Task 2-5: show plot a figure to show your predictions and real prices on the testing dataset
    #
    #
    # # <h4>Use numerical solution (gradient descent) to perform linear regression in the following cell</h4>
    #
    # # In[5]:
    #
    # # Feature scaling
    # scaler = preprocessing.StandardScaler().fit(train_data)
    # train_data = scaler.transform(train_data)
    # test_data = scaler.transform(test_data)
    #
    #
    # # Task 3-1: Implement a function performing gradient descent
    # # Inputs: Training data, training label, leaerning rate, number of iterations
    # # Output: Weights
    # def myGradientDescentFun(X, y, learning_rate, numItrs):
    #     return
    #
    #
    # # Task 3-2: Implement a function performing prediction
    # # Inputs: Testing data and weights
    # # Output: Predictions
    # def myPredictFun(X, w):
    #     return
    #
    #
    # # Here we insert a column of 1s into training_data and test_data (to be consistent with our lecture slides)
    # train_data_intercept = np.insert(train_data, 0, 1, axis=1)
    # test_data_intercept = np.insert(test_data, 0, 1, axis=1)
    #
    # # Here we call myGradientDescentFun to train the model and get weights
    # # Note: you need to figure out good learning rate value and the number of iterations
    # w = myGradientDescentFun(train_data_intercept, train_target, 0, 0)
    #
    # # Task 3-3: show intercept and coefficents
    #
    #
    # # Task 3-4: show errors on training dataset and testing dataset
    #
    #
    # # Task 3-5: show plot a figure to show your predictions and real prices on the testing dataset
    #
