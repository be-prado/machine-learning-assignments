''' This code seeks to predict whether a ray signal comes from gamma rays or background radiation.
    We implement kernel logistic regression using gradient descent. '''

'''
Author: Bernardo Bianco Prado
Date:  12/26/2021
The dataset used was the MAGIC Gamma Telescope Data Set.
'''


import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel


def linear_logistic_regression(x_train, y_train, x_test, y_test, step_size, reg_strength, num_iters):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1/reg_strength)
    clf.fit(x_train, y_train)
    test_acc = clf.score(x_test, y_test)
    return test_accx


def kernel_logistic_regression(x_train, y_train, x_test, y_test, step_size, reg_strength, num_iters, kernel_parameter):
    """
    x_train - (n_train, d)
    y_train - (n_train,)
    x_test - (n_test, d)
    y_test - (n_test,)
    step_size: gamma in problem description
    reg_strength: lambda in problem description
    num_iters: how many iterations of gradient descent to perform

    Implement KLR with the Gaussian Kernel.
    """
    # initialize alphas
    n_train, d = np.shape(x_train)
    n_test , = np.shape(y_test)
    alpha = np.zeros(n_train)
    b = 0
    
    # create kernelized matrix of training data
    K = rbf_kernel(x_train,x_train, gamma=kernel_parameter)
    
    # gradient descent
    for t in range(num_iters):
        # compute gradient direction
        beta = np.zeros(n_train)
        for i in range(n_train):
            argument = y_train[i]*(np.dot(alpha,K[i]) + b)
            beta[i] = -y_train[i] /(n_train*(1+np.exp(argument)))
        # update alpha
        alpha = alpha - step_size*(beta+2*reg_strength*alpha)
        # update b
        b = b - step_size*np.sum(beta)

    # compute test error
    y_predicted = np.zeros(n_test)
    K_test = rbf_kernel(x_test,x_train, gamma=kernel_parameter)
    # compute predicted classification
    for i in range(n_test):
        y_predicted[i] = np.sign(np.dot(alpha, K_test[i]) + b)
    # compute accuracy of our prediction on test data
    test_accuracy = np.count_nonzero(y_predicted+y_test)/n_test
    
    return test_accuracy



x_train = np.load("x_train.npy")    # shape (n_train, d)
x_test = np.load("x_test.npy")      # shape (n_test, d)

y_train = np.load("y_train.npy")    # shape (n_train,)
y_test = np.load("y_test.npy")        # shape (n_test,)


linear_acc = linear_logistic_regression(x_train, y_train, x_test, y_test, 1.0, 0.001, 200)
print("Linear LR accuracy:", linear_acc)

klr_acc = kernel_logistic_regression(x_train, y_train, x_test, y_test, 5.0, 0.001,200, 0.1)
print("Kernel LR accuracy:", klr_acc)
print("Kernel LR error:", 1 - klr_acc)
