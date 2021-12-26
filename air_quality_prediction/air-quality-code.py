''' This code seeks to predict cities air quality from data. We implement kernelized ridge regression
    and support vector  using the sklearn library. We used cross validation to tune the kernel and
    regularization parameters in the objective functions. '''

'''
Author: Bernardo Bianco Prado
Date:  12/26/2021
'''


import numpy as np
#from matplotlib import pyplot
#import matplotlib.pyplot as plt
import csv
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

# The csv file air-quality-train.csv contains the training data.
# After loaded, each row of X_train will correspond to CO, NO2, O3, SO2.
# The vector y_train will contain the PM2.5 concentrations.
# Each row of X_train corresponds to the same timestamp.
X_train = []
y_train = []

with open('air-quality-train.csv', 'r') as air_quality_train:
    air_quality_train_reader = csv.reader(air_quality_train)
    next(air_quality_train_reader)
    for row in air_quality_train_reader:
        row = [float(string) for string in row]
        row[0] = int(row[0])
        
        X_train.append([row[1], row[2], row[3], row[4]])
        y_train.append(row[5])
        
# The csv file air-quality-test.csv contains the testing data.
# After loaded, each row of X_test will correspond to CO, NO2, O3, SO2.
# The vector y_test will contain the PM2.5 concentrations.
# Each row of X_train corresponds to the same timestamp.
X_test = []
y_test = []

with open('air-quality-test.csv', 'r') as air_quality_test:
    air_quality_test_reader = csv.reader(air_quality_test)
    next(air_quality_test_reader)
    for row in air_quality_test_reader:
        row = [float(string) for string in row]
        row[0] = int(row[0])
        
        X_test.append([row[1], row[2], row[3], row[4]])
        y_test.append(row[5])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


#  implement 1. Use SVR loaded to train a SVR model with rbf kernel, regularizer (C) set to 1 and rbg kernel parameter (gamma) 0.1

svr_reg = SVR(kernel = 'rbf', gamma = 0.1, C = 1)
svr_reg.fit(X_train,y_train)

predicted1 = svr_reg.predict(X_test)

# compute RMSE
n,d = np.shape(X_test)
mse1 = 0
for i in range(n):
    mse1 += np.power(predicted1[i]-y_test[i],2)
mse1 = np.sqrt(mse1/n)
print("\nMean Squared Error for SVR:", mse1, "\n\n")


# Implement Kernel Ridge  model with rbf kernel, regularizer (C) set to 1 and rbg kernel parameter (gamma) 0.1

kernel_reg = KernelRidge(alpha = 0.5, kernel = 'rbf', gamma = 0.1)
kernel_reg.fit(X_train,y_train)

predicted2 = kernel_reg.predict(X_test)

# compute RMSE
mse2 = 0
for i in range(n):
    mse2 += np.power(predicted2[i]-y_test[i],2)
mse2 = np.sqrt(mse2/n)
print("\nMean Squared Error for Kernel Ridge Regression", mse2, "\n\n")



# Use this seed.
seed = 0
np.random.seed(seed) 

K = 5 #The number of folds we will create 


#vCreate a partition of training data into K=5 folds
n,d = np.shape(X_train)
randomized_indices = np.random.permutation(np.arange(n))

k = n//5
indices = []
for i in range(4):
    indices.append(randomized_indices[i*k:(i+1)*k])
indices.append(randomized_indices[3*k:n])

# Specify the grid search space 
reg_range = np.logspace(-1,1,3)     # Regularization paramters
kpara_range = np.logspace(-2, 0, 3) # Kernel parameters

# TODOs for part (d)
# Select the best parameters for both SVR and KernelRidge based on k-fold cross-validation error estimate (using RMSE as the performance metric)

# create matrix of errors
svr_cv_error = np.zeros((3,3))
kernel_cv_error = np.zeros((3,3))

# loop through folds
for k in range(K):
    for reg in range(3):
        for kpara in range(3):
        
            X = np.copy(X_train)
            y = np.copy(y_train)
            np.delete(X,indices[k],0)
            np.delete(y,indices[k],0)
            
            # SVR
            svr_reg = SVR(kernel = 'rbf', gamma = kpara_range[kpara], C = reg_range[reg])
            svr_reg.fit(X,y)
            
            # Kernel Ridge Regression
            kernel_reg = KernelRidge(alpha = 1/(2*reg_range[reg]), kernel = 'rbf', gamma = kpara_range[kpara])
            kernel_reg.fit(X,y)
            
            predicted1 = svr_reg.predict(X_train[indices[k]])
            svr_mse = 0
            
            predicted2 = kernel_reg.predict(X_train[indices[k]])
            ker_mse = 0
            
            # compute MSEs
            for i in range(len(indices[k])):
                svr_mse += np.power(predicted1[i]-y_train[indices[k][i]],2)
                ker_mse += np.power(predicted2[i]-y_train[indices[k][i]],2)
            svr_mse = np.sqrt(svr_mse/len(indices[k]))
            ker_mse = np.sqrt(ker_mse/len(indices[k]))
            
            # update svr error
            svr_cv_error[reg,kpara] += svr_mse
    
            # update kernel regression error
            kernel_cv_error[reg,kpara] += ker_mse
    

best_svr = np.argmin(svr_cv_error)
best_ker = np.argmin(kernel_cv_error)

#print(svr_cv_error, best_svr)
#print(kernel_cv_error, best_ker)


# compute best paramters for SVR
print("\n\nBest parameters for SVR:\n")
print("Regularization parameter:", reg_range[best_svr//3],"\nKernel parameter:", kpara_range[best_svr-3*best_svr//3],"\n\n")

# train SVR with best parameter
svr_reg = SVR(kernel = 'rbf', gamma = kpara_range[best_svr-3*best_svr//3], C = reg_range[best_svr//3])
svr_reg.fit(X_train,y_train)
predicted1 = svr_reg.predict(X_test)

# compute RMSE
n,d = np.shape(X_test)
svr_mse = 0
for i in range(n):
    svr_mse += np.power(predicted1[i]-y_test[i],2)
svr_mse = np.sqrt(svr_mse/n)

print("RMSE for best parameters:", svr_mse,"\n")


# compute best paramters for KernelRidge
print("\n\nBest parameters for Kernel Ridge Regression:\n")
print("Regularization parameter:", 1/(2*reg_range[best_ker//3]),"\nKernel parameter:", kpara_range[best_ker-3*best_ker//3],"\n\n")

# train KernelRidge with best parameter
kernel_reg = KernelRidge(alpha = 1/(2*reg_range[best_ker//3]), kernel = 'rbf', gamma = kpara_range[best_ker-3*best_ker//3])
kernel_reg.fit(X_train,y_train)

predicted2 = kernel_reg.predict(X_test)

# compute RMSE
ker_mse = 0
for i in range(n):
    ker_mse += np.power(predicted2[i]-y_test[i],2)
ker_mse = np.sqrt(ker_mse/n)

print("RMSE for best parameters:", ker_mse,"\n\n")
