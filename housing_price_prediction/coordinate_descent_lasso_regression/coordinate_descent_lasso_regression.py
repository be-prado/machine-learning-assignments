''' This code seeks to predict the price of homes in Iowa using coordinate descent ridge regression with l^1 penalty. '''

'''
Author: Bernardo Bianco Prado
Date:  12/26/2021

More information about the data set can be found at jse.amstat.org/v19n3/decock/DataDocumentation.txt
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
np.random.seed(0)


#_________________________ USEFUL FUNCTIONS _____________________________

# compute mean squared error of the found w
def mse(w,x,y):
    # number of instances
    n = np.size(y)
    # initialize MSE
    _MSE = 0
    # add MSE term for each instance
    for i in range(n):
        # (w^T * x_i + w_0 - y_i)^2
        _MSE += np.power(np.dot(w[1:],x[:,i])+w[0]-y[i],2)
    return _MSE/n
    
    
#________________________________________________________________________

Xtrain = np.load("housing_train_features.npy")
Xtest = np.load("housing_test_features.npy")
ytrain = np.load("housing_train_labels.npy")
ytest = np.load("housing_test_labels.npy")

feature_names = np.load("housing_feature_names.npy", allow_pickle=True)
#print("First feature name: ", feature_names[0])
#print("Lot frontage for first train sample:", Xtrain[0,0])


# SPHERE TRAINING DATA

d,n =np.shape(Xtrain)
# create sphered feature matrix
spheredX = np.copy(Xtrain)
# create feature empirical mean and variance vectors
mean = np.zeros(d)
sd = np.ones(d)

for i in range(d):
    # empirical mean of feature i
    mean[i] = np.mean(Xtrain[i])
    # empirical variance of feature i
    sd[i] = np.sqrt(np.var(Xtrain[i]))
    # sphered values of i-th feature of the x instances
    spheredX[i] = (spheredX[i] - mean[i]*np.ones(n))/sd[i]

#print(np.var(spheredX[0]), np.mean(spheredX[5]))
#print(mean)

#smean = np.zeros(d)
#svar = np.zeros(d)
#for i in range(d):
#    smean[i] = np.mean(spheredX[i])
#    svar[i] = np.var(spheredX[i])

#print("max mean:",np.max(smean),"min mean:", np.min(smean),"|", "max variance:", \
#                  np.max(svar[i]),"min variance:",np.min(svar[i]))


# COMPUTE WEIGHTS

_lambda = 100
# vector of the a_i used to compute the minimizer w_i
a = np.zeros(d)
# cycles to run the algorithm
cycles = 50
# weight vectors
w = np.ones(d+1)
# matrix of weight vectors per cycle
W = np.zeros((cycles,d+1))
# vector of mean squared error per iteration
MSE = np.zeros(cycles*d)
# count number of iterations
count = 0

# compute w[0]
w[0] = np.sum(ytrain)/n

# compute a
for i in range(d):
    a[i] = 2*np.dot(spheredX[i,:],spheredX[i,:])

    
# begin coordinate descent lasso loop
for l in range(cycles):
    # iterate over features
    for i in range(d):
        # compute c_i used to compute the minimizer w_{i+1}
        c = 0
        for j in range(n):
            y_hat = w[0]+np.dot(w[1:],spheredX[:,j]) - w[i+1]*spheredX[i,j]
            c += spheredX[i,j]*(ytrain[j] - y_hat)
        c *= 2
        
        # update w_{i+1}
        if c > _lambda:
            w[i+1] = (c-_lambda)/a[i]
        elif c < - _lambda:
            w[i+1] = (c+_lambda)/a[i]
        else:
            w[i+1] = 0
        
        # compute mean squared error for this iteration
        MSE[count] = mse(w,spheredX,ytrain)
        # update count
        count += 1
    
    # store weight vector computed after full cycle
    W[l] = np.copy(w)
    #print('cycle', l+1)

        
# print w_1,...,w_5
for i in range(1,6):
    print("w_" + str(i), "=", W[cycles-1,i])



# PLOT WEIGHTS

# plot weights on each feature as a function of lambda
for i in range(d):
    # use log scale to see the evolution more clearly
    #plt.yscale("symlog")
    #plt.xscale("symlog")
    plt.plot(np.arange(1,cycles+1),W[:,i+1],label = feature_names[i])

lgd=plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',ncol=4, borderaxespad=0.)

#plt.yscale("symlog")

plt.title('Weight Trajectories for CD Lasso with lambda=100')
plt.xlabel('Iteration')
plt.ylabel('Cycle')

# save plot
plt.savefig("weight_plot_p5.png",bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()



# Plot MSE trajectory


plt.plot(np.arange(1,cycles*d+1),MSE)


plt.title('MSE Trajectory for CD Lasso with lambda=100')
plt.xlabel('Iteration')
plt.ylabel('MSE')

# save plot
plt.savefig("mse_plot_p5.png")
plt.close()



# COMPUTE MSE FOR TEST DATA

td,tn = np.shape(Xtest)

spheredXtest = np.zeros((td,tn))

# sphere test data
for i in range(td):
    # sphered values of i-th feature of the x instances
    spheredXtest[i] = (Xtest[i] - mean[i]*np.ones(tn))/sd[i]

# print MSE from test data
print("Final test MSE:",mse(W[49],spheredXtest,ytest))



# FIND WEIGHTS THRESHOLDED TO ZERO

# array of indexes thresholded to zero
zero_indexes = []

# find indexes
for i in range(d):
    if W[cycles-1,i] == 0:
        zero_indexes.append(i)

# print labels of indexes thresholded to zero
print("\n\n\n\nLabels that were threshold to zero:\n")
for i in zero_indexes:
    print(feature_names[i])



