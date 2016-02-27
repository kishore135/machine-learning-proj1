import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import math


def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    return  1 / (1 + math.exp(-z))
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.
 
     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
 
     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
   
    mat = loadmat('D:\mnist_all.mat') #loads the MAT object as a Dictionary
   
    #Pick a reasonable size for validation data
   
    
    #Your code here
   
    train_data = np.array([])
    train_label = np.array([])
    validation_data = np.array([])
    validation_label = np.array([])
    test_data = np.array([])
    test_label = np.array([])
       
    train_data = np.vstack((mat['train0'],mat['train1'],mat['train2'],mat['train3'],mat['train4'],mat['train5'],mat['train6'],mat['train7'],mat['train8'],mat['train9']))
    print "\n\nThe stacked training data is:"
    print train_data
    print "The size of stacked training data is:"
    print len(train_data)
  
    array0 = [1,0,0,0,0,0,0,0,0,0]
    array1 = [0,1,0,0,0,0,0,0,0,0]
    array2 = [0,0,1,0,0,0,0,0,0,0]
    array3 = [0,0,0,1,0,0,0,0,0,0]
    array4 = [0,0,0,0,1,0,0,0,0,0]
    array5 = [0,0,0,0,0,1,0,0,0,0]
    array6 = [0,0,0,0,0,0,1,0,0,0]
    array7 = [0,0,0,0,0,0,0,1,0,0]
    array8 = [0,0,0,0,0,0,0,0,1,0]
    array9 = [0,0,0,0,0,0,0,0,0,1]
                       
    for i in range(0,len(mat['train0'])):
       
        train_label=np.append(train_label,array0);
        
    for i in range(0,len(mat['train1'])):
       
        train_label=np.append(train_label,array1);
       
    for i in range(0,len(mat['train2'])):
       
        train_label=np.append(train_label,array2);
       
    for i in range(0,len(mat['train3'])):
       
        train_label=np.append(train_label,array3);
       
    for i in range(0,len(mat['train4'])):
       
        train_label=np.append(train_label,array4);
       
    for i in range(0,len(mat['train5'])):
       
        train_label=np.append(train_label,array5);
       
    for i in range(0,len(mat['train6'])):
       
        train_label=np.append(train_label,array6);
       
    for i in range(0,len(mat['train7'])):
       
        train_label=np.append(train_label,array7);
       
    for i in range(0,len(mat['train8'])):
       
        train_label=np.append(train_label,array8);
       
    for i in range(0,len(mat['train9'])):
       
        train_label=np.append(train_label,array9);
   
    train_label.resize(60000,10)
    print "The training data true labels are:"
    print train_label
    print "The size of training data true labels is:"
    print len(train_label)
   
    """
    test_data = np.vstack((mat['test0'],mat['test1'],mat['test2'],mat['test3'],mat['test4'],mat['test5'],mat['test6'],mat['test7'],mat['test8'],mat['test9']))
    print "The stacked test data is:"
    print test_data
    print "The size of stacked training data is:"
    print len(test_data)
   
    for i in range(0,len(mat['test0'])):
       
        test_label=np.append(test_label,0);
        
    for i in range(0,len(mat['test1'])):
       
        test_label=np.append(test_label,1);
       
    for i in range(0,len(mat['test2'])):
       
        test_label=np.append(test_label,2);
       
    for i in range(0,len(mat['test3'])):
       
        test_label=np.append(test_label,3);
       
    for i in range(0,len(mat['test4'])):
       
        test_label=np.append(test_label,4);
       
    for i in range(0,len(mat['test5'])):
       
        test_label=np.append(test_label,5);
       
    for i in range(0,len(mat['test6'])):
       
        test_label=np.append(test_label,6);
       
    for i in range(0,len(mat['test7'])):
       
        test_label=np.append(test_label,7);
       
    for i in range(0,len(mat['test8'])):
       
        test_label=np.append(test_label,8);
       
    for i in range(0,len(mat['test9'])):
       
        test_label=np.append(test_label,9);
   
    print "The testing data true labels are:"
    print test_label
    print "The size of testing data true labels is:"
    print len(test_label)"""
   
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
    
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    
    #Your code here
    
    #to compute the net input at each hidden node and take its sigmoid
    net_input_for_hidden_node = np.array([])
    
    for h in range(0,n_hidden):
        train_ex = 0    #the training example
        for i in range(0,n_input):
            net_ip += w1[i,h]*training_data[train_ex,i]
            train_ex = train_ex + 1
            net_input_for_hidden_node.vstack(net_input_for_hidden_nodes,net_ip)
    
    #to compute the net input at each output node and take its sigmoid
    
    #to compute obj_val which is error at output node
    
    
    #to compute obj_grad vector which is the __?
     
    
    
    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    labels = np.array([])
    #Your code here
    
    return labels
    



"""**************Neural Network Script Starts here********************************"""

import os
os.system('cls')

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
