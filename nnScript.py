import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from scipy.stats import logistic
from scipy.special import expit
from math import sqrt
import math
import scipy
from stringold import digits

    
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    print "**********Inside initializeWeights**********"
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    print "**********Inside sigmoid**********"
    
    return  expit(z)
    
    

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

    #Arrays to store the entire training data (including validation data) for all digits:
    all_training_data = np.array([])
    all_training_label = np.array([])
    
    #Stacking the entire train data for all the digits into one matrix 'all_data'
    all_training_data = np.vstack((mat['train0'],mat['train1'],mat['train2'],mat['train3'],mat['train4'],mat['train5'],mat['train6'],mat['train7'],mat['train8'],mat['train9']))
    print "\nAll training data (stacked) is:"
    print all_training_data.shape
    
    #Validation data taken as 1/6th of the total training data:
    all_training_data_size = len(all_training_data)
    validation_data_size = all_training_data_size/6 
    train_data_size = all_training_data_size - validation_data_size 
    
    #Append the true labels of the entire train data to a label vector for all the digits:                        
    for i in range(0,len(mat['train0'])):
             
        all_training_label=np.append(all_training_label,0); 
         
    for i in range(0,len(mat['train1'])):
         
        all_training_label=np.append(all_training_label,1);
         
    for i in range(0,len(mat['train2'])):
        
        all_training_label=np.append(all_training_label,2);
        
    for i in range(0,len(mat['train3'])):
        
        all_training_label=np.append(all_training_label,3);
        
    for i in range(0,len(mat['train4'])):
        
        all_training_label=np.append(all_training_label,4);
        
    for i in range(0,len(mat['train5'])):
        
        all_training_label=np.append(all_training_label,5);
        
    for i in range(0,len(mat['train6'])):
        
        all_training_label=np.append(all_training_label,6);
        
    for i in range(0,len(mat['train7'])):
        
        all_training_label=np.append(all_training_label,7);
        
    for i in range(0,len(mat['train8'])):
        
        all_training_label=np.append(all_training_label,8);
        
    for i in range(0,len(mat['train9'])):
        
        all_training_label=np.append(all_training_label,9);
    
    
    print "\nAll traininig data true labels are:"
    print all_training_label.shape
    
    #Stack the test data into a single matrix test_data for all the digits:
    test_data = np.vstack((mat['test0'],mat['test1'],mat['test2'],mat['test3'],mat['test4'],mat['test5'],mat['test6'],mat['test7'],mat['test8'],mat['test9']))
    print "\nThe stacked test data is:"
    print test_data.shape
    
    #Append the true labels of test data to the test_label vector for all the digits:
    for i in range(len(mat['test0'])):
         
        test_label=np.append(test_label,0); 
         
    for i in range(len(mat['test1'])):
         
        test_label=np.append(test_label,1);
         
    for i in range(len(mat['test2'])):
         
        test_label=np.append(test_label,2);
         
    for i in range(len(mat['test3'])):
         
        test_label=np.append(test_label,3);
         
    for i in range(len(mat['test4'])):
         
        test_label=np.append(test_label,4);
         
    for i in range(len(mat['test5'])):
         
        test_label=np.append(test_label,5);
         
    for i in range(len(mat['test6'])):
         
        test_label=np.append(test_label,6);
         
    for i in range(len(mat['test7'])):
         
        test_label=np.append(test_label,7);
         
    for i in range(len(mat['test8'])):
         
        test_label=np.append(test_label,8);
         
    for i in range(len(mat['test9'])):
         
        test_label=np.append(test_label,9);
     
    print "\nThe testing data true labels are:"
    print test_label.shape
    
    #Randomize function for randomly shuffling the training data and labels into the same sequence:
    randomObject = np.random.RandomState()
    indices = np.arange(all_training_data_size)
    randomObject.shuffle(indices)
    
    #Shuffling the arrays all_data and all_label
    shuffled_training_data = all_training_data[indices]
    shuffled_training_label = all_training_label[indices]
    
    print "\nShuffled training data is:"
    print shuffled_training_data.shape

    print "\nShuffled training labels are:"
    print shuffled_training_label.shape
   
    #Splitting the entire training data 'all_data' into 'validation_data' and 'training_data' according to 'validation_data_size':
    validation_data = shuffled_training_data[:validation_data_size]
    train_data = shuffled_training_data[validation_data_size:]
    print "\nSplit Validation data is:"
    print validation_data.shape
    
    print "\nSplit Training data is:"
    print train_data.shape

    #Splitting all the training labels 'all_label' into 'validation_label' and 'training_label' according to 'validation_data_size':
    validation_label = shuffled_training_label[:validation_data_size]
    train_label = shuffled_training_label[validation_data_size:]
    print "\nSplit Validation labels are:"
    print validation_label.shape
    
    print "\nSplit Training labels are:"
    print train_label.shape
    
    #Feature Selection and normalizarion:
    
    #Stack the training, validation and test data to find out all columns which have only 0's:
    featured_data = np.array(np.vstack((train_data,validation_data,test_data)))
    print "\nCombined Featured data is:"
    print featured_data.shape
    
    #Normalize all the data between 0 and 255
    featured_data = featured_data/255.0
    
    #Search the columns with all 0's:
    col_zero = np.where(~featured_data.any(axis=0))[0]
#     print "\nColumns with all 0's are:"
#     print len(col_zero)
#     print col_zero
    
    #Delete the columns found in previous step
    for i in range((len(col_zero)-1),-1,-1):
        featured_data = scipy.delete(featured_data,col_zero[i],1)
    
    #Split the data back into training, validation and testing data:
    train_data = featured_data[:train_data_size]
    validation_data = featured_data[train_data_size:(train_data_size+validation_data_size)]
    test_data = featured_data[(train_data_size+validation_data_size):]
    
    print "\nAfter deletion, final data:"
    print "Combined data:"
    print featured_data.shape
    print "Train:"
    print train_data.shape
    print "Validationn:"
    print validation_data.shape
    print "Test:"
    print test_data.shape
    
    #returning all the data matrices and label vectors       
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
    
    print "**********Inside nnObjFunction**********"
    
    #-------------------Feed Forward Phase-------------------------------------------
    
    #for hidden nodes    
    #given training data does not consider the bias node, therefore adding it:
    input_bias_node = np.zeros(len(training_data))
    #inputs from bias nodes is 1
    input_bias_node.fill(1)
    training_data = np.column_stack([training_data,input_bias_node])
    #taking transpose of w1 for dot product
    w1T = np.transpose(w1)
    #taking dot product of inputs with weights
    aj = np.dot(training_data,w1T)
    #taking sigmoids of aj
    zj = sigmoid(aj)
    
    #for output nodes
    hidden_bias_node = np.zeros(len(zj))
    hidden_bias_node.fill(1)
    zj = np.column_stack([zj,hidden_bias_node])
    #taking transpose of w2 for dot product
    w2T = np.transpose(w2)
    bl = np.dot(zj,w2T)
    ol = sigmoid(bl)
           
    #-------------------Back Propagation Phase---------------------------------------
    
    #-------------------Gradiance Phase----------------------------------------------
    
    #to calculate the gradiance, we need the true labels in 1 of k notation
    #converting...
#     yl = np.zeros((len(training_data),10))
#     for i in range(0,len(training_data)):
#         yl[i][training_label[i]] = 1
    
    yl = np.array([])
    for i in range(0,len(training_data)):
        digits = [0,0,0,0,0,0,0,0,0,0]
        digits[int(training_label[i])]=1
        yl=np.append(yl,digits)
    yl.resize(len(training_label),n_class)
    #print "yl size is:"
    #print yl.shape
    
    #now calculating gradiance wrt w2
    print "yl and ol shapes"
    print yl.shape
    print ol.shape
    
    delta_l = -1 * (yl - ol) * (1 - ol) * ol
    print delta_l.shape
    delta_l_T = np.transpose(delta_l)
    grad_w2 = np.dot(delta_l_T,zj)  
    print grad_w2.shape
    
    #now calculating gradiance wrt w1
    gw1_term1 = -1 * (zj - 1) * zj
    gw1_term2 = np.dot(delta_l,w2)
    gw1_term3 = gw1_term1 * gw1_term2
    gw1_term3_T = np.transpose(gw1_term3)
    
    grad_w1 = np.dot(gw1_term3_T,training_data)
    grad_w1 = grad_w1[0:n_hidden,:]
    print grad_w1.shape
    
    #-------------------Error Function Phase-----------------------------------------
    #calculating error as squared loss error function
    error_term_n = 0
    for n in range(0,len(training_data)):
        error_term_k = 0
        for k in range(0,n_class):
            error_term_k = (math.pow((yl[n][k] - ol[n][k]),2))             
        error_term_n += (error_term_k / 2)
    error_term = (error_term_n / len(training_data))
    print "error term"
    print error_term
    
    #-------------------Regularization Phase-----------------------------------------
    
    #Regularization of gradiance    
    #Regularization of grad_w1
    grad_w1 = (grad_w1 + (lambdaval * w1)) / len(training_data)
    
    #Regularization of grad_w2
    grad_w2 = (grad_w2 + (lambdaval * w2)) / len(training_data)
    
    #final gradiance
    obj_grad = np.array([])
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    
    #Regularization of error value
    reg_term1 = np.sum(w1 * w1)
    reg_term2 = np.sum(w2 * w2)
    
    reg_term3 = (lambdaval/(2*len(training_data)))*(reg_term1 + reg_term2)
    obj_val = error_term + reg_term3 
    print "obj_val : "
    print obj_val
    
        
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
       
    
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
    
    print "**********Inside nnPredict**********"
    
    #for hidden nodes    
    #given training data does not consider the bias node, therefore adding it:
    input_bias_node = np.zeros(len(data))
    #inputs from bias nodes is 1
    input_bias_node.fill(1)
    training_data = np.column_stack([data,input_bias_node])
    #taking transpose of w1 for dot product
    w1T = np.transpose(w1)
    #taking dot product of inputs with weights
    aj = np.dot(training_data,w1T)
    #taking sigmoids of aj
    zj = sigmoid(aj)
    
    #for output nodes
    hidden_bias_node = np.zeros(len(zj))
    hidden_bias_node.fill(1)
    zj = np.column_stack([zj,hidden_bias_node])
    #taking transpose of w2 for dot product
    w2T = np.transpose(w2)
    bl = np.dot(zj,w2T)
    ol = sigmoid(bl)
    
    for i in range(ol.shape[0]):
       max_index = np.argmax(ol[i])
       labels = np.append(labels, max_index)
    
    return labels
    



"""**************Neural Network Script Starts here********************************"""

import os
os.system('cls')

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 5;
                   
# set the number of nodes in output unit
n_class = 10;                   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.1;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 5}    # Preferred value.

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

print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
