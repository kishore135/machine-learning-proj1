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
    print "\nThe stacked training data is:"
    print train_data
    print "\nThe size of stacked training data is:"
    print len(train_data)
   
    digits = ["1000000000","0100000000","0010000000","0001000000","0000100000","0000010000","0000001000","0000000100","0000000010","0000000001"]
                        
    for i in range(0,len(mat['train0'])):
        
        train_label=np.append(train_label,digits[0]); 
        
    for i in range(0,len(mat['train1'])):
        
        train_label=np.append(train_label,digits[1]);
        
    for i in range(0,len(mat['train2'])):
        
        train_label=np.append(train_label,digits[2]);
        
    for i in range(0,len(mat['train3'])):
        
        train_label=np.append(train_label,digits[3]);
        
    for i in range(0,len(mat['train4'])):
        
        train_label=np.append(train_label,digits[4]);
        
    for i in range(0,len(mat['train5'])):
        
        train_label=np.append(train_label,digits[5]);
        
    for i in range(0,len(mat['train6'])):
        
        train_label=np.append(train_label,digits[6]);
        
    for i in range(0,len(mat['train7'])):
        
        train_label=np.append(train_label,digits[7]);
        
    for i in range(0,len(mat['train8'])):
        
        train_label=np.append(train_label,digits[8]);
        
    for i in range(0,len(mat['train9'])):
        
        train_label=np.append(train_label,digits[9]);
    
    #train_label.resize(60000,10)
    
    print "\nThe training data true labels are:"
    print train_label
    print "\nThe size of training data true labels is:"
    print len(train_label)
    
    test_data = np.vstack((mat['test0'],mat['test1'],mat['test2'],mat['test3'],mat['test4'],mat['test5'],mat['test6'],mat['test7'],mat['test8'],mat['test9']))
    print "\nThe stacked test data is:"
    print test_data
    print "\nThe size of stacked training data is:"
    print len(test_data)
    
    for i in range(0,len(mat['test0'])):
        
        test_label=np.append(test_label,digits[0]); 
        
    for i in range(0,len(mat['test1'])):
        
        test_label=np.append(test_label,digits[1]);
        
    for i in range(0,len(mat['test2'])):
        
        test_label=np.append(test_label,digits[2]);
        
    for i in range(0,len(mat['test3'])):
        
        test_label=np.append(test_label,digits[3]);
        
    for i in range(0,len(mat['test4'])):
        
        test_label=np.append(test_label,digits[4]);
        
    for i in range(0,len(mat['test5'])):
        
        test_label=np.append(test_label,digits[5]);
        
    for i in range(0,len(mat['test6'])):
        
        test_label=np.append(test_label,digits[6]);
        
    for i in range(0,len(mat['test7'])):
        
        test_label=np.append(test_label,digits[7]);
        
    for i in range(0,len(mat['test8'])):
        
        test_label=np.append(test_label,digits[8]);
        
    for i in range(0,len(mat['test9'])):
        
        test_label=np.append(test_label,digits[9]);
    
    print "\nThe testing data true labels are:"
    print test_label
    print "\nThe size of testing data true labels is:"
    print len(test_label)
             
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
    
    #-----------------the algo corresponding to the code below-----------------
    #for each training example
        #for each hidden node
            #compute the net input at each hidden node
            #compute the sigmoid of the computed input and store it
        #for each output node
            #compute the net input at each output node
            #compute the sigmoid of the computed input and store it
        #calculate the error at output nodes
        #calculate the gradiant of the error
    #--------------------------------------------------------------------------
    
    #to perform the calculations in the above algo, we need transpose of the weight matrices
    #therefore, original w1(50*785) should be converted to w1T(785*50)
    #similarly, original w2(10*50) should be converted to w2T(50*10)
    w1T = np.transpose(w1)
    w2T = np.transpose(w2)
    #print "Printing w2T..."
    #print w2T
    
                
    #for each training example    
    for te in range(0,training_data.shape[0]-1):
        #input vector will be row(te) in the training_data
        inputs = np.array([])
        inputs = training_data[te,:]
        #print "Printing inputs 1"
        #print inputs
        
        #np array to store the sigmoids of net inputs at the hidden nodes
        sigmoid_at_hidden_nodes = np.array([])
        
        #for each hidden node
        for h in range(0,n_hidden-1):
            #compute the net input at each hidden node
            #calculate product of input and weight from each input node, sum them all together
            net_input_at_hidden_node = 0
            for n in range(0,inputs.size-1):
                #p = inputs[n] * w1T[n,h]
                #net_input_at_hidden_node += p
                net_input_at_hidden_node += (inputs[n] * w1T[n,h])
            sig_h = sigmoid(net_input_at_hidden_node)
            #print "Printing sigmoid..."
            #print sig
            np.append(sigmoid_at_hidden_nodes,sig_h)
            
            
        #np array to store the sigmoids of net inputs at the output nodes
        sigmoid_at_output_nodes = np.array([])
        
        #for each output node
        for o in range(0,n_class-1):
            #compute the net input at each output node
            #calculate the product of input and weight from each hidden node, sum them all together
            net_input_at_output_node = 0
            for s in range(0,sigmoid_at_hidden_nodes.size-1):
                #o = sigmoid_at_hidden_nodes[s] * w2T[s,o]
                #net_input_at_output_node += o
                net_input_at_output_node += (sigmoid_at_hidden_nodes[s] * w2T[s,o])
            sig_o = sigmoid(net_input_at_output_node)
            np.append(sigmoid_at_output_nodes,sig_o)
        
        #all the values in 'sigmoid_at_output_nodes' are in decimals
        #before calculating the error at output, we need to convert it to an output comparable to the true labels
        #that is, in the form of 0's and 1's
        #therefore, we need to consider the highest value in 'sigmoid_at_output_nodes' as the correct output
        index_of_label = np.argmax(sigmoid_at_output_nodes)
        observed_label = ""
        for i in range(0,9):
            if i == index_of_label:
                observed_label += 1
            else:
                observed_label += 0
        
        #compute the error at each output node
        #compute the total error at output by summing the individual errors        
        total_error = 0        
        for o in range(0,n_class-1):
            #error at each output node is the square of difference between the observed and expected output
            error = math.pow((training_label[te] - sigmoid_at_output_nodes[o]),2)
            total_error += error 
        obj_val = total_error / 2
        
        #compute the gradient of the error

        
    
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
"""
print "\n\nAFTER RETURN:"
print "Length of training data:"
print len(train_data)
print "Length of training labels:"
print len(train_label)

print "Length of testing data:"
print len(test_data)
print "Length of testing labels:"
print len(test_label)


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
"""