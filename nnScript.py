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
    

    mat = loadmat('D:\Work\Machine Learning\PA1\mnist_all.mat') #loads the MAT object as a Dictionary    

    #Pick a reasonable size for validation data
    
    #Your code here
    
    train_data = np.array([])
    train_label = np.array([])
    validation_data = np.array([])
    validation_label = np.array([])
    test_data = np.array([])
    test_label = np.array([])

    #Arrays to store the entire training and validation data for all digits:
    all_data = np.array([])
    all_label = np.array([])
    
    #Stacking the entire train data for all the digits into one matrix 'all_data'
    all_data = np.vstack((mat['train0'],mat['train1'],mat['train2'],mat['train3'],mat['train4'],mat['train5'],mat['train6'],mat['train7'],mat['train8'],mat['train9']))
    print "\nStacked all data is:"
    print all_data
    print "\nThe size of stacked all data is:"
    print len(all_data)
    
    all_data_size = len(all_data)
    validation_data_size = all_data_size/6  #Validation data size taken as 1/6th of the total training data size
    
    #Append the true labels of the entire train data to a label vector for all the digits:                        
    for i in range(0,len(mat['train0'])):
             
        all_label=np.append(all_label,0); 
         
    for i in range(0,len(mat['train1'])):
         
        all_label=np.append(all_label,1);
         
    for i in range(0,len(mat['train2'])):
        
        all_label=np.append(all_label,2);
        
    for i in range(0,len(mat['train3'])):
        
        all_label=np.append(all_label,3);
        
    for i in range(0,len(mat['train4'])):
        
        all_label=np.append(all_label,4);
        
    for i in range(0,len(mat['train5'])):
        
        all_label=np.append(all_label,5);
        
    for i in range(0,len(mat['train6'])):
        
        all_label=np.append(all_label,6);
        
    for i in range(0,len(mat['train7'])):
        
        all_label=np.append(all_label,7);
        
    for i in range(0,len(mat['train8'])):
        
        all_label=np.append(all_label,8);
        
    for i in range(0,len(mat['train9'])):
        
        all_label=np.append(all_label,9);
    
    
    print "\nAll data true labels are:"
    print all_label
    print "\nThe size of all data true labels is:"
    print len(all_label)
    
    #Stack the test data into a single matrix test_data for all the digits:
    test_data = np.vstack((mat['test0'],mat['test1'],mat['test2'],mat['test3'],mat['test4'],mat['test5'],mat['test6'],mat['test7'],mat['test8'],mat['test9']))
    print "\nThe stacked test data is:"
    print test_data
    print "\nThe size of stacked test data is:"
    print len(test_data)
    
    #Append    the true labels of test data to the test_label vector for all the digits:
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
    print test_label
    print "\nThe size of testing data true labels is:"
    print len(test_label)
    
    #Randomize function for randomly shuffling the training data and labels into the same sequence:
    randomObject = np.random.RandomState()
    indices = np.arange(all_data_size)
    randomObject.shuffle(indices)
    
    #Shuffling the arrays all_data and all_label
    shuffled_data = all_data[indices]
    shuffled_label = all_label[indices]
    
    print "\n\nShuffled data is:"
    print len(shuffled_data)
    print "\nShuffled labels are:"
    print len(shuffled_label)
   
    #Splitting the entire training data 'all_data' into 'validation_data' and 'training_data' according to 'validation_data_size':
    validation_data = shuffled_data[:validation_data_size]
    train_data = shuffled_data[validation_data_size:]
    print "\nValidation data is:"
    print len(validation_data)
    print "\nTraining data is:"
    print len(train_data)

    #Splitting all the training labels 'all_label' into 'validation_label' and 'training_label' according to 'validation_data_size':
    validation_label = shuffled_label[:validation_data_size]
    train_label = shuffled_label[validation_data_size:]
    print "\nValidation labels are:"
    print len(validation_label)
    print "\nTraining labels are:"
    print len(train_label)
    
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
    #w1T = np.transpose(w1)
    #w2T = np.transpose(w2)
    #print "Printing w2T..."
    #print w2T
    
    #total error and gradient for entire training data
    final_error = 0
    final_error_gradient = 0
                
    #for each training example    
    for te in range(0,training_data.shape[0]):
        #few declarations
        #total error for training example
        error_for_training_example = 0        
        #error gradients for training example wrt the 2 weight vectors
        intermediate_grad_w1 = np.array([])
        intermediate_grad_w2 = np.array([])
        
        #input vector will be row(te) in the training_data
        inputs = np.array([])
        inputs = training_data[te,:]
        #print "Printing inputs 1"
        #print inputs
        
        #np array to store the sigmoids of net inputs at the hidden nodes
        sigmoid_at_hidden_nodes = np.array([])
        
        #for each hidden node
        for h in range(0,n_hidden):
            #compute the net input at each hidden node
            #calculate product of input and weight from each input node, sum them all together
            net_input_at_hidden_node = 0
            for n in range(0,inputs.size+1):                
                #if input node is bias node
                if n == inputs.size:
                    net_input_at_hidden_node += (1 * w1[h,n])                    
                else:
                    net_input_at_hidden_node += (inputs[n] * w1[h,n])
            sig_h = sigmoid(net_input_at_hidden_node)            
            sigmoid_at_hidden_nodes = np.append(sigmoid_at_hidden_nodes,sig_h)
            
            
        #np array to store the sigmoids of net inputs at the output nodes
        sigmoid_at_output_nodes = np.array([])
        
        #for each output node
        for o in range(0,n_class):
            #compute the net input at each output node
            #calculate the product of input and weight from each hidden node, sum them all together
            net_input_at_output_node = 0
            for s in range(0,sigmoid_at_hidden_nodes.size+1):
                #if hidden node is a bias node
                if s == sigmoid_at_hidden_nodes.size:
                    net_input_at_output_node += (1 * w2[o,s])
                else:
                    net_input_at_output_node += (sigmoid_at_hidden_nodes[s] * w2[o,s])
            sig_o = sigmoid(net_input_at_output_node)
            sigmoid_at_output_nodes = np.append(sigmoid_at_output_nodes,sig_o)        
            
        
        """
        #printing sigmoid_at_output_node
        print "TE # : "
        print te
        for x in range(0,sigmoid_at_output_nodes.size):            
            print x
            print sigmoid_at_output_nodes[x]
        """
        
        #compute the error at each output node
        #compute the total error at output by summing the individual errors        
        total_error = 0        
        for o in range(0,n_class):
            #error at each output node is the square of difference between the observed and expected output
            if( training_label[te] == 0 or 
                training_label[te] == 1 or
                training_label[te] == 2 or
                training_label[te] == 3 or
                training_label[te] == 4 or
                training_label[te] == 5 or
                training_label[te] == 6 or
                training_label[te] == 7 or
                training_label[te] == 8 or
                training_label[te] == 9):
                error = math.pow((1 - sigmoid_at_output_nodes[o]),2)
            else:
                error = math.pow((0 - sigmoid_at_output_nodes[o]),2)
            total_error += error 
        error_for_training_example = total_error / 2
        
        #calculating the gradient of error for every training example
        #formula is -(1-)
        
    final_error += error_for_training_example
    
    #error value for all training data together
    #=sum of errors for all training examples / number of training examples 
    intermediate_obj_val = final_error / training_data.shape[0]    
    #obj_val ie equation 15 over intermediate_obj_val
    #obj_val = intermediate_obj_val + regularization term
    regularization_term = 0
    #calculating regularization of w1
    r1 = 0
    for i in range(0,n_hidden):
        r2 = 0
        for j in range(0,n_input+1):
            r2 += math.pow(w1[j,i],2)
        r1 += r2
    
    #calculating regularization of w1
    r3= 0
    for l in range(0,n_class):
        r4 = 0
        for j in range(0,n_hidden+1):
            r4 += math.pow(w2[l,j],2)
        r3 += r4
    
    total_r = (lambdaval / (2 * training_data.shape[0])) * (r1 + r3)
    obj_val = intermediate_obj_val + total_r
    
    #computing error gradiance
    #computing the gradient wrt w2 in the output layer
    grad_output_nodes = np.array([])
    for l in range(0,n_class):
        delta_l = (training_label[te] - sigmoid_at_output_nodes[l]) * (1 - sigmoid_at_output_nodes[l]) * sigmoid_at_output_nodes[l]
        gw2 = delta_l * sigmoid_at_hidden_nodes[]
        
    #compute the derivation of error functions and gradient of the error
    #first, derivation of error function wrt weights from hidden to output nodes
    
    
    #second, derivation of error function wrt weights from input to hidden nodes
    

        
    
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
n_hidden = 4;
                   
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

opts = {'maxiter' : 4}    # Preferred value.

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
