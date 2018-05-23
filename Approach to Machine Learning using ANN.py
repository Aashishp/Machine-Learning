# A simple approach to machine learning using neural networks

import numpy as np
import pandas as pd
import scipy.special
class neuralNetwork:
    ###############################################################
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes,self.inodes))

        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        self.lr = learningrate

        #Using sigmoid function as activation...
        self.activation_func = lambda x:scipy.special.expit(x)
        pass
    ##################################################################

    def train(self, inputs_list, targets_list):
        #converting to 2d Array...
        inputs = np.array(inputs_list,ndmin = 2).T
        targets = np.array(targets_list, ndmin = 2).T
        
        #Calculate signals into hidden layer...
        hidden_inputs = np.dot(self.wih, inputs)
        #Calculate inputs emerging from hidden layer...
        hidden_outputs = self.activation_function(hidden_inputs)

        #Signals in output layer...
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation(final_inputs)


        #Output layer error is target-actual
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        #Updating the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))

        #Updating the weights for links between tnput and hidden layers...
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

        pass
    ###############################################################


    def query(self, input_list):
        # conversion of list into 2d array...
        inputs = np.array(input_list, ndmin = 2).T

        #calculate signals emerging from hiddden layer
        hidden_inputs = np.dot(self.who, inputs)

        #calculate signals emerging from final output layer...
        final_outputs = self.activation_func(hidden_inputs)
        
        print(final_outputs)
        return final_outputs
    ##############################################################

input_nodes = 5
hidden_nodes = 3
output_nodes =1
learning_rate = 0.3
batch_size = 5

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = pd.read_csv("dataset.csv")
inputs = training_data_file.iloc[:,:-1].values
targets = training_data_file.iloc[:,1].values
inp = []
#(Add the code to divide the dataset and then pass it to network using the n.train method.(This code does not contain a method to divide data...))
#n.train(inputs,targets)

#n.query([1.0,0.5,-1.5])
