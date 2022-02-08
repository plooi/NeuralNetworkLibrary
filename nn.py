import numpy as np
import activation as act

from random import random
from time import time


import keyboard
from time import sleep
import pyautogui
import threading
import pickle
from datetime import datetime
import os


class NeuralNetwork:
    """
    dimensions is a list where each element is an int representing the layer size
        the first int is the input size, second int is first layer size, third
        is second layer size... last is the last layer size
    activations is a list where each element is an Activation object representing
    the activation for that layer
    
    length of dimensions will be one more than length of activations
    bc dimensions also has to have one extra element for the input,
    whereas there is no activation for the input
    
    hard coded for squared error but activation functions are customizable
    """
    def __init__(self, dimensions, activations, name = "NeuralNet"):
        
        self.init_dimensions = list(dimensions)
        
        if len(dimensions) - 1 != len(activations):
            raise Exception("dimensions needs to have one more element than activations")
        
        #each layer gets an extra neuron which always outputs 1 which functions as the bias
        #the last neuron is the bias neuron
        for i in range(len(dimensions)):
            dimensions[i] += 1
        
        self.weights = []
        self.activations = []
        for i in range(len(dimensions)-1):
            #self.weights.append(np.ones((dimensions[i+1],dimensions[i])))
            self.weights.append(np.random.rand(dimensions[i+1],dimensions[i]) * 2 - 1)
            self.activations = activations
        
        self.name = name
        
        self.o = None
        self.a = None
        self.pred = None
        
            
    def __str__(self, detailed=False):
        ret = "=======\nNeural Network:\n(the ' + 1' is for the bias)\n"
        ret += "Input size: " + str(self.weights[0].shape[1]-1) + " + 1\n"
        
        for i in range(len(self.weights)):
            ret += "Layer " + str(i+1) + ": " + str(self.weights[i].shape[0]-1) + " + 1 neurons | " + str(self.activations[i]) + "\n"
            if detailed:
                ret += "" + str(self.weights[i]) + "\n\n"
                
        ret += "Output size: " + str(self.weights[len(self.weights)-1].shape[0] - 1) + " + 1\n"
        ret += "=======\n"
        return ret
    
    def print_detailed(self):
        print(self.__str__(detailed=True))
    
            
    def __call__(self, input):
        self.check_x_shape(input)
        o_previous = np.append(input,[1])#for bias because for bias you need one feature to be constant at 1
        self.input = o_previous
        
            
        
        self.o = [None] * len(self.weights)
        self.a = [None] * len(self.weights)
        for i in range(len(self.weights)):
            self.a[i] = np.matmul(self.weights[i], o_previous)#calculate activation
            self.o[i] = self.f_of_x_with_bias(self.activations[i].function, self.a[i])
            o_previous = self.o[i]
        self.pred = o_previous
        return self.pred[0:-1]
        
    
    def sgd(self, x, y, learning_rate = .00001):
        self.check_y_shape(y)
        
        y = np.append(y,[1])#because output is one more because bias
        self(x)
        prediction = self.pred[0:-1]
        
        
        
        
        
        
        #https://haipeng-luo.net/courses/CSCI567/2021_fall/lec4-one-page.pdf
        num_layers = len(self.weights)
        weight_gradients = [None] * num_layers
        activation_gradients = [None] * num_layers
        
        
        for l in range(num_layers-1,-1,-1):
            
            if l == num_layers-1:
                
                activation_gradients[l] = 2 * (self.pred - y) * self.f_of_x_with_bias(self.activations[l].derivative, self.a[l])
                
            else:
                activation_gradients[l] = np.matmul(np.transpose(self.weights[l+1]), activation_gradients[l+1]) * self.f_of_x_with_bias(self.activations[l].derivative, self.a[l])
            
            previous_output = self.o[l-1] if l > 0 else self.input
            previous_output = np.reshape(previous_output, (1,-1))
            weight_gradients[l] = np.matmul(np.reshape(activation_gradients[l], (-1,1)), previous_output)
            
            
        
        #print("factors" + str(np.reshape(activation_gradients[l], (-1,1))) +  "\n" + str(previous_output))
        #print("wg",weight_gradients)
            
        for l in range(num_layers):
            self.weights[l] -= learning_rate * weight_gradients[l]
            
            
    #apply function f to vector x but last element is turned to one because bias
    def f_of_x_with_bias(self, f, x):
        x = f(x)
        x[len(x) - 1] = 1
        return x
        
    def error(self, x, y):
        if len(x) != len(y):
            raise Exception("Input and output array lengths dont match")
        error = 0
        for i in range(len(x)):
            input = x[i]
            label = y[i]
            dif = self(input) - label
            error += np.sum(dif * dif)
        return error
    def check_x_shape(self, x):
        if x.shape != (self.weights[0].shape[1]-1,):
            raise Exception("Input data is of length " + str(x.shape[0]) + " but needs to be length " + str(self.weights[0].shape[1]-1) + " is required")
    def check_y_shape(self, y):
        if y.shape != (self.weights[-1].shape[0]-1,):
            raise Exception("Label data is of length " + str(y.shape[0]) + " but needs to be length " + str(self.weights[-1].shape[0]-1) + " is required")
    def evaluation(self, x, y):
        for i in range(len(x)):
            print(x[i],"should be",y[i],"result is",self(x[i]))
            
    def encode_activations(self):
        for i in range(len(self.activations)):
            self.activations[i] = act.activations.index(self.activations[i])
    def decode_activations(self):
        for i in range(len(self.activations)):
            self.activations[i] = act.activations[self.activations[i]]

paused = True


def key_thread():
    global paused
    while True:
        if keyboard.read_key() == "end":
            paused = not paused

def save(nn, filename):
    f = open(filename,"wb")
    nn.encode_activations()
    pickle.dump(nn, f)
    nn.decode_activations()
    f.close()
def load(filename):
    if filename.endswith("nn"):
        f = open(filename, "rb")
        nn = pickle.load(f)
        nn.decode_activations()
        f.close()
        return nn
    else:
        ret = []
        f = open(filename, "r")
        for line in f:
            ret.append([float(x) for x in line.split(",")])
        f.close()
        ret = np.array(ret)
        return ret
def train(nn, x, y):
    global paused
    
    
    l = {
    "learning_rate" : .00001,
    "update_print_frequency_seconds" : 1,
    "epoch" : 0,
    "nn" : nn,
    "x" : x,
    "y" : y,
    "random_data" : False,
    }
    
    def restart():
        l['nn'] = NeuralNetwork(l['nn'].init_dimensions, l['nn'].activations, l['nn'].name)
        l['epoch'] = 0
        print("Initializing new neural network")
        return ""
    l["restart"] = restart
    def faster():
        l['learning_rate'] *= 1.8
    l["faster"] = faster
    def slower():
        l['learning_rate'] *= 1/1.8
    l["slower"] = slower
    
    def _load(nnfile):
        l["nn"] = load(nnfile)
    l["load"] = _load
    
    last_update_print = time()
    
    threading.Thread(target=key_thread).start()
    
    
    while True:
        for i in range(len(l['x'])):
        
            if paused:
                print("Training paused. To resume, type 'start' and enter. To pause the training, press and hold 'end' key. \nType 'vars' to see variables\nType 'fn' to see neural network functions")
                if not os.path.isdir("./weights"):
                    os.mkdir("./weights")
                filename = datetime.now().strftime("./weights/"+l["nn"].name + " %m-%d-%Y %H_%M %S" + str(int(random()*1000)) + ".nn")
                print("Neural network saved as " + filename)
                save(nn, filename)
                while True:
                    c = input(">>> ")
                    c = c.strip()
                    if c == "start":
                        paused = False
                        break
                    elif c == "vars":
                        for k in l:
                            
                            to_print = str(l[k])
                            if k == "x": to_print = "numpy matrix " + str(l['x'].shape[0]) + " X " + str(l['x'].shape[1])
                            if k == "y": to_print = "numpy matrix " + str(l['y'].shape[0]) + " X " + str(l['y'].shape[1])
                            if k == "nn": to_print = "neural network object"
                            if k == "learning_rate": to_print = '{:.20f}'.format(l["learning_rate"])
                            print(k + " = " + to_print)
                    elif c == "fn":
                        for attr in dir(l['nn']):
                            if isinstance(getattr(l['nn'], attr), type(l['nn'].__init__)) and not attr.startswith("__"):
                                print("nn."+attr)
                    else:
                        try:
                            print(eval(c, None, l))
                        except SyntaxError as e:
                            
                            try:
                                exec(c, None, l)
                            except Exception as e:
                                print(e)
                        except Exception as e:
                            print(e)
                    
            if l["random_data"]:
                i = int(random()*len(l["x"]))
            
        
            #do backpropogation
            l['nn'].sgd(l['x'][i], l['y'][i], learning_rate=l["learning_rate"])
            l['epoch'] += 1
            
            
            
            if time()-last_update_print > l["update_print_frequency_seconds"]:
                print("Epoch: " + str(l['epoch']) + " Error: " + str(l['nn'].error(l['x'], l['y'])))
                last_update_print = time()
        
        
        
    
    
def main():


    x = load("features.csv")
    
    y = load("labels.csv")
    nn = NeuralNetwork([4,10,10,1], [act.relu,act.relu,act.sigmoid])
    
    train(nn, x, y)
    
    
    nn.evaluation(x, y)
    
    
    
    
    
    
    
if __name__ == "__main__": main()
