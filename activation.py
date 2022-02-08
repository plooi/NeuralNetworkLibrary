import numpy as np
from math import e

class Activation:
    def __init__(self, function, derivative, name):
        self.function = np.vectorize(function)
        self.derivative = np.vectorize(derivative)
        self.name = name
        if function == None or derivative == None:
            raise Exception("no")
    def __str__(self):
        return "Activation: " + self.name
    def __repr__(self):
        return str(self)
    
sigmoid = Activation(lambda x: 1/(1 + e**(-x)), lambda x: 1/(1 + e**(-x)) * (1 - 1/(1 + e**(-x))), "Sigmoid")
one = Activation(lambda x: 1, lambda x: 0, "One")
relu = Activation(lambda x: x/10 if x < 0 else x, lambda x: .1 if x < 0 else 1, "Relu")
linear = Activation(lambda x: x, lambda x: 1, "Linear")

activations = [sigmoid, one, relu, linear]


