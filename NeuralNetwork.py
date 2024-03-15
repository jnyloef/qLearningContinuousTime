import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) # L + 1 input layer (l=0)
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights = [np.random.randn(layer_sizes[i+1], layer_sizes[i]) for i in range(self.num_layers - 1)]
        self.biases = [np.zeros((layer_sizes[i+1], 1)) for i in range(self.num_layers - 1)]

        # Initialize weights and bias derivatives
        self.weight_derivatives = [None]*(self.num_layers - 1)
        self.bias_derivatives = [None]*(self.num_layers - 1)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return x * (x >= 0)
    
    def relu_derivative(self, x):
        return 1*(x >= 0)
    
    def feedforward(self, X):
        activations = [X]
        z = [None]*(self.num_layers - 1)
        for i in range(self.num_layers - 1):
            z[i] = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            activations.append(self.relu(z[i]))
        return activations, z
    
    def eval(self, X):
        a,_ = self.feedforward(X)
        return a[-1][0][0]

    def backward(self, X, y, activations):
        # Compute gradients
        deltas = [None] * (self.num_layers - 1)
        deltas[-1] = (y - activations[-1]) * self.sigmoid_derivative(activations[-1])
        
        for l in range(self.num_layers - 2, 0, -1):
            deltas[l-1] = np.dot(deltas[l], self.weights[l].T) * self.sigmoid_derivative(activations[l])
        
        # Update weights and biases
        for l in range(self.num_layers - 1):
            self.weights[l] += np.dot(activations[l].T, deltas[l]) * self.learning_rate
            self.biases[l] += np.sum(deltas[l], axis=0, keepdims=True) * self.learning_rate
    
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            activations = self.feedforward(X)
            self.backward(X, y, activations)
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - activations[-1]))
                print(f'Epoch {epoch}: Loss = {loss}')

    def gradient(self, X):
        # Compute gradients
        activations, z = self.feedforward(X)
        delta_z = [None] * (self.num_layers - 1) # L values
        delta_z[-1] = self.relu_derivative(z[-1]) # L th layer
        
        for l in range(self.num_layers - 2, 0, -1): # L-1 to 1
            delta_z[l-1] = np.dot(self.weights[l].T, delta_z[l]) * self.relu_derivative(z[l-1])
        
        # Update weights and biases
        for l in range(self.num_layers - 1):
            self.weight_derivatives[l] =  np.dot(delta_z[l], activations[l].T)
            self.bias_derivatives[l] = delta_z[l]


# Example usage:

#y = np.array([[0], [1], [1], [0]]).T

# Define layer sizes (including input and output layers)
layer_sizes = [10, 30, 3, 1]

X = np.random.randn(layer_sizes[0],1)

nn = NeuralNetwork(layer_sizes)
nn.gradient(X)

# Test the trained network
print("Output after training:")
print(nn.eval(X))
print(nn.weight_derivatives)
print(nn.bias_derivatives)
