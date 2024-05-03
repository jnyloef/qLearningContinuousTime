import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rates=[0.2, 0.1]):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) # L + 1 input layer (l=0)
        self.learning_rates = learning_rates
        
        # Initialize weights and biases
        self.weights = [np.random.randn(layer_sizes[i+1], layer_sizes[i]) for i in range(self.num_layers - 1)]
        self.biases = [np.zeros((layer_sizes[i+1], 1)) for i in range(self.num_layers - 1)]

        # Initialize weights and bias derivatives
        self.weight_derivatives = [None]*(self.num_layers - 1)
        self.bias_derivatives = [None]*(self.num_layers - 1)
    
    def sigmoid(self, x):
        print(x)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, a): # as a function of activation
        return a * (1 - a)
    
    def relu(self, x):
        return x * (x >= 0)
    
    def relu_derivative(self, x):
        return 1*(x >= 0)
    
    def feedforward(self, X):
        activations = [X] # L + 1, 0,...,L+1
        Z = [None]*(self.num_layers - 1) # L, 1,...,L
        for i in range(self.num_layers - 1): # L times
            Z[i] = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            if i == self.num_layers - 2:
                activations.append(Z[i]) # output is a real number
            else:
                activations.append(self.sigmoid(Z[i]))
        return activations, Z
    
    def eval(self, X):
        a,_ = self.feedforward(X)
        if len(a) == 1:
            return a[-1][0][0] # real number
        else:
            return a[-1] # vector
    
    def backward(self, y, activations, Z, epoch, epochs):
        # Compute gradients
        dZ = [None] * (self.num_layers - 1)
        dZ[-1] = (y - activations[-1])
        
        for l in range(self.num_layers - 2, 0, -1): #L-1,...,1 i.e., L-1 times
            dZ[l-1] = np.dot(self.weights[l].T, dZ[l]) * self.sigmoid_derivative(activations[l]) # this is correct, since lth activation is the l-1th z, in our implementation. z at 0th layer doesnt exist. This means taking -1 on each z in the formulas when implementing in this algorithm. See it as we have removed a None in the z list. Same for W and b. There is no 0th weight and bias.
        
        # Update weights and biases
        for l in range(self.num_layers - 1):
            self.weights[l] += 1/100*np.dot(dZ[l], activations[l].T) * self.learning_rates[(len(self.learning_rates)*epoch - 1)//epochs]
            self.biases[l] += 1/100*np.sum(dZ[l], axis=1, keepdims=True) * self.learning_rates[(len(self.learning_rates)*epoch - 1)//epochs]
    
    def gradient(self, X):
        """
        Compute the gradients for of the neural network evalueted at X outputting a real number,
        with sigmoid activation functions for layers 1,...,L-1 and linear activation function for layer L.
        This is the same as backpropagation with cost function a[L].

        Args:
        - X (ndarray): Column vector input to the neural network of dim (n,1).

        Returns:
        - weight_grads_l (list, len = L): List of gradients wrt the weights for each layer evaluated at X.
        - bias_grads (list, len = L):  List of gradients wrt the biases for each layer evaluated at X.
        """

        activations, z = self.feedforward(X) # X dependance
        dZ = [None] * (self.num_layers - 1) # L values
        dZ[-1] = np.array([[1]]) # L th layer derivative is 1 in the case of no (i.e., linear) activation function in last layer.
        
        for l in range(self.num_layers - 2, 0, -1): # L-1 to 1
            dZ[l-1] = np.dot(self.weights[l].T, dZ[l]) * self.sigmoid_derivative(activations[l])
        
        # Append derivatives
        for l in range(self.num_layers - 1):
            self.weight_derivatives[l] =  np.dot(dZ[l], activations[l].T) # (n_{l-1}, n_l)
            self.bias_derivatives[l] = dZ[l]
        return self.weight_derivatives, self.bias_derivatives
    

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            activations, Z = self.feedforward(X)
            assert activations[-1].shape == y.shape
            self.backward(y, activations, Z, epoch, epochs)
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - activations[-1]))
                print(f'Epoch {epoch}: Loss = {loss}')

# Example usage:

#y = np.array([[0], [1], [1], [0]]).T

# Define layer sizes (including input and output layers)
if __name__ == "__main__":
    layer_sizes = [1, 100, 50, 10, 5, 1]

    X = np.linspace(-2*np.pi,2*np.pi, 100).reshape(1,-1)
    y = np.sin(X)
    nn = NeuralNetwork(layer_sizes)
    nn.train(X, y, 10000)
    y_nn = nn.eval(X)
    plt.figure()
    plt.plot(X[0], y[0])
    plt.plot(X[0], y_nn[0])
    plt.grid()
    plt.show()

    # Test the trained network
    print("Output after training:")
    print(nn.eval(X))
    #print(nn.weight_derivatives)
    #print(nn.bias_derivatives)
