# activation functions and their derivatives

# --relu--
def relu(self, z):
    return np.maximum(z, 0)

def relu_derivative(self, z):
    return np.where(z > 0, 1, 0)

# --sigmoid--
def sigmoid(self, z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(self, z):
    return self.sigmoid(z) * (1 - self.sigmoid(z))

# --tanh--
def tanh(self, z):
    return np.tanh(z)

def tanh_derivative(self, z):
    return 1 - np.tanh(z) ** 2

# --softmax--
def softmax(self, z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)

def softmax_derivative(self, z):
    return self.softmax(z) * (1 - self.softmax(z))

# --linear--
def linear(self, z):
    return z

def linear_derivative(self, z):
    return 1
