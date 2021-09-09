import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 784).float()  # Reshape input
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 784).float()  # Reshape input
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

class Softmax:
    def __init__(self):
        self.W = torch.tensor(np.random.rand([784, 10]))
        self.b = torch.tensor(np.random.rand(10))

    def f(self, x):
        return torch.softmax(x @ self.W + self.b)

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

model = Softmax()

optimizer = torch.optim.SGD([model.W, model.b], 0.10, momentum=0.99)



# Show the input of the first observation in the training set
plt.imshow(x_train[0, :].reshape(28, 28))

# Print the classification of the first observation in the training set
print(y_train[0, :])

# Save the input of the first observation in the training set
plt.imsave('x_train_1.png', x_train[0, :].reshape(28, 28))

plt.show()


