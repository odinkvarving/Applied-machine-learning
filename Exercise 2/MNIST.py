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
        self.W = torch.tensor(torch.zeros([784, 10]), requires_grad=True)
        self.b = torch.tensor(torch.zeros([10]), requires_grad=True)

    def f(self, x):
        return torch.softmax(self.logits(x), dim=-1)

    def logits(self, x):
        return x @ self.W + self.b

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.f(x), y)


model = Softmax()

optimizer = torch.optim.SGD([model.W, model.b], 0.0007)
for epoch in range(1000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()
    if(epoch % 100 == 0):
        print("epoch %s, loss %s acc %s " % (epoch, model.loss(x_train, y_train).item(), model.accuracy(x_test, y_test).item()))

print("W = %s, b = %s, loss = %s, accuracy = %s" % (model.W, model.b, model.loss(x_train, y_train), model.accuracy(x_test, y_test)))

model.W = model.W.detach()
model.b = model.b.detach()

for i in range(10):
    plt.imsave('W %s .png' % (i), model.W[:, i].reshape(28, 28))

