import torch
import matplotlib.pyplot as plt
import numpy as np


class LinearRegressionModel2D:
    def __init__(self):
        # Model variables
        self.W = torch.zeros(1,1, dtype=torch.float32, requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.zeros(1,1, dtype=torch.float32, requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))  # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numerical stability


def visualize2D(data_list):
    model = LinearRegressionModel2D()
    x_data = []
    y_data = []
    header = data_list.pop(0)

    for row in data_list:
        x_data.append(float(row[0]))
        y_data.append(float(row[1]))

    x_train = torch.tensor(x_data).reshape(-1, 1)
    y_train = torch.tensor(y_data).reshape(-1, 1)

    # Optimize: adjust W and b to minimize loss using stochastic gradient descent
    optimizer = torch.optim.SGD([model.W, model.b], 0.00015)
    for epoch in range(100000):
        model.loss(x_train, y_train).backward()
        optimizer.step()
        optimizer.zero_grad()

    # Print model variables and loss
    print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

    # Visualize result
    plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
    plt.xlabel(header[0])
    plt.ylabel(header[1])
    x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])  # x = [[1], [6]]]
    plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = xW+b$')
    plt.legend()
    plt.show()


class linearRegression3D:
    def __init__(self):
        # Model variables
        self.W = torch.zeros(2,1, dtype=torch.float32, requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.zeros(1,1, dtype=torch.float32, requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))  # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numerical stability


def visualize3D(data_list):
    model = linearRegression3D()
    x_data = []
    y_data = []
    header = data_list.pop(0)

    for row in data_list:
        x_data.append([float(row[1]), float(row[2])])
        y_data.append(float(row[0]))

    x_train = torch.tensor(x_data, dtype=torch.float32).reshape(-1, 2)
    y_train = torch.tensor(y_data, dtype=torch.float32).reshape(-1, 1)

    optimizer = torch.optim.SGD([model.W, model.b], 0.00011)
    for epoch in range(1000000):
        model.loss(x_train, y_train).backward()
        optimizer.step()
        optimizer.zero_grad()

    # Print model variables and loss
    print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

    x_vector = x_train[:, 0]
    y_vector = x_train[:, 1]
    z_vector = y_train[:, 0]

    fig = plt.figure("3D Linear regression")

    plot1 = fig.add_subplot(111, projection='3d')
    plot1.plot3D(x_vector.numpy(), y_vector.numpy(),
                 z_vector.numpy(), 'o')

    x = np.linspace(torch.min(x_vector), torch.max(x_vector), 10)
    y = np.linspace(torch.min(y_vector), torch.max(y_vector), 10)

    X, Y = np.meshgrid(x, y)
    Z = np.empty([10, 10])
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            Z[i, j] = model.f(torch.tensor(
                [[X[i, j], Y[i, j]]], dtype=torch.float32)).detach().numpy()
    plot1.plot_wireframe(X, Y, Z, color="green")

    plt.show()


def runLinearRegression(data_list):
    if len(data_list[0]) == 2:
        visualize2D(data_list)

    if len(data_list[0]) == 3:
        visualize3D(data_list)
