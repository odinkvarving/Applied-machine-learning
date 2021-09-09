import torch
import numpy as np
import matplotlib.pyplot as plt

W1_init = torch.tensor([[10.0, -10.0], [10.0, -10.0]])
b1_init = torch.tensor([[-5.0, 15.0]])
W2_init = torch.tensor([[10.0], [10.0]])
b2_init = torch.tensor([[-15.0]])


class XOROperatorModel:
    def __init__(self, W1 = W1_init.clone(), W2 = W2_init.clone(), b1 = b1_init.clone(), b2 = b2_init.clone()):
        self.W1 = W1
        self.W2 = W2
        self.b1 = b1
        self.b2 = b2

    # Predictor
    def f1(self, x):
        return torch.sigmoid(x @ self.W1 + self.b1)

    def f2(self, x):
        return torch.sigmoid(x @ self.W2 + self.b2)

    def f(self, x):
        return self.f2(self.f1(x))

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.f(x), y)


model = XOROperatorModel()

# Observed/training input and output
x_train = torch.tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).astype(np.float32), requires_grad=True)
y_train = torch.tensor(np.array([[0], [1], [1], [0]]).astype(np.float32), requires_grad=True)

optimizer = torch.optim.SGD([model.W1, model.W2, model.b1, model.b2], 0.10, momentum=0.99)

for epoch in range(10000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

print("W1 = %s, W2 = %s, b1 = %s, b2 = %s, loss = %s" % (model.W1, model.W2, model.b1, model.b2, model.loss(x_train, y_train)))

fig = plt.figure("Logistic regression: the logical XOR operator")

plot1 = fig.add_subplot(111, projection='3d')

plot1_f = plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green",
                               label="$\\hat y=f(\\mathbf{x})=\\sigma(\\mathbf{xW}+b)$")

plot1.plot(x_train[:, 0].detach().numpy().squeeze(), x_train[:, 1].detach().numpy().squeeze(),
           y_train[:, 0].detach().numpy().squeeze(), 'o', label="$(x_1^{(i)}, x_2^{(i)},y^{(i)})$", color="blue")

plot1_info = fig.text(0.01, 0.02, "")

plot1.set_xlabel("$x_1$")
plot1.set_ylabel("$x_2$")
plot1.set_zlabel("$y$")
plot1.legend(loc="upper left")
plot1.set_xticks([0, 1])
plot1.set_yticks([0, 1])
plot1.set_zticks([0, 1])
plot1.set_xlim(-0.25, 1.25)
plot1.set_ylim(-0.25, 1.25)
plot1.set_zlim(-0.25, 1.25)

def update_figure(event=None):
    global plot1_f
    plot1_f.remove()
    x1_grid, x2_grid = np.meshgrid(np.linspace(-0.25, 1.25, 10), np.linspace(-0.25, 1.25, 10))
    y_grid = np.empty([10, 10])
    for i in range(0, x1_grid.shape[0]):
        for j in range(0, x1_grid.shape[1]):
            y_grid[i, j] = model.f(torch.tensor(np.array([[x1_grid[i, j], x2_grid[i, j]]]).astype(np.float32)))
    plot1_f = plot1.plot_wireframe(x1_grid, x2_grid, y_grid, color="green")

update_figure()

plt.show()
