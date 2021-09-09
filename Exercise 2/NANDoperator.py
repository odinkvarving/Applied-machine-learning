import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, art3d

class NANDOperatorModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0], [0.0]], dtype=torch.float32, requires_grad=True)
        self.b = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    # Predictor
    def f(self, x):
        return torch.sigmoid(self.logits(x))

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)


model = NANDOperatorModel()

# Observed/training input and output
x_train = torch.tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).astype(np.float32), requires_grad=True)
y_train = torch.tensor(np.array([[0], [1], [1], [1]]).astype(np.float32), requires_grad=True)

optimizer = torch.optim.SGD([model.W, model.b], 0.10, momentum=0.99)

for epoch in range(10000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

fig = plt.figure("Logistic regression: the logical NAND operator")

plot1 = fig.add_subplot(111, projection='3d')

plot1_f = plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green", label="$\\hat y=f(\\mathbf{x})=\\sigma(\\mathbf{xW}+b)$")

plot1.plot(x_train[:, 0].detach().numpy().squeeze(), x_train[:, 1].detach().numpy().squeeze(), y_train[:, 0].detach().numpy().squeeze(), 'o', label="$(x_1^{(i)}, x_2^{(i)},y^{(i)})$", color="blue")

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
fig.canvas.mpl_connect('key_press_event', update_figure)

plt.show()

