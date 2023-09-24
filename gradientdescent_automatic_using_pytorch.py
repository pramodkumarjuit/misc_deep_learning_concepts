#
# Automatic Gradient Calculation
#

import torch
import torch.nn as nn

# A simple linear regression
# f = w * x

# f = 2 * x
# Training samples
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# Weights to optimize
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# Model/forward function
def forward(x):
    return w * x

print(f'Prediction before training: f(5) = {forward(5).item():.3f}')

# Training hyper parameter
learning_rate = 0.009
n_iters = 100

#learning_rate = 0.001
# n_iters = 20  --> change it to 100/150/200/250....see the impact.

# loss function
loss = nn.MSELoss()

# optimizer
optimizer = torch.optim.SGD([w], lr=learning_rate)

# Training loop
for epoch in range(n_iters):
    # predict = forward pass
    y_predicted = forward(X)

    # loss
    l = loss(Y, y_predicted)

    # calculate gradients = backward pass
    l.backward()

    # update weights
    optimizer.step()

    # zero the gradients after updating
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print('epoch ', epoch+1, ': w = ', w, ' loss = ', l.item() )

#print(f'Prediction after training: f(5) = {forward(5).item():.3f}')

######################## Inference ########################
print(f'Prediction after training: f(5) = {forward(5):.4f}')
print(f'Prediction after training: f(150) = {forward(150):.4f}')
print(f'Prediction after training: f(12.32) = {forward(12.32):.4f}')

