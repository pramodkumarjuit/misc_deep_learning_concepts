#
# Manual Gradient Calculation
#
import numpy as np


# A simple linear regression
# f = w * x

# f = 2 * x
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

print('X', X)
print('Y', Y)

# Weights initializaton
#   1. inital value of weight, doesn't matter much
#   2. w = 5.0 is also fine
w = 0.1

# model output
def forward(x):
    return w * x

# loss = MSE (Mean Square Error)
# 1. error = |y_pred - y|
# 2. Square of error: (y_pred - y)^2
# 3. Take mean of that
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

# Use gradient descent to minimize/reduce the loss
# J = MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x(w*x - y) = 1/N * 2x(y_pred - y)
def gradient(x, y, y_pred):
    return np.dot(2*x, y_pred - y).mean()

# Single prediction
print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training Hyper parameter
learning_rate = 0.001
n_iters = 20

# What if learning_rate is set as learning_rate = 0.001 ?
#   - Small steps hence slow convergence
#   - Need more time to converge, meaning need to train for more iters

#learning_rate = 0.001
# n_iters = 20  --> change it to 100/150/200/250....see the impact.


for epoch in range(n_iters):
    # predict = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # calculate gradients
    dw = gradient(X, Y, y_pred)
    print(f'dw={dw}')

    # update weights
    w = w - learning_rate * dw

    # print at every 2 epoch
    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')


######################## Inference ########################
print(f'Prediction after training: f(5) = {forward(5):.4f}')

print(f'Prediction after training: f(150) = {forward(150):.4f}')

print(f'Prediction after training: f(12.32) = {forward(12.32):.4f}')
