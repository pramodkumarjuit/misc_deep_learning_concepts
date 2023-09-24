#
# This is example code is to print initial wt & bias before/after optimizer.step().
# We compare the output manually.
#

import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(3, 2, bias=False)
        #self.fc1.weight.data.fill_(0.01)
        # self.fc1.bias.data.fill_(0.01)
        #nn.init.zeros_(self.fc1.weight)
        #nn.init.zeros_(self.fc1.bias)

    def forward(self, x):
        x = self.fc1(x)
        #x = self.fc2(x)
        return x

# Create an instance of the model
model = MyModel()

w=0
b=0

# Iterate over the named parameters and print initial values
print('==== print initial values ====')
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Parameter name: {name}")
        if name == 'fc1.weight':
            w = param.data.clone().detach() # clone is must otherwise w point param.data which later gets updated after optimizer.step()
        elif name == 'fc1.bias':
            b = param.data.clone().detach()
        print(f"Initial wt : {param.data}")

        print("-" * 20)

print("=" * 20)

# Define an example input and target
#torch.seed()
input_data = torch.randn(1, 3)
target = torch.randn(1, 2)

# Perform a forward pass
output = model(input_data)

print('input: ', input_data)
print('output: ', output)
print('')


# Compute the loss (e.g., using Mean Squared Error)
loss_fn = nn.MSELoss()
loss = loss_fn(output, target)

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Backpropagate and compute gradients
loss.backward()

# Update the parameters
optimizer.step()  # updates the learnable params for this experiment

dw=0
#=torch.zeros(3,3)
print('==== print after optimizer.step() ====')
# Print the loss and gradients of trainable parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Parameter name: {name}")
        print(f"Updated value: {param.data}")
        print(f"Loss: {loss.item()}")
        print(f"Gradients: {param.grad}")
        dw=param.grad
        print("-" * 20)


# Equivalent operation
x = input_data
# print(f'x: {x}')
# print(f'w: {w}')
# print(f'b: {b}')
# print(f'dw: {dw}')

# J = MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x(w*x - y)
# dJ/dw = 1/N * 2*x*y_pred
# def gradient(x, y, y_pred):
#     return np.mean(2*x*(y_pred - y))

# update weights
#w -= learning_rate * dw

# lr_dw = torch.mul(dw, float(0.01))
# print('W:', torch.sub(w,lr_dw))

optimizer.zero_grad()

##################################
output_manual = torch.matmul(x, w.t()) + b
print(f'output_manual: {output_manual}')

if torch.allclose(output, output_manual, atol=1e-4):
    print('### Pass: output & output_manual ###')
else:
    print('!!! MISMATCH: output & output_manual !!!')

