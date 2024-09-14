import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(2,2) # kernel_size, stride
        self.ln1 = nn.Linear(30,10)
        self.ln2 = nn.Linear(10,5)
        self.ln3 = nn.Linear(5,1)

    def forward(self, x):
        x1 = self.conv1(x)
        #print(f'x1 shape: {x1.shape}') # 10,60,60
        x2 = self.pool(x1)
        #print(f'x2 shape: {x2.shape}') # 10,30,30
        x3 = self.ln1(x2)
        #print(f'x3 shape: {x3.shape}') # 10,30,10
        x4 = self.ln2(x3)
        #print(f'x4 shape: {x4.shape}') # 10,30,5
        x5 = self.ln3(x4)
        #print(f'x5 shape: {x5.shape}') # 10,30,1
        return x5

MyToyModel = ToyModel()

input_tensor = torch.rand([1,64,64])
#print(input_tensor)
#print(MyToyModel(input_tensor))

summary(MyToyModel, (1,64,64))

## training code
criterion = nn.MSELoss()

# stochastic gradient descent (SGD)
# momentum=0.9: this means that the new velocity will be a weighted sum
# of 90% of the previous velocity and 10% of the current gradient.
# This can help the optimizer to avoid getting stuck in local minima and converge more quickly.
optimizer = optim.SGD(MyToyModel.parameters(), lr=0.001, momentum=0.9)
epochs = 10
steps = 12
for epoch in range(epochs):
    for i in range(steps): # for i, (x, y) in enumerate(train_loader): # x=image, y=label
        x = torch.rand([1,64,64])
        y = torch.rand([10,30,1])
        optimizer.zero_grad()
        y_pred = MyToyModel(x)
        loss = criterion(y, y_pred) 
        #computes the gradient of the loss function with respect to the model's parameters.
        loss.backward()

        # optimizer work:
        #  - update the model's parameters 
        #  - it incorporates the previous update direction into the current update (if momentum is used)
        #  - Learning Rate Adjustment (if applicable). Adam dynamically adjust LR
        #  - Weight Decay (if applicable): also known as L2 regularization, penalizing large weights.
        optimizer.step()

        print(f'Epoch:{epoch}/{epochs} step:{i}/{steps} loss: {loss.item():.4}')

