import torch
import torch.nn as nn
import torch.optim as optim

# toy dataset
#  - We have 3 samples and 3 classes
#  - The true labels are represented as class indices
true_labels = torch.tensor([2, 0, 1], dtype=torch.long)

# Example predicted scores (logits) from a neural network
# These scores are often the output of the final layer of the network
# before the softmax
predicted_scores = torch.tensor([
    [0.2, 0.5, 0.1, 0.2],
    [0.8, 0.1, 0.05, 0.05],
    [0.3, 0.2, 0.4, 0.1]
], dtype=torch.float32)

# Define the CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

# Calculate the loss
loss = criterion(predicted_scores, true_labels)

print("Cross-Entropy Loss:", loss.item())
