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

# CrossEntropyLoss function calculates a single loss value
# that represents the average loss across all the inputs in the batch.
# If you want to compute the total loss for a specific batch,
# you would simply multiply this average loss by the batch size (3 here).

criterion = nn.CrossEntropyLoss()

# Calculate the loss
loss = criterion(predicted_scores, true_labels)
print("Cross-Entropy Loss:", loss.item())

# How to print loss for each of the input in a batch?
# Calculate individual losses for each input
individual_losses = []

# Iterate over the batch
for i in range(len(true_labels)):
    # Get the true label and predicted scores for the current input
    true_label = true_labels[i]
    predicted_score = predicted_scores[i]

    # Calculate the loss for the current input
    loss = criterion(predicted_score.unsqueeze(0), torch.tensor([true_label]))

    # Append the loss to the list
    individual_losses.append(loss.item())

# Print individual losses
for i, loss in enumerate(individual_losses):
    print(f"Loss for input {i}: {loss}")
